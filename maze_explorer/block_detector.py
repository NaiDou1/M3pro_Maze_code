"""四色方块检测与三维定位。

从同步的 RGB-D 帧中检测红、绿、黄、蓝方块，并解算其相对车体 ``base_link`` 的
三维坐标，供对位与机械臂 IK 使用。

彩色与深度未配准
----------------
相机 ``depth_registration`` 为假，且彩色 640x480@30fps、深度 640x400@10fps，
两者像素不一一对应。本模块沿用现有 demo 的近似做法——按分辨率比例把彩色像素
映射到深度图坐标；取样时在窗口内取中位数，比现有 demo 的单点取值更抗噪。

坐标解算
--------
像素加深度转相机光学坐标系，z 向前、x 向右、y 向下::

    camera_x = (u - center_x) * depth / focal_x
    camera_y = (v - center_y) * depth / focal_y
    camera_z = depth

再用安装外参 ``mount_xyz`` 与 ``mount_rpy`` 变换到 ``base_link``。

.. warning::
   ``mount_rpy`` 随机械臂姿态变化，相机装在第 4 连杆上。只有在巡线姿态下完成
   定位且外参已标定时，变换结果才可信。未标定时只用 ``distance_m`` 与
   ``lateral_m``，它们在相机系内自洽，不依赖外参。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from maze_explorer._compat import StrEnum

#: 默认相机内参，来自现有 demo 标定，针对 640x480 彩色图，焦距与主点单位 px
DEFAULT_CAMERA_MATRIX = (
    (477.57421875, 0.0, 319.3820495605469),
    (0.0, 477.55718994140625, 238.64108276367188),
    (0.0, 0.0, 1.0),
)

#: 毫米换算到米的系数
MILLIMETERS_PER_METER = 1000.0

#: 调试节点打印检测摘要的周期，单位 s
REPORT_PERIOD_SEC = 2.0


class BlockColor(StrEnum):
    """方块颜色，取值 red、green、blue、yellow 四者之一。

    比赛规则不允许贴标记，识别只能走颜色路线；阈值来自
    ``config/hsv_params.yaml``，须现场重标。
    """

    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'


#: 全部合法颜色，顺序固定，供参数键与统计表按同一顺序遍历
COLOR_NAMES: Tuple[BlockColor, ...] = (
    BlockColor.RED,
    BlockColor.GREEN,
    BlockColor.BLUE,
    BlockColor.YELLOW,
)


@dataclass
class BlockDetection:
    """单个方块的一次观测。

    :ivar color: 方块颜色，取值见 ``BlockColor``。
    :ivar pixel_x: 彩色图上的质心横坐标，单位 px。
    :ivar pixel_y: 彩色图上的质心纵坐标，单位 px。
    :ivar area: 轮廓面积，单位像素平方，小于 ``min_area`` 的轮廓被忽略。
    :ivar distance_m: 沿相机光轴的深度，单位 m，由深度图中位数解出。
    :ivar lateral_m: 相机系横向偏移，单位 m，正为右。
    :ivar vertical_m: 相机系纵向偏移，单位 m，正为下。
    :ivar yaw_rad: 由 minAreaRect 估计的方块朝向，单位 rad，缺失时为 ``None``。
    :ivar position_base: 变换到 base_link 的坐标三分量 x 前、y 左、z 上，
        单位 m；外参未标定时为 ``None``。
    """

    color: BlockColor
    #: 质心横坐标，单位 px
    pixel_x: float
    #: 质心纵坐标，单位 px
    pixel_y: float
    #: 轮廓面积，单位像素平方
    area: float
    #: 光轴深度，单位 m
    distance_m: float
    #: 相机系横向偏移，单位 m，正为右
    lateral_m: float
    #: 相机系纵向偏移，单位 m，正为下
    vertical_m: float
    #: 方块朝向，单位 rad，缺失时缺省
    yaw_rad: Optional[float] = None
    #: base_link 坐标三分量，单位 m，外参未标定时缺省
    position_base: Optional[Tuple[float, float, float]] = None

    def horizontal_distance(self) -> float:
        """返回相机系 xz 平面内的水平距离，单位 m，用于与抓取包络比较。

        :returns: 光轴深度与横向偏移的欧氏距离，不小于 0。
        """
        return math.hypot(self.distance_m, self.lateral_m)


def euler_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """把 ZYX 内旋欧拉角转成 3x3 旋转矩阵。

    :param rpy: 依次为 roll、pitch、yaw，单位 rad，长度 3。
    :returns: 3x3 浮点数组，行列式为 1 的正交矩阵。
    """
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cos_roll, sin_roll = math.cos(roll), math.sin(roll)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [
                cos_yaw * cos_pitch,
                cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
            ],
            [
                sin_yaw * cos_pitch,
                sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
            ],
            [
                -sin_pitch,
                cos_pitch * sin_roll,
                cos_pitch * cos_roll,
            ],
        ],
        dtype=np.float64,
    )


class BlockDetector:
    """四色方块检测与定位，不依赖 ROS，可离线单测。"""

    def __init__(
        self,
        hsv_map: Dict[str, Sequence[int]],
        camera_matrix: Sequence[Sequence[float]] = DEFAULT_CAMERA_MATRIX,
        min_area: int = 300,
        morph_kernel: int = 5,
        depth_window: int = 1,
        #: 相机相对 base_link 的位置三分量，单位 m，x 前 y 左 z 上
        mount_xyz: Sequence[float] = (0.10, 0.0, 0.35),
        #: 相机相对 base_link 的姿态三分量，单位 rad，ZYX 内旋
        mount_rpy: Sequence[float] = (0.0, 0.0, 0.0),
        mount_calibrated: bool = False,
    ) -> None:
        """配置颜色阈值、相机内参与安装外参。

        :param hsv_map: 颜色到 HSV 六元组的映射，键取值见 ``BlockColor``。
            H 取值 0 到 180，S 与 V 取值 0 到 255；缺键的颜色不参与检测。
        :param camera_matrix: 3x3 相机内参矩阵，焦距与主点单位 px。
        :param min_area: 判定方块所需的最小轮廓面积，单位像素平方。
        :param morph_kernel: 形态学闭运算核的边长，单位像素，不小于 1。
        :param depth_window: 深度取样的半窗口边长，单位像素，不小于 0。
        :param mount_xyz: 相机相对 base_link 的平移，单位 m，须现场标定。
        :param mount_rpy: 相机相对 base_link 的欧拉角，单位 rad，须现场标定。
        :param mount_calibrated: 外参是否已标定。为假时不输出 ``position_base``。
        """
        self._hsv: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in COLOR_NAMES:
            if name not in hsv_map:
                continue
            values = [int(v) for v in hsv_map[name]]
            self._hsv[name] = (
                np.array(values[:3], dtype=np.uint8),
                np.array(values[3:], dtype=np.uint8),
            )
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self._focal_x = float(self._camera_matrix[0, 0])
        self._focal_y = float(self._camera_matrix[1, 1])
        self._principal_x = float(self._camera_matrix[0, 2])
        self._principal_y = float(self._camera_matrix[1, 2])
        self._min_area = int(min_area)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(morph_kernel), int(morph_kernel))
        )
        self._window_radius = max(0, int(depth_window))

        self._rotation = euler_to_matrix(mount_rpy)
        self._translation = np.asarray(mount_xyz, dtype=np.float64)
        self.mount_calibrated = bool(mount_calibrated)

    # ------------------------------------------------------------------ 检测

    def detect(self, bgr: np.ndarray, depth_mm: np.ndarray) -> List[BlockDetection]:
        """检测一帧中的全部方块，按距离由近到远排序。

        :param bgr: 彩色图，BGR 三通道，典型尺寸 640x480。
        :param depth_mm: 深度图，float32，单位 mm，典型尺寸 640x400，零与非
            有限值视为无效深度。
        :returns: 检出的方块列表，空图或无有效深度时为空列表。
        """
        if bgr is None or bgr.size == 0:
            return []

        hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        results: List[BlockDetection] = []

        for name, (lower, upper) in self._hsv.items():
            mask = cv2.inRange(hsv_img, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self._min_area:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] <= 0:
                    continue
                pixel_x = moments['m10'] / moments['m00']
                pixel_y = moments['m01'] / moments['m00']

                depth_m = self._sample_depth(depth_mm, bgr.shape, pixel_x, pixel_y)
                if depth_m is None:
                    continue

                results.append(
                    self._localize(name, pixel_x, pixel_y, area, depth_m, contour)
                )

        # 距离由近到远，便于优先处理最近目标
        results.sort(key=lambda detection: detection.distance_m)
        return results

    def _sample_depth(
        self,
        depth_mm: np.ndarray,
        color_shape: Tuple[int, ...],
        pixel_x: float,
        pixel_y: float,
    ) -> Optional[float]:
        """把彩色像素映射到深度图坐标，取窗口内有效深度的中位数。

        彩色与深度分辨率不同，按宽高比例换算坐标；窗口内先剔除非有限值与非
        正值再取中位数，比单点取值抗深度噪声。

        :param depth_mm: 深度图，float32，单位 mm。
        :param color_shape: 彩色图尺寸，高在前宽在后。
        :param pixel_x: 彩色图质心横坐标，单位 px。
        :param pixel_y: 彩色图质心纵坐标，单位 px。
        :returns: 深度值，单位 m；越界或窗口内无有效值时为 ``None``。
        """
        if depth_mm is None or depth_mm.size == 0:
            return None
        depth_height, depth_width = depth_mm.shape[:2]
        color_height, color_width = color_shape[:2]
        depth_x = int(round(pixel_x * depth_width / color_width))
        depth_y = int(round(pixel_y * depth_height / color_height))
        if not (0 <= depth_x < depth_width and 0 <= depth_y < depth_height):
            return None

        radius = self._window_radius
        window = depth_mm[
            max(0, depth_y - radius):depth_y + radius + 1,
            max(0, depth_x - radius):depth_x + radius + 1,
        ]
        valid = window[np.isfinite(window)]
        valid = valid[valid > 0.0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / MILLIMETERS_PER_METER

    def _localize(
        self,
        color: BlockColor,
        pixel_x: float,
        pixel_y: float,
        area: float,
        depth_m: float,
        contour,
    ) -> BlockDetection:
        """由像素与深度解算相机系坐标，外参已标定时再转 base 系。

        :param color: 方块颜色，取值见 ``BlockColor``。
        :param pixel_x: 彩色图质心横坐标，单位 px。
        :param pixel_y: 彩色图质心纵坐标，单位 px。
        :param area: 轮廓面积，单位像素平方。
        :param depth_m: 该处光轴深度，单位 m。
        :param contour: 轮廓点集，供估计方块朝向。
        :returns: 含相机系量的观测；``mount_calibrated`` 为假时
            ``position_base`` 为 ``None``。
        """
        camera_x = (pixel_x - self._principal_x) * depth_m / self._focal_x
        camera_y = (pixel_y - self._principal_y) * depth_m / self._focal_y
        camera_z = depth_m

        rect = cv2.minAreaRect(contour)
        yaw_rad = math.radians(rect[2])

        position_base: Optional[Tuple[float, float, float]] = None
        if self.mount_calibrated:
            point = (
                self._rotation
                @ np.array([camera_x, camera_y, camera_z], dtype=np.float64)
                + self._translation
            )
            position_base = (float(point[0]), float(point[1]), float(point[2]))

        return BlockDetection(
            color=color,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            area=area,
            distance_m=camera_z,
            lateral_m=camera_x,
            vertical_m=camera_y,
            yaw_rad=yaw_rad,
            position_base=position_base,
        )

    # ------------------------------------------------------------ 便捷查询

    @staticmethod
    def nearest(
        detections: Sequence[BlockDetection], color: Optional[BlockColor] = None
    ) -> Optional[BlockDetection]:
        """返回最近的方块，指定颜色时只在该颜色内筛选。

        :param detections: 待筛选的观测列表。
        :param color: 颜色过滤条件，取 ``None`` 表示不限颜色。
        :returns: 距离最小的观测；候选为空时为 ``None``。
        """
        pool = [
            detection
            for detection in detections
            if color is None or detection.color == color
        ]
        if not pool:
            return None
        return min(pool, key=lambda detection: detection.distance_m)

    @staticmethod
    def in_grasp_envelope(
        detection: BlockDetection, dist_min: float, dist_max: float
    ) -> bool:
        """判断方块是否落在机械臂抓取包络的水平距离区间内。

        :param detection: 待判断观测。
        :param dist_min: 包络下界，单位 m，机械臂低于该距离抓不到。
        :param dist_max: 包络上界，单位 m，超出则对位无效。
        :returns: 水平距离落在闭区间内时为真。
        """
        distance = detection.horizontal_distance()
        return dist_min <= distance <= dist_max


class BlockDetectorNode(Node):
    """独立调试节点，订阅 RGB-D 并打印检测结果。"""

    def __init__(self) -> None:
        super().__init__('block_detector')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('red_hsv', [0, 154, 107, 184, 253, 255])
        self.declare_parameter('green_hsv', [34, 130, 205, 125, 253, 255])
        self.declare_parameter('blue_hsv', [55, 196, 137, 125, 253, 255])
        self.declare_parameter('yellow_hsv', [23, 99, 235, 125, 253, 255])
        self.declare_parameter('min_contour_area', 300)
        self.declare_parameter('mount_xyz', [0.10, 0.0, 0.35])
        self.declare_parameter('mount_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('mount_calibrated', False)
        self.declare_parameter('sync_slop_sec', 0.08)

        hsv_map = {
            name: list(self.get_parameter(f'{name}_hsv').value)
            for name in COLOR_NAMES
        }
        self._det = BlockDetector(
            hsv_map=hsv_map,
            min_area=int(self.get_parameter('min_contour_area').value),
            mount_xyz=list(self.get_parameter('mount_xyz').value),
            mount_rpy=list(self.get_parameter('mount_rpy').value),
            mount_calibrated=bool(self.get_parameter('mount_calibrated').value),
        )
        self._bridge = CvBridge()

        import message_filters  # 局部导入，避免无相机环境下的 import 开销

        color_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('color_topic').value)
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('depth_topic').value)
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=1,
            slop=float(self.get_parameter('sync_slop_sec').value),
        )
        self._sync.registerCallback(self._on_rgbd)
        self.create_timer(REPORT_PERIOD_SEC, self._report)
        self._last_count = 0

        if not self._det.mount_calibrated:
            self.get_logger().warn(
                'mount_calibrated=false：只输出相机系量 distance_m 与 lateral_m，'
                'position_base 为 None。需现场标定相机外参后置 true'
            )
        self.get_logger().info(f'BlockDetector 就绪 | 颜色 {list(hsv_map)}')

    def _on_rgbd(self, color_msg: Image, depth_msg: Image) -> None:
        """转换同步帧、执行检测并打印摘要。

        :param color_msg: 彩色图消息。
        :param depth_msg: 与之时间配准的深度图消息。
        """
        try:
            bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'RGB-D 转换失败：{exc}', throttle_duration_sec=5.0)
            return

        detections = self._det.detect(bgr, np.asarray(depth, dtype=np.float32))
        self._last_count = len(detections)
        if detections:
            summary = ', '.join(
                f'{detection.color}@{detection.distance_m:.2f}m'
                f'/{(detection.lateral_m * 100):+.0f}cm'
                for detection in detections
            )
            self.get_logger().info(f'检测到 {len(detections)} 个方块：{summary}')

    def _report(self) -> None:
        """无方块时按节流周期提示一次。"""
        if self._last_count == 0:
            self.get_logger().info('当前视野内无方块', throttle_duration_sec=10.0)


def main(args: Optional[List[str]] = None) -> None:
    """调试节点入口：初始化、自旋、退出时按序关闭。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
    rclpy.init(args=args)
    node = BlockDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
