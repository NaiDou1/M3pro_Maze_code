"""四色方块检测与三维定位。

从同步的 RGB-D 帧中检测红/绿/黄/蓝方块，并解算其相对车体 ``base_link`` 的
三维坐标，供对位与机械臂 IK 使用。

彩色与深度**未配准**
--------------------
相机 ``depth_registration=false``，且彩色 640x480@30、深度 640x400@10，两者像素
不一一对应。本模块沿用现有 demo 的近似做法——按分辨率比例把彩色像素映射到深度
图坐标；但取样时在 3x3 窗口内取**中位数**，比现有 demo 的单点取值更抗噪。

坐标解算
--------
像素 + 深度 → 相机光学坐标系（z 向前、x 向右、y 向下）：

.. math::
    X_c = (u - c_x) \\cdot d / f_x, \\quad
    Y_c = (v - c_y) \\cdot d / f_y, \\quad
    Z_c = d

再用安装外参 ``mount_xyz`` / ``mount_rpy`` 变换到 ``base_link``。

.. warning::
   ``mount_rpy`` 随机械臂姿态变化（相机装在 4 连杆上）。只有**巡线姿态**
   下完成定位、且外参已标定，转换结果才可信。未标定时请只用
   ``distance_m``（光轴深度）与 ``lateral_m``（相机系横向），它们在相机系
   内自洽，不依赖外参。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

#: 默认相机内参（现有 demo 已标定，针对 640x480 彩色图）
DEFAULT_CAMERA_MATRIX = (
    (477.57421875, 0.0, 319.3820495605469),
    (0.0, 477.55718994140625, 238.64108276367188),
    (0.0, 0.0, 1.0),
)

COLOR_NAMES: Tuple[str, ...] = ('red', 'green', 'blue', 'yellow')


@dataclass
class BlockDetection:
    """单个方块的一次观测。"""

    color: str
    u: float                  # 像素中心 x（彩色图坐标）
    v: float                  # 像素中心 y
    area: float               # 轮廓面积（像素）
    distance_m: float         # 沿相机光轴的深度（m）
    lateral_m: float          # 相机系横向偏移（m），正=右
    vertical_m: float         # 相机系纵向偏移（m），正=下
    yaw_rad: Optional[float] = None  # minAreaRect 估计的方块朝向（弧度）
    #: 变换到 base_link 的坐标 (x 前, y 左, z 上)，外参未标定时为 None
    position_base: Optional[Tuple[float, float, float]] = None

    def horizontal_distance(self) -> float:
        """水平距离（相机系 xz 平面），用于与抓取包络比较。"""
        return math.hypot(self.distance_m, self.lateral_m)


def euler_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """欧拉角（roll, pitch, yaw，ZYX 内旋）转 3x3 旋转矩阵。"""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


class BlockDetector:
    """四色方块检测与定位（不依赖 ROS，可离线单测）。"""

    def __init__(
        self,
        hsv_map: Dict[str, Sequence[int]],
        camera_matrix: Sequence[Sequence[float]] = DEFAULT_CAMERA_MATRIX,
        min_area: int = 300,
        morph_kernel: int = 5,
        depth_window: int = 1,
        #: 相机相对 base_link 的位置 (x, y, z)
        mount_xyz: Sequence[float] = (0.10, 0.0, 0.35),
        #: 相机相对 base_link 的姿态 (roll, pitch, yaw)
        mount_rpy: Sequence[float] = (0.0, 0.0, 0.0),
        mount_calibrated: bool = False,
    ) -> None:
        self._hsv: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in COLOR_NAMES:
            if name not in hsv_map:
                continue
            values = [int(v) for v in hsv_map[name]]
            self._hsv[name] = (
                np.array(values[:3], dtype=np.uint8),
                np.array(values[3:], dtype=np.uint8),
            )
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self._fx = float(self._K[0, 0])
        self._fy = float(self._K[1, 1])
        self._cx = float(self._K[0, 2])
        self._cy = float(self._K[1, 2])
        self._min_area = int(min_area)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(morph_kernel), int(morph_kernel))
        )
        self._win = max(0, int(depth_window))

        self._R = euler_to_matrix(mount_rpy)
        self._t = np.asarray(mount_xyz, dtype=np.float64)
        self.mount_calibrated = bool(mount_calibrated)

    # ------------------------------------------------------------------ 检测

    def detect(self, bgr: np.ndarray, depth_mm: np.ndarray) -> List[BlockDetection]:
        """检测一帧中的所有方块。

        :param bgr: 彩色图（BGR，640x480）。
        :param depth_mm: 深度图（float32，单位 mm，640x400）。
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
                u = moments['m10'] / moments['m00']
                v = moments['m01'] / moments['m00']

                depth_m = self._sample_depth(depth_mm, bgr.shape, u, v)
                if depth_m is None:
                    continue

                detection = self._localize(name, u, v, area, depth_m, contour)
                results.append(detection)

        # 按距离由近到远排序，便于优先处理最近目标
        results.sort(key=lambda d: d.distance_m)
        return results

    def _sample_depth(
        self, depth_mm: np.ndarray, color_shape: Tuple[int, ...], u: float, v: float
    ) -> Optional[float]:
        """把彩色像素映射到深度图坐标，取窗口内的有效中位数（米）。"""
        if depth_mm is None or depth_mm.size == 0:
            return None
        dh, dw = depth_mm.shape[:2]
        ch, cw = color_shape[:2]
        du = int(round(u * dw / cw))
        dv = int(round(v * dh / ch))
        if not (0 <= du < dw and 0 <= dv < dh):
            return None

        w = self._win
        window = depth_mm[max(0, dv - w):dv + w + 1, max(0, du - w):du + w + 1]
        valid = window[np.isfinite(window)]
        valid = valid[valid > 0.0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _localize(
        self, name: str, u: float, v: float, area: float, depth_m: float, contour
    ) -> BlockDetection:
        """由像素与深度解算相机系与 base_link 坐标。"""
        x_c = (u - self._cx) * depth_m / self._fx
        y_c = (v - self._cy) * depth_m / self._fy
        z_c = depth_m

        rect = cv2.minAreaRect(contour)
        yaw = math.radians(rect[2])

        position_base: Optional[Tuple[float, float, float]] = None
        if self.mount_calibrated:
            p = self._R @ np.array([x_c, y_c, z_c], dtype=np.float64) + self._t
            position_base = (float(p[0]), float(p[1]), float(p[2]))

        return BlockDetection(
            color=name,
            u=u,
            v=v,
            area=area,
            distance_m=z_c,
            lateral_m=x_c,
            vertical_m=y_c,
            yaw_rad=yaw,
            position_base=position_base,
        )

    # ------------------------------------------------------------ 便捷查询

    @staticmethod
    def nearest(detections: Sequence[BlockDetection], color: Optional[str] = None) -> Optional[BlockDetection]:
        """返回最近的方块；``color`` 非空时只在指定颜色中筛选。"""
        pool = [d for d in detections if color is None or d.color == color]
        if not pool:
            return None
        return min(pool, key=lambda d: d.distance_m)

    @staticmethod
    def in_grasp_envelope(
        detection: BlockDetection, dist_min: float, dist_max: float
    ) -> bool:
        """判断方块是否落在机械臂抓取包络（水平距离区间）内。"""
        d = detection.horizontal_distance()
        return dist_min <= d <= dist_max


class BlockDetectorNode(Node):
    """独立调试用节点：订阅 RGB-D 并打印检测结果。"""

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
        self.create_timer(2.0, self._report)
        self._last_count = 0

        if not self._det.mount_calibrated:
            self.get_logger().warn(
                'mount_calibrated=false：只输出相机系量（distance_m/lateral_m），'
                'position_base 为 None。需现场标定相机外参后置 true'
            )
        self.get_logger().info(f'BlockDetector 就绪 | 颜色 {list(hsv_map)}')

    def _on_rgbd(self, color_msg: Image, depth_msg: Image) -> None:
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
                f'{d.color}@{d.distance_m:.2f}m/{(d.lateral_m * 100):+.0f}cm'
                for d in detections
            )
            self.get_logger().info(f'检测到 {len(detections)} 个方块：{summary}')

    def _report(self) -> None:
        if self._last_count == 0:
            self.get_logger().info('当前视野内无方块', throttle_duration_sec=10.0)


def main(args=None) -> None:
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
