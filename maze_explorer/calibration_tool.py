"""现场标定工具。

迷宫现场的光照、地面材质、自备方块材质与机械臂装配都与出厂商Demo不同，
以下三项必须现场标定，否则会直接导致循迹跑偏或抓取失败：

============  ==========================================================
模式           作用
============  ==========================================================
``hsv``        鼠标框选采样，标定黑线引导线与红/绿/黄/蓝四色 HSV 阈值
``line_pose``  交互式调整机械臂巡线姿态关节角，使相机同时看到线与方块
``motion``     走格距离与 90 度转角自测，统计里程计误差并复核标定系数
``follow``     纯巡线测试，只靠相机沿黑线走，不依赖挡板与激光拓扑
============  ==========================================================

用法::

    ros2 run maze_explorer calibration_tool --ros-args -p mode:=hsv
    ros2 run maze_explorer calibration_tool --ros-args -p mode:=line_pose
    ros2 run maze_explorer calibration_tool --ros-args -p mode:=motion
    ros2 run maze_explorer calibration_tool --ros-args -p mode:=follow

.. note::
   写回 yaml 时采用按行正则替换而非整文件重写，以保留配置文件中的中文注释。
   需要图形环境即 ``cv2.imshow``，请在桌面终端运行。
"""

from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from arm_msgs.msg import ArmJoints
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

from maze_explorer._compat import StrEnum
from maze_explorer.arm_controller import JOINT_MAX_DEG, JOINT_MIN_DEG
from maze_explorer.base_driver import yaw_from_quaternion
from maze_explorer.grid_mapper import angle_diff
from maze_explorer.sensors import (
    DEFAULT_SCAN_RANGE_MAX_M,
    DEFAULT_SCAN_RANGE_MIN_M,
    SCAN_QUEUE_DEPTH,
)


class CalibMode(StrEnum):
    """标定模式，取值即 ``mode`` 参数的合法取值。

    hsv 为 HSV 阈值框选；line_pose 为巡线姿态；motion 为里程计系数自测；
    follow 为纯巡线测试。
    """

    HSV = 'hsv'
    LINE_POSE = 'line_pose'
    MOTION = 'motion'
    FOLLOW = 'follow'


#: 归类按键 -> yaml 键与中文名，键名与 hsv_params.yaml 一致
HSV_KEYS: Dict[str, Tuple[str, str]] = {
    '1': ('line_hsv', '黑线'),
    '2': ('red_hsv', '红'),
    '3': ('green_hsv', '绿'),
    '4': ('blue_hsv', '蓝'),
    '5': ('yellow_hsv', '黄'),
}

#: 各话题的发布与订阅队列深度，单位条
CMD_VEL_QUEUE_DEPTH = 1
ARM_QUEUE_DEPTH = 10
IMAGE_QUEUE_DEPTH = 1
ODOM_QUEUE_DEPTH = 50
#: 激光订阅队列深度，复用 sensors 的同名常量，单位条
LASER_QUEUE_DEPTH = SCAN_QUEUE_DEPTH

#: 前方碰撞守卫扇区的半宽，单位 deg，中心为车体系正前方
FRONT_GUARD_SECTOR_HALF_WIDTH_DEG = 20.0
#: ROI 分位数取值，取 5 与 95 百分位以抗边缘离群像素，取值区间 0 到 100
ROI_PERCENTILE_LOW = 5
ROI_PERCENTILE_HIGH = 95

#: 粗粒度自旋时长，单位 s，用于等首帧与开环发布循环的节拍
LOOP_SPIN_SEC = 0.05
#: UI 主循环常规自旋时长，单位 s
UI_SPIN_SEC = 0.01
#: 姿态交互模式的按键轮询周期，单位 ms
KEY_POLL_MS = 30
#: 关节微调与粗调的步长，单位 deg
JOINT_STEP_DEG = 5
JOINT_COARSE_STEP_DEG = 15
#: 姿态预览下发的耗时，单位 ms
POSE_PUBLISH_TIME_MS = 500

#: 启动等待阶段的自旋时长，单位 s
STARTUP_SPIN_SEC = 0.1
#: 巡线测试等待里程计与首帧图像的上限，单位 s
FOLLOW_START_TIMEOUT_SEC = 10.0
#: 巡线测试节拍日志的周期，单位 s
FOLLOW_LOG_PERIOD_SEC = 1.0
#: 巡线测试主循环的自旋时长，单位 s
FOLLOW_LOOP_SPIN_SEC = 0.02
#: 巡线测试默认走的格数
FOLLOW_DEFAULT_CELLS = 5.0

#: 运动标定的开环时长，单位 s，命令值 = 速度 乘 时长
TEST_DURATION_SEC = 3.0
#: 直行与横移标定的命令速度，单位 m/s
TEST_ADVANCE_LINEAR_M_S = 0.15
TEST_STRAFE_LATERAL_M_S = 0.12
#: 转角标定的命令角速度，单位 rad/s，与命令转角，单位 deg
TEST_TURN_ANGULAR_RAD_S = 0.5
TEST_TURN_ANGLE_DEG = 90.0
#: 下发停止后的静置时长，单位 s，等底盘停稳再读里程计
STOP_SETTLE_SEC = 0.3
#: 可分辨的最小位移与最小转角，单位 m 与 rad，低于该值判底盘无响应
MIN_MEASURABLE_DISTANCE_M = 1e-3
MIN_MEASURABLE_ANGLE_RAD = 1e-3


class CalibrationTool(Node):
    """标定工具主节点，按 ``mode`` 参数进入不同标定流程。"""

    def __init__(self) -> None:
        """声明参数、创建收发接口并初始化各模式的状态，不进入任何等待。

        ``mode`` 在此解析为 ``CalibMode``，非法取值记一条 error 并置空，
        由 :func:`main` 静默跳过。
        """
        super().__init__('calibration_tool')

        self.declare_parameter('mode', 'hsv')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('odom_topic', '/odom_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('arm_topic', 'arm6_joints')
        #: 配置文件所在目录，默认取本包 install 之前的 src 目录
        self.declare_parameter('config_dir', '')
        # ---------------- 纯巡线测试即 mode=follow 所需，与任务同一套语义 ----------------
        # 以下键名与 config/*.yaml 一致；calib_entry 会把两份 yaml 一起传进来，
        # 因此这里读到的是已标定的值，而不是工具自己的默认值。
        self.declare_parameter('line_hsv', [0, 0, 0, 180, 255, 80])
        self.declare_parameter('line_roi_top_ratio', 0.5)
        self.declare_parameter('min_line_pixels', 200)
        self.declare_parameter('morph_kernel', 5)
        self.declare_parameter('line_lost_frames', 5)
        self.declare_parameter('line_pid', [1.2, 0.0, 0.2])
        self.declare_parameter('line_steer_sign', -1.0)
        self.declare_parameter('cruise_linear', 0.12)
        self.declare_parameter('max_angular_z', 0.60)
        self.declare_parameter('safety_range', 0.25)
        self.declare_parameter('cell_size', 0.40)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('vision_timeout', 2.0)

        mode_value = str(self.get_parameter('mode').value)
        self.mode: Optional[CalibMode] = None
        try:
            self.mode = CalibMode(mode_value)
        except ValueError:
            self.get_logger().error(
                f'未知 mode={mode_value}，可选 '
                + ' / '.join(item.value for item in CalibMode)
            )
        self._config_dir = self._resolve_config_dir()

        self._lock = threading.Lock()
        self._image: Optional[np.ndarray] = None
        self._image_ts = 0.0
        self._pose: Optional[Tuple[float, float, float]] = None
        self._last_odom_ts = 0.0

        self._bridge = CvBridge()
        self._pub_vel = self.create_publisher(
            Twist, str(self.get_parameter('cmd_vel_topic').value),
            CMD_VEL_QUEUE_DEPTH,
        )
        self._pub_arm = self.create_publisher(
            ArmJoints, str(self.get_parameter('arm_topic').value), ARM_QUEUE_DEPTH
        )
        self.create_subscription(
            Image, str(self.get_parameter('color_topic').value), self._on_image,
            IMAGE_QUEUE_DEPTH,
        )
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self._on_odom,
            ODOM_QUEUE_DEPTH,
        )
        # 巡线测试用：激光只做碰撞守卫，不做拓扑，因此没有挡板也能跑
        self._scan: Optional[LaserScan] = None
        self.create_subscription(
            LaserScan, str(self.get_parameter('scan_topic').value), self._on_scan,
            LASER_QUEUE_DEPTH,
        )

        # HSV 标定状态
        #: 鼠标框选矩形，取值为 xmin、ymin、xmax、ymax
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._dragging = False
        #: 待写回 yaml 的阈值，键为 hsv_params.yaml 的键名
        self._pending: Dict[str, List[int]] = {}
        #: 最近一次 ROI 的 HSV 分位数统计，六个元素依次为 H、S、V 的上下界
        self._roi_stats: Optional[List[int]] = None

        # 巡线姿态
        #: 六个关节角，单位度，取值区间 0 到 180
        self._joints: List[int] = [90, 90, 12, 20, 90, 0]
        #: 当前操作的关节编号，取值 1 到 6
        self._active_joint = 1

        # 运动标定
        self._motion_start_pose: Optional[Tuple[float, float, float]] = None

        self.get_logger().info(
            f'CalibrationTool 启动 | mode={mode_value} | config={self._config_dir}'
        )

    # ------------------------------------------------------------------ 工具

    def _resolve_config_dir(self) -> Path:
        """定位 config 目录。

        标定结果必须写回源码目录而非 install 下的副本，后者会在下次
        ``colcon build`` 时被覆盖。因此优先取源码工作区，可用参数显式覆盖。
        """
        explicit = str(self.get_parameter('config_dir').value)
        if explicit:
            return Path(explicit).expanduser()
        src = Path.home() / 'yahboomcar_ws' / 'src' / 'maze_explorer' / 'config'
        if (src / 'maze_params.yaml').is_file():
            return src
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / 'config'
            if (candidate / 'maze_params.yaml').is_file():
                return candidate
        return src

    @staticmethod
    def _update_yaml_line(path: Path, key: str, values: List[int | float]) -> bool:
        """按行替换该键所在行的取值，保留文件其余内容与中文注释。

        :param path: 目标 yaml 文件路径。
        :param key: 待替换的键名。
        :param values: 新取值，整数按原样渲染，浮点按六位有效数字渲染。
        :returns: 找到并替换了该键为真；文件不存在或键不存在为假。
        """
        if not path.is_file():
            return False
        text = path.read_text(encoding='utf-8')
        rendered = '[' + ', '.join(
            f'{v:.6g}' if isinstance(v, float) else str(v) for v in values
        ) + ']'
        pattern = re.compile(rf'^(\s*){re.escape(key)}:.*$', re.MULTILINE)
        if not pattern.search(text):
            return False
        new_text = pattern.sub(lambda m: f'{m.group(1)}{key}: {rendered}', text, count=1)
        path.write_text(new_text, encoding='utf-8')
        return True

    # ------------------------------------------------------------------ 回调

    def _on_image(self, msg: Image) -> None:
        """缓存最新彩色帧与到达时刻，转换失败时静默丢帧。

        :param msg: 彩色图消息，来自 ``color_topic``。
        """
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._image = img
            self._image_ts = time.monotonic()

    def _on_scan(self, msg: LaserScan) -> None:
        """缓存最新激光帧，只供前方碰撞守卫使用。

        :param msg: 激光消息，来自 ``scan_topic``。
        """
        self._scan = msg

    def _front_range(self) -> Optional[float]:
        """返回前方守卫扇区的最近有效距离，单位 m，无有效点时为 ``None``。

        扇区中心为车体系正前方，半宽见 ``FRONT_GUARD_SECTOR_HALF_WIDTH_DEG``；
        有效距离区间与 sensors 的量程默认值一致。
        """
        scan = self._scan
        if scan is None or not scan.ranges:
            return None
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(
            ranges.size, dtype=np.float32
        ) * scan.angle_increment
        selected = ranges[
            np.abs(np.arctan2(np.sin(angles), np.cos(angles)))
            <= math.radians(FRONT_GUARD_SECTOR_HALF_WIDTH_DEG)
        ]
        selected = selected[
            np.isfinite(selected)
            & (selected >= DEFAULT_SCAN_RANGE_MIN_M)
            & (selected <= DEFAULT_SCAN_RANGE_MAX_M)
        ]
        return float(selected.min()) if selected.size else None

    def _on_odom(self, msg: Odometry) -> None:
        """缓存最新位姿与到达时刻，只取位置与偏航。

        :param msg: 里程计消息，来自 ``odom_topic``。
        """
        position = msg.pose.pose.position
        with self._lock:
            self._pose = (
                position.x, position.y,
                yaw_from_quaternion(msg.pose.pose.orientation),
            )
            self._last_odom_ts = time.monotonic()

    def _snapshot(self) -> Optional[np.ndarray]:
        """返回最新彩色帧的副本，无数据时返回 ``None``。"""
        with self._lock:
            return None if self._image is None else self._image.copy()

    def _current_pose(self) -> Optional[Tuple[float, float, float]]:
        """返回最近位姿，取值为 x 与 y 单位 m 与 yaw 单位 rad，无数据为 ``None``。"""
        with self._lock:
            return self._pose

    # ------------------------------------------------------- 模式一：HSV 标定

    def _on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        """处理鼠标拖拽，维护框选矩形。

        :param event: OpenCV 鼠标事件码。
        :param x: 事件像素坐标横分量。
        :param y: 事件像素坐标纵分量。
        :param flags: OpenCV 鼠标标志位。
        :param param: 回调窗口指针，未使用。
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._roi = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            assert self._roi is not None
            self._roi = (self._roi[0], self._roi[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            assert self._roi is not None
            x0, y0, x1, y1 = self._roi
            self._roi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def run_hsv(self) -> None:
        """交互式框选采样 HSV：框选目标，按 1 到 5 归类，按 s 写回，按 q 退出。"""
        window = 'calibration: HSV'
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)
        self.get_logger().info(
            '框选目标后按 1=黑线 2=红 3=绿 4=蓝 5=黄 归类，s 保存，q 退出'
        )

        while rclpy.ok():
            img = self._snapshot()
            if img is None:
                rclpy.spin_once(self, timeout_sec=LOOP_SPIN_SEC)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            canvas = img.copy()
            if self._roi is not None:
                x0, y0, x1, y1 = self._roi
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # 实时预览：用当前归类结果或 ROI 统计值做二值化
                roi = img[y0:y1, x0:x1]
                if roi.size:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    # 取 5 与 95 分位而非极值，抗单像素噪声
                    h_min, s_min, v_min = np.percentile(
                        hsv.reshape(-1, 3), ROI_PERCENTILE_LOW, axis=0
                    )
                    h_max, s_max, v_max = np.percentile(
                        hsv.reshape(-1, 3), ROI_PERCENTILE_HIGH, axis=0
                    )
                    self._roi_stats = [int(h_min), int(s_min), int(v_min),
                                       int(h_max), int(s_max), int(v_max)]

            # 左上角显示待保存项
            y = 24
            for key, (yaml_key, cn) in HSV_KEYS.items():
                if yaml_key in self._pending:
                    cv2.putText(canvas, f'{key}={cn}: {self._pending[yaml_key]}',
                                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y += 22

            cv2.imshow(window, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                self._flush_pending()
            if chr(k) in HSV_KEYS and self._roi is not None:
                stats = getattr(self, '_roi_stats', None)
                if stats:
                    yaml_key, cn = HSV_KEYS[chr(k)]
                    self._pending[yaml_key] = list(stats)
                    self.get_logger().info(f'{cn} {yaml_key} = {stats}')

            rclpy.spin_once(self, timeout_sec=UI_SPIN_SEC)

        cv2.destroyAllWindows()

    def _flush_pending(self) -> None:
        """把待保存阈值逐键写回 hsv_params.yaml，无待保存项时记 warn。"""
        if not self._pending:
            self.get_logger().warn('没有待保存的阈值')
            return
        for key, values in self._pending.items():
            if self._write_config_key('hsv_params.yaml', key, values):
                self.get_logger().info(f'已写入 {key} = {values}')
            else:
                self.get_logger().warn(f'未能写入 {key}，请检查 {self._config_dir}')
        self._pending.clear()

    @staticmethod
    def _install_config_dir() -> Optional[Path]:
        """返回 install/share 下的 config 目录，未安装时返回 ``None``。"""
        try:
            from ament_index_python.packages import get_package_share_directory

            return Path(get_package_share_directory('maze_explorer')) / 'config'
        except Exception:  # noqa: BLE001 - 未安装即纯源码运行时忽略
            return None

    def _write_config_key(self, filename: str, key: str, values: List) -> bool:
        """写入配置：源码 config 为权威，同时镜像到 install 副本。

        为什么必须镜像：launch 读的是 install/share 下的 config，而本工具写的是
        源码目录。只写源码时，标定结果必须 colcon build 之后才生效，这是极易踩的
        坑，标定半天车仍是老样子，故在此一并写入副本。

        :param filename: 配置文件名，不含目录。
        :param key: 待替换的键名。
        :param values: 新取值。
        :returns: 源码目录写入成功为真。
        """
        ok = self._update_yaml_line(self._config_dir / filename, key, values)
        install_dir = self._install_config_dir()
        if install_dir is not None and install_dir != self._config_dir:
            target = install_dir / filename
            if target.is_file() and self._update_yaml_line(target, key, values):
                self.get_logger().info(f'已同步到 install 副本：{target}')
            else:
                self.get_logger().warn(
                    f'install 副本未同步（{target}），下次 colcon build 会带上'
                )
        return ok

    # -------------------------------------------------- 模式二：巡线姿态标定

    def run_line_pose(self) -> None:
        """交互调整关节角使相机对准黑线与前方通道，保存为 line_pose。"""
        window = 'calibration: line_pose'
        cv2.namedWindow(window)
        self.get_logger().info(
            '按 1~6 选关节；w/s 增减 5 度；] / [ 增减 15 度；p 打印；P 保存 line_pose；q 退出'
        )
        self._publish_joints()
        step = JOINT_STEP_DEG

        while rclpy.ok():
            img = self._snapshot()
            if img is not None:
                canvas = img.copy()
                cv2.putText(
                    canvas,
                    f'joint{self._active_joint} = {self._joints[self._active_joint - 1]}',
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.imshow(window, canvas)

            k = cv2.waitKey(KEY_POLL_MS) & 0xFF
            if k == 0xFF:  # 无按键
                rclpy.spin_once(self, timeout_sec=UI_SPIN_SEC)
                continue
            if k == ord('q'):
                break
            if ord('1') <= k <= ord('6'):
                self._active_joint = k - ord('0')
            elif k == ord('w'):
                self._bump_joint(+step)
            elif k == ord('s'):
                self._bump_joint(-step)
            elif k == ord(']'):
                self._bump_joint(+JOINT_COARSE_STEP_DEG)
            elif k == ord('['):
                self._bump_joint(-JOINT_COARSE_STEP_DEG)
            elif k == ord('p'):
                self.get_logger().info(f'当前关节角 {self._joints}')
            elif k == ord('P'):
                if self._write_config_key('arm_poses.yaml', 'line_pose', self._joints):
                    self.get_logger().info(f'已保存 line_pose = {self._joints}')
                else:
                    self.get_logger().warn(f'保存失败，请检查 {self._config_dir}')

            rclpy.spin_once(self, timeout_sec=UI_SPIN_SEC)

        cv2.destroyAllWindows()

    def _bump_joint(self, delta: int) -> None:
        """按增量调整当前关节角并立即下发，结果截断到 0 到 180 度。

        :param delta: 增量，单位 deg，正值增大负值减小。
        """
        idx = self._active_joint - 1
        self._joints[idx] = max(
            JOINT_MIN_DEG, min(JOINT_MAX_DEG, self._joints[idx] + delta)
        )
        self._publish_joints()

    def _publish_joints(self, time_ms: int = POSE_PUBLISH_TIME_MS) -> None:
        """下发当前六个关节角到机械臂。

        :param time_ms: 动作耗时，单位 ms，不小于 1。
        """
        msg = ArmJoints()
        (msg.joint1, msg.joint2, msg.joint3,
         msg.joint4, msg.joint5, msg.joint6) = self._joints
        msg.time = time_ms
        self._pub_arm.publish(msg)

    # -------------------------------------------------- 模式四：纯巡线测试

    def run_follow(self, cells: float = FOLLOW_DEFAULT_CELLS) -> None:
        """纯巡线测试：只靠相机沿黑线走，不依赖挡板与激光拓扑。

        用途是现场还没搭挡板时验证巡线。正式任务 ``maze_run`` 依赖激光测墙判开口，
        无挡板必失败；本模式则跳过拓扑。与任务里 ``MotionController.advance`` 使用
        同一套语义，即归一化偏差、同一个 ``line_steer_sign``、同样的丢线保护，
        所以这里能走通，任务里的巡线就走得通。

        :param cells: 走多少个格距，默认取 ``FOLLOW_DEFAULT_CELLS``，走满即停。
        """
        from maze_explorer.line_detector import LineDetector
        from maze_explorer.motion_controller import PID

        hsv = [int(v) for v in self.get_parameter('line_hsv').value]
        detector = LineDetector(
            hsv_range=(hsv[:3], hsv[3:]),
            roi_top_ratio=float(self.get_parameter('line_roi_top_ratio').value),
            min_pixels=int(self.get_parameter('min_line_pixels').value),
            morph_kernel=int(self.get_parameter('morph_kernel').value),
            lost_frames=int(self.get_parameter('line_lost_frames').value),
        )
        kp, ki, kd = (float(v) for v in self.get_parameter('line_pid').value)
        max_angular_z = float(self.get_parameter('max_angular_z').value)
        pid = PID(kp=kp, ki=ki, kd=kd, out_limit=max_angular_z)
        steer_sign = 1.0 if float(
            self.get_parameter('line_steer_sign').value
        ) >= 0 else -1.0
        cruise_linear = float(self.get_parameter('cruise_linear').value)
        safety_range = float(self.get_parameter('safety_range').value)
        vision_timeout = float(self.get_parameter('vision_timeout').value)
        target_distance = cells * float(self.get_parameter('cell_size').value)

        # 刚启动时订阅还没收到数据，先等里程计与首帧图像，不等会立刻退出
        deadline = time.monotonic() + FOLLOW_START_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=STARTUP_SPIN_SEC)
            if self._current_pose() is not None and self._snapshot() is not None:
                break
        start = self._current_pose()
        if start is None:
            self.get_logger().warn(
                f'{FOLLOW_START_TIMEOUT_SEC:.0f}s 内未收到里程计，无法巡线测试'
            )
            return
        self.get_logger().info(
            f'巡线测试开始：目标 {target_distance:.2f}m'
            f'（{cells:.0f} 格），速度 {cruise_linear:.2f}m/s，'
            f'PID=({kp},{ki},{kd})，转向符号 {steer_sign:+.0f}；Ctrl-C 随时停'
        )

        last_time = time.monotonic()
        last_log = last_time
        traveled = 0.0
        reason = '走满目标距离'
        while rclpy.ok():
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            front_distance = self._front_range()
            if front_distance is not None and front_distance < safety_range:
                reason = f'前方 {front_distance:.2f}m 触发碰撞保护'
                break

            with self._lock:
                img_age = now - self._image_ts if self._image_ts else float('inf')
            img = self._snapshot()
            if img is None or img_age > vision_timeout:
                reason = f'相机数据失效（年龄 {img_age:.1f}s）'
                break

            line_observation = detector.detect(img)
            if line_observation.valid:
                steering = steer_sign * pid.compute(line_observation.offset_norm, dt)
            elif detector.is_lost:
                reason = f'连续丢线 {line_observation.lost_frames} 帧'
                break
            else:
                steering = 0.0
            self._publish_vel(cruise_linear, 0.0, steering)

            current_pose = self._current_pose()
            if current_pose is not None:
                traveled = math.hypot(
                    current_pose[0] - start[0], current_pose[1] - start[1]
                )
                if traveled >= target_distance:
                    break
            if now - last_log >= FOLLOW_LOG_PERIOD_SEC:
                last_log = now
                flag = ''
                if not line_observation.valid:
                    flag = f'（本次无效，丢线 {line_observation.lost_frames}）'
                self.get_logger().info(
                    f'  {traveled:5.2f}m 偏差 {line_observation.offset_norm:+.3f}'
                    f'（{line_observation.offset_px:+.0f}px）'
                    f'→ 转向 {steering:+.3f} rad/s{flag}'
                )
            rclpy.spin_once(self, timeout_sec=FOLLOW_LOOP_SPIN_SEC)

        self._publish_vel(0.0, 0.0, 0.0)
        self.get_logger().info(f'巡线测试结束：{reason}，共走 {traveled:.2f}m')
        if reason.startswith('连续丢线'):
            self.get_logger().warn(
                '丢线排查：① 车是否压在黑线上、线是否在画面里（可用 line_detector 看掩膜）'
                ' ② line_hsv 是否框住黑线 ③ 若线在画面里却检不到，看掩膜缩略图'
            )

    # -------------------------------------------------- 模式三：运动标定自测

    def run_motion(self) -> None:
        """命令行交互式自测：开环走一段或转一段，用命令值反推里程计系数。"""
        self.get_logger().info(
            f'w=前进({TEST_ADVANCE_LINEAR_M_S}m/s×{TEST_DURATION_SEC:g}s) '
            f'a=左转({TEST_TURN_ANGULAR_RAD_S}rad/s×{TEST_DURATION_SEC:g}s) '
            f'd=右转 s=横移({TEST_STRAFE_LATERAL_M_S}m/s×{TEST_DURATION_SEC:g}s) '
            'q=退出'
        )
        self.get_logger().info(
            '说明：角速度系数用「命令角速度×时长」作基准即开环，'
            '故请确认底盘能达到指令角速度；更可靠可用地面基准复测'
        )
        while rclpy.ok():
            cmd = input('[w/a/d/s/q] > ').strip().lower()
            if cmd == 'q':
                break
            if cmd == 'w':
                self._test_advance(
                    linear_x=TEST_ADVANCE_LINEAR_M_S, linear_y=0.0
                )
            elif cmd == 's':
                self._test_advance(
                    linear_x=0.0, linear_y=TEST_STRAFE_LATERAL_M_S
                )
            elif cmd in ('a', 'd'):
                self._test_turn(
                    TEST_TURN_ANGLE_DEG if cmd == 'a' else -TEST_TURN_ANGLE_DEG
                )
            else:
                continue

    def _test_advance(self, linear_x: float, linear_y: float) -> None:
        """开环走一段，用命令速度乘时长作基准反推线速度系数。

        为什么按命令位移而非一格作基准：一格是 0.4 m，而 0.15 m/s 开环走
        3 s 实际命令位移是 0.45 m，拿格宽当基准会让建议值系统性偏大约 12.5%。
        此处直接用速度与 ``TEST_DURATION_SEC`` 的乘积作分母。

        :param linear_x: 前后命令速度，单位 m/s，正值前进。
        :param linear_y: 横移命令速度，单位 m/s，正值向左。
        """
        start = self._current_pose()
        if start is None:
            self.get_logger().warn('尚未收到里程计')
            return
        commanded = math.hypot(linear_x, linear_y) * TEST_DURATION_SEC
        self.get_logger().info(
            f'开始开环运动 {TEST_DURATION_SEC:.1f}s（命令位移 {commanded:.3f}m）...'
        )
        start_time = time.monotonic()
        while time.monotonic() - start_time < TEST_DURATION_SEC and rclpy.ok():
            self._publish_vel(linear_x, linear_y, 0.0)
            rclpy.spin_once(self, timeout_sec=LOOP_SPIN_SEC)
        self._publish_vel(0.0, 0.0, 0.0)
        time.sleep(STOP_SETTLE_SEC)
        end = self._current_pose()
        if end is None:
            return
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        distance = math.hypot(delta_x, delta_y)
        if distance <= MIN_MEASURABLE_DISTANCE_M:
            self.get_logger().warn('位移过小，请检查底盘是否响应 /cmd_vel')
            return
        self.get_logger().info(
            f'命令位移 {commanded:.3f}m，里程计读数 {distance:.3f}m'
            f'（dx={delta_x:+.3f} dy={delta_y:+.3f}）'
        )
        self.get_logger().info(
            f'odom_linear_scale_correction 建议 = {commanded / distance:.3f}'
            '（⚠️ 该系数目前未被任务代码使用，仅作记录）'
        )

    def _test_turn(self, angle_deg: float) -> None:
        """开环转角标定：固定角速度转固定时长，反推转角缩放系数。

        为什么必须开环：以里程计为停止条件再用里程计去测量，得到的比值恒约等于 1，
        在 5 度容差下还会打印约 0.94 的假建议，测不出真实误差。开环下真实转角由
        「角速度乘时长」给出，假定底盘达到指令角速度，与里程计读数之比才是系数，
        即真实转角等于系数乘 odom 读数。

        :param angle_deg: 目标转角，单位 deg，正值左转，实际转角由命令值决定。
        """
        start = self._current_pose()
        if start is None:
            self.get_logger().warn('尚未收到里程计')
            return
        angular_z = math.copysign(TEST_TURN_ANGULAR_RAD_S, angle_deg)
        commanded = TEST_TURN_ANGULAR_RAD_S * TEST_DURATION_SEC  # 命令转角，单位 rad
        self.get_logger().info(
            f'开始开环旋转 {TEST_DURATION_SEC:.1f}s（命令 '
            f'{TEST_TURN_ANGULAR_RAD_S:.2f}rad/s × {TEST_DURATION_SEC:.1f}s = '
            f'{math.degrees(commanded):+.1f}°）...'
        )
        start_time = time.monotonic()
        while time.monotonic() - start_time < TEST_DURATION_SEC and rclpy.ok():
            self._publish_vel(0.0, 0.0, angular_z)
            rclpy.spin_once(self, timeout_sec=LOOP_SPIN_SEC)
        self._publish_vel(0.0, 0.0, 0.0)
        time.sleep(STOP_SETTLE_SEC)

        end = self._current_pose()
        if end is None:
            return
        odom_delta = angle_diff(end[2], start[2])
        if abs(odom_delta) < MIN_MEASURABLE_ANGLE_RAD:
            self.get_logger().warn('里程计转角几乎为 0，请检查底盘是否响应 /cmd_vel')
            return
        suggested = commanded / odom_delta
        self.get_logger().info(
            f'命令转角 {math.degrees(commanded):+.1f}°，里程计读数 '
            f'{math.degrees(odom_delta):+.1f}°'
        )
        self.get_logger().info(
            f'odom_angular_scale_correction 建议 = {suggested:.3f}'
            '（填入 config/maze_params.yaml 后无需重建，工具已同步 install 副本）'
        )

    def _publish_vel(
        self, linear_x: float, linear_y: float, angular_z: float
    ) -> None:
        """发布一条速度指令到 ``cmd_vel_topic``。

        :param linear_x: 前后速度，单位 m/s。
        :param linear_y: 横移速度，单位 m/s。
        :param angular_z: 自转速度，单位 rad/s。
        """
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = linear_x, linear_y, angular_z
        self._pub_vel.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    """按 ``mode`` 分发到对应标定流程，退出前确保速度归零。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
    rclpy.init(args=args)
    node = CalibrationTool()
    try:
        # mode 为 None 时构造函数已记录 error，此处不重复
        if node.mode is CalibMode.HSV:
            node.run_hsv()
        elif node.mode is CalibMode.LINE_POSE:
            node.run_line_pose()
        elif node.mode is CalibMode.MOTION:
            node.run_motion()
        elif node.mode is CalibMode.FOLLOW:
            node.run_follow()
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_vel(0.0, 0.0, 0.0)  # noqa: SLF001 - 退出前确保停车
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
