"""现场标定工具。

迷宫现场的光照、地面材质、自备方块材质与机械臂装配都与出厂商Demo不同，
以下三项**必须现场标定**，否则会直接导致循迹跑偏或抓取失败：

============  ==========================================================
模式           作用
============  ==========================================================
``hsv``        鼠标框选采样，标定黑线引导线与红/绿/黄/蓝四色 HSV 阈值
``line_pose``  交互式调整机械臂巡线姿态关节角，使相机同时看到线与方块
``motion``     走格距离与 90 度转角自测，统计里程计误差并复核标定系数
============  ==========================================================

用法::

    ros2 run maze_explorer calibration_tool --ros-args -p mode:=hsv
    ros2 run maze_explorer calibration_tool --ros-args -p mode:=line_pose
    ros2 run maze_explorer calibration_tool --ros-args -p mode:=motion

.. note::
   写回 yaml 时采用**按行正则替换**而非整文件重写，以保留配置文件中的
   中文注释。需要图形环境（``cv2.imshow``），请在桌面终端运行。
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

from maze_explorer.base_driver import yaw_from_quaternion

#: 颜色名 -> (yaml 键, 中文名)，键名与 hsv_params.yaml 一致
HSV_KEYS: Dict[str, Tuple[str, str]] = {
    '1': ('line_hsv', '黑线'),
    '2': ('red_hsv', '红'),
    '3': ('green_hsv', '绿'),
    '4': ('blue_hsv', '蓝'),
    '5': ('yellow_hsv', '黄'),
}


class CalibrationTool(Node):
    """标定工具主节点。按 ``mode`` 参数进入不同标定流程。"""

    def __init__(self) -> None:
        super().__init__('calibration_tool')

        self.declare_parameter('mode', 'hsv')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('odom_topic', '/odom_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('arm_topic', 'arm6_joints')
        #: 配置文件所在目录（默认取本包 install 前的 src 目录）
        self.declare_parameter('config_dir', '')
        # ---------------- 纯巡线测试（mode=follow）所需，与任务同一套语义 ----------------
        #: 以下键名与 config/*.yaml 一致；calib_entry 会把两份 yaml 一起传进来，
        #: 因此这里读到的是**已标定**的值，而不是工具自己的默认值。
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

        self.mode = str(self.get_parameter('mode').value)
        self._config_dir = self._resolve_config_dir()

        self._lock = threading.Lock()
        self._image: Optional[np.ndarray] = None
        self._image_ts = 0.0
        self._pose: Optional[Tuple[float, float, float]] = None
        self._last_odom_ts = 0.0

        self._bridge = CvBridge()
        self._pub_vel = self.create_publisher(
            Twist, str(self.get_parameter('cmd_vel_topic').value), 1
        )
        self._pub_arm = self.create_publisher(
            ArmJoints, str(self.get_parameter('arm_topic').value), 10
        )
        self.create_subscription(
            Image, str(self.get_parameter('color_topic').value), self._on_image, 1
        )
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self._on_odom, 50
        )
        # 巡线测试用：激光只做碰撞守卫，不做拓扑（无挡板也能跑）
        self._scan: Optional[LaserScan] = None
        self.create_subscription(
            LaserScan, str(self.get_parameter('scan_topic').value), self._on_scan, 10
        )

        # HSV 标定状态
        self._roi: Optional[Tuple[int, int, int, int]] = None  # xmin, ymin, xmax, ymax
        self._dragging = False
        self._pending: Dict[str, List[int]] = {}  # 待写回 yaml 的阈值
        self._roi_stats: Optional[List[int]] = None  # 最近一次 ROI 的 HSV 分位数统计

        # 巡线姿态
        self._joints: List[int] = [90, 90, 12, 20, 90, 0]
        self._active_joint = 1

        # 运动标定
        self._motion_start_pose: Optional[Tuple[float, float, float]] = None

        self.get_logger().info(
            f'CalibrationTool 启动 | mode={self.mode} | config={self._config_dir}'
        )

    # ------------------------------------------------------------------ 工具

    def _resolve_config_dir(self) -> Path:
        """定位 config 目录。

        标定结果必须写回**源码目录**而非 install 下的副本——后者会在下次
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
        """按行替换 ``key: [...]`` 的值，保留文件其余内容与注释。"""
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
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._image = img
            self._image_ts = time.monotonic()

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg

    def _front_range(self) -> Optional[float]:
        """前方扇区（0°±20°）最近有效距离（米）；无有效点返回 None。"""
        scan = self._scan
        if scan is None or not scan.ranges:
            return None
        r = np.asarray(scan.ranges, dtype=np.float32)
        ang = scan.angle_min + np.arange(r.size, dtype=np.float32) * scan.angle_increment
        sel = r[np.abs(np.arctan2(np.sin(ang), np.cos(ang))) <= math.radians(20.0)]
        sel = sel[np.isfinite(sel) & (sel >= 0.05) & (sel <= 4.0)]
        return float(sel.min()) if sel.size else None

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        with self._lock:
            self._pose = (p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))
            self._last_odom_ts = time.monotonic()

    def _snapshot(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._image is None else self._image.copy()

    def _current_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self._pose

    # ------------------------------------------------------- 模式一：HSV 标定

    def _on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ANN001
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
        """框选采样 HSV：框选目标 → 按 1~5 归类 → s 写回 → q 退出。"""
        window = 'calibration: HSV'
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)
        self.get_logger().info(
            '框选目标后按 1=黑线 2=红 3=绿 4=蓝 5=黄 归类，s 保存，q 退出'
        )

        while rclpy.ok():
            img = self._snapshot()
            if img is None:
                rclpy.spin_once(self, timeout_sec=0.05)
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
                    h_min, s_min, v_min = np.percentile(hsv.reshape(-1, 3), 5, axis=0)
                    h_max, s_max, v_max = np.percentile(hsv.reshape(-1, 3), 95, axis=0)
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

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

    def _flush_pending(self) -> None:
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
        """install/share 下的 config 目录；未安装时返回 ``None``。"""
        try:
            from ament_index_python.packages import get_package_share_directory

            return Path(get_package_share_directory('maze_explorer')) / 'config'
        except Exception:  # noqa: BLE001 - 未安装（纯源码运行）时忽略
            return None

    def _write_config_key(self, filename: str, key: str, values: List) -> bool:
        """写入配置：源码 config 为权威，同时镜像到 install 副本。

        为什么必须镜像：launch 读的是 ``install/share/maze_explorer/config``，
        而本工具写的是源码目录。只写源码时，标定结果**必须 colcon build 之后
        才生效**——这是极易踩的坑（标了半天车还是老样子），故在此一并写。
        """
        ok = self._update_yaml_line(self._config_dir / filename, key, values)
        install_dir = self._install_config_dir()
        if install_dir is not None and install_dir != self._config_dir:
            target = install_dir / filename
            if target.is_file() and self._update_yaml_line(target, key, values):
                self.get_logger().info(f'已同步到 install 副本：{target}')
            else:
                self.get_logger().warn(f'install 副本未同步（{target}），下次 colcon build 会带上')
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
        step = 5

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

            k = cv2.waitKey(30) & 0xFF
            if k == 0xFF:  # 无按键
                rclpy.spin_once(self, timeout_sec=0.01)
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
                self._bump_joint(+15)
            elif k == ord('['):
                self._bump_joint(-15)
            elif k == ord('p'):
                self.get_logger().info(f'当前关节角 {self._joints}')
            elif k == ord('P'):
                if self._write_config_key('arm_poses.yaml', 'line_pose', self._joints):
                    self.get_logger().info(f'已保存 line_pose = {self._joints}')
                else:
                    self.get_logger().warn(f'保存失败，请检查 {self._config_dir}')

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

    def _bump_joint(self, delta: int) -> None:
        idx = self._active_joint - 1
        self._joints[idx] = max(0, min(180, self._joints[idx] + delta))
        self._publish_joints()

    def _publish_joints(self, time_ms: int = 500) -> None:
        msg = ArmJoints()
        (msg.joint1, msg.joint2, msg.joint3,
         msg.joint4, msg.joint5, msg.joint6) = self._joints
        msg.time = time_ms
        self._pub_arm.publish(msg)

    # -------------------------------------------------- 模式四：纯巡线测试

    def run_follow(self, cells: float = 5.0) -> None:
        """纯巡线测试：只靠相机沿黑线走，**不依赖挡板与激光拓扑**。

        用途：现场还没搭挡板时验证巡线（`--run` 会依赖激光测墙判开口，无挡板必失败）。
        与任务里 ``MotionController.advance`` 使用同一套语义——归一化偏差、同一个
        `line_steer_sign`、同样的丢线保护——所以这里能走通，任务里的巡线就走得通。

        :param cells: 走多少个格距（默认 5 格 = 2.0m），走满即停。
        """
        from maze_explorer.line_detector import LineDetector
        from maze_explorer.motion_controller import PID

        hsv = [int(v) for v in self.get_parameter('line_hsv').value]
        det = LineDetector(
            hsv_range=(hsv[:3], hsv[3:]),
            roi_top_ratio=float(self.get_parameter('line_roi_top_ratio').value),
            min_pixels=int(self.get_parameter('min_line_pixels').value),
            morph_kernel=int(self.get_parameter('morph_kernel').value),
            lost_frames=int(self.get_parameter('line_lost_frames').value),
        )
        kp, ki, kd = (float(v) for v in self.get_parameter('line_pid').value)
        max_wz = float(self.get_parameter('max_angular_z').value)
        pid = PID(kp=kp, ki=ki, kd=kd, out_limit=max_wz)
        sign = 1.0 if float(self.get_parameter('line_steer_sign').value) >= 0 else -1.0
        speed = float(self.get_parameter('cruise_linear').value)
        safety = float(self.get_parameter('safety_range').value)
        vision_timeout = float(self.get_parameter('vision_timeout').value)
        target = cells * float(self.get_parameter('cell_size').value)

        # 刚启动时订阅还没收到数据，先等里程计与首帧图像（实测不等待会立刻退出）
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._current_pose() is not None and self._snapshot() is not None:
                break
        start = self._current_pose()
        if start is None:
            self.get_logger().warn('10s 内未收到里程计，无法巡线测试')
            return
        self.get_logger().info(
            f'巡线测试开始：目标 {target:.2f}m（{cells:.0f} 格），速度 {speed:.2f}m/s，'
            f'PID=({kp},{ki},{kd})，转向符号 {sign:+.0f}；Ctrl-C 随时停'
        )

        last_t = time.monotonic()
        last_log = last_t
        traveled = 0.0
        reason = '走满目标距离'
        while rclpy.ok():
            now = time.monotonic()
            dt = now - last_t
            last_t = now

            front = self._front_range()
            if front is not None and front < safety:
                reason = f'前方 {front:.2f}m 触发碰撞保护'
                break

            with self._lock:
                img_age = now - self._image_ts if self._image_ts else float('inf')
            img = self._snapshot()
            if img is None or img_age > vision_timeout:
                reason = f'相机数据失效（年龄 {img_age:.1f}s）'
                break

            obs = det.detect(img)
            if obs.valid:
                steering = sign * pid.compute(obs.offset_norm, dt)
            elif det.is_lost:
                reason = f'连续丢线 {obs.lost_frames} 帧'
                break
            else:
                steering = 0.0
            self._publish_vel(speed, 0.0, steering)

            cur = self._current_pose()
            if cur is not None:
                traveled = math.hypot(cur[0] - start[0], cur[1] - start[1])
                if traveled >= target:
                    break
            if now - last_log >= 1.0:
                last_log = now
                flag = '' if obs.valid else f'（本次无效，丢线 {obs.lost_frames}）'
                self.get_logger().info(
                    f'  {traveled:5.2f}m 偏差 {obs.offset_norm:+.3f}'
                    f'（{obs.offset_px:+.0f}px）→ 转向 {steering:+.3f} rad/s{flag}'
                )
            rclpy.spin_once(self, timeout_sec=0.02)

        self._publish_vel(0.0, 0.0, 0.0)
        self.get_logger().info(f'巡线测试结束：{reason}，共走 {traveled:.2f}m')
        if reason.startswith('连续丢线'):
            self.get_logger().warn(
                '丢线排查：① 车是否压在黑线上、线是否在画面里（可用 line_detector 看掩膜）'
                ' ② line_hsv 是否框住黑线 ③ 若线在画面里却检不到，看掩膜缩略图'
            )

    # -------------------------------------------------- 模式三：运动标定自测

    def run_motion(self) -> None:
        """命令行交互式自测：开环走一段 / 开环转一段，用命令值反推里程计系数。"""
        self.get_logger().info(
            'w=前进(0.15m/s×3s) a=左转(0.5rad/s×3s) d=右转 s=横移(0.12m/s×3s) q=退出'
        )
        self.get_logger().info(
            '说明：角速度系数用「命令角速度×时长」作基准（开环），'
            '故请确认底盘能达到指令角速度；更可靠可用地面基准复测'
        )
        while rclpy.ok():
            cmd = input('[w/a/d/s/q] > ').strip().lower()
            if cmd == 'q':
                break
            if cmd == 'w':
                self._test_advance(linear=0.15, lateral=0.0)
            elif cmd == 's':
                self._test_advance(linear=0.0, lateral=0.12)
            elif cmd in ('a', 'd'):
                self._test_turn(+90.0 if cmd == 'a' else -90.0)
            else:
                continue

    #: 运动标定的固定时长（秒）与速度，命令值 = 速度 × 时长
    TEST_DURATION = 3.0

    def _test_advance(self, linear: float, lateral: float) -> None:
        """开环走一段，用「命令速度 × 时长」作基准反推线速度系数。

        旧实现拿 0.4m（一格）当基准，但 0.15m/s × 3s 实际是 0.45m，
        建议值会系统性偏大 12.5%；此处按命令位移计算。
        """
        start = self._current_pose()
        if start is None:
            self.get_logger().warn('尚未收到里程计')
            return
        commanded = math.hypot(linear, lateral) * self.TEST_DURATION
        self.get_logger().info(f'开始开环运动 {self.TEST_DURATION:.1f}s（命令位移 {commanded:.3f}m）...')
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.TEST_DURATION and rclpy.ok():
            self._publish_vel(linear, lateral, 0.0)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._publish_vel(0.0, 0.0, 0.0)
        time.sleep(0.3)
        end = self._current_pose()
        if end is None:
            return
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-3:
            self.get_logger().warn('位移过小，请检查底盘是否响应 /cmd_vel')
            return
        self.get_logger().info(
            f'命令位移 {commanded:.3f}m，里程计读数 {dist:.3f}m（dx={dx:+.3f} dy={dy:+.3f}）'
        )
        self.get_logger().info(
            f'odom_linear_scale_correction 建议 = {commanded / dist:.3f}'
            '（⚠️ 该系数目前未被任务代码使用，仅作记录）'
        )

    def _test_turn(self, angle_deg: float) -> None:
        """开环转角标定：固定角速度转固定时长，反推 ``odom_angular_scale_correction``。

        为什么必须开环：旧实现以里程计为停止条件、再用里程计去"测量"，
        得到的比值恒 ≈1（在 5° 容差下会打印 ≈0.94 的假建议），**测不出真实误差**。
        开环下"真实转角"由「角速度 × 时长」给出（假定底盘达到指令角速度），
        与里程计读数之比才是系数：真实转角 = 系数 × odom 读数。
        """
        start = self._current_pose()
        if start is None:
            self.get_logger().warn('尚未收到里程计')
            return
        speed = 0.5
        wz = math.copysign(speed, angle_deg)
        commanded = speed * self.TEST_DURATION          # 命令转角（rad）
        self.get_logger().info(
            f'开始开环旋转 {self.TEST_DURATION:.1f}s（命令 {speed:.2f}rad/s × '
            f'{self.TEST_DURATION:.1f}s = {math.degrees(commanded):+.1f}°）...'
        )
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.TEST_DURATION and rclpy.ok():
            self._publish_vel(0.0, 0.0, wz)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._publish_vel(0.0, 0.0, 0.0)
        time.sleep(0.3)

        end = self._current_pose()
        if end is None:
            return
        odom_delta = self._angle_diff(end[2], start[2])
        if abs(odom_delta) < 1e-3:
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

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """归一化角度差到 [-pi, pi]。"""
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _publish_vel(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self._pub_vel.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CalibrationTool()
    try:
        if node.mode == 'hsv':
            node.run_hsv()
        elif node.mode == 'line_pose':
            node.run_line_pose()
        elif node.mode == 'motion':
            node.run_motion()
        elif node.mode == 'follow':
            node.run_follow()
        else:
            node.get_logger().error(
                f'未知 mode={node.mode}，可选 hsv / line_pose / motion / follow'
            )
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_vel(0.0, 0.0, 0.0)  # noqa: SLF001 - 退出前确保停车
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
