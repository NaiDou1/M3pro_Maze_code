"""运动控制：巡线走格、原地转向、横移微调与碰撞保护。

把连续运动离散成两个原语，使误差不跨格累积：

* :meth:`MotionController.advance_one_cell` —— 沿黑线前进一格即 cell_size，
  巡线 PID 控横向，里程计闭环判到位；
* :meth:`MotionController.turn_to_heading` —— 麦轮原地转到目标网格朝向。

.. important::
   所有原语都是阻塞式的：内部统一通过 :meth:`MotionController._spin_once`
   驱动回调。组合多个节点时由调用方注入 ``spin_fn``，通常为
   ``executor.spin_once``。

   任何绕过 ``_spin_once`` 直接调用 ``rclpy.spin_once`` 的写法都会导致子节点
   回调不被驱动，闭环读到过期数据，表现为 yaw 永不更新、车以固定角速度转到
   超时。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from maze_explorer.base_driver import BaseDriver
from maze_explorer.grid_mapper import GridMapper, Heading, angle_diff
from maze_explorer.line_detector import LineDetector
from maze_explorer.sensors import SensorHub, SensorKind

#: 阻塞原语内每次自旋驱动的时长，单位 s，即闭环控制节拍
LOOP_SPIN_SEC = 0.02
#: 等待传感器恢复期间每次自旋的时长，单位 s
RECOVER_SPIN_SEC = 0.05
#: 停车后补一次自旋的时长，单位 s，让停车指令与回调处理落地
STOP_SPIN_SEC = 0.01
#: PID 计算的时间步下限，单位 s，同刻两次调用时防止除零
MIN_PID_DT_SEC = 1e-3
#: 判定转向缩放系数等于 1.0 的容差，无量纲
ANG_SCALE_EPS = 1e-6

#: 正前方扇区中心角，单位 deg，车体系，对应 config 的 front_angle_deg
FRONT_SECTOR_ANGLE_DEG = 0.0
#: 左侧扇区中心角，单位 deg，对应 config 的 left_angle_deg
LEFT_SECTOR_ANGLE_DEG = 90.0
#: 右侧扇区中心角，单位 deg，对应 config 的 right_angle_deg
RIGHT_SECTOR_ANGLE_DEG = -90.0
#: 开口判定扇区半宽，单位 deg，对应 config 的 opening_half_width_deg
OPENING_SECTOR_HALF_WIDTH_DEG = 20.0
#: 正前方障碍检测扇区半宽，单位 deg，窄于开口扇区以聚焦正前方通道
FRONT_SECTOR_HALF_WIDTH_DEG = 15.0

#: 脱离与回退时的反向速度为巡线巡航速度的比例，无量纲
BACKUP_SPEED_RATIO = 0.5
#: 转向比例增益，单位 s 的负一次方，作用于归一化后的航向误差
TURN_PROPORTIONAL_GAIN = 2.0
#: 转向静摩擦补偿的最小角速度，单位 rad/s，低于该值车体推不动静摩擦
MIN_TURN_ANGULAR_RAD_S = 0.12
#: 转向收敛判定容差的下限，单位 rad，与 yaw_tolerance 取大者，避免过严判失败
MIN_ACCEPT_YAW_TOLERANCE_RAD = 0.05

#: 横移默认超时，单位 s
DEFAULT_STRAFE_TIMEOUT_SEC = 6.0
#: 横移速度下限，单位 m/s，低于该值麦轮打滑不足以推动车体
STRAFE_MIN_SPEED_M_S = 0.06
#: 横移速度上限，单位 m/s，横移精度差故限制速度
STRAFE_MAX_SPEED_M_S = 0.12
#: 横移速度对目标距离的比例增益，无量纲，输出按上下限截断
STRAFE_SPEED_GAIN = 1.5

#: 碰撞回退距离占格宽的比例，无量纲，0.5 即半格
EMERGENCY_BACKOFF_CELL_RATIO = 0.5
#: 碰撞回退超时，单位 s，超时立即停车防止持续后退撞到后方
EMERGENCY_BACKOFF_TIMEOUT_SEC = 3.0
#: 碰撞回退速度，单位 m/s，负值为后退
EMERGENCY_BACKOFF_SPEED_M_S = -0.10


def _clamp(value: float, limit: float) -> float:
    """把数值截断到闭区间 负 limit 到正 limit。

    :param value: 待截断数值。
    :param limit: 绝对值上限，不小于 0。
    :returns: 截断后的数值。
    """
    return max(-limit, min(limit, value))


class PID:
    """极简增量式 PID，带输出限幅与积分限幅。"""

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        out_limit: Optional[float] = None,
        i_limit: Optional[float] = None,
    ) -> None:
        """记录增益与限幅，内部状态置零。

        :param kp: 比例增益，输出量纲随误差量纲而定。
        :param ki: 积分增益。
        :param kd: 微分增益。
        :param out_limit: 输出绝对值上限，取 ``None`` 表示不限幅。
        :param i_limit: 积分项绝对值上限，取 ``None`` 表示不限幅。
        """
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.out_limit = out_limit
        self.i_limit = i_limit
        self._integral = 0.0
        self._prev_error: Optional[float] = None

    def compute(self, error: float, dt: float) -> float:
        """按当前误差与时间步计算一次输出，并推进内部状态。

        :param error: 本拍误差，巡线场景为归一化横向偏差，无量纲。
        :param dt: 距上一拍的时长，单位 s，小于 ``MIN_PID_DT_SEC`` 时按该下限算。
        :returns: 控制输出，已按 ``out_limit`` 截断。
        """
        dt = max(dt, MIN_PID_DT_SEC)
        self._integral += error * dt
        if self.i_limit is not None:
            self._integral = _clamp(self._integral, self.i_limit)

        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.out_limit is not None:
            out = _clamp(out, self.out_limit)
        return out

    def reset(self) -> None:
        """清空积分与历史误差，动作切换前调用以免上一动作的积分残留。"""
        self._integral = 0.0
        self._prev_error = None


class MotionController:
    """基于底盘、传感器与巡线检测器的运动原语集合。"""

    def __init__(
        self,
        node: Node,
        base: BaseDriver,
        sensors: SensorHub,
        line_detector: LineDetector,
        *,
        spin_fn: Optional[Callable[[float], None]] = None,
        cell_size: float = 0.40,
        cruise_linear: float = 0.15,
        max_angular_z: float = 0.60,
        turn_angular: float = 0.50,
        line_pid: Tuple[float, float, float] = (1.2, 0.0, 0.2),
        cell_tolerance: float = 0.03,
        yaw_tolerance: float = 0.0873,
        safety_range: float = 0.25,
        opening_min_range: float = 0.55,
        advance_timeout: float = 12.0,
        turn_timeout: float = 8.0,
        angular_scale_correction: float = 1.0,
        scan_timeout: float = 0.5,
        vision_timeout: float = 2.0,
        line_steer_sign: float = -1.0,
        sensor_recover_wait: float = 2.0,
    ) -> None:
        """绑定三个子模块并装载运动与安全参数，默认值与 config 一致。

        :param node: 用于记日志的节点对象，不拥有其生命周期。
        :param base: 底盘驱动，提供位姿、速度与看门狗。
        :param sensors: 传感器集线器，提供激光与 RGB-D 保鲜查询。
        :param line_detector: 黑线检测器，提供巡线误差。
        :param spin_fn: 注入的自旋回调，形参为时长单位 s。组合多节点时传
            ``executor.spin_once``；取 ``None`` 时只驱动 ``node``。
        :param cell_size: 格心间距，单位 m，默认与通道宽 0.40 m 相等。
        :param cruise_linear: 巡线巡航速度，单位 m/s，取值区间 0 到
            config 的 max_linear_x 即 0.25。
        :param max_angular_z: 自转角速度上限，单位 rad/s，默认与 config 的
            max_angular_z 相等。
        :param turn_angular: 原地转 90 度的角速度，单位 rad/s。
        :param line_pid: 巡线 PID 三元组依次为 Kp、Ki、Kd，输出为 angular.z
            单位 rad/s，误差为归一化偏差，无量纲，1.0 表示线在图像边缘。不得用
            像素量纲的增益，否则输出恒饱和。
        :param cell_tolerance: 到位判定的位置容差，单位 m。
        :param yaw_tolerance: 转向收敛的航向容差，单位 rad，默认 0.0873 约 5 度。
        :param safety_range: 碰撞保护阈值，单位 m，正前方低于该值即停车。
        :param opening_min_range: 开口判定阈值，单位 m，扇区最近距离大于它即
            判开口。
        :param advance_timeout: 单次走格超时，单位 s，超时判失败并停车。
        :param turn_timeout: 单次转向超时，单位 s，超时按残余误差判成败。
        :param angular_scale_correction: odom 转角缩放系数，无量纲，语义为
            真实转角等于该系数乘 odom 读数，取 1.0 表示不修正。须用
            calibration_tool 的 motion 模式实测，否则每次转向系统性偏角。
        :param scan_timeout: 激光保鲜阈值，单位 s，年龄超过即禁止移动。
        :param vision_timeout: 相机保鲜阈值，单位 s，年龄超过即禁止前进。
        :param line_steer_sign: 巡线转向符号，取值 +1 或 -1。+1 表示图像右侧
            偏差对应左转，用于相机装反或图像旋转 180 度的场合；-1 表示右侧偏差
            对应右转。本机黑线近粗远细说明图像未旋转，故取 -1。判据见 AGENTS
            第 5 节，符号错误会使车朝偏离黑线的方向一直转直至丢线停车。
        :param sensor_recover_wait: 失效后的恢复等待上限，单位 s。RGB-D 首帧
            同步需完成 DDS 发现与配对，实测最坏 1.88 s，故默认 2.0 覆盖该值；
            没有这段宽限时启动瞬间必然误判失效并触发 FAULT。
        """
        self._node = node
        self._base_driver = base
        self._sensor_hub = sensors
        self._line_detector = line_detector
        self._spin_fn = spin_fn

        self.cell_size = float(cell_size)
        self._cruise_linear = float(cruise_linear)
        self._max_angular_z = float(max_angular_z)
        self._turn_angular = float(turn_angular)
        self._cell_tolerance = float(cell_tolerance)
        self._yaw_tolerance = float(yaw_tolerance)
        self._safety_range = float(safety_range)
        self._opening_min_range = float(opening_min_range)
        self._advance_timeout = float(advance_timeout)
        self._turn_timeout = float(turn_timeout)
        self._scan_timeout = float(scan_timeout)
        self._vision_timeout = float(vision_timeout)
        self._sensor_recover_wait = float(sensor_recover_wait)
        self._line_steer_sign = 1.0 if float(line_steer_sign) >= 0 else -1.0
        #: odom 转角缩放系数，语义与取值依据见构造函数 docstring
        self._angular_scale = float(angular_scale_correction)
        if abs(self._angular_scale - 1.0) > ANG_SCALE_EPS:
            self._node.get_logger().warn(
                f'已启用转向缩放修正 ang_scale={self._angular_scale:.3f}；'
                '该值应由 calibration_tool motion 模式实测得出，'
                '否则每次转向都会系统性偏角'
            )

        self._pid = PID(
            kp=line_pid[0],
            ki=line_pid[1],
            kd=line_pid[2],
            out_limit=self._max_angular_z,
        )
        self._node.get_logger().info(
            f'巡线控制：PID={list(line_pid)}（误差为归一化偏差，1.0=线在图像边缘）'
            f' 转向符号={self._line_steer_sign:+.0f}'
        )

    # ------------------------------------------------------------ 传感器辅助

    def _spin_once(self, timeout: float) -> None:
        """驱动一次回调，优先用注入的自旋回调，否则只转传入的节点。

        :param timeout: 本次自旋时长，单位 s。
        """
        if self._spin_fn is not None:
            self._spin_fn(timeout)
        else:
            rclpy.spin_once(self._node, timeout_sec=timeout)

    def _latest_image(self) -> Optional[np.ndarray]:
        """返回最新彩色帧，尚未同步到数据时返回 ``None``。"""
        rgbd = self._sensor_hub.get_rgbd()
        return None if rgbd is None else rgbd[0]

    # ------------------------------------------------------------ 传感器健康

    def _sensor_age(self, kind: SensorKind) -> float:
        """读取某项传感器数据的年龄，单位 s。

        传感器对象未提供对应接口时返回 0 即视为健康，这样纯逻辑单测可以传桩
        对象，不必搭 ROS 运行时。

        :param kind: 查询项，取值见 ``SensorKind``，值即集线器上的方法名。
        :returns: 年龄秒数，接口缺失或读取异常时为 0.0。
        """
        getter = getattr(self._sensor_hub, kind.value, None)
        if getter is None:
            return 0.0
        try:
            return float(getter())
        except Exception:  # noqa: BLE001 - 健康检查本身不应影响主流程
            return 0.0

    def scan_age(self) -> float:
        """返回激光数据年龄，单位 s，无数据时为 ``inf``。"""
        return self._sensor_age(SensorKind.SCAN)

    def vision_age(self) -> float:
        """返回相机数据年龄，单位 s，无数据时为 ``inf``。"""
        return self._sensor_age(SensorKind.RGBD)

    def sensors_ok(self, need_vision: bool = True) -> bool:
        """判断传感器是否新鲜可用。

        :param need_vision: 是否同时要求相机新鲜。前进循迹需要，原地转向不需要。
        :returns: 激光与相机年龄都不超过各自阈值时为真。
        """
        if self.scan_age() > self._scan_timeout:
            return False
        if need_vision and self.vision_age() > self._vision_timeout:
            return False
        return True

    def wait_until_healthy(
        self, need_vision: bool = True, timeout: Optional[float] = None
    ) -> bool:
        """持续自旋等待传感器恢复新鲜，返回是否在时限内可用。

        这是失效即停的宽限环节：数据陈旧未必代表硬件故障，机械臂动作期间不
        自旋就会让缓存变旧，RGB-D 首帧同步也需要秒级时间。等待期间持续自旋，
        让回调有机会刷新数据。

        :param need_vision: 是否要求相机新鲜，语义同 :meth:`sensors_ok`。
        :param timeout: 等待上限，单位 s，取 ``None`` 时用
            ``sensor_recover_wait``。
        :returns: 时限内恢复为真。
        """
        limit = self._sensor_recover_wait if timeout is None else float(timeout)
        if self.sensors_ok(need_vision):
            return True
        deadline = time.monotonic() + limit
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_once(RECOVER_SPIN_SEC)
            if self.sensors_ok(need_vision):
                return True
        return False

    def _stop_if_unhealthy(self, need_vision: bool) -> bool:
        """传感器失效时停车并给一次恢复机会，仍失效则放弃本次动作。

        为什么必须硬失败：激光失效时 ``sector_min_range`` 返回 ``None``，而
        ``is_path_clear`` 与碰撞保护都把 ``None`` 当通畅；相机失效时循迹偏差
        恒为 0，车会闷头直行。两者都会让保护退化成放任，因此宁可停车。

        :param need_vision: 是否要求相机新鲜，语义同 :meth:`sensors_ok`。
        :returns: 仍失效即放弃动作为真；恢复或本就健康为假。
        """
        if self.sensors_ok(need_vision):
            return False
        # 先停车，无论后续是否恢复动作都已中断
        self._base_driver.stop()
        if self.wait_until_healthy(need_vision, timeout=self._sensor_recover_wait):
            self._node.get_logger().info(
                f'传感器短暂失效后已恢复（等待上限 {self._sensor_recover_wait:.1f}s），继续动作'
            )
            return False
        parts = [f'激光年龄 {self.scan_age():.2f}s（阈值 {self._scan_timeout:.2f}s）']
        if need_vision:
            parts.append(
                f'相机年龄 {self.vision_age():.2f}s（阈值 {self._vision_timeout:.2f}s）'
            )
        self._node.get_logger().warn(
            '传感器数据失效，已停车并放弃本次动作：' + '，'.join(parts)
        )
        return True

    def front_range(self) -> Optional[float]:
        """返回正前方最近障碍距离，单位 m，扇区半宽见模块常量。

        :returns: 最近距离；扇区内无有效点或激光失效时为 ``None``。
        """
        return self._sensor_hub.sector_min_range(
            FRONT_SECTOR_ANGLE_DEG, FRONT_SECTOR_HALF_WIDTH_DEG
        )

    def is_front_blocked(self) -> bool:
        """判断正前方是否进入碰撞保护阈值。

        :returns: 测得距离小于 ``safety_range`` 时为真；无有效测距为假，
            须先用 :meth:`scan_age` 确认激光保鲜。
        """
        front_distance = self.front_range()
        return front_distance is not None and front_distance < self._safety_range

    def scan_openings(self) -> Tuple[bool, bool, bool]:
        """返回车体系前方、左方、右方三个扇区是否有开口。

        以激光扇区最近距离与 ``opening_min_range`` 比较；扇区无有效点即视为
        远处通透，也判为开口。

        :returns: 三元组依次为前方、左方、右方，开口为真。
        """
        clear_range = self._opening_min_range
        return (
            self._sensor_hub.is_path_clear(
                FRONT_SECTOR_ANGLE_DEG, OPENING_SECTOR_HALF_WIDTH_DEG, clear_range
            ),
            self._sensor_hub.is_path_clear(
                LEFT_SECTOR_ANGLE_DEG, OPENING_SECTOR_HALF_WIDTH_DEG, clear_range
            ),
            self._sensor_hub.is_path_clear(
                RIGHT_SECTOR_ANGLE_DEG, OPENING_SECTOR_HALF_WIDTH_DEG, clear_range
            ),
        )

    # ------------------------------------------------------------ 运动原语

    def advance_one_cell(self) -> bool:
        """沿黑线前进一格，即 ``cell_size`` 距离。

        :returns: 在位置容差内到位为真。
        """
        return self.advance(self.cell_size)

    def advance(self, distance: float, timeout: Optional[float] = None) -> bool:
        """沿黑线前进指定距离，负值为后退，返回是否在容差内到位。

        巡线 PID 只作用于前进方向；后退用于脱离贴墙等场景，不做巡线。

        安全门禁为传感器不新鲜一律不动。前进需要激光即碰撞保护与相机即循迹，
        后退只需要激光。门禁在动作开始前与每个控制周期内各查一次，因此行进
        途中掉线也会立刻停车。

        :param distance: 目标位移，单位 m，正值前进负值后退，绝对值不小于
            ``cell_tolerance`` 才可能判到位。
        :param timeout: 超时上限，单位 s，取 ``None`` 时用 ``advance_timeout``。
        :returns: 在容差内到位为真；传感器失效、无里程计、碰撞或丢线为假。
        """
        forward = distance >= 0
        if self._stop_if_unhealthy(need_vision=forward):
            return False

        start_pose = self._base_driver.get_pose()
        if start_pose is None:
            self._node.get_logger().warn('无里程计数据，无法走格')
            return False

        target_distance = abs(distance) - self._cell_tolerance
        time_limit = float(
            timeout if timeout is not None else self._advance_timeout
        )
        speed = (
            self._cruise_linear
            if forward
            else -BACKUP_SPEED_RATIO * self._cruise_linear
        )
        #: 出发时的前墙距离，单位 m，前方无墙或无回波时为 None，用于激光交叉验证
        wall_start_distance = self.front_range() if forward else None

        self._pid.reset()
        start_time = time.monotonic()
        last_time = start_time
        traveled = 0.0

        while rclpy.ok() and (time.monotonic() - start_time) < time_limit:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            # 行进途中掉线也要立刻停，否则保护退化为放任
            if self._stop_if_unhealthy(need_vision=forward):
                return False

            if forward and self.is_front_blocked():
                self._base_driver.stop()
                self._node.get_logger().warn(
                    f'前方 {self.front_range():.2f}m 触发碰撞保护，执行回退'
                )
                self._emergency_backoff()
                return False

            steering = 0.0
            if forward:
                image = self._latest_image()
                if image is not None:
                    line_observation = self._line_detector.detect(image)
                    if line_observation.valid:
                        # 误差用归一化偏差即无量纲量，1.0 表示线在图像边缘。
                        # 像素值量纲会因增益过大而恒饱和到角速度上限，等价于
                        # 满舵开关，车只会画龙。
                        # 乘 line_steer_sign 把符号映射到本机图像方向：正偏差
                        # 表示线在图像右侧，图像未旋转时应右转，而 angular.z
                        # 为负即右转，故默认符号为 -1。
                        steering = self._line_steer_sign * self._pid.compute(
                            line_observation.offset_norm, dt
                        )
                    elif self._line_detector.is_lost:
                        # 丢线保护：停止推进，交由上层处理重定位或退格
                        self._base_driver.stop()
                        self._node.get_logger().warn(
                            f'连续丢线 {line_observation.lost_frames} 帧，停止推进'
                        )
                        return False

            self._base_driver.set_velocity(speed, 0.0, steering)

            current_pose = self._base_driver.get_pose()
            if current_pose is not None:
                traveled = math.hypot(
                    current_pose[0] - start_pose[0],
                    current_pose[1] - start_pose[1],
                )
                # 激光交叉验证：出发时若前方有墙，可用距墙距离的减少量独立估计
                # 位移，取两者较小值作保守估计，抑制轮式里程计积分漂移导致的冲
                # 过头。前方是通道即无回波时自动跳过。
                if wall_start_distance is not None:
                    wall_now = self.front_range()
                    if wall_now is not None:
                        wall_traveled = wall_start_distance - wall_now
                        if wall_traveled > 0:
                            traveled = min(traveled, wall_traveled)
                if traveled >= target_distance:
                    break

            self._spin_once(LOOP_SPIN_SEC)

        self._base_driver.stop()
        self._spin_once(STOP_SPIN_SEC)
        arrived = traveled >= target_distance
        if not arrived:
            self._node.get_logger().warn(
                f'未到目标位移：{traveled:.3f}m（目标 {abs(distance):.3f}m）'
            )
        return arrived

    def turn_to_heading(
        self, target_heading: Heading, timeout: Optional[float] = None
    ) -> bool:
        """原地旋转到目标网格朝向，取值见 ``Heading``。

        安全门禁为激光失效时不启动转向。转向本身只看 yaw，但盲转意味着转完后
        无法确认周围环境，故门禁放在动作开始前，避免把车停在半路。

        :param target_heading: 目标朝向，取值 N、E、S、W 之一。
        :param timeout: 超时上限，单位 s，取 ``None`` 时用 ``turn_timeout``。
        :returns: 残余误差不超过容差为真；无里程计或传感器失效未恢复为假。
        """
        if self._stop_if_unhealthy(need_vision=False):
            return False

        current_yaw = self._base_driver.get_yaw()
        if current_yaw is None:
            self._node.get_logger().warn('无里程计数据，无法转向')
            return False

        target_yaw = GridMapper.heading_yaw(target_heading)
        # odom 转角存在系统性缩放，标定语义为真实转角等于 ang_scale 乘 odom
        # 读数，因此要达成目标真实转角，需要 odom 变化量为目标除以 ang_scale。
        # 此处必须是除不是乘：乘会让 90 度只转 67.5 度，在 ang_scale 为 0.75
        # 时逐格累积必然撞墙。
        yaw_delta = angle_diff(target_yaw, current_yaw) / self._angular_scale
        absolute_target_yaw = current_yaw + yaw_delta

        time_limit = float(timeout if timeout is not None else self._turn_timeout)
        start_time = time.monotonic()
        heading_error = yaw_delta
        while rclpy.ok() and (time.monotonic() - start_time) < time_limit:
            current_yaw = self._base_driver.get_yaw()
            if current_yaw is None:
                break
            heading_error = angle_diff(absolute_target_yaw, current_yaw)
            if abs(heading_error) <= self._yaw_tolerance:
                break
            angular_z = _clamp(
                TURN_PROPORTIONAL_GAIN * heading_error, self._turn_angular
            )
            # 静摩擦补偿：给一个最小有效角速度，避免小误差下卡住
            if abs(angular_z) < MIN_TURN_ANGULAR_RAD_S:
                angular_z = math.copysign(MIN_TURN_ANGULAR_RAD_S, angular_z)
            self._base_driver.set_velocity(0.0, 0.0, angular_z)
            self._spin_once(LOOP_SPIN_SEC)

        self._base_driver.stop()
        self._spin_once(STOP_SPIN_SEC)
        acceptable = abs(heading_error) <= max(
            self._yaw_tolerance, MIN_ACCEPT_YAW_TOLERANCE_RAD
        )
        if not acceptable:
            self._node.get_logger().warn(
                f'转向未收敛：残余 {math.degrees(heading_error):+.1f} 度'
            )
        return acceptable

    def strafe(
        self, distance: float, timeout: float = DEFAULT_STRAFE_TIMEOUT_SEC
    ) -> bool:
        """横移指定距离，正值向左，用里程计闭环判定到位。

        麦轮横移误差比纵向大，仅用于抓取对位等小距离场景。位移按车体左方向在
        世界系中投影后累加，故不受当前朝向影响。传感器不新鲜时不动。

        :param distance: 目标横移距离，单位 m，正值向左负值向右。
        :param timeout: 超时上限，单位 s。
        :returns: 在容差内到位为真；传感器失效或无里程计为假。
        """
        if self._stop_if_unhealthy(need_vision=False):
            return False

        start_pose = self._base_driver.get_pose()
        yaw = self._base_driver.get_yaw()
        if start_pose is None or yaw is None:
            return False

        sign = 1.0 if distance >= 0 else -1.0
        magnitude = abs(distance)
        # 速度按距离增益计算后截断到上下限，距离大时取上限避免耗时过长
        lateral_speed = (
            min(
                STRAFE_MAX_SPEED_M_S,
                max(magnitude * STRAFE_SPEED_GAIN, STRAFE_MIN_SPEED_M_S),
            )
            * sign
        )
        # 车体左方向在世界系中的单位向量
        left_x, left_y = -math.sin(yaw), math.cos(yaw)

        start_time = time.monotonic()
        moved = 0.0
        while rclpy.ok() and (time.monotonic() - start_time) < timeout:
            current_pose = self._base_driver.get_pose()
            if current_pose is not None:
                moved = (current_pose[0] - start_pose[0]) * left_x + (
                    current_pose[1] - start_pose[1]
                ) * left_y
                if abs(moved) >= magnitude - self._cell_tolerance:
                    break
            self._base_driver.set_velocity(0.0, lateral_speed, 0.0)
            self._spin_once(LOOP_SPIN_SEC)

        self._base_driver.stop()
        self._spin_once(STOP_SPIN_SEC)
        return abs(moved) >= magnitude - self._cell_tolerance

    def _emergency_backoff(self, distance: Optional[float] = None) -> None:
        """碰撞保护：后退指定距离并停稳，默认半格。

        :param distance: 后退距离，单位 m，取 ``None`` 时用半格宽。
        """
        back_distance = (
            EMERGENCY_BACKOFF_CELL_RATIO * self.cell_size
            if distance is None
            else float(distance)
        )
        start_pose = self._base_driver.get_pose()
        if start_pose is None:
            return
        start_time = time.monotonic()
        while rclpy.ok() and (time.monotonic() - start_time) < EMERGENCY_BACKOFF_TIMEOUT_SEC:
            current_pose = self._base_driver.get_pose()
            if current_pose is not None:
                traveled = math.hypot(
                    current_pose[0] - start_pose[0],
                    current_pose[1] - start_pose[1],
                )
                if traveled >= back_distance:
                    break
            self._base_driver.set_velocity(
                EMERGENCY_BACKOFF_SPEED_M_S, 0.0, 0.0
            )
            self._spin_once(LOOP_SPIN_SEC)
        self._base_driver.stop()
        self._spin_once(STOP_SPIN_SEC)

    def stop(self) -> None:
        """立即停止，可被外部中断处理调用。"""
        self._base_driver.stop()
        self._spin_once(STOP_SPIN_SEC)
