"""运动控制：巡线走格、原地转向、横移微调与碰撞保护。

把连续运动离散成两个原语，使误差**不跨格累积**：

* :meth:`MotionController.advance_one_cell` —— 沿黑线前进一格（0.4m），巡线 PID
  控横向，里程计闭环判到位；
* :meth:`MotionController.turn_to_heading` —— 麦轮原地转到目标网格朝向。

.. important::
   所有原语都是**阻塞式**的：内部统一通过 :meth:`MotionController._spin_once`
   驱动回调。组合多个节点时由调用方注入 ``spin_fn``（通常是
   ``executor.spin_once``）。

   **任何绕过 ``_spin_once`` 直接调用 ``rclpy.spin_once`` 的写法都会导致子节点
   回调不被驱动**，闭环将读到过期数据（本项目曾因此出现转向失控：yaw 永不更新，
   车以固定角速度转到超时）。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from maze_explorer.base_driver import BaseDriver
from maze_explorer.grid_mapper import GridMapper, angle_diff
from maze_explorer.line_detector import LineDetector
from maze_explorer.sensors import SensorHub


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class PID:
    """极简增量式 PID（带输出限幅与积分限幅）。"""

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        out_limit: Optional[float] = None,
        i_limit: Optional[float] = None,
    ) -> None:
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.out_limit = out_limit
        self.i_limit = i_limit
        self._integral = 0.0
        self._prev_error: Optional[float] = None

    def compute(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-3)
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
        self._integral = 0.0
        self._prev_error = None


class MotionController:
    """基于底盘/传感器/巡线检测器的运动原语集合。"""

    def __init__(
        self,
        node: Node,
        base: BaseDriver,
        sensors: SensorHub,
        line_detector: LineDetector,
        *,
        #: 注入的 spin 回调（timeout 秒）。组合多个节点时由调用方传入
        #: ``executor.spin_once``，否则默认只驱动传入的 ``node``。
        spin_fn: Optional[Callable[[float], None]] = None,
        cell_size: float = 0.40,
        cruise_linear: float = 0.15,
        max_angular_z: float = 0.60,
        turn_angular: float = 0.50,
        line_pid: Tuple[float, float, float] = (50.0, 0.0, 10.0),
        cell_tolerance: float = 0.03,
        yaw_tolerance: float = 0.0873,
        safety_range: float = 0.25,
        opening_min_range: float = 0.55,
        advance_timeout: float = 12.0,
        turn_timeout: float = 8.0,
        angular_scale_correction: float = 1.0,
        #: 激光保鲜阈值（秒）：超过即视为失效，禁止移动
        scan_timeout: float = 0.5,
        #: 相机保鲜阈值（秒）：超过即视为失效，禁止前进（循迹依赖它）
        vision_timeout: float = 2.0,
    ) -> None:
        self._node = node
        self._base = base
        self._sensors = sensors
        self._line = line_detector
        self._spin_fn = spin_fn

        self.cell_size = float(cell_size)
        self._cruise = float(cruise_linear)
        self._max_wz = float(max_angular_z)
        self._turn_wz = float(turn_angular)
        self._tolerance = float(cell_tolerance)
        self._yaw_tol = float(yaw_tolerance)
        self._safety = float(safety_range)
        self._opening_range = float(opening_min_range)
        self._advance_timeout = float(advance_timeout)
        self._turn_timeout = float(turn_timeout)
        self._scan_timeout = float(scan_timeout)
        self._vision_timeout = float(vision_timeout)
        #: odom 转角缩放系数：真实转角 = 系数 × odom 读数。
        #: 需用 calibration_tool 的 motion 模式实测得出，1.0 表示不做修正。
        #: 注意方向：闭环目标是「目标转角 / 系数」，见 turn_to_heading。
        self._ang_scale = float(angular_scale_correction)
        if abs(self._ang_scale - 1.0) > 1e-6:
            self._node.get_logger().warn(
                f'已启用转向缩放修正 ang_scale={self._ang_scale:.3f}；'
                '该值应由 calibration_tool motion 模式实测得出，'
                '否则每次转向都会系统性偏角'
            )

        self._pid = PID(
            kp=line_pid[0], ki=line_pid[1], kd=line_pid[2], out_limit=self._max_wz
        )

    # ------------------------------------------------------------ 传感器辅助

    def _spin_once(self, timeout: float) -> None:
        """驱动一次回调：优先用注入的 spin 回调，否则只转传入的节点。"""
        if self._spin_fn is not None:
            self._spin_fn(timeout)
        else:
            rclpy.spin_once(self._node, timeout_sec=timeout)

    def _latest_image(self) -> Optional[np.ndarray]:
        rgbd = self._sensors.get_rgbd()
        return None if rgbd is None else rgbd[0]

    # ------------------------------------------------------------ 传感器健康

    def _sensor_age(self, name: str) -> float:
        """读取某项传感器数据的年龄（秒）。

        ``name`` 取 ``'scan_age'`` 或 ``'rgbd_age'``。传感器对象未提供该接口时
        返回 0（视为健康）——这样纯逻辑单测可以传桩对象，不必搭 ROS 运行时。
        """
        getter = getattr(self._sensors, name, None)
        if getter is None:
            return 0.0
        try:
            return float(getter())
        except Exception:  # noqa: BLE001 - 健康检查本身不应影响主流程
            return 0.0

    def scan_age(self) -> float:
        """激光数据年龄（秒）；无数据为 ``inf``。"""
        return self._sensor_age('scan_age')

    def vision_age(self) -> float:
        """相机数据年龄（秒）；无数据为 ``inf``。"""
        return self._sensor_age('rgbd_age')

    def sensors_ok(self, need_vision: bool = True) -> bool:
        """传感器是否新鲜可用。

        :param need_vision: 是否同时要求相机新鲜（前进循迹需要，原地转向不需要）。
        """
        if self.scan_age() > self._scan_timeout:
            return False
        if need_vision and self.vision_age() > self._vision_timeout:
            return False
        return True

    def _stop_if_unhealthy(self, need_vision: bool) -> bool:
        """传感器失效时立即停车并返回 ``True``（调用方应放弃本次动作）。

        为什么必须硬失败：激光失效时 ``sector_min_range`` 返回 ``None``，而
        ``is_path_clear`` 与碰撞保护都把 ``None`` 当"通畅"；相机失效时循迹偏差
        恒为 0，车会闷头直行。两者都会让"保护"退化成"放任"，因此宁可停车。
        """
        if self.sensors_ok(need_vision):
            return False
        self._base.stop()
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
        """正前方最近障碍距离（米）。"""
        return self._sensors.sector_min_range(0.0, 15.0)

    def is_front_blocked(self) -> bool:
        """正前方是否进入碰撞保护阈值。"""
        d = self.front_range()
        return d is not None and d < self._safety

    def scan_openings(self) -> Tuple[bool, bool, bool]:
        """返回车体系 ``(前方, 左方, 右方)`` 是否有开口。

        以激光扇区最近距离与 ``opening_min_range`` 比较；扇区无有效点（视为
        远处通透）也判为开口。
        """
        r = self._opening_range
        return (
            self._sensors.is_path_clear(0.0, 20.0, r),
            self._sensors.is_path_clear(90.0, 20.0, r),
            self._sensors.is_path_clear(-90.0, 20.0, r),
        )

    # ------------------------------------------------------------ 运动原语

    def advance_one_cell(self) -> bool:
        """沿黑线前进一格（``cell_size``）。"""
        return self.advance(self.cell_size)

    def advance(self, distance: float, timeout: Optional[float] = None) -> bool:
        """沿黑线前进指定距离（米），负值为后退。返回是否在容差内到位。

        巡线 PID 只作用于前进方向；后退用于脱离贴墙等场景，不做巡线。

        安全门禁：**传感器不新鲜一律不动**。前进需要激光（碰撞保护）与相机
        （循迹）；后退只需要激光。门禁在动作开始前与每个控制周期内各查一次，
        因此行进途中掉线也会立刻停车。
        """
        forward = distance >= 0
        if self._stop_if_unhealthy(need_vision=forward):
            return False

        start = self._base.get_pose()
        if start is None:
            self._node.get_logger().warn('无里程计数据，无法走格')
            return False

        target = abs(distance) - self._tolerance
        limit = float(timeout if timeout is not None else self._advance_timeout)
        speed = self._cruise if forward else -0.5 * self._cruise
        #: 出发时的前墙距离（前方无墙/无回波则为 None），用于激光交叉验证
        wall_start = self.front_range() if forward else None

        self._pid.reset()
        t0 = time.monotonic()
        last_t = t0
        traveled = 0.0

        while rclpy.ok() and (time.monotonic() - t0) < limit:
            now = time.monotonic()
            dt = now - last_t
            last_t = now

            # 行进途中掉线也要立刻停：否则保护退化为"放任"
            if self._stop_if_unhealthy(need_vision=forward):
                return False

            if forward and self.is_front_blocked():
                self._base.stop()
                self._node.get_logger().warn(
                    f'前方 {self.front_range():.2f}m 触发碰撞保护，执行回退'
                )
                self._emergency_backoff()
                return False

            steering = 0.0
            if forward:
                img = self._latest_image()
                if img is not None:
                    obs = self._line.detect(img)
                    if obs.valid:
                        steering = self._pid.compute(obs.offset_px, dt)
                    elif self._line.is_lost:
                        # 丢线保护：停止推进，交由上层处理（重定位或退格）
                        self._base.stop()
                        self._node.get_logger().warn(
                            f'连续丢线 {obs.lost_frames} 帧，停止推进'
                        )
                        return False

            self._base.set_velocity(speed, 0.0, steering)

            cur = self._base.get_pose()
            if cur is not None:
                traveled = math.hypot(cur[0] - start[0], cur[1] - start[1])
                # 激光交叉验证：出发时若前方有墙，可用「距墙距离的减少量」独立
                # 估计位移，取两者较小值作保守估计，抑制轮式里程计积分漂移导致
                # 的冲过头。前方是通道（无回波）时自动跳过。
                if wall_start is not None:
                    wall_now = self.front_range()
                    if wall_now is not None:
                        wall_traveled = wall_start - wall_now
                        if wall_traveled > 0:
                            traveled = min(traveled, wall_traveled)
                if traveled >= target:
                    break

            self._spin_once(0.02)

        self._base.stop()
        self._spin_once(0.02)
        arrived = traveled >= target
        if not arrived:
            self._node.get_logger().warn(
                f'未到目标位移：{traveled:.3f}m（目标 {abs(distance):.3f}m）'
            )
        return arrived

    def turn_to_heading(self, target_heading: str, timeout: Optional[float] = None) -> bool:
        """原地旋转到目标网格朝向（'N'/'E'/'S'/'W'）。

        安全门禁：激光失效时**不启动转向**（转向本身只看 yaw，但盲转意味着
        转完后无法确认周围环境）。门禁放在动作开始前，避免把车停在半路。
        """
        if self._stop_if_unhealthy(need_vision=False):
            return False

        cur_yaw = self._base.get_yaw()
        if cur_yaw is None:
            self._node.get_logger().warn('无里程计数据，无法转向')
            return False

        target_yaw = GridMapper.heading_yaw(target_heading)
        # odom 转角存在系统性缩放。calibrate_angular.py 的实测语义是：
        #     真实转角 = ang_scale × odom 读数
        # 所以要达成目标真实转角，需要 odom 变化「目标 / ang_scale」。
        # 注意是**除**不是乘——乘会让 90 度只转 67.5 度（ang_scale=0.75 时），
        # 逐格累积必然撞墙。
        desired = angle_diff(target_yaw, cur_yaw) / self._ang_scale
        target = cur_yaw + desired

        limit = float(timeout if timeout is not None else self._turn_timeout)
        t0 = time.monotonic()
        err = desired
        while rclpy.ok() and (time.monotonic() - t0) < limit:
            cur_yaw = self._base.get_yaw()
            if cur_yaw is None:
                break
            err = angle_diff(target, cur_yaw)
            if abs(err) <= self._yaw_tol:
                break
            wz = _clamp(2.0 * err, self._turn_wz)
            # 静摩擦补偿：给一个最小有效角速度，避免小误差下卡住
            if abs(wz) < 0.12:
                wz = math.copysign(0.12, wz)
            self._base.set_velocity(0.0, 0.0, wz)
            self._spin_once(0.02)

        self._base.stop()
        self._spin_once(0.02)
        ok = abs(err) <= max(self._yaw_tol, 0.05)
        if not ok:
            self._node.get_logger().warn(
                f'转向未收敛：残余 {math.degrees(err):+.1f} 度'
            )
        return ok

    def strafe(self, distance: float, timeout: float = 6.0) -> bool:
        """横移指定距离（米），正值为向左，用里程计闭环。

        麦轮横移误差比纵向大，仅用于抓取对位等小距离场景。位移按车体左方向
        在世界系中投影后累加，故不受当前朝向影响。传感器不新鲜时不动。
        """
        if self._stop_if_unhealthy(need_vision=False):
            return False

        start = self._base.get_pose()
        yaw = self._base.get_yaw()
        if start is None or yaw is None:
            return False

        sign = 1.0 if distance >= 0 else -1.0
        magnitude = abs(distance)
        v = min(0.12, max(magnitude * 1.5, 0.06)) * sign
        # 车体左方向在世界系中的单位向量
        left_x, left_y = -math.sin(yaw), math.cos(yaw)

        t0 = time.monotonic()
        moved = 0.0
        while rclpy.ok() and (time.monotonic() - t0) < timeout:
            cur = self._base.get_pose()
            if cur is not None:
                moved = (cur[0] - start[0]) * left_x + (cur[1] - start[1]) * left_y
                if abs(moved) >= magnitude - self._tolerance:
                    break
            self._base.set_velocity(0.0, v, 0.0)
            self._spin_once(0.02)

        self._base.stop()
        self._spin_once(0.02)
        return abs(moved) >= magnitude - self._tolerance

    def _emergency_backoff(self, distance: Optional[float] = None) -> None:
        """碰撞保护：后退半格并停稳。"""
        back = 0.5 * self.cell_size if distance is None else float(distance)
        start = self._base.get_pose()
        if start is None:
            return
        t0 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t0) < 3.0:
            cur = self._base.get_pose()
            if cur is not None:
                traveled = math.hypot(cur[0] - start[0], cur[1] - start[1])
                if traveled >= back:
                    break
            self._base.set_velocity(-0.10, 0.0, 0.0)
            self._spin_once(0.02)
        self._base.stop()
        self._spin_once(0.02)

    def stop(self) -> None:
        """立即停止（可被外部中断处理调用）。"""
        self._base.stop()
        self._spin_once(0.01)
