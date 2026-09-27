"""运动控制：巡线走格、原地转向、横移微调与碰撞保护。

把连续运动离散成两个原语，使误差**不跨格累积**：

* :meth:`MotionController.advance_one_cell` —— 沿黑线前进一格（0.4m），巡线 PID
  控横向，里程计闭环判到位；
* :meth:`MotionController.turn_to_heading` —— 麦轮原地转到目标网格朝向。

.. important::
   所有原语都是**阻塞式**的，内部自行调用 ``rclpy.spin_once`` 驱动回调。
   因此**调用方不得再对该节点额外 spin**，否则会出现双重驱动。
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

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
    ) -> None:
        self._node = node
        self._base = base
        self._sensors = sensors
        self._line = line_detector

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
        #: odom 转角系统性缩放系数，需用 calibration_tool motion 模式实测确定
        self._ang_scale = float(angular_scale_correction)

        self._pid = PID(
            kp=line_pid[0], ki=line_pid[1], kd=line_pid[2], out_limit=self._max_wz
        )

    # ------------------------------------------------------------ 传感器辅助

    def _latest_image(self) -> Optional[np.ndarray]:
        rgbd = self._sensors.get_rgbd()
        return None if rgbd is None else rgbd[0]

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
        """沿黑线前进一格。返回是否在容差内到位。"""
        start = self._base.get_pose()
        if start is None:
            self._node.get_logger().warn('无里程计数据，无法走格')
            return False

        self._pid.reset()
        t0 = time.monotonic()
        last_t = t0
        traveled = 0.0
        target = self.cell_size - self._tolerance

        while rclpy.ok() and (time.monotonic() - t0) < self._advance_timeout:
            now = time.monotonic()
            dt = now - last_t
            last_t = now

            if self.is_front_blocked():
                self._base.stop()
                self._node.get_logger().warn(
                    f'前方 {self.front_range():.2f}m 触发碰撞保护，执行回退'
                )
                self._emergency_backoff()
                return False

            steering = 0.0
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

            self._base.set_velocity(self._cruise, 0.0, steering)

            cur = self._base.get_pose()
            if cur is not None:
                traveled = math.hypot(cur[0] - start[0], cur[1] - start[1])
                if traveled >= target:
                    break

            rclpy.spin_once(self._node, timeout_sec=0.02)

        self._base.stop()
        rclpy.spin_once(self._node, timeout_sec=0.02)
        arrived = traveled >= target
        if not arrived:
            self._node.get_logger().warn(
                f'走格未到位：位移 {traveled:.3f}m（目标 {self.cell_size:.3f}m）'
            )
        return arrived

    def turn_to_heading(self, target_heading: str, timeout: Optional[float] = None) -> bool:
        """原地旋转到目标网格朝向（'N'/'E'/'S'/'W'）。"""
        cur_yaw = self._base.get_yaw()
        if cur_yaw is None:
            self._node.get_logger().warn('无里程计数据，无法转向')
            return False

        target_yaw = GridMapper.heading_yaw(target_heading)
        # odom 转角存在系统性缩放，按标定系数补偿后闭环
        desired = angle_diff(target_yaw, cur_yaw) * self._ang_scale
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
            rclpy.spin_once(self._node, timeout_sec=0.02)

        self._base.stop()
        rclpy.spin_once(self._node, timeout_sec=0.02)
        ok = abs(err) <= max(self._yaw_tol, 0.05)
        if not ok:
            self._node.get_logger().warn(
                f'转向未收敛：残余 {math.degrees(err):+.1f} 度'
            )
        return ok

    def strafe(self, distance: float, timeout: float = 6.0) -> bool:
        """横移指定距离（米），正值为向左，用里程计闭环。

        麦轮横移误差比纵向大，仅用于抓取对位等小距离场景。位移按车体左方向
        在世界系中投影后累加，故不受当前朝向影响。
        """
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
            rclpy.spin_once(self._node, timeout_sec=0.02)

        self._base.stop()
        rclpy.spin_once(self._node, timeout_sec=0.02)
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
            rclpy.spin_once(self._node, timeout_sec=0.02)
        self._base.stop()
        rclpy.spin_once(self._node, timeout_sec=0.02)

    def stop(self) -> None:
        """立即停止（可被外部中断处理调用）。"""
        self._base.stop()
        rclpy.spin_once(self._node, timeout_sec=0.01)
