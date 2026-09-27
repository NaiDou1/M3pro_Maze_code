"""机械臂抓取状态机。

替代 ``M3Pro_demo/grasp.py``——后者用 ``time.sleep(2.5)`` 硬编码各步节拍，
既无法中断也难以压缩时间，而抓取是整场最大的耗时来源（8 块 x 10~20s）。
本模块改为显式状态机：每步下发动作后等待''预计到位时刻''，支持超时与中断，
失败自动重试一次，二次失败登记待办而不阻塞整体流程。

IK 请求约定（沿用现有 demo 实测可用的取值）
-------------------------------------------
* ``tar_x = target_x + 0.03``：前向补偿 3cm，避免夹爪撞到方块
* ``roll = pi``：夹爪朝下
* ``pitch / yaw``：取当前末端姿态（FK），保证姿态连续、避免 IK 跳解
* ``joint6``（夹爪）**不参与 IK**，由抓取逻辑直接给定

.. warning::
   现有 demo 对 joint5 有一组依赖方块朝向的经验修正。本模块把它抽象为
   ``joint5_mode``（``fixed`` / ``from_yaw``）。默认 ``fixed`` 取中位 90 度。
   **现场必须按实际夹爪装配与方块摆放复核或重标**，否则可能夹偏。
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import List, Optional, Sequence

import rclpy
from rclpy.node import Node

from maze_explorer.arm_controller import ArmController
from maze_explorer.block_detector import BlockDetection


class GraspState(Enum):
    """抓取流程状态。"""

    IDLE = auto()
    PREPARE = auto()    # 张开夹爪
    SOLVE = auto()      # IK 求解
    DESCEND = auto()    # 下探到目标
    CLOSE = auto()      # 闭合夹爪
    LIFT = auto()       # 抬起
    PLACE = auto()      # 移到收集筐上方
    RELEASE = auto()    # 释放
    RETREAT = auto()    # 回到巡线姿态
    DONE = auto()
    FAILED = auto()


class GraspFSM:
    """单个方块的抓取流程。"""

    def __init__(self, node: Node, arm: ArmController) -> None:
        self._node = node
        self._arm = arm

        # 位姿与夹爪（来自 config/arm_poses.yaml）
        self._line_pose = self._joints_param('line_pose', [90, 90, 12, 20, 90, 0])
        self._grasp_home = self._joints_param('grasp_home', [90, 150, 12, 20, 90, 30])
        self._place_pose = self._joints_param('place_pose', [180, 90, 12, 20, 90, 30])
        self._gripper_open = self._int_param('gripper_open_angle', 30)
        self._gripper_closed = self._int_param('gripper_closed_angle', 120)

        # 包络与时长
        self._dist_min = self._float_param('grasp_distance_min', 0.13)
        self._dist_max = self._float_param('grasp_distance_max', 0.25)
        self._move_ms = self._int_param('move_time_ms', 2000)
        self._grasp_ms = self._int_param('grasp_time_ms', 1200)
        self._settle_sec = self._float_param('settle_sec', 0.3)

        #: 前向补偿（m），沿用 demo 的 3cm
        self._forward_comp = self._float_param('forward_compensation', 0.03)
        #: 抬起时的 joint2 角度
        self._lift_joint2 = self._int_param('lift_joint2', 120)
        #: joint5 策略：fixed 用固定值；from_yaw 由方块角点朝向推算
        self._joint5_mode = str(self._node.declare_parameter(
            'joint5_mode', 'fixed').value)
        self._joint5_fixed = self._int_param('joint5_fixed', 90)

        self._state = GraspState.IDLE
        self._interrupted = False

    # ------------------------------------------------------------------ 参数

    def _joints_param(self, name: str, default: Sequence[int]) -> List[int]:
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, list(default))
        return [int(v) for v in self._node.get_parameter(name).value]

    def _int_param(self, name: str, default: int) -> int:
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, int(default))
        return int(self._node.get_parameter(name).value)

    def _float_param(self, name: str, default: float) -> float:
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, float(default))
        return float(self._node.get_parameter(name).value)

    # ------------------------------------------------------------------ 查询

    @property
    def state(self) -> GraspState:
        return self._state

    def interrupt(self) -> None:
        """请求中断（协作式，在状态切换点生效）。"""
        self._interrupted = True

    # ------------------------------------------------------------------ 主流程

    def in_envelope(self, target: BlockDetection) -> bool:
        """目标是否落在有效抓取包络内。"""
        d = target.horizontal_distance()
        return self._dist_min <= d <= self._dist_max

    def execute(self, target: BlockDetection) -> bool:
        """执行一次完整抓取（含一次重试）。成功返回 ``True``。"""
        self._interrupted = False

        if not self.in_envelope(target):
            d = target.horizontal_distance()
            self._node.get_logger().warn(
                f'{target.color} 方块水平距离 {d:.3f}m 不在包络 '
                f'[{self._dist_min:.2f}, {self._dist_max:.2f}]，需先对位'
            )
            return False

        for attempt in (1, 2):
            if self._interrupted:
                break
            if self._run_sequence(target):
                self._retreat()
                self._state = GraspState.DONE
                self._node.get_logger().info(
                    f'抓取成功：{target.color}（第 {attempt} 次尝试）'
                )
                return True
            self._node.get_logger().warn(
                f'抓取 {target.color} 第 {attempt} 次失败'
                + ('，将重试一次' if attempt == 1 else '，登记为待办')
            )
            # 重试前先回准备位，避免姿态残留
            self._arm.send_joints(self._grasp_home, self._move_ms)
            self._arm.wait_until_ready()

        self._state = GraspState.FAILED
        return False

    # -------------------------------------------------------------- 单次序列

    def _run_sequence(self, target: BlockDetection) -> bool:
        # 1) 张开夹爪
        self._state = GraspState.PREPARE
        self._arm.set_gripper(False, self._grasp_ms)
        if not self._arm.wait_until_ready():
            return False

        # 2) IK 求解
        self._state = GraspState.SOLVE
        position = target.position_base
        if position is None:
            # 外参未标定：退化为相机系近似（前=水平距离，左=横向，上=0）
            self._node.get_logger().warn(
                'position_base 为空（相机外参未标定），改用相机系近似定位'
            )
            position = (target.horizontal_distance(), -target.lateral_m, 0.0)

        fk = self._arm.solve_fk()
        pitch = fk[4] if fk is not None else 0.0
        yaw = fk[5] if fk is not None else 0.0

        joints = self._arm.solve_ik(
            x=position[0] + self._forward_comp,
            y=position[1],
            z=position[2],
            roll=math.pi,
            pitch=pitch,
            yaw=yaw,
        )
        if joints is None:
            return False

        joints = self._sanitize(joints, target)

        # 3) 下探到目标
        self._state = GraspState.DESCEND
        self._arm.send_joints(joints, self._move_ms)
        if not self._arm.wait_until_ready(self._move_ms / 1000.0 + 2.0):
            return False
        self._settle()

        # 4) 闭合夹爪
        self._state = GraspState.CLOSE
        self._arm.set_gripper(True, self._grasp_ms)
        if not self._arm.wait_until_ready(self._grasp_ms / 1000.0 + 2.0):
            return False
        self._settle()

        # 5) 抬起
        self._state = GraspState.LIFT
        lifted = list(joints)
        lifted[1] = self._lift_joint2
        self._arm.send_joints(lifted, self._move_ms)
        if not self._arm.wait_until_ready(self._move_ms / 1000.0 + 2.0):
            return False

        # 6) 移到收集筐
        self._state = GraspState.PLACE
        self._arm.send_joints(self._place_pose, self._move_ms)
        if not self._arm.wait_until_ready(self._move_ms / 1000.0 + 2.0):
            return False

        # 7) 释放
        self._state = GraspState.RELEASE
        self._arm.set_gripper(False, self._grasp_ms)
        if not self._arm.wait_until_ready(self._grasp_ms / 1000.0 + 2.0):
            return False
        self._settle()
        return True

    def _retreat(self) -> None:
        """回到巡线姿态，使相机恢复看地视角。"""
        self._state = GraspState.RETREAT
        self._arm.send_joints(self._line_pose, self._move_ms)
        self._arm.wait_until_ready()

    def _settle(self) -> None:
        if self._settle_sec > 0:
            import time

            time.sleep(self._settle_sec)

    def _sanitize(self, joints: Sequence[float], target: BlockDetection) -> List[int]:
        """按现有 demo 的经验规则裁剪关节角，直接可用于下发。"""
        clamped = [max(0, min(180, int(round(v)))) for v in joints[:6]]
        # joint4 超过 90 会导致腕部翻转，demo 里直接截断
        if clamped[3] > 90:
            clamped[3] = 90
        # joint5：夹爪朝向，默认取中位
        clamped[4] = self._resolve_joint5(target)
        # joint6 由抓取逻辑决定，不采用 IK 结果
        clamped[5] = self._gripper_open
        return clamped

    def _resolve_joint5(self, target: BlockDetection) -> int:
        if self._joint5_mode == 'from_yaw' and target.yaw_rad is not None:
            # 由方块角点朝向推算夹爪旋转角，并限制在可用区间
            angle = math.degrees(target.yaw_rad) % 180.0
            if angle > 135:
                angle -= 90
            elif angle < 45:
                angle += 90
            return max(0, min(180, int(round(angle))))
        return self._joint5_fixed
