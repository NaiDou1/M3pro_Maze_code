"""机械臂抓取状态机。

替代 ``M3Pro_demo/grasp.py``：后者用固定 ``time.sleep`` 硬编码各步节拍，既无法
中断也难以压缩时间，而抓取是整场最大的耗时来源，8 块每块 10 到 20 s。本模块
改为显式状态机：每步下发动作后等待预计到位时刻，支持超时与中断，失败自动重试
一次，二次失败登记待办而不阻塞整体流程。

IK 请求约定，沿用现有 demo 实测可用的取值
-------------------------------------------
* ``tar_x`` 为 target_x 加前向补偿 0.03 m，避免夹爪撞到方块
* ``roll`` 取 pi 即夹爪朝下
* ``pitch`` 与 ``yaw`` 取当前末端姿态即 FK 结果，保证姿态连续、避免 IK 跳解
* ``joint6`` 即夹爪不参与 IK，由抓取逻辑直接给定

.. warning::
   现有 demo 对 joint5 有一组依赖方块朝向的经验修正。本模块把它抽象为
   ``joint5_mode``，取值 fixed 或 from_yaw，默认 fixed 取中位 90 度。现场必须按
   实际夹爪装配与方块摆放复核或重标，否则可能夹偏。
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import List, Sequence

from rclpy.node import Node

from maze_explorer._compat import StrEnum
from maze_explorer.arm_controller import (
    JOINT_MAX_DEG,
    JOINT_MIN_DEG,
    MILLISECONDS_PER_SECOND,
    ArmController,
)
from maze_explorer.block_detector import BlockDetection

#: 每步等待到位时在动作耗时之外附加的宽限，单位 s，覆盖下发链路延迟
WAIT_GRACE_SEC = 2.0
#: joint4 的允许上限，单位度，超过会导致腕部翻转
WRIST_JOINT4_MAX_DEG = 90
#: joint5 按方块朝向折叠时的上下界，单位度，区间外按 90 度步长折回可用区间
JOINT5_FOLD_UPPER_DEG = 135
JOINT5_FOLD_LOWER_DEG = 45
JOINT5_FOLD_STEP_DEG = 90
#: 连续 2 次尝试后仍失败即登记待办
MAX_GRASP_ATTEMPTS = 2


class GraspState(Enum):
    """抓取流程状态，顺序即正常流转顺序。"""

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


class Joint5Mode(StrEnum):
    """joint5 取值策略，取值 fixed 或 from_yaw。

    fixed 使用固定角度，装配关系简单时使用；from_yaw 由方块角点朝向推算，用于
    夹爪需对准方块棱边的场合。
    """

    FIXED = 'fixed'
    FROM_YAW = 'from_yaw'


class GraspFSM:
    """单个方块的抓取流程，一次实例管理一条流程记录。"""

    def __init__(self, node: Node, arm: ArmController) -> None:
        """读取位姿、包络与节拍参数，默认值与 config 一致。

        :param node: 提供参数声明与日志的节点。
        :param arm: 机械臂控制器，状态机不拥有其生命周期。
        """
        self._node = node
        self._arm = arm

        # 位姿与夹爪，来自 config/arm_poses.yaml
        self._line_pose = self._joints_param('line_pose', [90, 90, 12, 20, 90, 0])
        self._grasp_home = self._joints_param(
            'grasp_home', [90, 150, 12, 20, 90, 30]
        )
        self._place_pose = self._joints_param('place_pose', [180, 90, 12, 20, 90, 30])
        self._gripper_open_deg = self._int_param('gripper_open_angle', 30)
        self._gripper_closed_deg = self._int_param('gripper_closed_angle', 120)

        # 包络与时长
        self._grasp_distance_min_m = self._float_param('grasp_distance_min', 0.13)
        self._grasp_distance_max_m = self._float_param('grasp_distance_max', 0.25)
        self._move_time_ms = self._int_param('move_time_ms', 2000)
        self._grasp_time_ms = self._int_param('grasp_time_ms', 1200)
        self._settle_sec = self._float_param('settle_sec', 0.3)

        #: 前向补偿，单位 m，沿用 demo 的 3 cm，避免夹爪撞到方块
        self._forward_compensation_m = self._float_param(
            'forward_compensation', 0.03
        )
        #: 抬起时的 joint2 角度，单位度
        self._lift_joint2_deg = self._int_param('lift_joint2', 120)
        #: joint5 策略，取值见 Joint5Mode
        self._joint5_mode = Joint5Mode(self._str_param('joint5_mode', 'fixed'))
        #: fixed 策略下 joint5 的固定角度，单位度
        self._joint5_fixed_deg = self._int_param('joint5_fixed', 90)
        #: 相机外参未标定时退化定位所用的方块中心高度，单位 m。不能取 0，那是
        #: 地面，会让 IK 去够地面而抓空或撞地
        self._block_center_z_m = self._float_param('block_center_z', 0.03)

        self._state = GraspState.IDLE
        self._interrupted = False

    # ------------------------------------------------------------------ 参数

    def _joints_param(self, name: str, default: Sequence[int]) -> List[int]:
        """读取六轴角度参数，未声明时按默认值声明。

        :param name: 参数名。
        :param default: 默认角度序列，单位度，长度 6。
        :returns: 参数当前值，元素为整数角度。
        """
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, list(default))
        return [int(v) for v in self._node.get_parameter(name).value]

    def _int_param(self, name: str, default: int) -> int:
        """读取整数参数，未声明时按默认值声明。

        :param name: 参数名。
        :param default: 默认值。
        :returns: 参数当前值。
        """
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, int(default))
        return int(self._node.get_parameter(name).value)

    def _float_param(self, name: str, default: float) -> float:
        """读取浮点参数，未声明时按默认值声明。

        :param name: 参数名。
        :param default: 默认值。
        :returns: 参数当前值。
        """
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, float(default))
        return float(self._node.get_parameter(name).value)

    def _str_param(self, name: str, default: str) -> str:
        """读取字符串参数，未声明时按默认值声明。

        :param name: 参数名。
        :param default: 默认值。
        :returns: 参数当前值。
        """
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, str(default))
        return str(self._node.get_parameter(name).value)

    # ------------------------------------------------------------------ 查询

    @property
    def state(self) -> GraspState:
        """当前流程状态，取值见 ``GraspState``。"""
        return self._state

    def interrupt(self) -> None:
        """请求中断，协作式，在状态切换点生效。"""
        self._interrupted = True

    # ------------------------------------------------------------------ 主流程

    def in_envelope(self, target: BlockDetection) -> bool:
        """判断目标是否落在有效抓取包络的水平距离区间内。

        :param target: 待判断观测。
        :returns: 水平距离落在闭区间内即为真，区间两端由参数给出，单位 m。
        """
        horizontal_distance = target.horizontal_distance()
        return (
            self._grasp_distance_min_m
            <= horizontal_distance
            <= self._grasp_distance_max_m
        )

    def execute(self, target: BlockDetection) -> bool:
        """执行一次完整抓取，含一次重试，成功返回真。

        不在包络内直接失败并提示先对位；尝试 ``MAX_GRASP_ATTEMPTS`` 次仍失败时
        状态置为 FAILED，由上层登记待办。

        :param target: 待抓取方块观测，须已通过 :meth:`in_envelope` 或接受其
            内部的包络检查。
        :returns: 抓取成功为真。
        """
        self._interrupted = False

        if not self.in_envelope(target):
            horizontal_distance = target.horizontal_distance()
            self._node.get_logger().warn(
                f'{target.color} 方块水平距离 {horizontal_distance:.3f}m 不在包络 '
                f'[{self._grasp_distance_min_m:.2f}, {self._grasp_distance_max_m:.2f}]，需先对位'
            )
            return False

        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
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
                + ('，将重试一次' if attempt < MAX_GRASP_ATTEMPTS else '，登记为待办')
            )
            # 重试前先回准备位，避免姿态残留
            self._arm.send_joints(self._grasp_home, self._move_time_ms)
            self._arm.wait_until_ready()

        self._state = GraspState.FAILED
        return False

    # -------------------------------------------------------------- 单次序列

    def _run_sequence(self, target: BlockDetection) -> bool:
        """执行一次不含重试的抓取序列，任一步超时即返回假。

        :param target: 待抓取方块观测，须已确认在包络内。
        :returns: 七个步骤全部完成为真。
        """
        # 第 1 步：张开夹爪
        self._state = GraspState.PREPARE
        self._arm.set_gripper(False, self._grasp_time_ms)
        if not self._arm.wait_until_ready():
            return False

        # 第 2 步：IK 求解
        self._state = GraspState.SOLVE
        position = target.position_base
        if position is None:
            # 外参未标定时退化为相机系近似：前取水平距离，左取负 lateral 即
            # lateral 正为右，上取块心高度。高度不能取 0 即地面，否则 IK 会去
            # 够地面而抓空或撞地。
            self._node.get_logger().warn(
                'position_base 为空（相机外参未标定），改用相机系近似定位，'
                f'块心高度取 {self._block_center_z_m:.3f}m'
            )
            position = (
                target.horizontal_distance(),
                -target.lateral_m,
                self._block_center_z_m,
            )

        fk_pose = self._arm.solve_fk()
        pitch = fk_pose[4] if fk_pose is not None else 0.0
        yaw = fk_pose[5] if fk_pose is not None else 0.0

        joints = self._arm.solve_ik(
            x=position[0] + self._forward_compensation_m,
            y=position[1],
            z=position[2],
            roll=math.pi,
            pitch=pitch,
            yaw=yaw,
        )
        if joints is None:
            return False

        joints = self._sanitize(joints, target)

        # 第 3 步：下探到目标
        self._state = GraspState.DESCEND
        self._arm.send_joints(joints, self._move_time_ms)
        if not self._arm.wait_until_ready(
            self._move_time_ms / MILLISECONDS_PER_SECOND + WAIT_GRACE_SEC
        ):
            return False
        self._settle()

        # 第 4 步：闭合夹爪
        self._state = GraspState.CLOSE
        self._arm.set_gripper(True, self._grasp_time_ms)
        if not self._arm.wait_until_ready(
            self._grasp_time_ms / MILLISECONDS_PER_SECOND + WAIT_GRACE_SEC
        ):
            return False
        self._settle()

        # 第 5 步：抬起
        self._state = GraspState.LIFT
        lifted = list(joints)
        lifted[1] = self._lift_joint2_deg
        self._arm.send_joints(lifted, self._move_time_ms)
        if not self._arm.wait_until_ready(
            self._move_time_ms / MILLISECONDS_PER_SECOND + WAIT_GRACE_SEC
        ):
            return False

        # 第 6 步：移到收集筐
        self._state = GraspState.PLACE
        self._arm.send_joints(self._place_pose, self._move_time_ms)
        if not self._arm.wait_until_ready(
            self._move_time_ms / MILLISECONDS_PER_SECOND + WAIT_GRACE_SEC
        ):
            return False

        # 第 7 步：释放
        self._state = GraspState.RELEASE
        self._arm.set_gripper(False, self._grasp_time_ms)
        if not self._arm.wait_until_ready(
            self._grasp_time_ms / MILLISECONDS_PER_SECOND + WAIT_GRACE_SEC
        ):
            return False
        self._settle()
        return True

    def _retreat(self) -> None:
        """回到巡线姿态，使相机恢复看地视角。"""
        self._state = GraspState.RETREAT
        self._arm.send_joints(self._line_pose, self._move_time_ms)
        self._arm.wait_until_ready()

    def _settle(self) -> None:
        """等待舵机稳定，替代固定 sleep，时长由 ``settle_sec`` 配置。"""
        if self._settle_sec > 0:
            time.sleep(self._settle_sec)

    def _sanitize(
        self, joints: Sequence[float], target: BlockDetection
    ) -> List[int]:
        """按现有 demo 的经验规则裁剪关节角，返回可直接下发的整数序列。

        :param joints: IK 返回的关节角，单位度，取前 6 个。
        :param target: 待抓取观测，供 joint5 策略推算夹爪朝向。
        :returns: 六个整数关节角，单位度，取值区间 0 到 180。
        """
        clamped = [
            max(JOINT_MIN_DEG, min(JOINT_MAX_DEG, int(round(value))))
            for value in joints[:6]
        ]
        # joint4 超过 90 度会导致腕部翻转，demo 里直接截断
        if clamped[3] > WRIST_JOINT4_MAX_DEG:
            clamped[3] = WRIST_JOINT4_MAX_DEG
        # joint5 决定夹爪朝向，默认取中位
        clamped[4] = self._resolve_joint5(target)
        # joint6 由抓取逻辑决定，不采用 IK 结果
        clamped[5] = self._gripper_open_deg
        return clamped

    def _resolve_joint5(self, target: BlockDetection) -> int:
        """按策略给出 joint5 角度，单位度。

        :param target: 待抓取观测，``from_yaw`` 策略下用其 ``yaw_rad``。
        :returns: 取值区间 0 到 180 的整数角度；``from_yaw`` 且朝向缺失时退回
            ``joint5_fixed``。
        """
        if self._joint5_mode == Joint5Mode.FROM_YAW and target.yaw_rad is not None:
            # 由方块角点朝向推算夹爪旋转角，并按 90 度步长折回可用区间
            angle = math.degrees(target.yaw_rad) % JOINT_MAX_DEG
            if angle > JOINT5_FOLD_UPPER_DEG:
                angle -= JOINT5_FOLD_STEP_DEG
            elif angle < JOINT5_FOLD_LOWER_DEG:
                angle += JOINT5_FOLD_STEP_DEG
            return max(JOINT_MIN_DEG, min(JOINT_MAX_DEG, int(round(angle))))
        return self._joint5_fixed_deg
