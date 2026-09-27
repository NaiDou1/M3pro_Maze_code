"""机械臂硬件抽象层。

硬件
----
6 自由度 DOFBOT 系舵机臂，**关节角为 0~180 度**（不是弧度）。

接口（均在 yahboomcar_ws 内已有定义）
-------------------------------------
========================  ==========================  ====================================
方向                      话题 / 服务                  消息类型
========================  ==========================  ====================================
发布                      ``arm6_joints``             ``arm_msgs/ArmJoints``（joint1..6 + time(ms)）
发布                      ``arm_joint``               ``arm_msgs/ArmJoint``（id, joint, time）
调用                      ``get_kinemarics``          ``arm_interface/ArmKinemarics``（ik / fk）
========================  ==========================  ====================================

约定（详见 AGENTS.md）
----------------------
* **夹爪 = joint6**：``30`` 张开、``120~140`` 夹紧
* 服务端角度映射 ``(joint - 90) * DE2RA``，工具（夹爪）长度 ``0.12m``
* 抓取包络：水平距离 ``0.13~0.25m``，车体对位目标 ``0.20m``

.. warning::
   舵机**无位置反馈**，本模块的"到位"只能依据下发消息里的 ``time`` 字段估算，
   属于开环等待。若要精确节拍需后续加装反馈或做时间标定。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import rclpy
from arm_interface.srv import ArmKinemarics
from arm_msgs.msg import ArmJoint, ArmJoints
from rclpy.node import Node

#: 末端位姿 ``(x, y, z, roll, pitch, yaw)``，单位 m / rad
EndPose = Tuple[float, float, float, float, float, float]


class ArmController(Node):
    """机械臂控制器：关节下发、夹爪开合、IK/FK 求解。"""

    def __init__(
        self,
        *,
        spin_until_fn: Optional[Callable[[object, float], None]] = None,
    ) -> None:
        super().__init__('arm_controller')
        #: 等待服务 future 的注入回调。组合多节点时**必须**传入
        #: ``executor.spin_until_future_complete``——若直接调用
        #: ``rclpy.spin_until_future_complete(self, ...)``，会因本节点已被加入
        #: 外部 executor 而抛 "Node has already been added to an executor"。
        self._spin_until = spin_until_fn

        self.declare_parameter('joints_topic', 'arm6_joints')
        self.declare_parameter('joint_topic', 'arm_joint')
        self.declare_parameter('ik_service', 'get_kinemarics')
        #: 各动作默认运行时长（ms），写入消息的 time 字段
        self.declare_parameter('default_time_ms', 2000)
        #: 夹爪张/合角度，需按自备方块尺寸实测标定
        self.declare_parameter('gripper_open_angle', 30)
        self.declare_parameter('gripper_closed_angle', 120)
        #: 巡线姿态下的初始关节角（沿用现有 demo 经验值）
        self.declare_parameter('initial_joints', [90, 150, 12, 20, 90, 0])

        joints_topic = str(self.get_parameter('joints_topic').value)
        joint_topic = str(self.get_parameter('joint_topic').value)
        self._ik_service = str(self.get_parameter('ik_service').value)
        self._default_time_ms = int(self.get_parameter('default_time_ms').value)
        self._gripper_open = int(self.get_parameter('gripper_open_angle').value)
        self._gripper_closed = int(self.get_parameter('gripper_closed_angle').value)

        self._lock = threading.Lock()
        #: 最近一次下发的关节角，作为 IK 的 ``cur_joint*`` 默认值
        self._last_joints: List[int] = [int(a) for a in self.get_parameter('initial_joints').value]
        #: 预计运动完成时刻（monotonic）
        self._ready_at = 0.0

        self._pub_joints = self.create_publisher(ArmJoints, joints_topic, 10)
        self._pub_joint = self.create_publisher(ArmJoint, joint_topic, 10)
        self._ik_client = self.create_client(ArmKinemarics, self._ik_service)

        self.get_logger().info(
            f'ArmController 就绪 | joints={joints_topic} | joint={joint_topic} | '
            f'ik={self._ik_service}'
        )

    # ------------------------------------------------------------ 关节下发

    def send_joints(self, joints: Sequence[float], time_ms: Optional[int] = None) -> float:
        """下发六个关节角（度）；返回预计运动完成时刻（monotonic 秒）。

        :param joints: 长度 6 的序列，元素为 0~180 的角度。
        :param time_ms: 运行时长（ms），``None`` 时用 ``default_time_ms``。
        """
        if len(joints) != 6:
            raise ValueError(f'需要 6 个关节角，实际收到 {len(joints)} 个')
        t = int(self._default_time_ms if time_ms is None else time_ms)
        angles = [int(round(float(a))) for a in joints]

        msg = ArmJoints()
        msg.joint1, msg.joint2, msg.joint3 = angles[0], angles[1], angles[2]
        msg.joint4, msg.joint5, msg.joint6 = angles[3], angles[4], angles[5]
        msg.time = t
        self._pub_joints.publish(msg)

        ready_at = time.monotonic() + t / 1000.0
        with self._lock:
            self._last_joints = angles
            self._ready_at = ready_at
        return ready_at

    def send_joint(self, joint_id: int, angle: float, time_ms: Optional[int] = None) -> float:
        """下发单个关节角（度），``joint_id`` 取 1~6；返回预计完成时刻。"""
        if not 1 <= int(joint_id) <= 6:
            raise ValueError(f'joint_id 必须在 1~6，实际 {joint_id}')
        t = int(self._default_time_ms if time_ms is None else time_ms)

        msg = ArmJoint()
        msg.id = int(joint_id)
        msg.joint = int(round(float(angle)))
        msg.time = t
        self._pub_joint.publish(msg)

        ready_at = time.monotonic() + t / 1000.0
        with self._lock:
            self._last_joints[int(joint_id) - 1] = msg.joint
            self._ready_at = max(self._ready_at, ready_at)
        return ready_at

    def set_gripper(self, closed: bool, time_ms: Optional[int] = None) -> float:
        """开合夹爪：``closed=True`` 夹紧，``False`` 张开。"""
        angle = self._gripper_closed if closed else self._gripper_open
        return self.send_joint(6, angle, time_ms)

    def last_joints(self) -> List[int]:
        """返回最近一次下发的六关节角（副本）。"""
        with self._lock:
            return list(self._last_joints)

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """阻塞等待至预计运动完成。

        :param timeout: 额外等待上限（秒），``None`` 表示一直等到预计时刻。
        :return: ``True`` 表示已到预计完成时刻；``False`` 表示因 ``timeout`` 提前返回。
        """
        with self._lock:
            deadline = self._ready_at
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return True
        if timeout is not None and remaining > timeout:
            time.sleep(float(timeout))
            return False
        time.sleep(remaining)
        return True

    def wait_for_subscribers(self, timeout: float = 5.0) -> bool:
        """等待 ``arm6_joints`` 出现订阅者（下位机机械臂驱动上线）。

        现有 demo 在启动时也会做同样的等待，否则早期消息会丢失。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._pub_joints.get_subscription_count() > 0:
                return True
            time.sleep(0.1)
        self.get_logger().warn(
            f'{timeout:.1f}s 内未检测到 arm6_joints 订阅者，机械臂可能不会响应'
        )
        return False

    # --------------------------------------------------------------- IK / FK

    def wait_for_ik_service(self, timeout: float = 5.0) -> bool:
        """等待 IK 服务可用（``arm_kin/kin_srv`` 需先启动）。"""
        return self._ik_client.wait_for_service(timeout_sec=timeout)

    def _wait_future(self, future, timeout: float) -> None:
        """等待服务 future 完成。

        组合多节点时走注入的 ``spin_until_fn``；单节点独立运行时退回
        ``rclpy.spin_until_future_complete``。
        """
        if self._spin_until is not None:
            self._spin_until(future, timeout)
        else:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

    def solve_ik(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        cur_joints: Optional[Iterable[float]] = None,
        timeout: float = 2.0,
    ) -> Optional[List[float]]:
        """逆运动学求解，返回六个关节角（度）；失败返回 ``None``。

        :param x, y, z: 目标末端位置（米，基座坐标系）。
        :param roll, pitch, yaw: 目标末端姿态（弧度）。
        :param cur_joints: 当前关节角，``None`` 时用最近一次下发的值。

        .. note::
           内部通过 ``_wait_future`` 驱动，**不可在回调中调用**（会与单线程
           executor 互锁）。组合多节点时必须给构造函数注入 ``spin_until_fn``，
           否则本节点已归属外部 executor 时会抛异常。
        """
        if not self._ik_client.service_is_ready() and not self.wait_for_ik_service(timeout):
            self.get_logger().warn(f'IK 服务 {self._ik_service} 不可用')
            return None

        cur = [float(v) for v in (cur_joints if cur_joints is not None else self.last_joints())]
        req = ArmKinemarics.Request()
        req.tar_x, req.tar_y, req.tar_z = float(x), float(y), float(z)
        req.roll, req.pitch, req.yaw = float(roll), float(pitch), float(yaw)
        req.cur_joint1, req.cur_joint2, req.cur_joint3 = cur[0], cur[1], cur[2]
        req.cur_joint4, req.cur_joint5, req.cur_joint6 = cur[3], cur[4], cur[5]
        req.kin_name = 'ik'

        future = self._ik_client.call_async(req)
        self._wait_future(future, timeout)
        if not future.done() or future.result() is None:
            self.get_logger().warn('IK 求解超时或无解')
            return None

        res = future.result()
        return [
            float(res.joint1), float(res.joint2), float(res.joint3),
            float(res.joint4), float(res.joint5), float(res.joint6),
        ]

    def solve_fk(
        self,
        cur_joints: Optional[Iterable[float]] = None,
        timeout: float = 2.0,
    ) -> Optional[EndPose]:
        """正运动学求解，返回末端位姿 ``(x, y, z, roll, pitch, yaw)``；失败返回 ``None``。"""
        if not self._ik_client.service_is_ready() and not self.wait_for_ik_service(timeout):
            self.get_logger().warn(f'FK 服务 {self._ik_service} 不可用')
            return None

        cur = [float(v) for v in (cur_joints if cur_joints is not None else self.last_joints())]
        req = ArmKinemarics.Request()
        req.kin_name = 'fk'
        req.cur_joint1, req.cur_joint2, req.cur_joint3 = cur[0], cur[1], cur[2]
        req.cur_joint4, req.cur_joint5, req.cur_joint6 = cur[3], cur[4], cur[5]

        future = self._ik_client.call_async(req)
        self._wait_future(future, timeout)
        if not future.done() or future.result() is None:
            self.get_logger().warn('FK 求解超时或失败')
            return None

        res = future.result()
        return (res.x, res.y, res.z, res.roll, res.pitch, res.yaw)


def main(args=None) -> None:
    """独立运行：把机械臂归位到 ``initial_joints``，用于联调验证关节通道。"""
    rclpy.init(args=args)
    node = ArmController()
    try:
        node.wait_for_subscribers(timeout=5.0)
        init = [int(a) for a in node.get_parameter('initial_joints').value]
        node.get_logger().info(f'归位到初始关节角 {init}')
        node.send_joints(init)
        node.wait_until_ready()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
