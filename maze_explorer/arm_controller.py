"""机械臂硬件抽象层。

硬件
----
6 自由度 DOFBOT 系舵机臂，关节角单位为度，取值区间 0 到 180，不是弧度。

接口，均在 yahboomcar_ws 内已有定义
------------------------------------
=======================  ==========================  ====================================
方向                      话题 / 服务                  消息类型
=======================  ==========================  ====================================
发布                      ``arm6_joints``             ``arm_msgs/ArmJoints``，含 6 轴角度与耗时 ms
发布                      ``arm_joint``               ``arm_msgs/ArmJoint``，含 id、角度与耗时 ms
调用                      ``get_kinemarics``          ``arm_interface/ArmKinemarics``，含 ik 与 fk
=======================  ==========================  ====================================

约定，详见 AGENTS.md
--------------------
* 夹爪为 joint6，张开角度 30，夹紧角度 120 到 140
* 服务端角度映射为 ``90`` 度基准的偏移量换算，工具即夹爪长度 0.12 m
* 抓取包络为水平距离 0.13 到 0.25 m，车体对位目标 0.20 m

.. warning::
   舵机无位置反馈，本模块的到位只能依据下发消息里的 ``time`` 字段推算，属于开环
   等待。若要精确节拍需后续加装反馈或做时间标定。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import rclpy
from arm_interface.srv import ArmKinemarics
from arm_msgs.msg import ArmJoint, ArmJoints
from rclpy.node import Node

from maze_explorer._compat import StrEnum

#: 末端位姿六分量，依次为 x、y、z 单位 m 与 roll、pitch、yaw 单位 rad
EndPose = Tuple[float, float, float, float, float, float]

#: 关节角单位度，取值区间 0 到 180
JOINT_MIN_DEG = 0
#: 关节角上限，单位度
JOINT_MAX_DEG = 180
#: 关节编号下限
MIN_JOINT_ID = 1
#: 关节数量
JOINT_COUNT = 6

#: 各动作默认耗时，单位 ms，写入消息的 time 字段
DEFAULT_TIME_MS = 2000
#: 夹爪张开角度，单位度，需按自备方块尺寸实测标定
GRIPPER_OPEN_ANGLE = 30
#: 夹爪夹紧角度，单位度，实测 120 到 140 皆可夹紧
GRIPPER_CLOSED_ANGLE = 120
#: 夹爪关节编号
GRIPPER_JOINT_ID = 6
#: 等待下位机订阅端上线的上限，单位 s
DEFAULT_SUBSCRIBER_WAIT_SEC = 5.0
#: 等待 IK 服务上线的上限，单位 s
DEFAULT_SERVICE_WAIT_SEC = 5.0
#: 单次运动学求解的超时，单位 s
DEFAULT_SOLVE_TIMEOUT_SEC = 2.0
#: 等待订阅端建连的轮询间隔，单位 s
SUBSCRIBER_POLL_INTERVAL_SEC = 0.1
#: 关节话题的发布队列深度，单位条
JOINT_QUEUE_DEPTH = 10
#: 毫秒每秒
MILLISECONDS_PER_SECOND = 1000.0


class KinName(StrEnum):
    """运动学求解类型，取值 ik 或 fk，即服务请求字段 ``kin_name`` 的合法取值。

    ik 为逆解，输入位姿返回六个关节角；fk 为正解，输入关节角返回末端位姿。
    """

    IK = 'ik'
    FK = 'fk'


class ArmController(Node):
    """机械臂控制器：关节下发、夹爪开合、IK 与 FK 求解。"""

    def __init__(
        self,
        *,
        spin_until_fn: Optional[Callable[[object, float], None]] = None,
    ) -> None:
        """声明参数、创建话题与服务客户端，不做任何阻塞等待。

        :param spin_until_fn: 等待服务 future 的注入回调。组合多节点时必须传入
            ``executor.spin_until_future_complete``；直接调用
            ``rclpy.spin_until_future_complete`` 会因本节点已被加入外部 executor
            而抛异常。取 ``None`` 时仅适用于单节点独立运行。
        """
        super().__init__('arm_controller')
        self._spin_until = spin_until_fn

        self.declare_parameter('joints_topic', 'arm6_joints')
        self.declare_parameter('joint_topic', 'arm_joint')
        self.declare_parameter('ik_service', 'get_kinemarics')
        #: 各动作默认耗时，单位 ms
        self.declare_parameter('default_time_ms', DEFAULT_TIME_MS)
        #: 夹爪张开与夹紧角度，单位度，需按自备方块尺寸实测标定
        self.declare_parameter('gripper_open_angle', GRIPPER_OPEN_ANGLE)
        self.declare_parameter('gripper_closed_angle', GRIPPER_CLOSED_ANGLE)
        #: 巡线姿态下的初始关节角，单位度，沿用现有 demo 经验值
        self.declare_parameter('initial_joints', [90, 150, 12, 20, 90, 0])

        joints_topic = str(self.get_parameter('joints_topic').value)
        joint_topic = str(self.get_parameter('joint_topic').value)
        self._ik_service = str(self.get_parameter('ik_service').value)
        self._default_time_ms = int(self.get_parameter('default_time_ms').value)
        self._gripper_open = int(self.get_parameter('gripper_open_angle').value)
        self._gripper_closed = int(self.get_parameter('gripper_closed_angle').value)

        self._lock = threading.Lock()
        #: 最近一次下发的关节角，作为 IK 的 cur_joint 默认值
        self._last_joints: List[int] = [
            int(a) for a in self.get_parameter('initial_joints').value
        ]
        #: 预计运动完成时刻，单位 s，monotonic 时基
        self._ready_at = 0.0

        self._pub_joints = self.create_publisher(
            ArmJoints, joints_topic, JOINT_QUEUE_DEPTH
        )
        self._pub_joint = self.create_publisher(
            ArmJoint, joint_topic, JOINT_QUEUE_DEPTH
        )
        self._ik_client = self.create_client(ArmKinemarics, self._ik_service)

        self.get_logger().info(
            f'ArmController 就绪 | joints={joints_topic} | joint={joint_topic} | '
            f'ik={self._ik_service}'
        )

    # ------------------------------------------------------------ 关节下发

    def send_joints(
        self, joints: Sequence[float], time_ms: Optional[int] = None
    ) -> float:
        """下发六个关节角，返回预计运动完成时刻，单位 s，monotonic 时基。

        下发即返回，无位置反馈，到位只能靠 :meth:`wait_until_ready` 按耗时等待。

        :param joints: 长度 6 的序列，元素为取值区间 0 到 180 的角度，单位度。
        :param time_ms: 动作耗时，单位 ms，取 ``None`` 时用 ``default_time_ms``。
        :returns: 预计完成时刻，与 ``time.monotonic`` 同时基。
        """
        if len(joints) != JOINT_COUNT:
            raise ValueError(
                f'需要 {JOINT_COUNT} 个关节角，实际收到 {len(joints)} 个'
            )
        duration_ms = int(self._default_time_ms if time_ms is None else time_ms)
        angles = [int(round(float(a))) for a in joints]

        msg = ArmJoints()
        msg.joint1, msg.joint2, msg.joint3 = angles[0], angles[1], angles[2]
        msg.joint4, msg.joint5, msg.joint6 = angles[3], angles[4], angles[5]
        msg.time = duration_ms
        self._pub_joints.publish(msg)

        ready_at = time.monotonic() + duration_ms / MILLISECONDS_PER_SECOND
        with self._lock:
            self._last_joints = angles
            self._ready_at = ready_at
        return ready_at

    def send_joint(
        self, joint_id: int, angle: float, time_ms: Optional[int] = None
    ) -> float:
        """下发单个关节角，返回预计完成时刻，单位 s，monotonic 时基。

        :param joint_id: 关节编号，取值 1 到 6。
        :param angle: 目标角，单位度，取值区间 0 到 180。
        :param time_ms: 动作耗时，单位 ms，取 ``None`` 时用 ``default_time_ms``。
        :returns: 预计完成时刻，与 ``time.monotonic`` 同时基。
        """
        if not MIN_JOINT_ID <= int(joint_id) <= JOINT_COUNT:
            raise ValueError(
                f'joint_id 必须在 {MIN_JOINT_ID} 到 {JOINT_COUNT}，实际 {joint_id}'
            )
        duration_ms = int(self._default_time_ms if time_ms is None else time_ms)

        msg = ArmJoint()
        msg.id = int(joint_id)
        msg.joint = int(round(float(angle)))
        msg.time = duration_ms
        self._pub_joint.publish(msg)

        ready_at = time.monotonic() + duration_ms / MILLISECONDS_PER_SECOND
        with self._lock:
            self._last_joints[int(joint_id) - 1] = msg.joint
            self._ready_at = max(self._ready_at, ready_at)
        return ready_at

    def set_gripper(self, closed: bool, time_ms: Optional[int] = None) -> float:
        """开合夹爪，返回预计完成时刻，单位 s，monotonic 时基。

        :param closed: 为真夹紧，为假张开。
        :param time_ms: 动作耗时，单位 ms，取 ``None`` 时用 ``default_time_ms``。
        :returns: 预计完成时刻。
        """
        angle = self._gripper_closed if closed else self._gripper_open
        return self.send_joint(GRIPPER_JOINT_ID, angle, time_ms)

    def last_joints(self) -> List[int]:
        """返回最近一次下发的六关节角副本，单位度，长度为 6。"""
        with self._lock:
            return list(self._last_joints)

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """阻塞等待至预计运动完成时刻。

        :param timeout: 额外等待上限，单位 s，取 ``None`` 表示一直等到预计时刻。
        :returns: 已到预计完成时刻为真；因超时提前返回为假。
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

    def wait_for_subscribers(
        self, timeout: float = DEFAULT_SUBSCRIBER_WAIT_SEC
    ) -> bool:
        """等待 ``arm6_joints`` 出现订阅者，即下位机机械臂驱动上线。

        现有 demo 在启动时也做同样的等待，否则早期消息会丢失。轮询间隔为
        ``SUBSCRIBER_POLL_INTERVAL_SEC``。

        :param timeout: 等待上限，单位 s，不小于 0。
        :returns: 出现订阅端为真；到时仍无订阅端为假并记一条 WARN。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._pub_joints.get_subscription_count() > 0:
                return True
            time.sleep(SUBSCRIBER_POLL_INTERVAL_SEC)
        self.get_logger().warn(
            f'{timeout:.1f}s 内未检测到 arm6_joints 订阅者，机械臂可能不会响应'
        )
        return False

    # --------------------------------------------------------------- IK / FK

    def wait_for_ik_service(
        self, timeout: float = DEFAULT_SERVICE_WAIT_SEC
    ) -> bool:
        """等待 IK 服务可用，``arm_kin/kin_srv`` 需先启动。

        :param timeout: 等待上限，单位 s。
        :returns: 服务可用为真。
        """
        return self._ik_client.wait_for_service(timeout_sec=timeout)

    def _wait_future(self, future, timeout: float) -> None:
        """等待服务 future 完成，超时不抛异常，由调用方检查 ``future`` 是否完成。

        组合多节点时走注入的 ``spin_until_fn``；单节点独立运行时退回
        ``rclpy.spin_until_future_complete``。

        :param future: 已发起的服务调用 future。
        :param timeout: 等待上限，单位 s。
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
        timeout: float = DEFAULT_SOLVE_TIMEOUT_SEC,
    ) -> Optional[List[float]]:
        """逆运动学求解，返回六个关节角，单位度；失败返回 ``None``。

        :param x: 目标末端 x 分量，单位 m，基座坐标系。
        :param y: 目标末端 y 分量，单位 m。
        :param z: 目标末端 z 分量，单位 m。
        :param roll: 目标末端横滚角，单位 rad。
        :param pitch: 目标末端俯仰角，单位 rad。
        :param yaw: 目标末端偏航角，单位 rad。
        :param cur_joints: 当前关节角，单位度；取 ``None`` 时用最近一次下发的值。
        :param timeout: 服务等待上限，单位 s。
        :returns: 六个关节角，单位度；服务不可用、超时或无解时为 ``None``。

        .. note::
           内部通过 ``_wait_future`` 驱动，不可在回调中调用，否则会与单线程
           executor 互锁。组合多节点时必须给构造函数注入 ``spin_until_fn``，
           否则本节点已归属外部 executor 时会抛异常。
        """
        if not self._ik_client.service_is_ready() and not self.wait_for_ik_service(
            timeout
        ):
            self.get_logger().warn(f'IK 服务 {self._ik_service} 不可用')
            return None

        current_joints = [
            float(v)
            for v in (cur_joints if cur_joints is not None else self.last_joints())
        ]
        req = ArmKinemarics.Request()
        req.tar_x, req.tar_y, req.tar_z = float(x), float(y), float(z)
        req.roll, req.pitch, req.yaw = float(roll), float(pitch), float(yaw)
        req.cur_joint1, req.cur_joint2, req.cur_joint3 = (
            current_joints[0],
            current_joints[1],
            current_joints[2],
        )
        req.cur_joint4, req.cur_joint5, req.cur_joint6 = (
            current_joints[3],
            current_joints[4],
            current_joints[5],
        )
        req.kin_name = KinName.IK.value

        future = self._ik_client.call_async(req)
        self._wait_future(future, timeout)
        if not future.done() or future.result() is None:
            self.get_logger().warn('IK 求解超时或无解')
            return None

        res = future.result()
        return [
            float(res.joint1),
            float(res.joint2),
            float(res.joint3),
            float(res.joint4),
            float(res.joint5),
            float(res.joint6),
        ]

    def solve_fk(
        self,
        cur_joints: Optional[Iterable[float]] = None,
        timeout: float = DEFAULT_SOLVE_TIMEOUT_SEC,
    ) -> Optional[EndPose]:
        """正运动学求解，返回末端位姿六分量；失败返回 ``None``。

        :param cur_joints: 当前关节角，单位度；取 ``None`` 时用最近一次下发的值。
        :param timeout: 服务等待上限，单位 s。
        :returns: 依次为 x、y、z 单位 m 与 roll、pitch、yaw 单位 rad；失败为
            ``None``。

        .. note::
           与 :meth:`solve_ik` 相同的 executor 约束，不可在回调中调用。
        """
        if not self._ik_client.service_is_ready() and not self.wait_for_ik_service(
            timeout
        ):
            self.get_logger().warn(f'FK 服务 {self._ik_service} 不可用')
            return None

        current_joints = [
            float(v)
            for v in (cur_joints if cur_joints is not None else self.last_joints())
        ]
        req = ArmKinemarics.Request()
        req.kin_name = KinName.FK.value
        req.cur_joint1, req.cur_joint2, req.cur_joint3 = (
            current_joints[0],
            current_joints[1],
            current_joints[2],
        )
        req.cur_joint4, req.cur_joint5, req.cur_joint6 = (
            current_joints[3],
            current_joints[4],
            current_joints[5],
        )

        future = self._ik_client.call_async(req)
        self._wait_future(future, timeout)
        if not future.done() or future.result() is None:
            self.get_logger().warn('FK 求解超时或失败')
            return None

        res = future.result()
        return (res.x, res.y, res.z, res.roll, res.pitch, res.yaw)


def main(args: Optional[List[str]] = None) -> None:
    """独立运行：把机械臂归位到 ``initial_joints``，用于联调验证关节通道。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
    rclpy.init(args=args)
    node = ArmController()
    try:
        node.wait_for_subscribers(timeout=DEFAULT_SUBSCRIBER_WAIT_SEC)
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
