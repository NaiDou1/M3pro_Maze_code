"""底盘硬件抽象层。

对应硬件
--------
麦轮全向底盘，由下位机 STM32 经 micro-ROS 串口（``/dev/myserial`` @2Mbps）接入
ROS 2 图，**没有上位机侧的底盘驱动节点**。

* 控制：发布 ``geometry_msgs/Twist`` 到 ``/cmd_vel``
  （``linear.x`` 前后 m/s、``linear.y`` 左右横移 m/s、``angular.z`` 自转 rad/s）
* 反馈：订阅 ``/odom_raw``（STM32 直出，高频）

关键约束（详见 AGENTS.md）
--------------------------
1. ``/cmd_vel`` **无超时停车保护**，必须 >=10Hz 持续发布，停止时显式发全 0；
   本模块用定时器以 20Hz 持续重发缓存的速度指令来满足该要求。
2. 闭环控制使用 ``/odom_raw``，**不要**用 6Hz 的 ``/odom``（EKF 输出）。
3. 位置由速度积分得到，长时间必然漂移，仅可短时信任；需配合激光做格心校正。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

#: 里程计位姿表示为 ``(x, y, yaw)``，单位 m / m / rad
Pose = Tuple[float, float, float]


def yaw_from_quaternion(q) -> float:
    """从四元数提取偏航角（绕 Z 轴），弧度。

    只关心平面运动，故直接由 z/w 分量求 atan2，避免引入 tf_transformations 依赖。
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BaseDriver(Node):
    """底盘驱动器：持续发布速度指令，并缓存里程计反馈供上层查询。"""

    def __init__(self) -> None:
        super().__init__('base_driver')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom_raw')
        #: 重发频率，必须 >=10Hz 才能维持下位机运动（无超时保护）
        self.declare_parameter('publish_rate_hz', 20.0)
        #: 超过该时长未收到里程计则告警（只告警一次，避免刷屏）
        self.declare_parameter('odom_timeout_sec', 1.0)

        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self._odom_timeout = float(self.get_parameter('odom_timeout_sec').value)

        if rate_hz < 10.0:
            self.get_logger().warn(
                f'publish_rate_hz={rate_hz:.1f} 低于 10Hz，下位机可能因指令间隔过长发散'
            )

        # 回调在 executor 线程执行，主线程也会读，故用锁保护
        self._lock = threading.Lock()
        self._cmd = Twist()  # 目标速度，默认全 0
        self._pose: Optional[Pose] = None
        self._last_odom_ts = 0.0
        self._stale_warned = False

        self._pub = self.create_publisher(Twist, cmd_topic, 1)
        self._sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 50
        )
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            f'BaseDriver 就绪 | cmd_vel={cmd_topic} | odom={odom_topic} | {rate_hz:.0f}Hz'
        )

    # ------------------------------------------------------------------ 指令

    def set_velocity(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """设置目标速度（m/s, m/s, rad/s）。下个定时器周期起持续发布。"""
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(wz)
        with self._lock:
            self._cmd = cmd

    def stop(self) -> None:
        """立即停止：显式发送全 0 速度。"""
        self.set_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ 反馈

    def has_odom(self) -> bool:
        """是否已收到过至少一帧里程计。"""
        with self._lock:
            return self._pose is not None

    def get_pose(self) -> Optional[Pose]:
        """返回缓存位姿 ``(x, y, yaw)``；尚未收到里程计时返回 ``None``。"""
        with self._lock:
            return self._pose

    def get_yaw(self) -> Optional[float]:
        """返回当前偏航角（rad），无数据时 ``None``。"""
        with self._lock:
            return None if self._pose is None else self._pose[2]

    def odom_age(self) -> float:
        """距最近一帧里程计的时长（秒）；从未收到时返回 ``inf``。"""
        with self._lock:
            if self._last_odom_ts <= 0.0:
                return math.inf
            return time.monotonic() - self._last_odom_ts

    # -------------------------------------------------------------- 内部回调

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        with self._lock:
            self._pose = (p.x, p.y, yaw)
            self._last_odom_ts = time.monotonic()
            self._stale_warned = False

    def publish_now(self) -> None:
        """立即发布当前缓存的速度指令，不等定时器周期。"""
        with self._lock:
            cmd = self._cmd
        self._pub.publish(cmd)

    def _tick(self) -> None:
        self.publish_now()

        # 里程计失联监测：只在状态翻转时告警一次，避免每帧刷屏
        age = self.odom_age()
        if age > self._odom_timeout and not self._stale_warned:
            with self._lock:
                self._stale_warned = True
            self.get_logger().warn(
                f'超过 {self._odom_timeout:.1f}s 未收到里程计，请检查 micro-ROS agent '
                '是否启动（start_agent.sh）及 ROS_DOMAIN_ID=30'
            )


def main(args=None) -> None:
    """独立运行：仅维持底盘话题在线，用于联调时确认接口可用。"""
    rclpy.init(args=args)
    node = BaseDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.publish_now()  # 退出前确保下发停止指令
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
