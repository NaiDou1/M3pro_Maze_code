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
from typing import Callable, Optional, Tuple

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


class CommandWatchdog:
    """速度指令看门狗：非零指令超时未刷新即强制归零。

    语义：每个速度指令调用一次 :meth:`feed`（喂狗）。若**最后一条指令为非零**
    且超过 ``timeout`` 秒没有再次喂狗，判定上层已失联，调用 ``on_timeout``。

    为什么必须是独立线程
    --------------------
    本机 ``/cmd_vel`` **无超时停车保护**，下位机会一直执行最后收到的指令。而
    指令上线依赖单线程 executor 里的主线程 spin，一旦主线程阻塞（例如机械臂
    ``time.sleep``）或 executor 卡死，定时器不再发布，最后的速度就被锁存。
    本类跑在独立线程、只依赖 DDS 发布，不经过 executor，因此仍能发出停车指令。

    .. warning::
       进程被 ``SIGKILL`` / OOM 杀死时线程一并消失，**本看门狗无法兜底**。
       该场景只能靠物理断电，详见 ``SAFETY.md``。

    ``timeout <= 0`` 表示禁用看门狗。
    """

    def __init__(
        self,
        timeout: float,
        on_timeout: Callable[[], None],
        period: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout = float(timeout)
        self._on_timeout = on_timeout
        self._period = max(0.01, float(period))
        self._clock = clock
        self._lock = threading.Lock()
        self._cmd_ts = 0.0
        self._nonzero = False
        self._thread: Optional[threading.Thread] = None
        self._running = False
        #: 累计触发次数，供日志与自检查询
        self.trips = 0

    @property
    def timeout(self) -> float:
        """看门狗超时阈值（秒）。"""
        return self._timeout

    @property
    def enabled(self) -> bool:
        """是否启用（``timeout > 0``）。"""
        return self._timeout > 0.0

    def feed(self, nonzero: bool) -> None:
        """喂狗：记录最近一次指令的时刻，并标记它是否为非零指令。"""
        with self._lock:
            self._cmd_ts = self._clock()
            self._nonzero = bool(nonzero)

    def check(self, now: Optional[float] = None) -> bool:
        """判定一次；超时且最后指令为非零时回调归零并返回 ``True``。

        :param now: 当前时刻（秒），``None`` 时取 ``clock()``。显式传入便于单测。
        """
        if not self.enabled:
            return False
        now = self._clock() if now is None else float(now)
        with self._lock:
            tripped = self._nonzero and (now - self._cmd_ts) > self._timeout
            if tripped:
                # 归零标记，避免同一次失联被反复触发（回调会重新喂零速指令）
                self._nonzero = False
                self._cmd_ts = now
        if tripped:
            self.trips += 1
            self._on_timeout()
        return tripped

    def start(self) -> None:
        """启动看门狗线程（禁用或已启动时为空操作）。"""
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name='cmd_watchdog', daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止看门狗线程并等待其退出。"""
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            try:
                self.check()
            except Exception:  # noqa: BLE001 - 看门狗自身绝不能因异常退出
                pass
            time.sleep(self._period)


class BaseDriver(Node):
    """底盘驱动器：持续发布速度指令，并缓存里程计反馈供上层查询。"""

    def __init__(
        self,
        *,
        yaw_source: Optional[str] = None,
        watchdog_timeout: Optional[float] = None,
    ) -> None:
        super().__init__('base_driver')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom_raw')
        #: 备用航向源话题：EKF 输出（融合 IMU，航向通常比轮式直出更准，但仅 6Hz）
        self.declare_parameter('alt_odom_topic', '/odom')
        #: 航向来源：'odom_raw'（默认，与位置同源、高频）或 'odom'（EKF，航向更准）。
        #: ⚠️ 两者误差特性不同，odom_angular_scale_correction 必须按所选源现场重标
        self.declare_parameter('yaw_source', 'odom_raw')
        #: 重发频率，必须 >=10Hz 才能维持下位机运动（无超时保护）
        self.declare_parameter('publish_rate_hz', 20.0)
        #: 超过该时长未收到里程计则告警（只告警一次，避免刷屏）
        self.declare_parameter('odom_timeout_sec', 1.0)
        #: 速度指令看门狗阈值（秒）：非零指令超过该时长未被刷新即强制归零。0=禁用
        self.declare_parameter('cmd_watchdog_timeout', 0.5)

        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        alt_odom_topic = str(self.get_parameter('alt_odom_topic').value)
        # 构造参数优先：本节点由 mission_manager 在进程内创建，收不到 launch
        # 注入的 yaml，故需由调用方把值转发进来
        self._yaw_source = str(
            yaw_source if yaw_source is not None
            else self.get_parameter('yaw_source').value
        )
        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self._odom_timeout = float(self.get_parameter('odom_timeout_sec').value)
        watchdog_timeout = float(
            watchdog_timeout if watchdog_timeout is not None
            else self.get_parameter('cmd_watchdog_timeout').value
        )

        if rate_hz < 10.0:
            self.get_logger().warn(
                f'publish_rate_hz={rate_hz:.1f} 低于 10Hz，下位机可能因指令间隔过长发散'
            )

        # 回调在 executor 线程执行，主线程也会读，故用锁保护
        self._lock = threading.Lock()
        self._cmd = Twist()  # 目标速度，默认全 0
        self._pose: Optional[Pose] = None
        self._alt_yaw: Optional[float] = None
        self._last_odom_ts = 0.0
        self._stale_warned = False

        self._pub = self.create_publisher(Twist, cmd_topic, 1)
        self._sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 50
        )
        # 备用航向源：只取 yaw，不参与位置解算
        self._alt_sub = self.create_subscription(
            Odometry, alt_odom_topic, self._on_alt_odom, 10
        )
        self.create_timer(1.0 / rate_hz, self._tick)

        # 速度看门狗跑在独立线程：主线程阻塞或 executor 卡死时仍能强制归零
        self._watchdog = CommandWatchdog(watchdog_timeout, self._force_zero)
        self._watchdog.start()

        if self._yaw_source not in ('odom_raw', 'odom'):
            self.get_logger().warn(
                f'yaw_source={self._yaw_source!r} 非法，回退为 odom_raw'
            )
            self._yaw_source = 'odom_raw'

        self.get_logger().info(
            f'BaseDriver 就绪 | cmd_vel={cmd_topic} | odom={odom_topic} | {rate_hz:.0f}Hz'
        )
        if self._yaw_source == 'odom':
            self.get_logger().info(
                f'航向来源 yaw_source=odom（EKF，仅约 6Hz）| 备用源 {alt_odom_topic}；'
                '换源后必须用 calibration_tool motion 模式重标角速度系数'
            )
        else:
            self.get_logger().info(
                f'航向来源 yaw_source=odom_raw（与位置同源）| 备用源 {alt_odom_topic}'
            )
        if self._watchdog.enabled:
            self.get_logger().info(
                f'速度看门狗已启用：非零指令超过 {watchdog_timeout:.2f}s 未刷新即强制归零'
            )
        else:
            self.get_logger().warn(
                '速度看门狗被禁用（cmd_watchdog_timeout<=0）——主线程阻塞时'
                '最后的速度会被下位机锁存，存在失控风险，详见 SAFETY.md'
            )

    # ------------------------------------------------------------------ 指令

    def set_velocity(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        """设置目标速度（m/s, m/s, rad/s）。下个定时器周期起持续发布。

        同时喂一次看门狗：**非零指令必须被持续刷新**，否则看门狗会强制归零。
        """
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(wz)
        with self._lock:
            self._cmd = cmd
        self._watchdog.feed(bool(vx or vy or wz))

    def stop(self) -> None:
        """立即停止：显式发送全 0 速度。"""
        self.set_velocity(0.0, 0.0, 0.0)

    def last_command(self) -> Tuple[float, float, float]:
        """返回缓存的速度指令 ``(vx, vy, wz)``，供自检与单测查询。"""
        with self._lock:
            return (
                self._cmd.linear.x,
                self._cmd.linear.y,
                self._cmd.angular.z,
            )

    @property
    def watchdog_trips(self) -> int:
        """看门狗累计触发（强制归零）次数。"""
        return self._watchdog.trips

    def _force_zero(self) -> None:
        """看门狗超时回调：强制归零并立即发布。

        可能在**看门狗线程**中执行，因此只使用锁保护的最小状态。
        """
        cmd = Twist()
        with self._lock:
            self._cmd = cmd
        self._pub.publish(cmd)
        self.get_logger().warn(
            f'看门狗触发：{self._watchdog.timeout:.2f}s 未收到新的速度指令，'
            f'已强制归零（累计 {self._watchdog.trips} 次）'
        )

    def shutdown(self) -> None:
        """退出前收尾：停止看门狗并下发归零指令。"""
        self._watchdog.stop()
        self.stop()
        self.publish_now()

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
        """返回当前偏航角（rad），无数据时 ``None``。

        按 ``yaw_source`` 选择来源：``odom_raw``（默认）取主里程计；``odom``
        优先取 EKF 备用源（融合 IMU，航向更准），其尚无数据时退回主源，以免
        上电初期返回 ``None`` 导致转向原语直接失败。
        """
        with self._lock:
            if self._yaw_source == 'odom' and self._alt_yaw is not None:
                return self._alt_yaw
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

    def _on_alt_odom(self, msg: Odometry) -> None:
        """备用航向源回调：只取 yaw，不使用其位置。"""
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        with self._lock:
            self._alt_yaw = yaw

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
        node.shutdown()  # 停看门狗 + 确保下发停止指令
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
