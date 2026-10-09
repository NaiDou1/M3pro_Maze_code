"""底盘硬件抽象层。

对应硬件
--------
麦轮全向底盘，由下位机 STM32 经 micro-ROS 串口接入 ROS 2 图，没有上位机侧的
底盘驱动节点。

* 控制：发布 ``geometry_msgs/Twist`` 到 ``/cmd_vel``，``linear.x`` 为前后速度
  单位 m/s，``linear.y`` 为横移速度单位 m/s，``angular.z`` 为自转单位 rad/s
* 反馈：订阅 ``/odom_raw``，STM32 直出，高频

关键约束，详见 AGENTS.md
------------------------
1. ``/cmd_vel`` 无超时停车保护，必须不低于 10Hz 持续发布，停止时显式发全 0。
   本模块用定时器以 20Hz 持续重发缓存的速度指令来满足该要求。
2. 闭环控制使用 ``/odom_raw``，不要用 6Hz 的 ``/odom``，后者为 EKF 输出。
3. 位置由速度积分得到，长时间必然漂移，仅可短时信任，需配合激光做格心校正。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from maze_explorer._compat import StrEnum

#: 里程计位姿三分量，依次为 x 与 y 单位 m、yaw 单位 rad
Pose = Tuple[float, float, float]

#: 速度指令重发频率，单位 Hz，下位机要求不低于 10Hz
DEFAULT_PUBLISH_RATE_HZ = 20.0
#: 重发频率下限，单位 Hz，低于该值下位机可能因指令间隔过长发散
MIN_PUBLISH_RATE_HZ = 10.0
#: 里程计失联告警阈值，单位 s
DEFAULT_ODOM_TIMEOUT_SEC = 1.0
#: 速度看门狗阈值，单位 s，非零指令超过该时长未刷新即归零，取 0 表示禁用
DEFAULT_WATCHDOG_TIMEOUT_SEC = 0.5
#: 看门狗轮询周期，单位 s
DEFAULT_WATCHDOG_PERIOD_SEC = 0.05
#: 看门狗轮询周期下限，单位 s，防止忙轮询
MIN_WATCHDOG_PERIOD_SEC = 0.01
#: 看门狗线程退出的等待上限，单位 s
WATCHDOG_JOIN_TIMEOUT_SEC = 1.0
#: 速度指令话题的发布队列深度，单位条
CMD_VEL_QUEUE_DEPTH = 1
#: 主里程计订阅的队列深度，单位条
ODOM_QUEUE_DEPTH = 50
#: 备用航向源订阅的队列深度，单位条
ALT_ODOM_QUEUE_DEPTH = 10


class YawSource(StrEnum):
    """航向来源，取值 odom_raw 或 odom。

    odom_raw 为 STM32 直出，与位置同源且高频；odom 为 EKF 输出，融合 IMU 航向
    通常更准但仅约 6Hz。两者误差特性不同，``odom_angular_scale_correction`` 必须
    按所选来源现场重标。
    """

    ODOM_RAW = 'odom_raw'
    EKF_ODOM = 'odom'


def yaw_from_quaternion(quaternion) -> float:
    """从四元数提取绕 Z 轴的偏航角，单位 rad。

    只关心平面运动，故直接由 z 与 w 分量求 atan2，避免引入 tf_transformations
    依赖。

    :param quaternion: 四元数消息对象，需含 x、y、z、w 分量。
    :returns: 偏航角，取值区间负 π 到正 π。
    """
    yaw_sin_term = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    yaw_cos_term = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(yaw_sin_term, yaw_cos_term)


class CommandWatchdog:
    """速度指令看门狗：非零指令超时未刷新即强制归零。

    每次速度指令调用一次 :meth:`feed` 喂狗。最后一条指令为非零且超过
    ``timeout`` 秒没有再次喂狗时，判定上层已失联，调用 ``on_timeout``。

    为什么必须是独立线程
    --------------------
    本机 ``/cmd_vel`` 无超时停车保护，下位机会一直执行最后收到的指令。而指令
    上线依赖单线程 executor 里的主线程 spin，一旦主线程阻塞，例如机械臂
    ``time.sleep``，或 executor 卡死，定时器不再发布，最后的速度就被锁存。本类
    跑在独立线程、只依赖 DDS 发布，不经过 executor，因此仍能发出停车指令。

    .. warning::
       进程被 ``SIGKILL`` 或 OOM 杀死时线程一并消失，本看门狗无法兜底。该场景
       只能靠物理断电，详见 ``SAFETY.md``。

    ``timeout`` 不大于 0 表示禁用看门狗。
    """

    def __init__(
        self,
        timeout: float,
        on_timeout: Callable[[], None],
        period: float = DEFAULT_WATCHDOG_PERIOD_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """配置阈值与轮询周期，线程需另调 :meth:`start` 启动。

        :param timeout: 超时阈值，单位 s，不大于 0 表示禁用。
        :param on_timeout: 触发时的回调，应下发全 0 速度。
        :param period: 轮询周期，单位 s，下限为 ``MIN_WATCHDOG_PERIOD_SEC``。
        :param clock: 时钟函数，返回 monotonic 秒，注入假时钟便于单测。
        """
        self._timeout = float(timeout)
        self._on_timeout = on_timeout
        self._period = max(MIN_WATCHDOG_PERIOD_SEC, float(period))
        self._clock = clock
        self._lock = threading.Lock()
        self._cmd_ts = 0.0
        self._nonzero = False
        self._thread: Optional[threading.Thread] = None
        self._running = False
        #: 累计触发次数，供日志与自检查询，取值不小于 0
        self.trips = 0

    @property
    def timeout(self) -> float:
        """看门狗超时阈值，单位 s。"""
        return self._timeout

    @property
    def enabled(self) -> bool:
        """看门狗是否启用，即阈值大于 0。"""
        return self._timeout > 0.0

    def feed(self, nonzero: bool) -> None:
        """喂狗：记录最近一次指令的时刻，并标记它是否为非零指令。

        :param nonzero: 本条速度指令是否含非零分量。
        """
        with self._lock:
            self._cmd_ts = self._clock()
            self._nonzero = bool(nonzero)

    def check(self, now: Optional[float] = None) -> bool:
        """判定一次，超时且最后指令为非零时回调归零。

        :param now: 当前时刻，单位 s，取 ``None`` 时读注入时钟。显式传入便于单测。
        :returns: 本次是否触发。
        """
        if not self.enabled:
            return False
        now = self._clock() if now is None else float(now)
        with self._lock:
            tripped = self._nonzero and (now - self._cmd_ts) > self._timeout
            if tripped:
                # 归零标记，避免同一次失联被反复触发，回调会重新喂零速指令
                self._nonzero = False
                self._cmd_ts = now
        if tripped:
            self.trips += 1
            self._on_timeout()
        return tripped

    def start(self) -> None:
        """启动看门狗线程，禁用或已启动时为空操作。"""
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name='cmd_watchdog', daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止看门狗线程并等待其退出，上限为 ``WATCHDOG_JOIN_TIMEOUT_SEC``。"""
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=WATCHDOG_JOIN_TIMEOUT_SEC)

    def _loop(self) -> None:
        """轮询线程主体：周期判定，异常不得导致线程退出。"""
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
        """声明参数、建立话题连接并启动看门狗线程。

        构造参数优先于 ``config/*.yaml``：本节点由 mission_manager 在进程内创建，
        收不到 launch 注入的 yaml，故需由调用方把值转发进来。

        :param yaw_source: 航向来源，取值见 ``YawSource``，取 ``None`` 时读参数。
        :param watchdog_timeout: 看门狗阈值，单位 s，取 ``None`` 时读参数。
        """
        super().__init__('base_driver')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom_raw')
        self.declare_parameter('alt_odom_topic', '/odom')
        self.declare_parameter('yaw_source', YawSource.ODOM_RAW.value)
        self.declare_parameter('publish_rate_hz', DEFAULT_PUBLISH_RATE_HZ)
        self.declare_parameter('odom_timeout_sec', DEFAULT_ODOM_TIMEOUT_SEC)
        self.declare_parameter(
            'cmd_watchdog_timeout', DEFAULT_WATCHDOG_TIMEOUT_SEC
        )

        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        alt_odom_topic = str(self.get_parameter('alt_odom_topic').value)
        raw_yaw_source = (
            yaw_source if yaw_source is not None
            else str(self.get_parameter('yaw_source').value)
        )
        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self._odom_timeout = float(self.get_parameter('odom_timeout_sec').value)
        watchdog_timeout = float(
            watchdog_timeout if watchdog_timeout is not None
            else self.get_parameter('cmd_watchdog_timeout').value
        )

        if rate_hz < MIN_PUBLISH_RATE_HZ:
            self.get_logger().warn(
                f'publish_rate_hz={rate_hz:.1f} 低于 {MIN_PUBLISH_RATE_HZ:.0f}Hz，'
                '下位机可能因指令间隔过长发散'
            )

        # 回调在 executor 线程执行，主线程也会读，故用锁保护
        self._lock = threading.Lock()
        self._cmd = Twist()  # 目标速度，默认全 0
        self._pose: Optional[Pose] = None
        self._alt_yaw: Optional[float] = None
        self._last_odom_ts = 0.0
        self._stale_warned = False

        self._pub = self.create_publisher(Twist, cmd_topic, CMD_VEL_QUEUE_DEPTH)
        self._sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, ODOM_QUEUE_DEPTH
        )
        # 备用航向源只取 yaw，不参与位置解算
        self._alt_sub = self.create_subscription(
            Odometry, alt_odom_topic, self._on_alt_odom, ALT_ODOM_QUEUE_DEPTH
        )
        self.create_timer(1.0 / rate_hz, self._tick)

        # 速度看门狗跑在独立线程：主线程阻塞或 executor 卡死时仍能强制归零
        self._watchdog = CommandWatchdog(watchdog_timeout, self._force_zero)
        self._watchdog.start()

        try:
            self._yaw_source = YawSource(raw_yaw_source)
        except ValueError:
            self.get_logger().warn(
                f'yaw_source={raw_yaw_source!r} 非法，回退为 '
                f'{YawSource.ODOM_RAW.value}'
            )
            self._yaw_source = YawSource.ODOM_RAW

        self.get_logger().info(
            f'BaseDriver 就绪 | cmd_vel={cmd_topic} | odom={odom_topic} | {rate_hz:.0f}Hz'
        )
        if self._yaw_source == YawSource.EKF_ODOM:
            self.get_logger().info(
                f'航向来源 {YawSource.EKF_ODOM.value} 即 EKF，仅约 6Hz | '
                f'备用源 {alt_odom_topic}；换源后必须用 calibration_tool motion '
                '模式重标角速度系数'
            )
        else:
            self.get_logger().info(
                f'航向来源 {YawSource.ODOM_RAW.value} 与位置同源 | '
                f'备用源 {alt_odom_topic}'
            )
        if self._watchdog.enabled:
            self.get_logger().info(
                f'速度看门狗已启用：非零指令超过 {watchdog_timeout:.2f}s 未刷新即强制归零'
            )
        else:
            self.get_logger().warn(
                '速度看门狗被禁用，即 cmd_watchdog_timeout 不大于 0——主线程阻塞时'
                '最后的速度会被下位机锁存，存在失控风险，详见 SAFETY.md'
            )

    # ------------------------------------------------------------------ 指令

    def set_velocity(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        """设置目标速度，下个定时器周期起持续发布。

        同时喂一次看门狗：非零指令必须被持续刷新，否则看门狗会强制归零。

        :param linear_x: 前后速度，单位 m/s，正为前进。
        :param linear_y: 横移速度，单位 m/s，正为向左。
        :param angular_z: 自转角速度，单位 rad/s，正为逆时针。
        """
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.linear.y = float(linear_y)
        cmd.angular.z = float(angular_z)
        with self._lock:
            self._cmd = cmd
        self._watchdog.feed(bool(linear_x or linear_y or angular_z))

    def stop(self) -> None:
        """立即停止，显式发送全 0 速度。"""
        self.set_velocity(0.0, 0.0, 0.0)

    def last_command(self) -> Tuple[float, float, float]:
        """返回缓存的速度指令三分量，供自检与单测查询。

        :returns: 依次为前后 m/s、横移 m/s、自转 rad/s。
        """
        with self._lock:
            return (
                self._cmd.linear.x,
                self._cmd.linear.y,
                self._cmd.angular.z,
            )

    @property
    def watchdog_trips(self) -> int:
        """看门狗累计强制归零次数，取值不小于 0。"""
        return self._watchdog.trips

    def _force_zero(self) -> None:
        """看门狗超时回调：强制归零并立即发布。

        可能在看门狗线程中执行，因此只使用锁保护的最小状态。
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
        """返回缓存位姿，尚未收到里程计时返回 ``None``。

        :returns: 三分量依次为 x 与 y 单位 m、yaw 单位 rad。
        """
        with self._lock:
            return self._pose

    def get_yaw(self) -> Optional[float]:
        """返回当前偏航角，单位 rad，无数据时返回 ``None``。

        按 ``yaw_source`` 选择来源：``odom_raw`` 取主里程计；``odom`` 优先取
        EKF 备用源，其尚无数据时退回主源，以免上电初期返回 ``None`` 导致转向
        原语直接失败。
        """
        with self._lock:
            if self._yaw_source == YawSource.EKF_ODOM and self._alt_yaw is not None:
                return self._alt_yaw
            return None if self._pose is None else self._pose[2]

    def odom_age(self) -> float:
        """返回距最近一帧里程计的时长，单位 s，从未收到时为 ``inf``。

        :returns: 非负秒数或正无穷。
        """
        with self._lock:
            if self._last_odom_ts <= 0.0:
                return math.inf
            return time.monotonic() - self._last_odom_ts

    # -------------------------------------------------------------- 内部回调

    def _on_odom(self, msg: Odometry) -> None:
        """刷新位姿缓存与时间戳，并复位失联告警标记。

        :param msg: 主里程计消息，来自 ``/odom_raw``。
        """
        position = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        with self._lock:
            self._pose = (position.x, position.y, yaw)
            self._last_odom_ts = time.monotonic()
            self._stale_warned = False

    def _on_alt_odom(self, msg: Odometry) -> None:
        """备用航向源回调：只取 yaw，不使用其位置。

        :param msg: 备用里程计消息，来自 ``/odom``。
        """
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        with self._lock:
            self._alt_yaw = yaw

    def publish_now(self) -> None:
        """立即发布当前缓存的速度指令，不等定时器周期。"""
        with self._lock:
            cmd = self._cmd
        self._pub.publish(cmd)

    def _tick(self) -> None:
        """定时器主体：重发速度指令并做里程计失联监测。"""
        self.publish_now()

        # 里程计失联监测只在状态翻转时告警一次，避免每帧刷屏
        age = self.odom_age()
        if age > self._odom_timeout and not self._stale_warned:
            with self._lock:
                self._stale_warned = True
            self.get_logger().warn(
                f'超过 {self._odom_timeout:.1f}s 未收到里程计，请检查 micro-ROS agent '
                '是否启动，即 start_agent.sh，以及 ROS_DOMAIN_ID=30'
            )


def main(args: Optional[List[str]] = None) -> None:
    """独立运行入口：维持底盘话题在线，用于联调时确认接口可用。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
    rclpy.init(args=args)
    node = BaseDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()  # 停看门狗并确保下发停止指令
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
