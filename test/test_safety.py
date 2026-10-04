"""安全保护单测：速度看门狗、激光/相机保鲜度门禁。

覆盖 ``SAFETY.md`` 的三项最小改动：

1. :class:`CommandWatchdog` —— 非零速度指令超时未刷新即强制归零（注入假时钟，纯算法）；
2. ``SensorHub.scan_age()`` —— 激光保鲜度可查（需要 ROS 上下文，环境不允许则跳过）；
3. ``MotionController`` 门禁 —— 激光/相机失效时禁止移动（注入桩对象，不需要 ROS）。

注意：``advance``/``turn_to_heading`` 的**失效路径**在进入闭环前就返回，
因此这些用例无需 ``rclpy.init()``；只有直接构造 ROS 节点的用例才需要上下文。
"""

import math
import time
from typing import List, Tuple

import pytest

from maze_explorer.base_driver import CommandWatchdog
from maze_explorer.motion_controller import MotionController


# ---------------------------------------------------------------- 测试替身

class FakeClock:
    """可控时钟：替代 ``time.monotonic`` 让超时判定完全确定。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


class FakeBase:
    """最小底盘桩：记录每一条速度指令。"""

    def __init__(self, yaw: float = 0.0) -> None:
        self.yaw = yaw
        self.commands: List[Tuple[float, float, float]] = []

    def get_pose(self) -> Tuple[float, float, float]:
        return (0.0, 0.0, self.yaw)

    def get_yaw(self) -> float:
        return self.yaw

    def set_velocity(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        self.commands.append((float(vx), float(vy), float(wz)))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0, 0.0)

    def moved(self) -> bool:
        """是否出现过非零速度指令。"""
        return any(cmd != (0.0, 0.0, 0.0) for cmd in self.commands)


class FakeSensors:
    """最小传感器桩：只提供保鲜度查询。"""

    def __init__(self, scan_age: float = 0.0, rgbd_age: float = 0.0) -> None:
        self.scan = float(scan_age)
        self.rgbd = float(rgbd_age)

    def scan_age(self) -> float:
        return self.scan

    def rgbd_age(self) -> float:
        return self.rgbd


class FakeLogger:
    """吞掉日志，保留最近一条 warn 供断言。"""

    def __init__(self) -> None:
        self.warnings: List[str] = []

    def warn(self, msg: str, **_kwargs) -> None:
        self.warnings.append(str(msg))

    def info(self, *_args, **_kwargs) -> None:
        pass


class FakeNode:
    """只提供 MotionController 用到的 logger。"""

    def __init__(self) -> None:
        self.logger = FakeLogger()

    def get_logger(self) -> FakeLogger:
        return self.logger


# ------------------------------------------------------------ 一、看门狗

def test_watchdog_trips_after_timeout_for_nonzero_command() -> None:
    """非零指令超时未刷新 → 回调归零，且只触发一次。"""
    clock = FakeClock()
    trips: List[float] = []
    watchdog = CommandWatchdog(0.5, lambda: trips.append(clock.now), clock=clock)

    watchdog.feed(True)
    clock.advance(0.4)
    assert watchdog.check() is False          # 未超时

    clock.advance(0.2)                        # 累计 0.6s > 0.5s
    assert watchdog.check() is True
    assert len(trips) == 1
    assert watchdog.trips == 1

    clock.advance(10.0)
    assert watchdog.check() is False          # 不重复触发
    assert len(trips) == 1


def test_watchdog_ignores_zero_command() -> None:
    """零速指令永远安全，不触发。"""
    clock = FakeClock()
    watchdog = CommandWatchdog(0.5, lambda: pytest.fail('零速指令不应触发看门狗'),
                               clock=clock)
    watchdog.feed(False)
    clock.advance(100.0)

    assert watchdog.check() is False
    assert watchdog.trips == 0


def test_watchdog_feed_renews_deadline() -> None:
    """持续喂狗（运动循环每 20ms 一次）不应触发。"""
    clock = FakeClock()
    watchdog = CommandWatchdog(0.5, lambda: pytest.fail('持续喂狗不应触发'),
                               clock=clock)

    for _ in range(5):
        watchdog.feed(True)
        clock.advance(0.4)
        assert watchdog.check() is False


def test_watchdog_disabled_when_timeout_not_positive() -> None:
    """timeout<=0 表示禁用（仅用于联调，不建议）。"""
    clock = FakeClock()
    watchdog = CommandWatchdog(0.0, lambda: pytest.fail('禁用时不应触发'),
                               clock=clock)
    assert watchdog.enabled is False

    watchdog.feed(True)
    clock.advance(100.0)
    assert watchdog.check() is False


def test_watchdog_thread_trips_and_stops() -> None:
    """真实线程烟雾测试：会触发，且 stop() 后不再触发。"""
    trips: List[float] = []
    watchdog = CommandWatchdog(0.05, lambda: trips.append(time.monotonic()))
    watchdog.start()
    try:
        watchdog.feed(True)
        deadline = time.monotonic() + 3.0
        while watchdog.trips == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert watchdog.trips >= 1, '看门狗线程未在超时后触发'
    finally:
        watchdog.stop()

    settled = watchdog.trips
    time.sleep(0.2)
    assert watchdog.trips == settled, 'stop() 之后不应再触发'


# ------------------------------------------------ 二、SensorHub 保鲜度

@pytest.fixture(scope='module')
def ros_context():
    """提供 ROS 上下文；环境不允许（例如 ~/.ros 只读）时跳过用例。"""
    rclpy = pytest.importorskip('rclpy')
    if not rclpy.ok():
        try:
            rclpy.init()
        except RuntimeError as exc:  # pragma: no cover - 环境受限
            pytest.skip(f'ROS 上下文不可用：{exc}')
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_sensor_hub_scan_age(ros_context) -> None:  # noqa: ARG001
    """未收到激光时为 inf；注入一帧后应立刻变新鲜。"""
    from sensor_msgs.msg import LaserScan

    from maze_explorer.sensors import SensorHub

    hub = SensorHub()
    try:
        assert hub.scan_age() == math.inf
        assert hub.rgbd_age() == math.inf

        hub._on_scan(LaserScan())  # noqa: SLF001 - 直接注入一帧，避免起发布者
        assert hub.scan_age() < 1.0
    finally:
        hub.destroy_node()


def test_base_driver_watchdog_wiring(ros_context) -> None:  # noqa: ARG001
    """BaseDriver 接线：非零指令停止刷新后，看门狗强制把缓存归零。"""
    from maze_explorer.base_driver import BaseDriver

    node = BaseDriver(watchdog_timeout=0.2)
    try:
        node.set_velocity(0.15, 0.0, 0.1)
        assert node.last_command() != (0.0, 0.0, 0.0)

        deadline = time.monotonic() + 3.0
        while node.watchdog_trips == 0 and time.monotonic() < deadline:
            time.sleep(0.02)

        assert node.watchdog_trips >= 1, '看门狗未在超时后强制归零'
        assert node.last_command() == (0.0, 0.0, 0.0), '触发后缓存应已归零'
    finally:
        node.shutdown()
        node.destroy_node()


# --------------------------------------- 三、运动原语的传感器门禁

def _make_controller(
    base: FakeBase,
    sensors: FakeSensors,
    scan_timeout: float = 0.5,
    vision_timeout: float = 2.0,
) -> Tuple[MotionController, FakeNode]:
    node = FakeNode()
    controller = MotionController(
        node, base, sensors, None,  # type: ignore[arg-type]
        spin_fn=lambda _timeout: None,
        scan_timeout=scan_timeout,
        vision_timeout=vision_timeout,
    )
    return controller, node


def test_sensors_ok_boundary() -> None:
    """阈值边界：等于阈值算健康，超过才算失效。"""
    base = FakeBase()
    controller, _ = _make_controller(base, FakeSensors(scan_age=0.5, rgbd_age=2.0))

    assert controller.sensors_ok() is True
    assert controller.sensors_ok(need_vision=False) is True


def test_sensors_ok_detects_stale_scan_and_vision() -> None:
    """激光过期 → 全部动作禁止；相机过期 → 仅禁止需要视觉的动作。"""
    base = FakeBase()
    stale_scan, _ = _make_controller(base, FakeSensors(scan_age=5.0))
    assert stale_scan.sensors_ok() is False
    assert stale_scan.sensors_ok(need_vision=False) is False

    stale_vision, _ = _make_controller(base, FakeSensors(rgbd_age=10.0))
    assert stale_vision.sensors_ok() is False                    # 前进要视觉 → 禁止
    assert stale_vision.sensors_ok(need_vision=False) is True    # 转向不要视觉 → 放行


def test_sensors_ok_without_age_interface_defaults_healthy() -> None:
    """桩对象不提供保鲜度接口时按健康处理（纯逻辑单测可用）。"""
    controller, _ = _make_controller(FakeBase(), None)  # type: ignore[arg-type]
    assert controller.sensors_ok() is True


def test_advance_refuses_when_scan_stale() -> None:
    """激光失效：一步都不许动，且不下发非零速度。"""
    base = FakeBase()
    controller, node = _make_controller(base, FakeSensors(scan_age=5.0))

    assert controller.advance(0.4) is False
    assert base.moved() is False
    assert any('传感器数据失效' in msg for msg in node.logger.warnings)


def test_advance_refuses_when_vision_stale() -> None:
    """相机失效：前进被拒（否则循迹偏差恒 0，车会闷头直行）。"""
    base = FakeBase()
    controller, _ = _make_controller(base, FakeSensors(rgbd_age=10.0))

    assert controller.advance(0.4) is False
    assert base.moved() is False


def test_advance_backward_needs_scan_only() -> None:
    """后退不需要相机：相机过期时门禁应放行到闭环阶段。"""
    base = FakeBase()
    controller, _ = _make_controller(base, FakeSensors(rgbd_age=10.0))

    assert controller.sensors_ok(need_vision=False) is True
    assert controller._stop_if_unhealthy(need_vision=False) is False  # noqa: SLF001


def test_turn_refuses_when_scan_stale() -> None:
    """激光失效时不允许盲转：动作开始前即返回 False。"""
    base = FakeBase()
    controller, _ = _make_controller(base, FakeSensors(scan_age=5.0))

    assert controller.turn_to_heading('N') is False
    assert base.moved() is False


def test_strafe_refuses_when_scan_stale() -> None:
    """横移同样受激光门禁约束。"""
    base = FakeBase()
    controller, _ = _make_controller(base, FakeSensors(scan_age=5.0))

    assert controller.strafe(0.1) is False
    assert base.moved() is False


def test_scan_age_infinite_without_data() -> None:
    """激光从未到达时年龄为 inf，必然判定失效。"""
    sensors = FakeSensors(scan_age=math.inf)
    controller, _ = _make_controller(FakeBase(), sensors)

    assert controller.scan_age() == math.inf
    assert controller.sensors_ok() is False
