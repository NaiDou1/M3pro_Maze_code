"""安全保护单测：速度看门狗、激光与相机保鲜度门禁。

覆盖 ``SAFETY.md`` 的三项最小改动：

1. :class:`CommandWatchdog` —— 非零速度指令超时未刷新即强制归零，注入假时钟，纯算法；
2. ``SensorHub`` 的 ``scan_age`` —— 激光保鲜度可查，需要 ROS 上下文，不允许则跳过；
3. ``MotionController`` 门禁 —— 激光与相机失效时禁止移动，注入桩对象，不需要 ROS。

注意：``advance`` 与 ``turn_to_heading`` 的**失效路径**在进入闭环前就返回，
因此这些用例无需 ``rclpy`` 的 init 接口；只有直接构造 ROS 节点的用例才需要上下文。
"""

import math
import time
from typing import List, Tuple

import numpy as np

import pytest

from maze_explorer.base_driver import CommandWatchdog
from maze_explorer.line_detector import LineObservation
from maze_explorer.motion_controller import MotionController


# ---------------------------------------------------------------- 测试替身

class FakeClock:
    """可控时钟：替代 ``time.monotonic`` 让超时判定完全确定。"""

    def __init__(self, start: float = 1000.0) -> None:
        """给定起始时刻，后续由 advance 推进。"""
        self.now = float(start)

    def __call__(self) -> float:
        """以可调用形式替代 time.monotonic。"""
        return self.now

    def advance(self, dt: float) -> None:
        """推进时钟，单位 s。"""
        self.now += float(dt)


class FakeBase:
    """最小底盘桩：记录每一条速度指令。"""

    def __init__(self, yaw: float = 0.0) -> None:
        """给定初始 yaw，指令记录列表置空。"""
        self.yaw = yaw
        self.commands: List[Tuple[float, float, float]] = []

    def get_pose(self) -> Tuple[float, float, float]:
        """返回 x 与 y 恒 0 的位姿，只有 yaw 参与判定。"""
        return (0.0, 0.0, self.yaw)

    def get_yaw(self) -> float:
        """返回当前 yaw。"""
        return self.yaw

    def set_velocity(
        self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0
    ) -> None:
        """记录一条速度指令，形参名与 BaseDriver 保持一致。"""
        self.commands.append((float(linear_x), float(linear_y), float(angular_z)))

    def stop(self) -> None:
        """补记一条全零指令，模拟放弃动作前停车。"""
        self.set_velocity(0.0, 0.0, 0.0)

    def moved(self) -> bool:
        """是否出现过非零速度指令。"""
        return any(cmd != (0.0, 0.0, 0.0) for cmd in self.commands)

    def last_is_zero(self) -> bool:
        """最后一条速度指令是否为全 0，用于验证放弃动作前先停车。"""
        return bool(self.commands) and self.commands[-1] == (0.0, 0.0, 0.0)


class FakeSensors:
    """最小传感器桩：只提供保鲜度查询。"""

    def __init__(self, scan_age: float = 0.0, rgbd_age: float = 0.0) -> None:
        """给定激光与相机的初始保鲜度，单位 s。"""
        self.scan = float(scan_age)
        self.rgbd = float(rgbd_age)

    def scan_age(self) -> float:
        """返回构造时设定的激光年龄。"""
        return self.scan

    def rgbd_age(self) -> float:
        """返回构造时设定的相机年龄。"""
        return self.rgbd


class RecoveringSensors(FakeSensors):
    """失效后能在若干次 spin 内恢复新鲜的桩，用于测"宽限等待"。"""

    def __init__(self, scan_age: float, rgbd_age: float, recover_after: int) -> None:
        """给定初值与恢复所需的自旋次数。"""
        super().__init__(scan_age, rgbd_age)
        self.spins = 0
        self.recover_after = int(recover_after)

    def tick(self) -> None:
        """模拟一次 spin：回调收到新数据，保鲜度恢复。"""
        self.spins += 1
        if self.spins >= self.recover_after:
            self.scan = 0.0
            self.rgbd = 0.0


class FakeLogger:
    """吞掉日志，保留最近一条 warn 供断言。"""

    def __init__(self) -> None:
        """告警列表置空。"""
        self.warnings: List[str] = []

    def warn(self, msg: str, **_kwargs) -> None:
        """记录告警文本供断言。"""
        self.warnings.append(str(msg))

    def info(self, *_args, **_kwargs) -> None:
        """丢弃 info 级日志。"""
        pass


class FakeNode:
    """只提供 MotionController 用到的 logger。"""

    def __init__(self) -> None:
        """创建自带的测试 logger。"""
        self.logger = FakeLogger()

    def get_logger(self) -> FakeLogger:
        """返回测试用 logger。"""
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
    """持续喂狗，运动循环每 20ms 一次，不应触发。"""
    clock = FakeClock()
    watchdog = CommandWatchdog(0.5, lambda: pytest.fail('持续喂狗不应触发'),
                               clock=clock)

    for _ in range(5):
        watchdog.feed(True)
        clock.advance(0.4)
        assert watchdog.check() is False


def test_watchdog_disabled_when_timeout_not_positive() -> None:
    """超时值不为正即禁用看门狗，仅用于联调，不建议。"""
    clock = FakeClock()
    watchdog = CommandWatchdog(0.0, lambda: pytest.fail('禁用时不应触发'),
                               clock=clock)
    assert watchdog.enabled is False

    watchdog.feed(True)
    clock.advance(100.0)
    assert watchdog.check() is False


def test_watchdog_thread_trips_and_stops() -> None:
    """真实线程烟雾测试：会触发，且 stop 之后不再触发。"""
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
    """提供 ROS 上下文，环境不允许即例如 home 目录只读时跳过用例。"""
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
    recover_wait: float = 1.5,
    spin_fn=None,
) -> Tuple[MotionController, FakeNode]:
    """按超时与恢复参数构造控制器，自旋回调可注入。"""
    node = FakeNode()
    controller = MotionController(
        node, base, sensors, None,  # type: ignore[arg-type]
        spin_fn=spin_fn if spin_fn is not None else (lambda _timeout: None),
        scan_timeout=scan_timeout,
        vision_timeout=vision_timeout,
        sensor_recover_wait=recover_wait,
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
    """桩对象不提供保鲜度接口时按健康处理，纯逻辑单测可用。"""
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
    """相机失效即前进被拒，否则循迹偏差恒 0，车会闷头直行。"""
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


# --------------------------------- 四、失效后的宽限等待，启动竞态修复

def test_stop_if_unhealthy_waits_for_recovery(ros_context) -> None:  # noqa: ARG001
    """短暂失效不应直接放弃动作：等待期内恢复就放行。

    背景：SensorHub 首帧同步 RGB-D 需 1.6 到 1.8 s，没有宽限会在 EXPLORE 第一步
    就误判相机失效并连环失败进 FAULT。
    """
    sensors = RecoveringSensors(scan_age=5.0, rgbd_age=9.0, recover_after=2)
    controller, _ = _make_controller(
        FakeBase(), sensors, recover_wait=2.0, spin_fn=lambda _t: sensors.tick()
    )
    assert controller.sensors_ok() is False

    assert controller._stop_if_unhealthy(need_vision=True) is False  # noqa: SLF001
    assert controller.sensors_ok() is True
    assert sensors.spins >= 2


def test_stop_if_unhealthy_aborts_when_never_recovers(ros_context) -> None:  # noqa: ARG001
    """始终不恢复才放弃动作，并给出年龄明细。"""
    base = FakeBase()
    controller, node = _make_controller(
        base, FakeSensors(scan_age=5.0, rgbd_age=9.0), recover_wait=0.2
    )

    assert controller._stop_if_unhealthy(need_vision=True) is True  # noqa: SLF001
    assert base.last_is_zero() is True, '放弃动作前必须先停车'
    assert any('传感器数据失效' in msg for msg in node.logger.warnings)


def test_wait_until_healthy_times_out_without_recovery(ros_context) -> None:  # noqa: ARG001
    """wait_until_healthy 在超时后返回 False，不无限等待。"""
    controller, _ = _make_controller(
        FakeBase(), FakeSensors(scan_age=5.0), recover_wait=0.2
    )
    started = time.monotonic()
    assert controller.wait_until_healthy(need_vision=False, timeout=0.3) is False
    assert time.monotonic() - started < 3.0, '超时保护失效，等待过久'


def test_wait_until_healthy_returns_true_when_already_fresh() -> None:
    """本来就新鲜时立即返回 True，不做任何等待，走快路径。"""
    controller, _ = _make_controller(FakeBase(), FakeSensors())

    started = time.monotonic()
    assert controller.wait_until_healthy(need_vision=True, timeout=5.0) is True
    assert time.monotonic() - started < 0.1


# ------------------------------------ 五、巡线转向符号，实测回归

class FakeLine:
    """固定偏差的巡线桩：``offset_norm > 0`` 表示线在图像右侧。"""

    def __init__(self, offset_px: float, width: float = 640.0) -> None:
        """把像素偏差换算成归一化偏差并构造观测。"""
        self._obs = LineObservation(
            valid=True, offset_px=offset_px, offset_norm=offset_px / (width / 2.0)
        )
        self.is_lost = False

    def detect(self, _img):
        """恒返回构造时的观测，忽略图像内容。"""
        return self._obs


class DriveSensors(FakeSensors):
    """advance 所需的完整传感器桩：新鲜、有图、前方无遮挡。"""

    def get_rgbd(self):
        """返回极小图像，只为通过存在性检查。"""
        return (np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.float32))

    def sector_min_range(self, _angle_deg, _half_width_deg):
        """恒返回 2.0 m，模拟前方无遮挡。"""
        return 2.0          # 远高于 safety_range，不触发碰撞保护


def _drive_once(controller: MotionController, base: FakeBase, timeout: float = 0.2):
    """让 advance 跑一小段，靠超时退出，返回期间下发的 angular.z 序列。"""
    base.commands.clear()
    controller.advance(0.1, timeout=timeout)
    return [c[2] for c in base.commands]


def test_steering_turns_toward_line_on_right(ros_context) -> None:  # noqa: ARG001
    """线在图像右侧即 offset_norm 大于 0 必须右转，angular.z 为负。

    实测背景：黑线近粗远细，即行 280 宽 12px 而行 460 宽 17px，说明图像下半部为
    近处且未旋转，故图像右侧就是车体右侧；旧实现给正 Kp 乘像素偏差即左转，
    车越转越偏随后丢线，现场表现为一直往左转不跟随。
    """
    base = FakeBase()
    controller, _ = _make_controller(
        base, DriveSensors(), recover_wait=0.2, spin_fn=lambda _t: None
    )
    controller._line_detector = FakeLine(offset_px=+40.0)  # noqa: SLF001

    zs = _drive_once(controller, base)

    assert zs, '未下发任何速度指令'
    assert all(z <= 0.0 for z in zs), f'线在右侧却给了左转指令：{zs}'
    assert any(z < 0.0 for z in zs), '转向量恒为 0，符号检查失效'


def test_steering_turns_left_when_line_on_left(ros_context) -> None:  # noqa: ARG001
    """线在左侧即 offset_norm 小于 0 必须左转，angular.z 为正。"""
    base = FakeBase()
    controller, _ = _make_controller(
        base, DriveSensors(), recover_wait=0.2, spin_fn=lambda _t: None
    )
    controller._line_detector = FakeLine(offset_px=-40.0)  # noqa: SLF001

    zs = _drive_once(controller, base)

    assert any(z > 0.0 for z in zs), f'线在左侧却给了右转指令：{zs}'


def test_steering_sign_is_configurable(ros_context) -> None:  # noqa: ARG001
    """line_steer_sign 为 +1 时符号翻转，供相机装反的场合使用。"""
    base = FakeBase()
    controller, _ = _make_controller(
        base, DriveSensors(), recover_wait=0.2, spin_fn=lambda _t: None
    )
    controller._line_detector = FakeLine(offset_px=+40.0)  # noqa: SLF001
    controller._line_steer_sign = +1.0  # noqa: SLF001

    zs = _drive_once(controller, base)

    assert any(z > 0.0 for z in zs), '符号开关未生效'


def test_steering_not_saturated_by_small_offset(ros_context) -> None:  # noqa: ARG001
    """归一化误差配合理增益，小偏差给出小转向，不再满舵。

    旧实现 Kp=50 配像素误差：40px 偏差直接被限幅到正负 0.6rad/s，即满舵开关。
    """
    base = FakeBase()
    controller, _ = _make_controller(
        base, DriveSensors(), recover_wait=0.2, spin_fn=lambda _t: None
    )
    controller._line_detector = FakeLine(offset_px=+20.0)  # offset_norm=0.0625  # noqa: SLF001

    zs = [z for z in _drive_once(controller, base) if z != 0.0]

    assert zs, '未产生转向'
    assert all(abs(z) < 0.6 for z in zs), f'小偏差被饱和成满舵：{zs[:5]}…'
    assert abs(zs[0]) < 0.2, f'归一化偏差 0.0625 转向过大：{zs[0]:.3f}'
