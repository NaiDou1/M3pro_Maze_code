"""转向缩放语义的回归测试。

重点锁定 B2 修复的语义：出厂标定实测关系为

    真实转角 = ang_scale × odom 读数

因此要达成目标真实转角，闭环目标必须是「目标转角 / ang_scale」。先前实现误用
乘法，导致 ang_scale=0.75 时 90 度只转 67.5 度，逐格累积必然撞墙。

这里用最小底盘桩 + 注入 spin 回调，使闭环可离线运行。
"""

import math
from typing import Optional

import pytest

import rclpy

from maze_explorer.grid_mapper import angle_diff
from maze_explorer.motion_controller import MotionController


@pytest.fixture(scope='module', autouse=True)
def _rclpy_context():
    """``turn_to_heading`` 以 ``rclpy`` 的 ok 接口作循环条件，须先初始化上下文。"""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


class FakeBase:
    """最小底盘桩：记录速度，由 spin 推进 yaw。"""

    def __init__(self, yaw: float = 0.0) -> None:
        """给定初始 yaw，速度初值三轴为 0。"""
        self.yaw = yaw
        self.vel = (0.0, 0.0, 0.0)
        self._yaw_override: Optional[bool] = None

    def get_pose(self):
        """返回 x 与 y 恒 0 的位姿，只有 yaw 参与闭环。"""
        return (0.0, 0.0, self.yaw)

    def get_yaw(self):
        """里程计可用时返回 yaw，模拟失效时返回 None。"""
        if self._yaw_override is None:
            return self.yaw
        return None

    def set_velocity(
        self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0
    ) -> None:
        """记录速度指令供闭环与断言读取。"""
        self.vel = (linear_x, linear_y, angular_z)

    def stop(self) -> None:
        """清零速度，供放弃动作路径调用。"""
        self.vel = (0.0, 0.0, 0.0)

    def drop_odom(self) -> None:
        """模拟里程计不可用。"""
        self._yaw_override = False


class FakeLogger:
    def warn(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass


class FakeNode:
    """只提供 MotionController 用到的 logger。"""

    def get_logger(self) -> FakeLogger:
        """返回吞日志的 logger，仅满足接口。"""
        return FakeLogger()


def _make_controller(base: FakeBase, ang_scale: float, dt: float = 0.02) -> MotionController:
    """构造带角速度积分桩的控制器，系数与容差按参数注入。"""
    def spin(_timeout: float) -> None:
        """每次自旋把角速度按 dt 积分进 yaw，模拟底盘响应。"""
        # 模拟 0.02 秒的角速度积分
        base.yaw += base.vel[2] * dt

    return MotionController(
        FakeNode(), base, None, None,  # type: ignore[arg-type]
        spin_fn=spin,
        angular_scale_correction=ang_scale,
        yaw_tolerance=math.radians(2.0),
        turn_timeout=10.0,
        turn_angular=0.5,
    )


# ---------------------------------------------------------------- 缩放方向

@pytest.mark.parametrize(
    'ang_scale,expected_odom_delta',
    [
        (1.0, math.pi / 2),    # 系数 1：odom 变化 = 目标真实转角
        (0.5, math.pi),        # 真实 = 0.5 × 读数 → 读数须为目标的 2 倍
        (2.0, math.pi / 4),    # 真实 = 2 × 读数 → 读数只须目标的一半
    ],
)
def test_turn_scaling_uses_division(ang_scale: float, expected_odom_delta: float) -> None:
    """转向到正北即 +90 度后，odom 的 yaw 变化应为「目标 / 系数」。"""
    base = FakeBase(yaw=0.0)
    controller = _make_controller(base, ang_scale)

    assert controller.turn_to_heading('N') is True
    assert base.yaw == pytest.approx(expected_odom_delta, abs=0.05)


def test_turn_scaling_negative_direction() -> None:
    """向南即 -90 度时方向也要正确，且同样遵守除法语义。"""
    base = FakeBase(yaw=0.0)
    controller = _make_controller(base, 0.5)

    assert controller.turn_to_heading('S') is True
    # 目标 -90 度，系数 0.5 → odom 应变化 -180 度
    assert base.yaw == pytest.approx(-math.pi, abs=0.05)


# ------------------------------------------------------------------ 边界

def test_turn_when_already_aligned_does_not_move() -> None:
    """已对准时不应再下发转动指令。"""
    base = FakeBase(yaw=math.pi / 2)  # 已朝北
    controller = _make_controller(base, 1.0)

    assert controller.turn_to_heading('N') is True
    assert abs(angle_diff(base.yaw, math.pi / 2)) < 0.05


def test_turn_without_odom_returns_false() -> None:
    """里程计失效时返回 False，由上层转入故障处理。"""
    base = FakeBase(yaw=0.0)
    base.drop_odom()
    controller = _make_controller(base, 1.0)

    assert controller.turn_to_heading('N') is False


@pytest.mark.parametrize('heading', ['N', 'E', 'S', 'W'])
def test_turn_to_every_heading_converges(heading: str) -> None:
    """四个方向都能收敛到对应 yaw。"""
    base = FakeBase(yaw=0.0)
    controller = _make_controller(base, 1.0)

    assert controller.turn_to_heading(heading) is True
    from maze_explorer.grid_mapper import GridMapper

    target = GridMapper.heading_yaw(heading)
    assert abs(angle_diff(base.yaw, target)) < 0.05
