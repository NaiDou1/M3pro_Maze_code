"""对位触发与步长的回归测试（锁定 B3 修复）。

B3 原实现只要视野内出现方块就转入对位，而对位最多 4 步 × 0.15m，而相机巡线
姿态可看到 1~2m 外的方块，导致：
  * 从格心直接前冲，脱离格心、打乱 DFS 拓扑；
  * 距离 >0.8m 的方块必然「未能进入包络」而被误判失败。

修复后：仅在进入触发距离内才转入对位；对位步长按剩余距离自适应并限幅。
"""

import pytest

from maze_explorer.block_detector import BlockDetection, BlockDetector
from maze_explorer.mission_manager import approach_step, should_approach


# -------------------------------------------------------------- 触发判定

def test_should_not_approach_far_blocks() -> None:
    """B3 的根因：远处方块绝不能触发对位。"""
    assert should_approach(2.00, 0.60) is False
    assert should_approach(1.00, 0.60) is False
    assert should_approach(0.61, 0.60) is False


def test_should_approach_within_trigger() -> None:
    assert should_approach(0.60, 0.60) is True   # 边界含等号
    assert should_approach(0.40, 0.60) is True
    assert should_approach(0.05, 0.60) is True   # 过近也应转入（由对位后退处理）


# -------------------------------------------------------------- 步长计算

@pytest.mark.parametrize(
    'distance,target,expected',
    [
        (0.50, 0.20, 0.20),    # 远：被限幅到 0.20
        (0.30, 0.20, 0.10),    # 中：按差值
        (0.21, 0.20, 0.01),    # 近：微调
        (0.20, 0.20, 0.00),    # 已在目标
        (0.15, 0.20, -0.05),   # 过近：后退
        (0.05, 0.20, -0.15),   # 更近
        (0.00, 0.20, -0.20),   # 负方向同样限幅
    ],
)
def test_approach_step_adaptive_and_limited(
    distance: float, target: float, expected: float
) -> None:
    assert approach_step(distance, target) == pytest.approx(expected)


def test_approach_step_custom_limit() -> None:
    assert approach_step(2.0, 0.2, limit=0.05) == pytest.approx(0.05)
    assert approach_step(0.0, 0.2, limit=0.05) == pytest.approx(-0.05)


# -------------------------------------------------------------- 收敛性

@pytest.mark.parametrize('start', [2.00, 1.50, 0.80, 0.50])
def test_approach_converges_within_max_steps(start: float) -> None:
    """从任意触发范围内的距离出发，都应在 20 步内进入抓取包络。"""
    distance = start
    target = 0.20
    envelope = (0.13, 0.25)

    for _ in range(20):
        if envelope[0] <= distance <= envelope[1]:
            break
        distance -= approach_step(distance, target)

    assert envelope[0] <= distance <= envelope[1], f'未收敛，停在 {distance:.3f}m'


def test_approach_does_not_overshoot_envelope() -> None:
    """单步限幅不应让距离一次跨过包络。"""
    distance = 0.30
    step = approach_step(distance, 0.20)
    assert distance - step >= 0.13


def test_in_grasp_envelope_boundaries() -> None:
    """包络判定与 ArmController/抓取状态机使用同一区间语义。"""
    block = BlockDetection(
        color='red', u=320.0, v=240.0, area=600.0,
        distance_m=0.20, lateral_m=0.0, vertical_m=0.0,
    )
    assert BlockDetector.in_grasp_envelope(block, 0.13, 0.25) is True
    assert BlockDetector.in_grasp_envelope(block, 0.25, 0.40) is False
    assert BlockDetector.in_grasp_envelope(block, 0.05, 0.13) is False
