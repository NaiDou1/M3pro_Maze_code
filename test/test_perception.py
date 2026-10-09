"""感知层算法单测：用合成图像验证，不依赖相机与硬件。

运行::

    source /opt/ros/humble/setup.bash
    cd ~/yahboomcar_ws/src/maze_explorer
    python3 -m pytest test/test_perception.py -v
"""

import math

import cv2
import numpy as np
import pytest

from maze_explorer.block_detector import BlockDetector
from maze_explorer.line_detector import LineDetector

#: 黑线阈值，取低明度
LINE_HSV = ((0, 0, 0), (180, 255, 80))
#: 红色阈值，沿用现有 demo 标定值
RED_HSV = {'red': [0, 154, 107, 184, 253, 255]}
#: 高饱和度方块的代表色 BGR 分量，饱和度约 233，落在阈值内
RED_BGR = (20, 20, 230)


def _blank(h=480, w=640):
    """生成灰度 200 的空白底图，供画线与画块。"""
    return np.full((h, w, 3), 200, np.uint8)


# ------------------------------------------------------------------ 黑线检测

def test_line_offset_reports_right_deviation():
    """线在中心右侧 80px 时偏差为正且归一化约 0.25。"""
    img = _blank()
    img[:, 395:405] = 0  # 竖线位于 x=400，图像中心 320
    obs = LineDetector(hsv_range=LINE_HSV).detect(img)

    assert obs.valid
    assert obs.offset_px == pytest.approx(80.0, abs=2.0)
    assert obs.offset_norm == pytest.approx(0.25, abs=0.01)
    assert obs.angle_rad == pytest.approx(0.0, abs=1e-3)


def test_line_offset_reports_left_deviation():
    """线在中心左侧时偏差为负，符号与右侧相反。"""
    img = _blank()
    img[:, 235:245] = 0  # x=240，中心左侧 80px
    obs = LineDetector(hsv_range=LINE_HSV).detect(img)

    assert obs.valid
    assert obs.offset_px == pytest.approx(-80.0, abs=2.0)


def test_line_lost_protection_triggers_after_threshold():
    """连续丢帧达到阈值即进入丢线保护状态。"""
    det = LineDetector(hsv_range=LINE_HSV, lost_frames=3)
    obs = None
    for _ in range(3):
        obs = det.detect(_blank())

    assert obs is not None and not obs.valid
    assert obs.lost_frames == 3
    assert det.is_lost


def test_line_recovers_after_being_found():
    """重新检出线后丢帧计数清零并退出保护。"""
    det = LineDetector(hsv_range=LINE_HSV, lost_frames=3)
    det.detect(_blank())
    img = _blank()
    img[:, 315:325] = 0
    obs = det.detect(img)

    assert obs.valid
    assert obs.lost_frames == 0
    assert not det.is_lost


# ---------------------------------------------------------------- 方块检测

def _detector(**kwargs):
    """构造只认红色的方块检测器，关键字透传其余参数。"""
    return BlockDetector(hsv_map=RED_HSV, **kwargs)


def test_block_distance_and_lateral_from_depth():
    """由深度与像素差解算距离与横向，数值须符合理论投影。"""
    img = np.zeros((480, 640, 3), np.uint8)
    img[200:280, 300:380] = RED_BGR          # 中心像素横 340 纵 240
    depth = np.full((400, 640), 500.0, np.float32)  # 500 mm

    res = _detector().detect(img, depth)

    assert len(res) == 1
    block = res[0]
    assert block.color == 'red'
    assert block.distance_m == pytest.approx(0.5, abs=1e-6)
    # 理论横向为像素差 340 减 319.38 乘深度 0.5 m 再除焦距 477.57，约 0.0216 m
    assert block.lateral_m == pytest.approx(0.0216, abs=2e-3)
    assert block.horizontal_distance() == pytest.approx(
        math.hypot(block.distance_m, block.lateral_m), abs=1e-9
    )


def test_block_skips_invalid_zero_depth():
    """深度全 0 视为无效，不产生任何目标。"""
    img = np.zeros((480, 640, 3), np.uint8)
    img[200:280, 300:380] = RED_BGR

    assert _detector().detect(img, np.zeros((400, 640), np.float32)) == []


def test_block_ignores_tiny_contour():
    """小于面积阈值的轮廓须丢弃，抗噪点误检。"""
    img = np.zeros((480, 640, 3), np.uint8)
    img[240:245, 320:325] = RED_BGR  # 25 像素，远小于 min_area

    depth = np.full((400, 640), 500.0, np.float32)
    assert _detector(min_area=300).detect(img, depth) == []


def test_block_multiple_targets_sorted_by_distance():
    """多目标按距离升序返回，深度分界须落在正确像素列。"""
    img = np.zeros((480, 640, 3), np.uint8)
    img[100:160, 100:160] = RED_BGR
    img[300:380, 400:480] = RED_BGR
    depth = np.full((400, 640), 500.0, np.float32)
    depth[:, 400:] = 800.0  # 右侧更远，分界点须落在右侧方块中心左侧

    res = _detector().detect(img, depth)

    assert len(res) == 2
    assert [round(d.distance_m, 1) for d in res] == [0.5, 0.8]


def test_grasp_envelope_check():
    """抓取包络判定须与配置的距离区间端点关系一致。"""
    img = np.zeros((480, 640, 3), np.uint8)
    img[200:280, 300:380] = RED_BGR
    depth = np.full((400, 640), 500.0, np.float32)
    block = _detector().detect(img, depth)[0]

    assert BlockDetector.in_grasp_envelope(block, 0.13, 0.25) is False  # 0.5m 太远
    assert BlockDetector.in_grasp_envelope(block, 0.40, 0.60) is True


def test_position_base_none_when_mount_uncalibrated():
    """未标定外参时基座坐标为空，标定后按外参平移可算。"""
    img = np.zeros((480, 640, 3), np.uint8)
    img[200:280, 300:380] = RED_BGR
    depth = np.full((400, 640), 500.0, np.float32)

    block = _detector(mount_calibrated=False).detect(img, depth)[0]
    assert block.position_base is None

    block = _detector(mount_calibrated=True).detect(img, depth)[0]
    assert block.position_base is not None
    x, y, z = block.position_base
    # 外参为单位旋转加平移 0.10、0、0.35，故基座坐标为相机系坐标逐项加平移
    assert x == pytest.approx(0.10 + block.lateral_m, abs=1e-6)
    assert y == pytest.approx(block.vertical_m, abs=1e-6)
    assert z == pytest.approx(0.35 + block.distance_m, abs=1e-6)
