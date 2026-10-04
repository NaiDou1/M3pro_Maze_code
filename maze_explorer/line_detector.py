"""黑线检测：提取通道中央引导线的横向偏差与航向偏差。

算法
----
1. 只取图像**下半部分 ROI**（近处地面），降低计算量并排除远处干扰；
2. HSV 二值化 + 形态学开运算去噪；
3. 按行采样，取每行前景的质心横坐标；
4. 对 ``(y, x)`` 做一次线性拟合 ``x = a·y + b``，抗单行噪声；
5. 由拟合线在图像**底边**的取值得到横向偏差，由斜率得到航向偏差；
6. 前景像素不足判为丢线，连续丢线超过阈值则 ``valid=False``。

.. note::
   HSV 阈值来自 ``config/hsv_params.yaml``，**必须现场用 calibration_tool 重标**。
   现有 ``LineFollowHSV.text`` 的 ``V_min=74`` 偏高，不符合黑色电工胶布的低明度
   特征，故未采用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


@dataclass
class LineObservation:
    """一帧黑线观测结果。"""

    valid: bool
    #: 底边处线中心相对图像中心线的像素偏差，**正=线在右侧**
    offset_px: float = 0.0
    #: 归一化横向偏差，``offset_px / (width/2)``，范围约 [-1, 1]
    offset_norm: float = 0.0
    #: 线相对竖直方向的倾角（弧度），正=线向右倾斜
    angle_rad: float = 0.0
    #: 底边处线中心的像素横坐标
    center_x: float = 0.0
    #: 连续丢线帧数
    lost_frames: int = 0


class LineDetector:
    """黑线检测算法（不依赖 ROS，可离线单测）。"""

    def __init__(
        self,
        hsv_range: Sequence[Sequence[int]],
        roi_top_ratio: float = 0.5,
        min_pixels: int = 200,
        morph_kernel: int = 5,
        lost_frames: int = 5,
        row_samples: int = 20,
    ) -> None:
        self._lower = np.array(hsv_range[0], dtype=np.uint8)
        self._upper = np.array(hsv_range[1], dtype=np.uint8)
        self._roi_top_ratio = float(roi_top_ratio)
        self._min_pixels = int(min_pixels)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(morph_kernel), int(morph_kernel))
        )
        self._lost_limit = int(lost_frames)
        self._row_samples = max(2, int(row_samples))
        self._lost = 0
        self._last = LineObservation(valid=False)
        #: 最近一次的 ROI 二值化掩膜，供调试节点可视化（调 HSV 阈值时最有用）
        self._last_mask = None

    def detect(self, bgr: np.ndarray) -> LineObservation:
        """对一帧 BGR 图像做检测。"""
        if bgr is None or bgr.size == 0:
            return self._mark_lost()

        h, w = bgr.shape[:2]
        top = int(h * self._roi_top_ratio)
        roi = bgr[top:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        self._last_mask = mask

        if cv2.countNonZero(mask) < self._min_pixels:
            return self._mark_lost()

        ys, xs = self._scan_rows(mask)
        if xs.size < 2:
            return self._mark_lost()

        slope, intercept = np.polyfit(ys, xs, 1)
        y_bottom = float(mask.shape[0] - 1)
        x_bottom = float(slope * y_bottom + intercept)
        offset_px = x_bottom - w / 2.0

        self._lost = 0
        self._last = LineObservation(
            valid=True,
            offset_px=offset_px,
            offset_norm=float(offset_px / (w / 2.0)),
            # slope = dx/dy：>0 表示线随 y 增大右移，即向右倾斜
            angle_rad=float(math.atan(slope)),
            center_x=x_bottom,
            lost_frames=0,
        )
        return self._last

    def _scan_rows(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mask.shape[0]
        step = max(1, h // self._row_samples)
        ys, xs = [], []
        for y in range(0, h, step):
            idx = np.nonzero(mask[y])[0]
            if idx.size:
                ys.append(float(y))
                xs.append(float(idx.mean()))
        return np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)

    def _mark_lost(self) -> LineObservation:
        self._lost += 1
        self._last = LineObservation(valid=False, lost_frames=self._lost)
        return self._last

    @property
    def consecutive_lost(self) -> int:
        """连续丢线帧数。"""
        return self._lost

    @property
    def is_lost(self) -> bool:
        """连续丢线是否超过容忍上限（应触发丢线保护）。"""
        return self._lost >= self._lost_limit

    def last(self) -> LineObservation:
        """返回最近一次检测结果（不重新计算）。"""
        return self._last

    def last_mask(self) -> Optional[np.ndarray]:
        """返回最近一次 ROI 的二值化掩膜；尚未检测过时为 ``None``。

        用途：调 HSV 阈值时直接看掩膜——黑线应该是一条连贯的白色带；
        若掩膜空白说明阈值没框住黑线，若满是噪点说明阈值太宽。
        """
        return self._last_mask


class LineDetectorNode(Node):
    """独立调试用节点：订阅相机画面并统计检测情况。"""

    def __init__(self) -> None:
        super().__init__('line_detector')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('line_hsv', [0, 0, 0, 180, 255, 80])
        self.declare_parameter('roi_top_ratio', 0.5)
        self.declare_parameter('min_line_pixels', 200)
        self.declare_parameter('line_lost_frames', 5)
        self.declare_parameter('show', False)

        hsv = [int(v) for v in self.get_parameter('line_hsv').value]
        self._det = LineDetector(
            hsv_range=(hsv[:3], hsv[3:]),
            roi_top_ratio=float(self.get_parameter('roi_top_ratio').value),
            min_pixels=int(self.get_parameter('min_line_pixels').value),
            lost_frames=int(self.get_parameter('line_lost_frames').value),
        )
        self._show = bool(self.get_parameter('show').value)
        self._bridge = CvBridge()
        self._frames = 0
        self._valid = 0
        self._warned = False

        self.create_subscription(
            Image, str(self.get_parameter('color_topic').value), self._on_image, 1
        )
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f'LineDetector 就绪 | hsv={hsv} | show={self._show}（阈值需现场重标）'
        )

    def _on_image(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'图像转换失败：{exc}', throttle_duration_sec=5.0)
            return

        obs = self._det.detect(img)
        self._frames += 1
        if obs.valid:
            self._valid += 1

        if not obs.valid and self._det.is_lost and not self._warned:
            self._warned = True
            self.get_logger().warn(
                f'连续丢线 {obs.lost_frames} 帧，请检查 line_hsv 阈值与地面光照'
            )
        elif obs.valid:
            self._warned = False

        if self._show:
            self._visualize(img, obs)

    def _visualize(self, img: np.ndarray, obs: LineObservation) -> None:
        canvas = img.copy()
        h, w = canvas.shape[:2]
        self._draw_mask(canvas, w)
        cv2.line(canvas, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        if obs.valid:
            cv2.circle(canvas, (int(obs.center_x), int(h * 0.95)), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas, f'offset={obs.offset_px:+.0f}px ang={math.degrees(obs.angle_rad):+.1f}deg',
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
            )
        else:
            cv2.putText(
                canvas, 'LINE LOST', (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
        cv2.imshow('line_detector', canvas)
        cv2.waitKey(1)

    def _draw_mask(self, canvas: np.ndarray, width: int) -> None:
        """把二值化掩膜缩略图贴到右上角，便于边看边调 HSV。"""
        mask = self._det.last_mask()
        if mask is None:
            return
        inset_w = max(80, width // 4)
        scale = inset_w / float(mask.shape[1])
        inset_h = max(1, int(mask.shape[0] * scale))
        small = cv2.resize(mask, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
        canvas[0:inset_h, width - inset_w:width] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    def _report(self) -> None:
        rate = (self._valid / self._frames * 100.0) if self._frames else 0.0
        self.get_logger().info(
            f'检测 {self._frames} 帧，有效 {self._valid} 帧（{rate:.0f}%）'
        )
        self._frames = 0
        self._valid = 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
