"""黑线检测：提取通道中央引导线的横向偏差与航向偏差。

算法
----
1. 只取图像下半部分作 ROI，即近处地面，降低计算量并排除远处干扰；
2. HSV 二值化加形态学开运算去噪；
3. 按行采样，取每行前景的质心横坐标；
4. 对采样点做一次线性拟合 ``x = a*y + b``，抗单行噪声；
5. 由拟合线在图像底边的取值得到横向偏差，由斜率得到航向偏差；
6. 前景像素不足判为丢线，连续丢线超过上限则 ``valid`` 为假。

.. note::
   HSV 阈值来自 ``config/hsv_params.yaml``，必须现场用 calibration_tool 重标。
   现有 ``LineFollowHSV.text`` 的 ``V_min`` 为 74 偏高，不符合黑色电工胶布的低
   明度特征，故未采用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

#: 调试节点输出统计的周期，单位 s
REPORT_PERIOD_SEC = 5.0
#: 调试画面中掩膜缩略图的最小宽度，单位 px
THUMBNAIL_MIN_WIDTH_PX = 80
#: 调试画面中掩膜缩略图占画幅宽度的比例分母
THUMBNAIL_WIDTH_DIVISOR = 4
#: 调试画面中线中心标记点的纵坐标比例，取值 0 到 1
CENTER_MARKER_Y_RATIO = 0.95
#: 调试画面中线中心标记点的半径，单位 px
CENTER_MARKER_RADIUS_PX = 6
#: 调试画面叠加文字的左上角坐标，单位 px
OVERLAY_TEXT_POS_PX = (8, 26)


@dataclass
class LineObservation:
    """一帧黑线观测结果。

    :ivar valid: 本帧是否检出有效黑线。
    :ivar offset_px: 底边处线中心相对图像中心线的像素偏差，单位 px，
        正值表示线在图像右侧，取值区间负半幅宽到正半幅宽。
    :ivar offset_norm: 归一化横向偏差，等于 offset_px 除以半幅宽，无量纲，
        取值区间 -1 到 1，1 表示线在图像边缘，巡线 PID 用它作误差。
    :ivar angle_rad: 线相对竖直方向的倾角，单位 rad，正表示线向右倾斜，
        取值区间负 90 度到正 90 度。
    :ivar center_x: 底边处线中心的像素横坐标，单位 px。
    :ivar lost_frames: 连续丢线帧数，单位帧，不小于 0。
    """

    valid: bool
    #: 像素偏差，单位 px，正为线在图像右侧
    offset_px: float = 0.0
    #: 归一化偏差，无量纲，PID 误差用它而非像素值
    offset_norm: float = 0.0
    #: 线倾角，单位 rad，正为向右倾斜
    angle_rad: float = 0.0
    #: 底边处线中心的像素横坐标，单位 px
    center_x: float = 0.0
    #: 连续丢线帧数，单位帧
    lost_frames: int = 0


class LineDetector:
    """黑线检测算法，不依赖 ROS，可离线单测。"""

    def __init__(
        self,
        hsv_range: Sequence[Sequence[int]],
        roi_top_ratio: float = 0.5,
        min_pixels: int = 200,
        morph_kernel: int = 5,
        lost_frames: int = 5,
        row_samples: int = 20,
    ) -> None:
        """配置阈值与采样参数，不持有图像状态。

        :param hsv_range: 黑线 HSV 双阈值，上下界各三个分量。H 取值 0 到 180，
            S 与 V 取值 0 到 255，来源为 ``config/hsv_params.yaml``。
        :param roi_top_ratio: ROI 顶边相对图像高度的比例，无量纲，取值 0 到 1，
            0.5 表示只分析下半幅。
        :param min_pixels: 判定有效所需的前景像素数下限，单位像素。低于该值判
            为丢线，现场标定 HSV 后按掩膜实际像素量调整。
        :param morph_kernel: 形态学开运算核的边长，单位像素，不小于 1。
        :param lost_frames: 触发丢线保护的连续空帧上限，单位帧，不小于 1。
        :param row_samples: 逐行采样的目标行数，单位行，不小于 2，实际步长由
            ROI 高度除以它再取整。
        """
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
        #: 最近一次的 ROI 二值化掩膜，供调试节点可视化，调阈值时直接看它
        self._last_mask = None

    def detect(self, bgr: np.ndarray) -> LineObservation:
        """对一帧 BGR 图像做检测，内部更新连续丢线计数。

        :param bgr: BGR 三通道图像，宽高不小于 1。
        :returns: 本帧观测；空图、前景不足或采样不足时为无效观测。
        """
        if bgr is None or bgr.size == 0:
            return self._mark_lost()

        height, width = bgr.shape[:2]
        roi_top = int(height * self._roi_top_ratio)
        roi = bgr[roi_top:, :]

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
        offset_px = x_bottom - width / 2.0

        self._lost = 0
        self._last = LineObservation(
            valid=True,
            offset_px=offset_px,
            offset_norm=float(offset_px / (width / 2.0)),
            # slope 为 dx/dy，大于 0 表示线随 y 增大右移即向右倾斜
            angle_rad=float(math.atan(slope)),
            center_x=x_bottom,
            lost_frames=0,
        )
        return self._last

    def _scan_rows(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """按固定步长逐行取前景质心横坐标。

        :param mask: ROI 二值掩膜，取值 0 或 255。
        :returns: 行坐标数组与对应前景质心横坐标数组，单位 px，长度相等。
        """
        height = mask.shape[0]
        step = max(1, height // self._row_samples)
        ys, xs = [], []
        for y in range(0, height, step):
            idx = np.nonzero(mask[y])[0]
            if idx.size:
                ys.append(float(y))
                xs.append(float(idx.mean()))
        return np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)

    def _mark_lost(self) -> LineObservation:
        """累加连续丢线计数并返回无效观测。"""
        self._lost += 1
        self._last = LineObservation(valid=False, lost_frames=self._lost)
        return self._last

    @property
    def consecutive_lost(self) -> int:
        """连续丢线帧数，单位帧，检出有效线后归零。"""
        return self._lost

    @property
    def is_lost(self) -> bool:
        """连续丢线是否达到容忍上限，为真时上层应触发丢线保护。"""
        return self._lost >= self._lost_limit

    def last(self) -> LineObservation:
        """返回最近一次检测结果，不重新计算。"""
        return self._last

    def last_mask(self) -> Optional[np.ndarray]:
        """返回最近一次 ROI 的二值化掩膜，尚未检测过时为 ``None``。

        调 HSV 阈值时直接看掩膜：黑线应为一条连贯白色带；掩膜空白说明阈值没框
        住黑线，满是噪点说明阈值太宽。
        """
        return self._last_mask


class LineDetectorNode(Node):
    """独立调试节点，订阅相机画面并按周期统计检测情况。"""

    def __init__(self) -> None:
        """声明循线参数并装配检测器，订阅画面后按周期打印统计报告。"""
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
        self.create_timer(REPORT_PERIOD_SEC, self._report)
        self.get_logger().info(
            f'LineDetector 就绪 | hsv={hsv} | show={self._show}（阈值需现场重标）'
        )

    def _on_image(self, msg: Image) -> None:
        """转换单帧图像、执行检测并按需可视化与告警。

        :param msg: 相机彩色图消息。
        """
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'图像转换失败：{exc}', throttle_duration_sec=5.0)
            return

        observation = self._det.detect(img)
        self._frames += 1
        if observation.valid:
            self._valid += 1

        if not observation.valid and self._det.is_lost and not self._warned:
            self._warned = True
            self.get_logger().warn(
                f'连续丢线 {observation.lost_frames} 帧，请检查 line_hsv 阈值与地面光照'
            )
        elif observation.valid:
            self._warned = False

        if self._show:
            self._visualize(img, observation)

    def _visualize(self, img: np.ndarray, observation: LineObservation) -> None:
        """在调试窗口绘制中心线、线中心标记与偏差读数。

        :param img: 原始 BGR 图像。
        :param observation: 本帧观测结果。
        """
        canvas = img.copy()
        height, width = canvas.shape[:2]
        self._draw_mask(canvas, width)
        cv2.line(canvas, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)
        if observation.valid:
            cv2.circle(
                canvas,
                (int(observation.center_x), int(height * CENTER_MARKER_Y_RATIO)),
                CENTER_MARKER_RADIUS_PX,
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                canvas,
                f'offset={observation.offset_px:+.0f}px '
                f'ang={math.degrees(observation.angle_rad):+.1f}deg',
                OVERLAY_TEXT_POS_PX,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )
        else:
            cv2.putText(
                canvas,
                'LINE LOST',
                OVERLAY_TEXT_POS_PX,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        cv2.imshow('line_detector', canvas)
        cv2.waitKey(1)

    def _draw_mask(self, canvas: np.ndarray, width: int) -> None:
        """把二值化掩膜缩略图贴到画布右上角，便于边看边调 HSV。

        :param canvas: 待叠加的 BGR 画布，会被就地修改。
        :param width: 画布宽度，单位 px。
        """
        mask = self._det.last_mask()
        if mask is None:
            return
        inset_width = max(THUMBNAIL_MIN_WIDTH_PX, width // THUMBNAIL_WIDTH_DIVISOR)
        scale = inset_width / float(mask.shape[1])
        inset_height = max(1, int(mask.shape[0] * scale))
        small = cv2.resize(
            mask, (inset_width, inset_height), interpolation=cv2.INTER_NEAREST
        )
        canvas[0:inset_height, width - inset_width:width] = cv2.cvtColor(
            small, cv2.COLOR_GRAY2BGR
        )

    def _report(self) -> None:
        """按周期打印有效帧比例并清零计数。"""
        rate = (self._valid / self._frames * 100.0) if self._frames else 0.0
        self.get_logger().info(
            f'检测 {self._frames} 帧，有效 {self._valid} 帧（{rate:.0f}%）'
        )
        self._frames = 0
        self._valid = 0


def main(args: Optional[List[str]] = None) -> None:
    """调试节点入口：初始化、自旋、退出时按序关闭。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
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
