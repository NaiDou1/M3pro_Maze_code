"""传感器硬件抽象层：RGB-D 相机与激光雷达。

相机（Orbbec DaBai DCW2）
------------------------
======================  ==========================  ==============================
话题                    类型                        说明
======================  ==========================  ==============================
``/camera/color/image_raw``  ``sensor_msgs/Image`` 640x480 @30fps，``bgr8``
``/camera/depth/image_raw``  ``sensor_msgs/Image`` 640x400 @10fps，``32FC1``，**单位 mm**
======================  ==========================  ==============================

.. warning::
   相机 ``depth_registration`` 默认为 ``false``，彩色图与深度图**未配准**，
   且二者帧率不同（30 vs 10）。因此必须用 ``ApproximateTimeSynchronizer`` 对齐，
   并按内参做彩色像素 → 深度像素的映射，不可假定同索引像素对应同一点。

激光雷达
--------
数据链为 ``/scan0`` + ``/scan1`` → merger → ``/scan_multi`` → filter → ``/scan``。
**上层统一订阅 ``/scan``**：``frame_id=base_link``、360°、角分辨率 1°、量程 0.05~4.0m。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

#: 同步后的一帧 RGB-D，``(rgb_bgr, depth_mm)``，均为 ndarray
RgbdFrame = Tuple[np.ndarray, np.ndarray]


class SensorHub(Node):
    """汇聚相机与激光数据，供感知/建图模块按需查询。"""

    def __init__(self) -> None:
        super().__init__('sensors')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        #: 彩色/深度时间戳允许的最大偏差（秒）
        self.declare_parameter('sync_slop_sec', 0.08)
        #: 激光有效量程下限（米），低于此值视为车体自遮挡
        self.declare_parameter('scan_range_min', 0.05)
        self.declare_parameter('scan_range_max', 4.0)

        color_topic = str(self.get_parameter('color_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        scan_topic = str(self.get_parameter('scan_topic').value)
        slop = float(self.get_parameter('sync_slop_sec').value)
        self._range_min = float(self.get_parameter('scan_range_min').value)
        self._range_max = float(self.get_parameter('scan_range_max').value)

        self._bridge = None  # 延迟导入 cv_bridge，避免无相机环境下 import 失败
        try:
            from cv_bridge import CvBridge

            self._bridge = CvBridge()
        except ImportError:  # pragma: no cover - 环境缺依赖时给出明确提示
            self.get_logger().error('未找到 cv_bridge，RGB-D 功能不可用')

        self._lock = threading.Lock()
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._rgbd_ts = 0.0
        self._scan: Optional[LaserScan] = None
        self._scan_ts = 0.0

        # 帧率统计（只用于低频 INFO 日志，不逐帧打印）
        self._frame_count = 0
        self._last_fps_log = time.monotonic()

        # RGB-D 同步：队列 1 + ApproximateTime，避免积压旧帧
        if self._bridge is not None:
            rgb_sub = message_filters.Subscriber(self, Image, color_topic)
            depth_sub = message_filters.Subscriber(self, Image, depth_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=1, slop=slop
            )
            self._sync.registerCallback(self._on_rgbd)

        self._scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._on_scan, 10
        )
        self.create_timer(20.0, self._log_rate)

        self.get_logger().info(
            f'SensorHub 就绪 | color={color_topic} | depth={depth_topic} | scan={scan_topic}'
        )

    # -------------------------------------------------------------- RGB-D 查询

    def get_rgbd(self) -> Optional[RgbdFrame]:
        """返回最近一帧同步的 ``(rgb_bgr, depth_mm)``；无数据返回 ``None``。"""
        with self._lock:
            if self._rgb is None or self._depth is None:
                return None
            return self._rgb, self._depth

    def rgbd_age(self) -> float:
        """距最近一帧同步 RGB-D 的时长（秒）；从未收到返回 ``inf``。"""
        with self._lock:
            if self._rgbd_ts <= 0.0:
                return math.inf
            return time.monotonic() - self._rgbd_ts

    # ---------------------------------------------------------------- 激光查询

    def get_scan(self) -> Optional[LaserScan]:
        """返回最近一帧激光数据；无数据返回 ``None``。"""
        with self._lock:
            return self._scan

    def scan_age(self) -> float:
        """距最近一帧激光的时长（秒）；从未收到返回 ``inf``。

        用途：激光失效时 ``sector_min_range`` 会返回 ``None``，而
        ``is_path_clear`` / 碰撞保护都把 ``None`` 当作"通畅"。因此**必须先查
        保鲜度**再相信测距结果，否则雷达掉线会被误判成"前方无阻挡"。
        """
        with self._lock:
            if self._scan_ts <= 0.0:
                return math.inf
            return time.monotonic() - self._scan_ts

    def sector_min_range(self, angle_deg: float, half_width_deg: float) -> Optional[float]:
        """取指定角度扇区内的**最近**有效距离（米）。

        :param angle_deg: 扇区中心角，车体坐标系，0=正前方，逆时针为正。
        :param half_width_deg: 扇区半宽（度）。
        :return: 最近距离；扇区内无有效点时返回 ``None``。
        """
        scan = self.get_scan()
        if scan is None:
            return None

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return None

        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
        center = math.radians(angle_deg)
        half = math.radians(half_width_deg)
        # 归一化角差到 [-pi, pi]，正确处理 ±180° 跨界
        diff = np.arctan2(np.sin(angles - center), np.cos(angles - center))
        sector = ranges[np.abs(diff) <= half]

        valid = sector[np.isfinite(sector)]
        valid = valid[(valid >= self._range_min) & (valid <= self._range_max)]
        if valid.size == 0:
            return None
        return float(valid.min())

    def is_path_clear(self, angle_deg: float, half_width_deg: float, threshold_m: float) -> bool:
        """判断题定扇区是否通畅（最近距离大于阈值）。

        扇区内无有效点（例如全是 inf）时视为通畅。
        """
        d = self.sector_min_range(angle_deg, half_width_deg)
        return d is None or d > threshold_m

    # ------------------------------------------------------------------ 回调

    def _on_rgbd(self, color_msg: Image, depth_msg: Image) -> None:
        try:
            rgb = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # 深度为 32FC1、单位 mm；统一转 float32 便于后续测距
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception as exc:  # noqa: BLE001 - 转换失败不应中断节点
            self.get_logger().warn(f'RGB-D 转换失败：{exc}', throttle_duration_sec=5.0)
            return

        with self._lock:
            self._rgb = rgb
            self._depth = np.asarray(depth, dtype=np.float32)
            self._rgbd_ts = time.monotonic()
            self._frame_count += 1

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self._scan = msg
            self._scan_ts = time.monotonic()

    def _log_rate(self) -> None:
        with self._lock:
            count = self._frame_count
            self._frame_count = 0
        now = time.monotonic()
        elapsed = max(now - self._last_fps_log, 1e-6)
        self._last_fps_log = now
        self.get_logger().info(f'RGB-D 同步帧率 ≈ {count / elapsed:.1f} Hz')


def main(args=None) -> None:
    """独立运行：用于确认相机与激光话题正常、查看同步帧率。"""
    rclpy.init(args=args)
    node = SensorHub()
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
