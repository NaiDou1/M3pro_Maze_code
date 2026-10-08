"""传感器硬件抽象层：RGB-D 相机与激光雷达。

相机，型号 Orbbec DaBai DCW2
----------------------------
=========================  ==========================  ==============================
话题                        类型                        说明
=========================  ==========================  ==============================
``/camera/color/image_raw``  ``sensor_msgs/Image`` 640x480@30fps，``bgr8``
``/camera/depth/image_raw``  ``sensor_msgs/Image`` 640x400@10fps，``32FC1``，单位 mm
=========================  ==========================  ==============================

.. warning::
   相机 ``depth_registration`` 默认为假，彩色图与深度图未配准，且帧率不同，
   30fps 对 10fps。因此必须用 ``ApproximateTimeSynchronizer`` 对齐，并按内参做
   彩色像素到深度像素的映射，不可假定同索引像素对应同一点。

激光雷达
--------
数据链为 ``/scan0`` 与 ``/scan1`` 经 merger 得 ``/scan_multi``，再经 filter
得 ``/scan``。上层统一订阅 ``/scan``：``frame_id`` 为 ``base_link``，360 度，
角分辨率 1 度，量程 0.05 到 4.0 m。
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

from maze_explorer._compat import StrEnum

#: 同步后的一帧 RGB-D，元素为彩色 BGR 图与深度图，单位 mm
RgbdFrame = Tuple[np.ndarray, np.ndarray]

#: 彩色与深度时间戳允许的最大偏差，单位 s
DEFAULT_SYNC_SLOP_SEC = 0.08
#: 激光有效量程下限，单位 m，低于此值视为车体自遮挡
DEFAULT_SCAN_RANGE_MIN_M = 0.05
#: 激光有效量程上限，单位 m，由雷达规格给出
DEFAULT_SCAN_RANGE_MAX_M = 4.0
#: 帧率统计的打印周期，单位 s
RATE_LOG_PERIOD_SEC = 20.0
#: 帧率计算中时间间隔的下限，单位 s，防止同刻两次采样导致除零
RATE_LOG_MIN_ELAPSED_SEC = 1e-6
#: RGB-D 转换失败告警的节流周期，单位 s
WARN_THROTTLE_SEC = 5.0
#: 激光订阅的队列深度，单位帧
SCAN_QUEUE_DEPTH = 10


class SensorKind(StrEnum):
    """保鲜度查询项，取值 scan_age 或 rgbd_age。

    取值即 ``SensorHub`` 上的方法名，调用方按它做属性分发，属性缺失时可安全
    退化，便于单测注入桩对象。
    """

    SCAN = 'scan_age'
    RGBD = 'rgbd_age'


class SensorHub(Node):
    """汇聚相机与激光数据，供感知与建图模块按需查询。

    所有查询接口都加锁返回快照，不暴露内部缓冲引用的并发修改风险。
    """

    def __init__(self) -> None:
        """构造期完成参数声明、RGB-D 同步订阅与带锁缓冲装配。"""
        super().__init__('sensors')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('sync_slop_sec', DEFAULT_SYNC_SLOP_SEC)
        self.declare_parameter('scan_range_min', DEFAULT_SCAN_RANGE_MIN_M)
        self.declare_parameter('scan_range_max', DEFAULT_SCAN_RANGE_MAX_M)

        color_topic = str(self.get_parameter('color_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        scan_topic = str(self.get_parameter('scan_topic').value)
        slop = float(self.get_parameter('sync_slop_sec').value)
        self._range_min = float(self.get_parameter('scan_range_min').value)
        self._range_max = float(self.get_parameter('scan_range_max').value)

        # cv_bridge 延迟导入，缺依赖时只降级 RGB-D 功能而不让节点启动失败
        self._bridge = None
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

        # 帧率统计只用于低频 INFO 日志，不逐帧打印
        self._frame_count = 0
        self._last_fps_log = time.monotonic()

        # 队列 1 加 ApproximateTime，避免积压旧帧
        if self._bridge is not None:
            rgb_sub = message_filters.Subscriber(self, Image, color_topic)
            depth_sub = message_filters.Subscriber(self, Image, depth_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=1, slop=slop
            )
            self._sync.registerCallback(self._on_rgbd)

        self._scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._on_scan, SCAN_QUEUE_DEPTH
        )
        self.create_timer(RATE_LOG_PERIOD_SEC, self._log_rate)

        self.get_logger().info(
            f'SensorHub 就绪 | color={color_topic} | depth={depth_topic} | scan={scan_topic}'
        )

    # -------------------------------------------------------------- RGB-D 查询

    def get_rgbd(self) -> Optional[RgbdFrame]:
        """返回最近一帧同步的 RGB-D，尚未同步到数据时返回 ``None``。

        :returns: 二元组，首项为彩色 BGR 图，次项为 float32 深度图单位 mm。
        """
        with self._lock:
            if self._rgb is None or self._depth is None:
                return None
            return self._rgb, self._depth

    def rgbd_age(self) -> float:
        """返回距最近一帧同步 RGB-D 的时长，单位 s，从未收到时为 ``inf``。

        :returns: 非负秒数或正无穷。
        """
        with self._lock:
            if self._rgbd_ts <= 0.0:
                return math.inf
            return time.monotonic() - self._rgbd_ts

    # ---------------------------------------------------------------- 激光查询

    def get_scan(self) -> Optional[LaserScan]:
        """返回最近一帧激光数据，尚未收到时返回 ``None``。

        :returns: 激光消息对象，只读不复制。
        """
        with self._lock:
            return self._scan

    def scan_age(self) -> float:
        """返回距最近一帧激光的时长，单位 s，从未收到时为 ``inf``。

        激光失效时 ``sector_min_range`` 返回 ``None``，而 ``is_path_clear`` 与
        碰撞保护都把 ``None`` 当作通畅。因此必须先查保鲜度再相信测距结果，否则
        雷达掉线会被误判成前方无阻挡。

        :returns: 非负秒数或正无穷。
        """
        with self._lock:
            if self._scan_ts <= 0.0:
                return math.inf
            return time.monotonic() - self._scan_ts

    def sector_min_range(
        self, angle_deg: float, half_width_deg: float
    ) -> Optional[float]:
        """取指定角度扇区内的最近有效距离，单位 m。

        :param angle_deg: 扇区中心角，单位 deg，车体系，0 为正前方，逆时针为正。
        :param half_width_deg: 扇区半宽，单位 deg，不小于 0。
        :returns: 最近距离；扇区内无有效点时为 ``None``。
        """
        scan = self.get_scan()
        if scan is None:
            return None

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return None

        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * (
            scan.angle_increment
        )
        center = math.radians(angle_deg)
        half = math.radians(half_width_deg)
        # 角差归一化到负 π 到正 π，才能正确处理 ±180 度跨界
        diff = np.arctan2(np.sin(angles - center), np.cos(angles - center))
        sector = ranges[np.abs(diff) <= half]

        valid = sector[np.isfinite(sector)]
        valid = valid[(valid >= self._range_min) & (valid <= self._range_max)]
        if valid.size == 0:
            return None
        return float(valid.min())

    def is_path_clear(
        self, angle_deg: float, half_width_deg: float, threshold_m: float
    ) -> bool:
        """判断给定扇区是否通畅，即最近距离大于阈值。

        扇区内无有效点，例如回波全为 inf 时，视为通畅。调用方必须先确认
        ``scan_age`` 在保鲜阈值内，否则失效会被误读为通畅。

        :param angle_deg: 扇区中心角，单位 deg，车体系。
        :param half_width_deg: 扇区半宽，单位 deg，不小于 0。
        :param threshold_m: 通畅阈值，单位 m，大于该距离才算通畅。
        :returns: 通畅为真。
        """
        min_range = self.sector_min_range(angle_deg, half_width_deg)
        return min_range is None or min_range > threshold_m

    # ------------------------------------------------------------------ 回调

    def _on_rgbd(self, color_msg: Image, depth_msg: Image) -> None:
        """转换同步帧并刷新缓冲与时间戳。

        :param color_msg: 彩色图消息，编码 bgr8。
        :param depth_msg: 深度图消息，编码 32FC1，单位 mm。
        """
        try:
            rgb = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # 深度为 32FC1、单位 mm；统一转 float32 便于后续测距
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception as exc:  # noqa: BLE001 - 转换失败不应中断节点
            self.get_logger().warn(
                f'RGB-D 转换失败：{exc}', throttle_duration_sec=WARN_THROTTLE_SEC
            )
            return

        with self._lock:
            self._rgb = rgb
            self._depth = np.asarray(depth, dtype=np.float32)
            self._rgbd_ts = time.monotonic()
            self._frame_count += 1

    def _on_scan(self, msg: LaserScan) -> None:
        """刷新激光缓冲与时间戳。

        :param msg: 激光扫描消息。
        """
        with self._lock:
            self._scan = msg
            self._scan_ts = time.monotonic()

    def _log_rate(self) -> None:
        """按周期打印同步帧率并清零计数。"""
        with self._lock:
            count = self._frame_count
            self._frame_count = 0
        now = time.monotonic()
        elapsed = max(now - self._last_fps_log, RATE_LOG_MIN_ELAPSED_SEC)
        self._last_fps_log = now
        self.get_logger().info(f'RGB-D 同步帧率 ≈ {count / elapsed:.1f} Hz')


def main(args: Optional[List[str]] = None) -> None:
    """独立运行入口：确认相机与激光话题正常并查看同步帧率。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
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
