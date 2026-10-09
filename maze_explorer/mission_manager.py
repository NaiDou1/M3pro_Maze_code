"""顶层任务编排状态机。

主流程::

    INIT -> EXPLORE -- 发现方块 --> APPROACH -> GRASP --+
                     ^   |                            |
                     +---+ <--------------------------+
                         |
                    无未访问分支
                         v
                       RETURN -> FINISH

* ``EXPLORE``：每到一个格心就扫描路口写入拓扑，检测视野内方块，按 DFS 决策
  转向并走格；遇块即抓。
* ``APPROACH``：微调车体，使方块落入机械臂抓取包络。
* ``GRASP``：调用 :class:`GraspFSM` 完成抓取并放入车载收集筐。
* ``RETURN``：探索完成后沿已建拓扑走 BFS 最短路返回出口。
* ``FAULT``：碰撞、丢线或连续失败，停车等待人工介入。

.. note::
   本模块独占驱动：内部用 ``SingleThreadedExecutor`` 统一驱动自身与三个子节点
   即底盘、传感器、机械臂，因此所有阻塞式运动原语才能正确收到回调。
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import List, Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from maze_explorer.arm_controller import ArmController
from maze_explorer.base_driver import BaseDriver
from maze_explorer.block_detector import (
    COLOR_NAMES,
    BlockColor,
    BlockDetection,
    BlockDetector,
)
from maze_explorer.dfs_planner import DecisionKind, DfsPlanner
from maze_explorer.grasp_fsm import GraspFSM
from maze_explorer.grid_mapper import GridMapper
from maze_explorer.line_detector import LineDetector
from maze_explorer.motion_controller import MotionController
from maze_explorer.sensors import SensorHub

#: INIT 阶段等待里程计与激光的上限，单位 s
INIT_ODOM_SCAN_TIMEOUT_SEC = 15.0
#: 等待机械臂订阅端上线的上限，单位 s
ARM_SUBSCRIBER_WAIT_SEC = 3.0
#: 状态机内常规自旋时长，单位 s
SPIN_SEC = 0.1
#: 重试退避期间的自旋时长，单位 s
RETRY_SPIN_SEC = 0.05
#: 抓取后重新观测前的自旋时长，单位 s，留给相机回到巡线视角
POST_GRASP_SPIN_SEC = 0.2
#: 远距方块提示日志的节流周期，单位 s，避免每拍刷屏
LOG_THROTTLE_SEC = 5.0
#: 对位单步的限幅，单位 m，避免一次冲过头越过目标
APPROACH_STEP_LIMIT_M = 0.20
#: 对位步长小于该值即认为已在目标附近并退出，单位 m
APPROACH_STEP_MIN_M = 0.02
#: 连续走格或转向失败上限，单位次，超过即进入 FAULT
MAX_CONSECUTIVE_FAILURES = 3


class MissionState(Enum):
    """任务状态，成员顺序即正常流转顺序。"""

    INIT = auto()      # 加载标定、臂归巡线姿态
    EXPLORE = auto()   # 逐格推进 + 路口扫描 + DFS 选向
    APPROACH = auto()  # 车体对位至抓取包络
    GRASP = auto()     # 机械臂抓取并放入收集筐
    RETURN = auto()    # 沿拓扑最短路返出口
    FINISH = auto()
    FAULT = auto()     # 碰撞、丢线或连续失败，等待人工介入


def should_approach(distance: float, trigger_distance: float) -> bool:
    """判断是否应转入对位。

    相机在巡线姿态下能看到 1 到 2 m 外的方块；若远处就转去对位，会脱离格心、
    打乱 DFS 拓扑，因此只在进入触发距离后才转入。

    :param distance: 方块相对车体的水平距离，单位 m。
    :param trigger_distance: 触发阈值，单位 m。
    :returns: 距离不大于阈值时为真。
    """
    return distance <= trigger_distance


def approach_step(
    distance: float,
    target_distance: float,
    limit: float = APPROACH_STEP_LIMIT_M,
) -> float:
    """由当前水平距离算对位时应前进的量，正为前进负为后退。

    按剩余距离自适应，并限幅避免单步冲过头。返回 0 表示已在目标附近，
    调用方应据此结束循环。

    :param distance: 方块当前水平距离，单位 m。
    :param target_distance: 目标停靠距离，单位 m，即抓取包络中心距离。
    :param limit: 单步限幅，单位 m，不小于 0。
    :returns: 步长，单位 m，取值区间 负 limit 到正 limit。
    """
    step = distance - target_distance
    return max(-limit, min(limit, step))


class MissionManager(Node):
    """任务编排节点：持有四个子节点与四个算法模块并驱动其流转。"""

    def __init__(self) -> None:
        """声明参数、构造子节点与算法模块，并把全部节点加入同一 executor。"""
        super().__init__('mission_manager')
        self._declare_params()

        # ---------------- 子节点，共同由 executor 驱动 ----------------
        # 子节点在进程内创建，收不到 launch 注入的 yaml，故显式转发
        self.base = BaseDriver(
            yaw_source=self._yaw_source,
            watchdog_timeout=self._watchdog_timeout,
        )
        self.sensors = SensorHub()
        # 注入 executor 驱动的 future 等待回调：三个子节点都已加入本节点的
        # executor，若 ArmController 内部直接调用 rclpy 的自旋接口并传入 self
        # 会因节点重复加入 executor 而抛异常。
        self.arm = ArmController(spin_until_fn=self._spin_until_future)

        # ---------------- 纯算法模块 ----------------
        self.line_det = LineDetector(
            hsv_range=(self._line_hsv[:3], self._line_hsv[3:]),
            roi_top_ratio=self._roi_top_ratio,
            min_pixels=self._min_line_pixels,
            morph_kernel=self._morph_kernel,
            lost_frames=self._lost_frames,
        )
        self.block_det = BlockDetector(
            hsv_map=self._hsv_map,
            min_area=self._min_contour_area,
            morph_kernel=self._morph_kernel,
            mount_xyz=self._mount_xyz,
            mount_rpy=self._mount_rpy,
            mount_calibrated=self._mount_calibrated,
        )
        self.mapper = GridMapper(
            grid_size=self._grid_size,
            cell_size=self._cell_size,
            opening_min_range=self._opening_range,
            origin_rc=self._origin_rc,
        )
        self.planner = DfsPlanner(self.mapper, exit_rc=self._exit_rc)
        self.motion = MotionController(
            self, self.base, self.sensors, self.line_det,
            spin_fn=self._spin,
            cell_size=self._cell_size,
            cruise_linear=self._cruise_linear,
            max_angular_z=self._max_angular_z,
            turn_angular=self._turn_angular,
            line_pid=self._line_pid,
            cell_tolerance=self._cell_tolerance,
            yaw_tolerance=self._yaw_tolerance,
            safety_range=self._safety_range,
            opening_min_range=self._opening_range,
            advance_timeout=self._advance_timeout,
            angular_scale_correction=self._angular_scale,
            scan_timeout=self._scan_timeout,
            vision_timeout=self._vision_timeout,
            sensor_recover_wait=self._recover_wait,
            line_steer_sign=self._line_steer_sign,
        )
        self.grasp = GraspFSM(self, self.arm)

        self._executor = SingleThreadedExecutor()
        for child in (self, self.base, self.sensors, self.arm):
            self._executor.add_node(child)

        self.state = MissionState.INIT
        self._failed_advances = 0
        #: 当前待处理的方块，APPROACH 与 GRASP 两状态共用
        self._pending_block: Optional[BlockDetection] = None
        #: 各颜色已收集数量。达到配额后不再触发该色，防止同一物理方块被反复
        #: 识别，即方块被拿走后若筐内方块落入视野会造成重复抓取
        self._collected_by_color = {name: 0 for name in COLOR_NAMES}
        #: 抓取成功后的检测抑制截止时刻，单位 s，monotonic 时基
        self._suppress_until = 0.0
        self.get_logger().info(
            f'MissionManager 就绪 | 场地 {self._grid_size}x{self._grid_size} '
            f'格距 {self._cell_size}m | 入口 {self._origin_rc} 出口 {self._exit_rc}'
        )

    # ------------------------------------------------------------------ 参数

    def _declare_params(self) -> None:
        """声明全部参数并读入带语义名的实例属性，默认值与 config 一致。"""
        # 场地
        self.declare_parameter('grid_size', 7)
        self.declare_parameter('cell_size', 0.40)
        self.declare_parameter('origin_rc', [0, 0])
        self.declare_parameter('exit_rc', [6, 6])
        # 运动
        self.declare_parameter('cruise_linear', 0.15)
        self.declare_parameter('max_angular_z', 0.60)
        self.declare_parameter('turn_angular', 0.50)
        #: 巡线 PID 三元组：误差是归一化偏差，无量纲，1.0 表示线在图像边缘，
        #: 不是像素值。像素量纲的旧默认值会让输出恒饱和即 bang-bang
        self.declare_parameter('line_pid', [1.2, 0.0, 0.2])
        self.declare_parameter('cell_advance_tolerance', 0.03)
        self.declare_parameter('yaw_tolerance', 0.0873)
        self.declare_parameter('advance_timeout', 12.0)
        self.declare_parameter('odom_angular_scale_correction', 1.0)
        #: 速度看门狗阈值，单位 s，看门狗独立线程超时未收到新指令即强制归零，
        #: 取 0 表示禁用
        self.declare_parameter('cmd_watchdog_timeout', 0.5)
        #: 巡线转向符号：-1 表示图像右侧偏差对应右转即本机实测，+1 用于相机装反
        self.declare_parameter('line_steer_sign', -1.0)
        #: 激光保鲜阈值，单位 s，失效即禁止移动，否则激光失效会被误判为前方通畅
        self.declare_parameter('scan_timeout', 0.5)
        #: 相机保鲜阈值，单位 s，失效即禁止前进，否则循迹偏差恒 0 会闷头直行
        self.declare_parameter('vision_timeout', 2.0)
        #: 传感器失效后的等待上限，单位 s，先给一次恢复机会仍失效才判 FAULT
        self.declare_parameter('sensor_wait_timeout', 3.0)
        #: 传感器短暂失效后允许自旋等待恢复的时长，单位 s，仍失效才放弃动作
        self.declare_parameter('sensor_recover_wait', 2.0)
        #: INIT 阶段等相机首帧的上限，单位 s，RGB-D 首帧同步需完成 DDS 发现
        #: 与配对，实测最坏 1.88 s
        self.declare_parameter('init_sensor_timeout', 15.0)
        #: 走格或转向失败后到下次重试的退避时长，单位 s
        self.declare_parameter('retry_pause_sec', 0.5)
        #: 航向来源，取值见 base_driver.YawSource：odom_raw 与位置同源且高频，
        #: odom 即 EKF 输出融合 IMU 航向通常更准但仅约 6Hz
        self.declare_parameter('yaw_source', 'odom_raw')
        # 激光
        self.declare_parameter('opening_min_range', 0.55)
        self.declare_parameter('safety_range', 0.25)
        # HSV
        self.declare_parameter('line_hsv', [0, 0, 0, 180, 255, 80])
        self.declare_parameter('red_hsv', [0, 154, 107, 184, 253, 255])
        self.declare_parameter('green_hsv', [34, 130, 205, 125, 253, 255])
        self.declare_parameter('blue_hsv', [55, 196, 137, 125, 253, 255])
        self.declare_parameter('yellow_hsv', [23, 99, 235, 125, 253, 255])
        self.declare_parameter('line_roi_top_ratio', 0.5)
        self.declare_parameter('min_line_pixels', 200)
        self.declare_parameter('min_contour_area', 300)
        self.declare_parameter('morph_kernel', 5)
        self.declare_parameter('line_lost_frames', 5)
        # 相机外参，巡线姿态下相对 base_link
        self.declare_parameter('mount_xyz', [0.10, 0.0, 0.35])
        self.declare_parameter('mount_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('mount_calibrated', False)
        # 目标触发与对位
        self.declare_parameter('detect_trigger_distance', 0.60)
        self.declare_parameter('approach_max_steps', 20)
        self.declare_parameter('approach_target_distance', 0.20)
        self.declare_parameter('per_color_quota', 2)
        self.declare_parameter('block_suppress_sec', 3.0)

        get_parameter = self.get_parameter
        self._grid_size = int(get_parameter('grid_size').value)
        self._cell_size = float(get_parameter('cell_size').value)
        self._origin_rc = tuple(int(v) for v in get_parameter('origin_rc').value)
        self._exit_rc = tuple(int(v) for v in get_parameter('exit_rc').value)
        self._cruise_linear = float(get_parameter('cruise_linear').value)
        self._max_angular_z = float(get_parameter('max_angular_z').value)
        self._turn_angular = float(get_parameter('turn_angular').value)
        self._line_pid = tuple(float(v) for v in get_parameter('line_pid').value)
        self._line_steer_sign = float(get_parameter('line_steer_sign').value)
        self._cell_tolerance = float(
            get_parameter('cell_advance_tolerance').value
        )
        self._yaw_tolerance = float(get_parameter('yaw_tolerance').value)
        self._advance_timeout = float(get_parameter('advance_timeout').value)
        self._angular_scale = float(
            get_parameter('odom_angular_scale_correction').value
        )
        self._watchdog_timeout = float(
            get_parameter('cmd_watchdog_timeout').value
        )
        self._scan_timeout = float(get_parameter('scan_timeout').value)
        self._vision_timeout = float(get_parameter('vision_timeout').value)
        self._sensor_wait_timeout = float(
            get_parameter('sensor_wait_timeout').value
        )
        self._recover_wait = float(get_parameter('sensor_recover_wait').value)
        self._init_sensor_timeout = float(
            get_parameter('init_sensor_timeout').value
        )
        self._retry_pause_sec = float(get_parameter('retry_pause_sec').value)
        self._yaw_source = str(get_parameter('yaw_source').value)
        self._opening_range = float(get_parameter('opening_min_range').value)
        self._safety_range = float(get_parameter('safety_range').value)
        self._line_hsv = [int(v) for v in get_parameter('line_hsv').value]
        self._hsv_map = {
            name: [int(v) for v in get_parameter(f'{name}_hsv').value]
            for name in COLOR_NAMES
        }
        self._roi_top_ratio = float(get_parameter('line_roi_top_ratio').value)
        self._min_line_pixels = int(get_parameter('min_line_pixels').value)
        self._min_contour_area = int(get_parameter('min_contour_area').value)
        self._morph_kernel = int(get_parameter('morph_kernel').value)
        self._lost_frames = int(get_parameter('line_lost_frames').value)
        self._mount_xyz = [float(v) for v in get_parameter('mount_xyz').value]
        self._mount_rpy = [float(v) for v in get_parameter('mount_rpy').value]
        self._mount_calibrated = bool(get_parameter('mount_calibrated').value)
        self._trigger_distance = float(
            get_parameter('detect_trigger_distance').value
        )
        self._approach_max_steps = int(
            get_parameter('approach_max_steps').value
        )
        self._approach_target = float(
            get_parameter('approach_target_distance').value
        )
        self._per_color_quota = int(get_parameter('per_color_quota').value)
        self._suppress_sec = float(get_parameter('block_suppress_sec').value)

    def _spin(self, timeout: float) -> None:
        """统一自旋入口：驱动本节点与三个子节点。

        :param timeout: 本次自旋时长，单位 s。
        """
        self._executor.spin_once(timeout_sec=timeout)

    def _spin_until_future(self, future, timeout: float) -> None:
        """驱动 executor 直到服务 future 完成，供 ArmController 的 IK 与 FK 使用。

        不依赖返回值，调用方统一检查 ``future`` 是否完成，以兼容不同 rclpy
        版本的返回值差异。

        :param future: 已发起的服务调用 future。
        :param timeout: 等待上限，单位 s。
        """
        self._executor.spin_until_future_complete(future, timeout_sec=timeout)

    # ------------------------------------------------------------------ 主循环

    def run(self) -> None:
        """按状态分发处理器直至 FINISH 或 FAULT，结束时停车并打印进度。"""
        handlers = {
            MissionState.INIT: self._run_init,
            MissionState.EXPLORE: self._run_explore,
            MissionState.APPROACH: self._run_approach,
            MissionState.GRASP: self._run_grasp,
            MissionState.RETURN: self._run_return,
        }
        self.state = MissionState.INIT
        while rclpy.ok() and self.state not in (
            MissionState.FINISH,
            MissionState.FAULT,
        ):
            handler = handlers.get(self.state)
            if handler is None:
                self.get_logger().error(f'状态 {self.state.name} 无处理器')
                self.state = MissionState.FAULT
                break
            previous = self.state
            self.state = handler()
            if self.state != previous:
                self.get_logger().info(f'{previous.name} -> {self.state.name}')

        self.base.stop()
        if self.state is MissionState.FAULT:
            self.get_logger().error(f'任务中断：{self.planner.summary()}')
        else:
            self.get_logger().info(f'任务完成：{self.planner.summary()}')

    # -------------------------------------------------------------- 各状态实现

    def _run_init(self) -> MissionState:
        """等待传感器与机械臂就绪，并把臂摆到巡线姿态。

        :returns: 全部就绪为 EXPLORE；里程计、激光或相机超时为 FAULT。
        """
        deadline = time.monotonic() + INIT_ODOM_SCAN_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin(SPIN_SEC)
            if self.base.has_odom() and self.sensors.get_scan() is not None:
                break
        else:
            self.get_logger().error(
                f'{INIT_ODOM_SCAN_TIMEOUT_SEC:.0f}s 内未收到里程计或激光，'
                '检查 start_agent.sh 与 ROS_DOMAIN_ID=30'
            )
            return MissionState.FAULT

        line_pose = [
            int(v)
            for v in self._param_list('line_pose', [90, 90, 12, 20, 90, 0])
        ]
        self.arm.wait_for_subscribers(timeout=ARM_SUBSCRIBER_WAIT_SEC)
        self.arm.send_joints(line_pose)
        self.arm.wait_until_ready()

        # 等相机首帧：RGB-D 首帧同步需完成 DDS 发现与配对，实测最坏 1.88 s，而
        # 上面两条机械臂指令期间不自旋，全程没有回调机会。不等它，EXPLORE 第一步
        # 就会因相机年龄为无穷被安全门禁拦下并判 FAULT。
        if not self.motion.wait_until_healthy(
            need_vision=True, timeout=self._init_sensor_timeout
        ):
            self.get_logger().error(
                f'{self._init_sensor_timeout:.0f}s 内未取得同步 RGB-D'
                f'（相机年龄 {self.motion.vision_age():.1f}s），无视觉无法巡线，任务中止'
            )
            return MissionState.FAULT

        # 以里程计当前朝向初始化机器人朝向
        yaw = self.base.get_yaw()
        if yaw is not None:
            self.mapper.robot_heading = self.mapper.yaw_to_heading(yaw)
        self.get_logger().info(
            f'初始化完成，朝向 {self.mapper.robot_heading}'
            f'（激光 {self.motion.scan_age():.2f}s，相机 {self.motion.vision_age():.2f}s）'
        )
        return MissionState.EXPLORE

    def _run_explore(self) -> MissionState:
        """扫描路口、检测方块，按 DFS 决策转向并推进一格。

        :returns: 有方块在触发距离内为 APPROACH；无未访问分支为 RETURN；走格
            或转向失败退回 EXPLORE；连续失败或激光失效为 FAULT。
        """
        current_rc = self.mapper.robot_rc
        heading = self.mapper.robot_heading

        # 第 0 步：激光失效时禁止勘测路口。扇区取距失效会被读成四处开口，
        # 拓扑会被写脏，进而规划出撞墙路径。先给一次恢复机会，仍失效则 FAULT。
        if not self._wait_for_fresh_scan():
            self.get_logger().error(
                f'激光数据失效（年龄 {self.motion.scan_age():.1f}s > '
                f'{self._scan_timeout:.1f}s），拓扑不可信，任务中止'
            )
            return MissionState.FAULT

        # 第 1 步：路口扫描并写入拓扑
        front, left, right = self.motion.scan_openings()
        self.mapper.observe(current_rc, heading, front, left, right)

        # 第 2 步：视野内是否有足够近的方块即遇块即抓。相机巡线姿态下能看到
        # 1 到 2 m 外的方块，远处就转去对位会脱离格心打乱 DFS 拓扑，因此只在
        # 进入触发距离后才转入 APPROACH，更远的方块等走格自然靠近后再处理。
        block = self._nearest_block()
        if block is not None:
            horizontal_distance = block.horizontal_distance()
            if should_approach(horizontal_distance, self._trigger_distance):
                self._pending_block = block
                self.get_logger().info(
                    f'发现 {block.color} 方块，距离 {horizontal_distance:.3f}m，转入对位'
                )
                return MissionState.APPROACH
            self.get_logger().info(
                f'检出 {block.color} 方块但距离 {horizontal_distance:.3f}m 超过触发阈值 '
                f'{self._trigger_distance:.2f}m，继续探索靠近',
                throttle_duration_sec=LOG_THROTTLE_SEC,
            )

        # 第 3 步：DFS 决策
        decision = self.planner.decide()
        if decision.kind == DecisionKind.FINISH:
            self.get_logger().info('所有分支探索完毕，开始返航')
            return MissionState.RETURN

        # 第 4 步：转向与走格
        if heading != decision.direction:
            if not self.motion.turn_to_heading(decision.direction):
                self._failed_advances += 1
                self._retry_pause()
                return self._fault_if_repeated()
            self.mapper.robot_heading = decision.direction

        if not self.motion.advance_one_cell():
            self._failed_advances += 1
            self.get_logger().warn(f'走格失败第 {self._failed_advances} 次')
            self._retry_pause()
            return self._fault_if_repeated()

        self._failed_advances = 0
        target_rc = decision.target_rc
        if target_rc is not None:
            self.mapper.set_depth(target_rc, current_rc)
            self.mapper.mark_arrival(target_rc, decision.direction)
            self.planner.commit(decision)
        return MissionState.EXPLORE

    def _run_approach(self) -> MissionState:
        """微调车体，使目标落入抓取包络。

        :returns: 进入包络为 GRASP；目标丢失或步数耗尽仍不在包络则登记待补抓
            并退回 EXPLORE；无待处理方块为 EXPLORE。
        """
        block = self._pending_block
        if block is None:
            return MissionState.EXPLORE

        for _ in range(self._approach_max_steps):
            horizontal_distance = block.horizontal_distance()
            if self.grasp.in_envelope(block):
                break
            # 自适应步长：按剩余距离走，单步限幅避免一次冲过头
            step = approach_step(horizontal_distance, self._approach_target)
            if abs(step) < APPROACH_STEP_MIN_M:
                break
            self.motion.advance(step)
            refreshed = self._nearest_block(color=block.color)
            if refreshed is None:
                self.get_logger().warn('对位过程中目标丢失，登记为待补抓')
                failed = self.planner.register_target(
                    block.color, self.mapper.robot_rc
                )
                self.planner.mark_failed(failed)
                self._pending_block = None
                return MissionState.EXPLORE
            block = refreshed
            self._pending_block = block

        if not self.grasp.in_envelope(block):
            self.get_logger().warn('未能进入抓取包络，登记为待补抓')
            target = self.planner.register_target(
                block.color, self.mapper.robot_rc
            )
            self.planner.mark_failed(target)
            self._pending_block = None
            return MissionState.EXPLORE

        return MissionState.GRASP

    def _run_grasp(self) -> MissionState:
        """登记目标并执行抓取，成功则累计配额并进入检测抑制期。

        :returns: 一律退回 EXPLORE；无待处理方块时同样为 EXPLORE。
        """
        block = self._pending_block
        if block is None:
            return MissionState.EXPLORE

        target = self.planner.register_target(
            block.color,
            self.mapper.robot_rc,
            pos_base=block.position_base,
            yaw=block.yaw_rad,
        )
        succeeded = self.grasp.execute(block)
        if succeeded:
            self.planner.mark_done(target)
            self._collected_by_color[block.color] = (
                self._collected_by_color.get(block.color, 0) + 1
            )
            # 抑制期内不再触发检测：方块刚被拿走，视野里可能仍残留，或收集筐
            # 里的方块进入视野，都会造成重复抓取
            self._suppress_until = time.monotonic() + self._suppress_sec
            self.get_logger().info(
                f'已收集 {block.color}，该色累计 '
                f'{self._collected_by_color[block.color]} 个'
            )
        else:
            self.planner.mark_failed(target)

        # 抓取后原地重新观测，相机回到巡线姿态
        self._spin(POST_GRASP_SPIN_SEC)
        self._pending_block = None
        return MissionState.EXPLORE

    def _run_return(self) -> MissionState:
        """沿拓扑最短路返回出口，途中顺路补抓待补目标。

        :returns: 到达出口或无通路为 FINISH；转向或走格失败为 FAULT。
        """
        directions = self.planner.return_home_directions()
        if not directions:
            self.get_logger().warn('未找到返回出口的通路，直接结束')
            return MissionState.FINISH

        for direction in directions:
            if not rclpy.ok():
                break
            if self.mapper.robot_heading != direction:
                if not self.motion.turn_to_heading(direction):
                    return MissionState.FAULT
                self.mapper.robot_heading = direction
            if not self.motion.advance_one_cell():
                return MissionState.FAULT

            # 顺路检测并补抓
            block = self._nearest_block()
            if block is not None and self.grasp.in_envelope(block):
                self._pending_block = block
                self._run_grasp()

            next_rc = self.mapper.neighbor_rc(self.mapper.robot_rc, direction)
            if next_rc is not None:
                self.mapper.mark_arrival(next_rc, direction)
        return MissionState.FINISH

    # ------------------------------------------------------------------ 工具

    def _fault_if_repeated(self) -> MissionState:
        """按连续失败计数决定退回探索还是进入 FAULT。

        :returns: 失败次数达到 ``MAX_CONSECUTIVE_FAILURES`` 为 FAULT，否则
            为 EXPLORE。
        """
        if self._failed_advances >= MAX_CONSECUTIVE_FAILURES:
            self.get_logger().error(
                f'连续 {MAX_CONSECUTIVE_FAILURES} 次走格或转向失败，进入 FAULT'
            )
            return MissionState.FAULT
        return MissionState.EXPLORE

    def _retry_pause(self) -> None:
        """失败后到下次重试之间的退避与自旋。

        没有它时连续走格失败会在十余毫秒内烧完，因为 EXPLORE 循环不自旋就再次
        进入，任何瞬时异常都会直接判 FAULT；这段停顿既给传感器恢复的机会，也让
        重试之间有真实的观测间隔。

        必须用墙钟循环加多次 ``_spin``。单次长自旋不起作用：``spin_once`` 的语义
        是最多等待指定时长、有回调就立即返回，而本机 odom 与 20Hz 定时器一直在
        刷，单次调用几乎立刻返回，退避会形同虚设。
        """
        deadline = time.monotonic() + self._retry_pause_sec
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin(RETRY_SPIN_SEC)

    def _wait_for_fresh_scan(self) -> bool:
        """等待激光恢复新鲜，返回是否可用。

        抖动容忍：传感器偶发掉一两帧属正常，先自旋等待 ``sensor_wait_timeout``
        秒，期间持续自旋让回调有机会更新数据。

        :returns: 激光年龄回到阈值内为真；到时仍失效时先停车再返回假。
        """
        if self.motion.scan_age() <= self._scan_timeout:
            return True
        self.get_logger().warn(
            f'激光数据不新鲜（年龄 {self.motion.scan_age():.2f}s），'
            f'等待最多 {self._sensor_wait_timeout:.1f}s 恢复'
        )
        deadline = time.monotonic() + self._sensor_wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin(SPIN_SEC)
            if self.motion.scan_age() <= self._scan_timeout:
                self.get_logger().info('激光已恢复')
                return True
        self.base.stop()
        return False

    def _nearest_block(
        self, color: Optional[BlockColor] = None
    ) -> Optional[BlockDetection]:
        """返回视野内最近的可收集方块。

        过滤两类目标以避免重复抓取：

        * 抓取抑制期内一律返回 ``None``，即刚抓完时视野可能仍有残留，或收集
          筐内方块进入视野；
        * 已达颜色配额的目标不再触发，赛题规定每色 2 个。

        :param color: 颜色过滤条件，取 ``None`` 表示不限颜色。
        :returns: 距离最近的可收集方块；无候选或处于抑制期为 ``None``。
        """
        if time.monotonic() < self._suppress_until:
            return None
        rgbd = self.sensors.get_rgbd()
        if rgbd is None:
            return None
        detections = [
            detection
            for detection in self.block_det.detect(rgbd[0], rgbd[1])
            if self._collected_by_color.get(detection.color, 0)
            < self._per_color_quota
        ]
        return self.block_det.nearest(detections, color)

    def _param_list(self, name: str, default: List[int]) -> List[int]:
        """读取整数列表参数，未声明时按默认值声明。

        :param name: 参数名。
        :param default: 默认值序列。
        :returns: 参数当前值的浅拷贝。
        """
        if not self.has_parameter(name):
            self.declare_parameter(name, list(default))
        return list(self.get_parameter(name).value)


def main(args: Optional[List[str]] = None) -> None:
    """任务入口：构造编排节点、运行主循环并按序收尾。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
    rclpy.init(args=args)
    node = MissionManager()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('人工中断')
    finally:
        node.base.shutdown()  # 停看门狗并确保下发停止
        node._spin(SPIN_SEC)  # noqa: SLF001 - 退出前确保下发停止
        for child in (node.arm, node.sensors, node.base, node):
            child.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
