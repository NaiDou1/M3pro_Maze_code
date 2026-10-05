"""顶层任务编排状态机。

主流程::

    INIT ──► EXPLORE ──(发现方块)──► APPROACH ──► GRASP ──┐
              ▲   │                                        │
              └───┘◄───────────────────────────────────────┘
              │
         (无未访问分支)
              ▼
            RETURN ──► FINISH

* ``EXPLORE``：每到一个格心就扫描路口写入拓扑，检测视野内方块，按 DFS 决策
  转向并走格；遇块即抓（决策 3）。
* ``APPROACH``：微调车体，使方块落入机械臂抓取包络。
* ``GRASP``：调用 :class:`GraspFSM` 完成抓取并放入车载收集筐。
* ``RETURN``：探索完成后沿已建拓扑走 BFS 最短路返回出口。

.. note::
   本模块**独占驱动**：内部用 ``SingleThreadedExecutor`` 统一驱动自身与三个
   子节点（底盘/传感器/机械臂），因此所有阻塞式运动原语才能正确收到回调。
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
from maze_explorer.block_detector import COLOR_NAMES, BlockDetection, BlockDetector
from maze_explorer.dfs_planner import DfsPlanner
from maze_explorer.grasp_fsm import GraspFSM
from maze_explorer.grid_mapper import GridMapper
from maze_explorer.line_detector import LineDetector
from maze_explorer.motion_controller import MotionController
from maze_explorer.sensors import SensorHub


class MissionState(Enum):
    INIT = auto()      # 加载标定、臂归巡线姿态
    EXPLORE = auto()   # 逐格推进 + 路口扫描 + DFS 选向
    APPROACH = auto()  # 车体对位至抓取包络
    GRASP = auto()     # 机械臂抓取并放入收集筐
    RETURN = auto()    # 沿拓扑最短路返出口
    FINISH = auto()
    FAULT = auto()     # 碰撞/丢线/连续失败，等待人工介入


def should_approach(distance: float, trigger_distance: float) -> bool:
    """判断是否应转入对位。

    相机在巡线姿态下能看到 1~2m 外的方块；若远处就转去对位，会脱离格心、
    打乱 DFS 拓扑，因此**只在进入触发距离后**才转入。

    :param distance: 方块相对车体的水平距离（米）。
    :param trigger_distance: 触发阈值（米）。
    """
    return distance <= trigger_distance


def approach_step(distance: float, target_distance: float, limit: float = 0.20) -> float:
    """由当前水平距离算对位时应前进的量（米）：正为前进、负为后退。

    按剩余距离自适应，并限幅避免单步冲过头。返回 0 表示已在目标附近，
    调用方应据此外结束循环。
    """
    step = distance - target_distance
    return max(-limit, min(limit, step))


class MissionManager(Node):
    """任务编排节点。"""

    def __init__(self) -> None:
        super().__init__('mission_manager')
        self._declare_params()

        # ---------------- 子节点（共同由 executor 驱动）----------------
        # 子节点在进程内创建，收不到 launch 注入的 yaml，故显式转发
        self.base = BaseDriver(
            yaw_source=self._yaw_source,
            watchdog_timeout=self._watchdog_timeout,
        )
        self.sensors = SensorHub()
        # 注入 executor 驱动的 future 等待回调：三个子节点都已加入本节点的
        # executor，若 ArmController 内部直接 spin_until_future_complete(self)
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
            cruise_linear=self._cruise,
            max_angular_z=self._max_wz,
            turn_angular=self._turn_wz,
            line_pid=self._line_pid,
            cell_tolerance=self._cell_tolerance,
            yaw_tolerance=self._yaw_tolerance,
            safety_range=self._safety_range,
            opening_min_range=self._opening_range,
            advance_timeout=self._advance_timeout,
            angular_scale_correction=self._ang_scale,
            scan_timeout=self._scan_timeout,
            vision_timeout=self._vision_timeout,
            sensor_recover_wait=self._recover_wait,
            line_steer_sign=self._steer_sign,
        )
        self.grasp = GraspFSM(self, self.arm)

        self._executor = SingleThreadedExecutor()
        for node in (self, self.base, self.sensors, self.arm):
            self._executor.add_node(node)

        self.state = MissionState.INIT
        self._failed_advances = 0
        #: 各颜色已收集数量。达到配额后不再触发该色，防止同一物理方块被
        #: 反复识别（方块被拿走后若筐内方块落入视野，会造成重复抓取）
        self._collected_by_color = {name: 0 for name in COLOR_NAMES}
        #: 抓取成功后的检测抑制截止时刻（monotonic）
        self._suppress_until = 0.0
        self.get_logger().info(
            f'MissionManager 就绪 | 场地 {self._grid_size}x{self._grid_size} '
            f'格距 {self._cell_size}m | 入口 {self._origin_rc} 出口 {self._exit_rc}'
        )

    # ------------------------------------------------------------------ 参数

    def _declare_params(self) -> None:
        # 场地
        self.declare_parameter('grid_size', 7)
        self.declare_parameter('cell_size', 0.40)
        self.declare_parameter('origin_rc', [0, 0])
        self.declare_parameter('exit_rc', [6, 6])
        # 运动
        self.declare_parameter('cruise_linear', 0.15)
        self.declare_parameter('max_angular_z', 0.60)
        self.declare_parameter('turn_angular', 0.50)
        #: 巡线 PID：误差是**归一化**偏差（1.0 = 线在图像边缘），不是像素值。
        #: 旧默认 [50,0,10] 是像素量纲，会让输出恒饱和（bang-bang），故一并改掉。
        self.declare_parameter('line_pid', [1.2, 0.0, 0.2])
        self.declare_parameter('cell_advance_tolerance', 0.03)
        self.declare_parameter('yaw_tolerance', 0.0873)
        self.declare_parameter('advance_timeout', 12.0)
        self.declare_parameter('odom_angular_scale_correction', 1.0)
        #: 速度看门狗阈值（秒）：看门狗独立线程超时未收到新指令即强制归零。0=禁用
        self.declare_parameter('cmd_watchdog_timeout', 0.5)
        #: 巡线转向符号：-1 = 图像右侧偏差对应右转（本机实测），+1 = 相机装反时
        self.declare_parameter('line_steer_sign', -1.0)
        #: 激光保鲜阈值（秒）：失效即禁止移动（否则激光失效会被误判为"前方通畅"）
        self.declare_parameter('scan_timeout', 0.5)
        #: 相机保鲜阈值（秒）：失效即禁止前进（否则循迹偏差恒 0，闷头直行）
        self.declare_parameter('vision_timeout', 2.0)
        #: 传感器失效后的等待上限（秒）：先给一次恢复机会，仍失效才判 FAULT
        self.declare_parameter('sensor_wait_timeout', 3.0)
        #: 传感器短暂失效后允许自旋等待恢复的时长（秒），仍失效才放弃动作
        self.declare_parameter('sensor_recover_wait', 2.0)
        #: INIT 阶段等相机首帧的上限（秒）：SensorHub 首帧同步实测需 1.6~1.8s
        self.declare_parameter('init_sensor_timeout', 15.0)
        #: 走格/转向失败后到下次重试的退避时长（秒）
        self.declare_parameter('retry_pause_sec', 0.5)
        #: 航向来源：'odom_raw'（默认，与位置同源、高频）或 'odom'
        #: （EKF 输出，融合 IMU 航向通常更准，但仅约 6Hz）
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
        # 相机外参（巡线姿态下相对 base_link）
        self.declare_parameter('mount_xyz', [0.10, 0.0, 0.35])
        self.declare_parameter('mount_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('mount_calibrated', False)
        # 目标触发与对位
        self.declare_parameter('detect_trigger_distance', 0.60)
        self.declare_parameter('approach_max_steps', 20)
        self.declare_parameter('approach_target_distance', 0.20)
        self.declare_parameter('per_color_quota', 2)
        self.declare_parameter('block_suppress_sec', 3.0)

        p = self.get_parameter
        self._grid_size = int(p('grid_size').value)
        self._cell_size = float(p('cell_size').value)
        self._origin_rc = tuple(int(v) for v in p('origin_rc').value)
        self._exit_rc = tuple(int(v) for v in p('exit_rc').value)
        self._cruise = float(p('cruise_linear').value)
        self._max_wz = float(p('max_angular_z').value)
        self._turn_wz = float(p('turn_angular').value)
        self._line_pid = tuple(float(v) for v in p('line_pid').value)
        self._steer_sign = float(p('line_steer_sign').value)
        self._cell_tolerance = float(p('cell_advance_tolerance').value)
        self._yaw_tolerance = float(p('yaw_tolerance').value)
        self._advance_timeout = float(p('advance_timeout').value)
        self._ang_scale = float(p('odom_angular_scale_correction').value)
        self._watchdog_timeout = float(p('cmd_watchdog_timeout').value)
        self._scan_timeout = float(p('scan_timeout').value)
        self._vision_timeout = float(p('vision_timeout').value)
        self._sensor_wait_timeout = float(p('sensor_wait_timeout').value)
        self._recover_wait = float(p('sensor_recover_wait').value)
        self._init_sensor_timeout = float(p('init_sensor_timeout').value)
        self._retry_pause_sec = float(p('retry_pause_sec').value)
        self._yaw_source = str(p('yaw_source').value)
        self._opening_range = float(p('opening_min_range').value)
        self._safety_range = float(p('safety_range').value)
        self._line_hsv = [int(v) for v in p('line_hsv').value]
        self._hsv_map = {
            name: [int(v) for v in p(f'{name}_hsv').value] for name in COLOR_NAMES
        }
        self._roi_top_ratio = float(p('line_roi_top_ratio').value)
        self._min_line_pixels = int(p('min_line_pixels').value)
        self._min_contour_area = int(p('min_contour_area').value)
        self._morph_kernel = int(p('morph_kernel').value)
        self._lost_frames = int(p('line_lost_frames').value)
        self._mount_xyz = [float(v) for v in p('mount_xyz').value]
        self._mount_rpy = [float(v) for v in p('mount_rpy').value]
        self._mount_calibrated = bool(p('mount_calibrated').value)
        self._trigger_distance = float(p('detect_trigger_distance').value)
        self._approach_max_steps = int(p('approach_max_steps').value)
        self._approach_target = float(p('approach_target_distance').value)
        self._per_color_quota = int(p('per_color_quota').value)
        self._suppress_sec = float(p('block_suppress_sec').value)

    def _spin(self, timeout: float) -> None:
        """统一 spin 入口：驱动本节点与三个子节点。"""
        self._executor.spin_once(timeout_sec=timeout)

    def _spin_until_future(self, future, timeout: float) -> None:
        """驱动 executor 直到服务 future 完成（供 ArmController 的 IK/FK 使用）。

        不依赖返回值，调用方统一用 ``future.done()`` 判定是否完成，以兼容
        不同 rclpy 版本的返回值差异。
        """
        self._executor.spin_until_future_complete(future, timeout_sec=timeout)

    # ------------------------------------------------------------------ 主循环

    def run(self) -> None:
        handlers = {
            MissionState.INIT: self._run_init,
            MissionState.EXPLORE: self._run_explore,
            MissionState.APPROACH: self._run_approach,
            MissionState.GRASP: self._run_grasp,
            MissionState.RETURN: self._run_return,
        }
        self.state = MissionState.INIT
        while rclpy.ok() and self.state not in (MissionState.FINISH, MissionState.FAULT):
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
        """等待传感器与机械臂就绪，并把臂摆到巡线姿态。"""
        deadline = time.monotonic() + 15.0
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin(0.1)
            if self.base.has_odom() and self.sensors.get_scan() is not None:
                break
        else:
            self.get_logger().error(
                '15s 内未收到里程计或激光，检查 start_agent.sh 与 ROS_DOMAIN_ID=30'
            )
            return MissionState.FAULT

        line_pose = [int(v) for v in self._param_list('line_pose', [90, 90, 12, 20, 90, 0])]
        self.arm.wait_for_subscribers(timeout=3.0)
        self.arm.send_joints(line_pose)
        self.arm.wait_until_ready()

        # 等相机首帧：SensorHub 首帧同步 RGB-D 实测需 1.6~1.8s（DDS 发现 + 配对），
        # 而上面两条机械臂指令是 time.sleep（不 spin），全程没有回调机会。
        # 不等它，EXPLORE 第一步就会因"相机年龄 inf"被安全门禁拦下（实测 FAULT）。
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
        cur = self.mapper.robot_rc
        heading = self.mapper.robot_heading

        # 0) 激光失效时禁止勘测路口：扇区取距失效会被读成"四处开口"，
        #    拓扑会被写脏，进而规划出撞墙路径。先给一次恢复机会，仍失效则 FAULT。
        if not self._wait_for_fresh_scan():
            self.get_logger().error(
                f'激光数据失效（年龄 {self.motion.scan_age():.1f}s > '
                f'{self._scan_timeout:.1f}s），拓扑不可信，任务中止'
            )
            return MissionState.FAULT

        # 1) 路口扫描并写入拓扑
        front, left, right = self.motion.scan_openings()
        self.mapper.observe(cur, heading, front, left, right)

        # 2) 视野内是否有足够近的方块（遇块即抓）
        #    相机巡线姿态下能看到 1~2m 外的方块。若远处就转去对位，会脱离格心、
        #    打乱 DFS 拓扑，因此只在进入触发距离后才转入 APPROACH；更远的方块
        #    等走格自然靠近后再处理。
        block = self._nearest_block()
        if block is not None:
            d = block.horizontal_distance()
            if should_approach(d, self._trigger_distance):
                self._pending_block = block
                self.get_logger().info(
                    f'发现 {block.color} 方块，距离 {d:.3f}m，转入对位'
                )
                return MissionState.APPROACH
            self.get_logger().info(
                f'检出 {block.color} 方块但距离 {d:.3f}m 超过触发阈值 '
                f'{self._trigger_distance:.2f}m，继续探索靠近',
                throttle_duration_sec=5.0,
            )

        # 3) DFS 决策
        decision = self.planner.decide()
        if decision.kind == 'finish':
            self.get_logger().info('所有分支探索完毕，开始返航')
            return MissionState.RETURN

        # 4) 转向 + 走格
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
            self.mapper.set_depth(target_rc, cur)
            self.mapper.mark_arrival(target_rc, decision.direction)
            self.planner.commit(decision)
        return MissionState.EXPLORE

    def _run_approach(self) -> MissionState:
        """微调车体，使目标落入抓取包络。"""
        block = self._pending_block
        if block is None:
            return MissionState.EXPLORE

        for _ in range(self._approach_max_steps):
            d = block.horizontal_distance()
            if self.grasp.in_envelope(block):
                break
            # 自适应步长：按剩余距离走，单步限幅避免一次冲过头
            step = approach_step(d, self._approach_target)
            if abs(step) < 0.02:
                break
            self.motion.advance(step)
            refreshed = self._nearest_block(color=block.color)
            if refreshed is None:
                self.get_logger().warn('对位过程中目标丢失，登记为待补抓')
                failed = self.planner.register_target(block.color, self.mapper.robot_rc)
                self.planner.mark_failed(failed)
                self._pending_block = None
                return MissionState.EXPLORE
            block = refreshed
            self._pending_block = block

        if not self.grasp.in_envelope(block):
            self.get_logger().warn('未能进入抓取包络，登记为待补抓')
            target = self.planner.register_target(block.color, self.mapper.robot_rc)
            self.planner.mark_failed(target)
            self._pending_block = None
            return MissionState.EXPLORE

        return MissionState.GRASP

    def _run_grasp(self) -> MissionState:
        block = self._pending_block
        if block is None:
            return MissionState.EXPLORE

        target = self.planner.register_target(
            block.color, self.mapper.robot_rc, pos_base=block.position_base, yaw=block.yaw_rad
        )
        ok = self.grasp.execute(block)
        if ok:
            self.planner.mark_done(target)
            self._collected_by_color[block.color] = (
                self._collected_by_color.get(block.color, 0) + 1
            )
            # 抑制期内不再触发检测：方块刚被拿走，视野里可能仍残留、
            # 或收集筐里的方块进入视野，都会造成重复抓取
            self._suppress_until = time.monotonic() + self._suppress_sec
            self.get_logger().info(
                f'已收集 {block.color}，该色累计 '
                f'{self._collected_by_color[block.color]} 个'
            )
        else:
            self.planner.mark_failed(target)

        # 抓取后原地重新观测（相机回到巡线姿态）
        self._spin(0.2)
        self._pending_block = None
        return MissionState.EXPLORE

    def _run_return(self) -> MissionState:
        """沿拓扑最短路返回出口，途中顺路补抓待补目标。"""
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

            nxt = self.mapper.neighbor_rc(self.mapper.robot_rc, direction)
            if nxt is not None:
                self.mapper.mark_arrival(nxt, direction)
        return MissionState.FINISH

    # ------------------------------------------------------------------ 工具

    def _fault_if_repeated(self) -> MissionState:
        if self._failed_advances >= 3:
            self.get_logger().error('连续 3 次走格/转向失败，进入 FAULT')
            return MissionState.FAULT
        return MissionState.EXPLORE

    def _retry_pause(self) -> None:
        """失败后到下次重试之间的退避与自旋。

        必要性：没有它时 3 次"走格失败"实测在 **11 毫秒**内烧完（EXPLORE 循环
        不 spin 就再次进入），任何瞬时异常都会直接判 FAULT；这段停顿既给传感器
        恢复的机会，也让重试之间有真实的观测间隔。

        注意：必须用"墙钟循环 + 多次 spin_once"，**不能**只调一次
        ``self._spin(0.5)``——``spin_once`` 语义是"最多等 0.5s，有回调就立即
        返回"，本机 odom 与 20Hz 定时器一直在刷，单次调用几乎立刻返回
        （2026-10-04 实测：退避形同虚设，3 次失败仍在 26ms 内烧完）。
        """
        deadline = time.monotonic() + self._retry_pause_sec
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin(0.05)

    def _wait_for_fresh_scan(self) -> bool:
        """等待激光恢复新鲜；返回是否可用。

        抖动容忍：传感器偶发掉一两帧属正常，先自旋等待 ``sensor_wait_timeout``
        秒；期间持续 spin 让回调有机会更新数据。
        """
        if self.motion.scan_age() <= self._scan_timeout:
            return True
        self.get_logger().warn(
            f'激光数据不新鲜（年龄 {self.motion.scan_age():.2f}s），'
            f'等待最多 {self._sensor_wait_timeout:.1f}s 恢复'
        )
        deadline = time.monotonic() + self._sensor_wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin(0.1)
            if self.motion.scan_age() <= self._scan_timeout:
                self.get_logger().info('激光已恢复')
                return True
        self.base.stop()
        return False

    def _nearest_block(self, color: Optional[str] = None) -> Optional[BlockDetection]:
        """返回视野内最近的可收集方块。

        过滤两类目标，避免重复抓取：

        * 抓取抑制期内一律返回 ``None``（刚抓完时视野可能仍有残留/筐内方块）；
        * 已达颜色配额的目标（赛题规定每色 2 个）不再触发。
        """
        if time.monotonic() < self._suppress_until:
            return None
        rgbd = self.sensors.get_rgbd()
        if rgbd is None:
            return None
        detections = [
            d
            for d in self.block_det.detect(rgbd[0], rgbd[1])
            if self._collected_by_color.get(d.color, 0) < self._per_color_quota
        ]
        return self.block_det.nearest(detections, color)

    def _param_list(self, name: str, default: List[int]) -> List[int]:
        if not self.has_parameter(name):
            self.declare_parameter(name, list(default))
        return list(self.get_parameter(name).value)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionManager()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('人工中断')
    finally:
        node.base.shutdown()  # 停看门狗 + 确保下发停止
        node._spin(0.1)  # noqa: SLF001 - 退出前确保下发停止
        for child in (node.arm, node.sensors, node.base, node):
            child.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
