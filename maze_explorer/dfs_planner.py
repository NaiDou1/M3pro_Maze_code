"""DFS 探索决策与先深后浅的目标调度。

迷宫是以入口为根的树形结构，因此用 DFS 即可覆盖全部通道：每到一个格子就观测
路口，优先沿未访问的开口深入；无未访问分支时回溯到最近仍有分支的格。

同时维护方块目标队列。计罚按漏取方块的深度计算，故补抓顺序为 depth 降序——
时间不足时被放弃的是浅层低罚时目标。

纯算法实现，不依赖 ROS，可直接单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from maze_explorer._compat import StrEnum
from maze_explorer.grid_mapper import DIRECTIONS, CellRC, GridMapper, Heading


class DecisionKind(StrEnum):
    """探索决策种类，取值 advance、backtrack、finish 三者之一。

    advance 表示前进到相邻未访问格，backtrack 表示回溯到来路格，finish 表示
    全部分支已探索完毕。
    """

    ADVANCE = 'advance'
    BACKTRACK = 'backtrack'
    FINISH = 'finish'


class TargetState(StrEnum):
    """方块目标状态，取值 pending、done、failed 三者之一。

    pending 表示尚未抓取，done 表示已入筐，failed 表示抓取失败待重试。
    """

    PENDING = 'pending'
    DONE = 'done'
    FAILED = 'failed'


@dataclass
class BlockTarget:
    """一个待收集或已收集的方块记录。

    :ivar color: 方块颜色，取值域由 ``block_detector.BlockColor`` 定义，
        此处保持裸字符串以避免建图决策层依赖视觉库。
    :ivar cell_rc: 登记时所在格坐标。
    :ivar depth: 所在格树深度，抓取优先级排序依据。
    :ivar pos_base: 停车前测得的 base 系位置三分量，单位 m；外参未标定时为
        ``None``。
    :ivar yaw: 由角点估计的方块朝向，单位 rad；缺失时为 ``None``。
    :ivar state: 抓取状态，取值见 ``TargetState``。
    """

    color: str
    cell_rc: CellRC
    depth: int
    #: base 系位置三分量，单位 m，外参未标定时缺省
    pos_base: Optional[Tuple[float, float, float]] = None
    #: 方块朝向，单位 rad，缺失时缺省
    yaw: Optional[float] = None
    #: 抓取状态
    state: TargetState = TargetState.PENDING

    def key(self) -> Tuple[str, CellRC]:
        """返回颜色与格坐标组成的身份键，同色同格视为同一目标。

        :returns: 二元组，可直接用于字典索引。
        """
        return self.color, self.cell_rc


@dataclass
class Decision:
    """一次探索决策及其目标。

    :ivar kind: 决策种类，取值见 ``DecisionKind``。
    :ivar direction: 前进或回溯所用的网格方向，finish 时为 ``None``。
    :ivar target_rc: 目标格坐标，finish 时为 ``None``。
    """

    kind: DecisionKind
    #: 前进或回溯所用方向，finish 时缺省
    direction: Optional[Heading] = None
    #: 目标格坐标，finish 时缺省
    target_rc: Optional[CellRC] = None

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        """返回单行决策摘要，供日志输出。"""
        return f'{self.kind}({self.direction} -> {self.target_rc})'


class DfsPlanner:
    """基于网格拓扑的 DFS 决策器。

    决策与状态提交分离：:meth:`decide` 只读，:meth:`commit` 在动作成功后才改
    路径栈，保证内部路径与实车位置一致。
    """

    def __init__(self, mapper: GridMapper, exit_rc: Optional[CellRC] = None) -> None:
        """绑定拓扑并初始化路径栈与目标表。

        :param mapper: 已创建的拓扑实例，入口格已定。
        :param exit_rc: 出口格坐标；缺省时与入口同格，由 launch 的 ``exit_rc``
            注入实际出口。
        """
        self._mapper = mapper
        #: 出口格坐标，缺省与入口同格
        self.exit_rc: CellRC = exit_rc if exit_rc is not None else mapper.origin_rc
        #: 从入口到当前位置的路径栈，栈顶为当前位置，用于回溯
        self.path: List[CellRC] = [mapper.origin_rc]
        #: 已登记目标，键为颜色与格坐标的身份键
        self.targets: Dict[Tuple[str, CellRC], BlockTarget] = {}

    # ------------------------------------------------------------------ 决策

    def decide(self) -> Decision:
        """基于当前所在格给出下一步决策，不修改内部状态。

        :returns: 有未访问开口时为 advance，否则路径栈长度大于 1 时为
            backtrack，两者皆无时为 finish。
        """
        current_rc = self._mapper.robot_rc
        candidates = self._mapper.unvisited_openings(current_rc)

        if candidates:
            direction = self._pick_direction(current_rc, candidates)
            return Decision(
                DecisionKind.ADVANCE,
                direction,
                self._mapper.neighbor_rc(current_rc, direction),
            )

        # 无未访问分支时回溯
        if len(self.path) > 1:
            previous_rc = self.path[-2]
            direction = self._direction_between(current_rc, previous_rc)
            return Decision(DecisionKind.BACKTRACK, direction, previous_rc)

        return Decision(DecisionKind.FINISH)

    def commit(self, decision: Decision) -> None:
        """在动作成功执行后提交状态变更，保证路径与实车一致。

        :param decision: 已被上层成功执行的决策。
        """
        if decision.kind == DecisionKind.ADVANCE and decision.target_rc is not None:
            self.path.append(decision.target_rc)
        elif decision.kind == DecisionKind.BACKTRACK:
            if len(self.path) > 1:
                self.path.pop()

    def _pick_direction(
        self, current_rc: CellRC, candidates: Sequence[Heading]
    ) -> Heading:
        """在多个未访问开口中选一个。

        优先选与当前朝向一致的方向以省去一次 90 度转向；其次选只差一次转向的
        方向；最后取列表首项，保证决策可复现，便于回放调试。

        :param current_rc: 当前格坐标。
        :param candidates: 候选方向列表，非空。
        :returns: 选中的方向。
        """
        heading = self._mapper.robot_heading
        if heading in candidates:
            return heading
        for direction in candidates:
            if self._mapper.turn_direction(heading, direction) in (1, -1):
                return direction
        return candidates[0]

    def _direction_between(self, src: CellRC, dst: CellRC) -> Heading:
        """返回从源格走到相邻目标格所需的方向。

        :param src: 源格坐标。
        :param dst: 与源格相邻的目标格坐标。
        :returns: 方向；两格不相邻时抛 ``ValueError``。
        """
        for direction, (dr, dc, _) in DIRECTIONS.items():
            if (src[0] + dr, src[1] + dc) == dst:
                return direction
        raise ValueError(f'{src} 与 {dst} 不相邻')

    # -------------------------------------------------------------- 目标管理

    def register_target(
        self,
        color: str,
        cell_rc: CellRC,
        pos_base: Optional[Tuple[float, float, float]] = None,
        yaw: Optional[float] = None,
    ) -> BlockTarget:
        """登记一个方块，树深度取自所在格，重复登记返回首次记录。

        :param color: 方块颜色，取值域见 ``block_detector.BlockColor``。
        :param cell_rc: 方块所在格坐标。
        :param pos_base: base 系位置三分量，单位 m，外参未标定时为 ``None``。
        :param yaw: 方块朝向，单位 rad，缺失时为 ``None``。
        :returns: 已登记的目标记录。
        """
        target = BlockTarget(
            color=color,
            cell_rc=cell_rc,
            depth=self._mapper.cell(*cell_rc).depth,
            pos_base=pos_base,
            yaw=yaw,
        )
        self.targets.setdefault(target.key(), target)
        return self.targets[target.key()]

    def mark_done(self, target: BlockTarget) -> None:
        """把目标标为已入筐，同步更新表内同一记录。

        :param target: 待更新目标。
        """
        target.state = TargetState.DONE
        if target.key() in self.targets:
            self.targets[target.key()].state = TargetState.DONE

    def mark_failed(self, target: BlockTarget) -> None:
        """把目标标为抓取失败待重试，同步更新表内同一记录。

        :param target: 待更新目标。
        """
        target.state = TargetState.FAILED
        if target.key() in self.targets:
            self.targets[target.key()].state = TargetState.FAILED

    def pending_targets(self) -> List[BlockTarget]:
        """返回待补抓目标，按 depth 降序，深层优先即罚时最大。

        :returns: 未抓取与抓取失败两类目标构成的列表。
        """
        pool = [
            target
            for target in self.targets.values()
            if target.state in (TargetState.PENDING, TargetState.FAILED)
        ]
        return sorted(pool, key=lambda target: target.depth, reverse=True)

    def done_count(self) -> int:
        """返回已入筐目标数，取值区间 0 到已登记目标总数。"""
        return sum(
            1 for target in self.targets.values() if target.state == TargetState.DONE
        )

    def summary(self) -> str:
        """返回一行进度摘要，供日志与终局报表输出。"""
        return (
            f'已收集 {self.done_count()}/{len(self.targets)}，'
            f'待补 {len(self.pending_targets())}，已探索 '
            f'{self._mapper.visited_count()}/{self._mapper.grid_size ** 2} 格'
        )

    # ---------------------------------------------------------- 返航路径规划

    def shortest_path(self, src: CellRC, dst: CellRC) -> List[CellRC]:
        """在已探明拓扑上做 BFS 最短路，返回含起点与终点的格序列。

        :param src: 起点格坐标。
        :param dst: 终点格坐标。
        :returns: 格序列；无通路或仅起点时按相等与否返回，不相等且无通路为空。
        """
        if src == dst:
            return [src]

        queue: List[CellRC] = [src]
        came_from: Dict[CellRC, Optional[CellRC]] = {src: None}

        while queue:
            current_rc = queue.pop(0)
            for direction in self._mapper.known_neighbors(current_rc):
                next_rc = self._mapper.neighbor_rc(current_rc, direction)
                if next_rc is None or next_rc in came_from:
                    continue
                came_from[next_rc] = current_rc
                if next_rc == dst:
                    return self._reconstruct(came_from, dst)
                queue.append(next_rc)

        return []

    @staticmethod
    def _reconstruct(
        came_from: Dict[CellRC, Optional[CellRC]], dst: CellRC
    ) -> List[CellRC]:
        """沿 came_from 链从终点回溯到起点，再反转为起点到终点。

        :param came_from: 每格的来路格映射，起点映射为 ``None``。
        :param dst: 终点格坐标，必在 ``came_from`` 内。
        :returns: 含起点与终点的格序列。
        """
        path: List[CellRC] = []
        current: Optional[CellRC] = dst
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def path_directions(self, path: Sequence[CellRC]) -> List[Heading]:
        """把格序列转成方向序列，供运动控制逐步执行。

        :param path: 相邻格构成的序列，长度为 1 时返回空列表。
        :returns: 长度比路径少 1 的方向列表。
        """
        return [
            self._direction_between(path[index], path[index + 1])
            for index in range(len(path) - 1)
        ]

    def return_home_directions(self) -> List[Heading]:
        """规划从当前位置到出口的方向序列。

        :returns: 方向列表，当前位置即出口时为空列表。
        """
        path = self.shortest_path(self._mapper.robot_rc, self.exit_rc)
        return self.path_directions(path)
