"""DFS 探索决策与"先深后浅"目标调度。

迷宫是以入口为根的**树形结构**，因此用 DFS 即可覆盖全部通道：每到一个格子就
观测路口，优先沿未访问的开口深入；无未访问分支时回溯到最近的仍有分支的格。

同时维护方块目标队列。题目按"漏取方块的深度"计罚时，故补抓顺序按 **depth
降序** —— 万一时间不足，被放弃的也是浅层低罚时目标。

纯算法实现，不依赖 ROS，可直接单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from maze_explorer.grid_mapper import DIRECTIONS, CellRC, GridMapper

#: 方块颜色 -> 目标状态
PENDING = 'pending'
DONE = 'done'
FAILED = 'failed'


@dataclass
class BlockTarget:
    """一个待收集 / 已收集的方块。"""

    color: str
    cell_rc: CellRC
    depth: int
    #: 停车前测得的 base_link 坐标 (x, y, z)，米；未标定外参时为 None
    pos_base: Optional[Tuple[float, float, float]] = None
    #: 由角点估计的方块朝向（弧度），供 joint5 使用
    yaw: Optional[float] = None
    state: str = PENDING

    def key(self) -> Tuple[str, CellRC]:
        return self.color, self.cell_rc


@dataclass
class Decision:
    """探索决策。"""

    #: 'advance' 前进 | 'backtrack' 回溯 | 'finish' 探索完成
    kind: str
    #: advance / backtrack 时的目标网格方向
    direction: Optional[str] = None
    target_rc: Optional[CellRC] = None

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f'{self.kind}({self.direction} -> {self.target_rc})'


class DfsPlanner:
    """基于网格拓扑的 DFS 决策器。"""

    def __init__(self, mapper: GridMapper, exit_rc: Optional[CellRC] = None) -> None:
        self._mapper = mapper
        #: 出口格；默认与入口同格（题目中入口出口分列两侧，需现场配置）
        self.exit_rc: CellRC = exit_rc if exit_rc is not None else mapper.origin_rc
        #: 从入口到当前位置的路径（栈），用于回溯
        self.path: List[CellRC] = [mapper.origin_rc]
        #: 已登记的目标
        self.targets: Dict[Tuple[str, CellRC], BlockTarget] = {}

    # ------------------------------------------------------------------ 决策

    def decide(self) -> Decision:
        """基于当前所在格给出下一步决策（不修改内部状态）。"""
        cur = self._mapper.robot_rc
        candidates = self._mapper.unvisited_openings(cur)

        if candidates:
            direction = self._pick_direction(cur, candidates)
            return Decision('advance', direction, self._mapper.neighbor_rc(cur, direction))

        # 无未访问分支：尝试回溯
        if len(self.path) > 1:
            prev = self.path[-2]
            direction = self._direction_between(cur, prev)
            return Decision('backtrack', direction, prev)

        return Decision('finish')

    def commit(self, decision: Decision) -> None:
        """在动作**成功执行后**提交状态变更，保证路径与实车一致。"""
        if decision.kind == 'advance' and decision.target_rc is not None:
            self.path.append(decision.target_rc)
        elif decision.kind == 'backtrack':
            if len(self.path) > 1:
                self.path.pop()

    def _pick_direction(self, cur: CellRC, candidates: Sequence[str]) -> str:
        """在多个未访问开口中选一个。

        优先选与当前朝向一致的方向，减少一次 90 度转向；其次按固定顺序，
        保证决策可复现（便于回放调试）。
        """
        heading = self._mapper.robot_heading
        if heading in candidates:
            return heading
        # 其次选只差一次转向的
        for direction in candidates:
            if self._mapper.turn_direction(heading, direction) in (1, -1):
                return direction
        return candidates[0]

    def _direction_between(self, src: CellRC, dst: CellRC) -> str:
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
        """登记一个方块（深度取自所在格）。"""
        target = BlockTarget(
            color=color,
            cell_rc=cell_rc,
            depth=self._mapper.cell(*cell_rc).depth,
            pos_base=pos_base,
            yaw=yaw,
        )
        self.targets.setdefault(target.key(), target)
        return target

    def mark_done(self, target: BlockTarget) -> None:
        target.state = DONE
        if target.key() in self.targets:
            self.targets[target.key()].state = DONE

    def mark_failed(self, target: BlockTarget) -> None:
        target.state = FAILED
        if target.key() in self.targets:
            self.targets[target.key()].state = FAILED

    def pending_targets(self) -> List[BlockTarget]:
        """按 **depth 降序** 返回待补抓目标（深层优先，罚时最大）。"""
        pool = [t for t in self.targets.values() if t.state in (PENDING, FAILED)]
        return sorted(pool, key=lambda t: t.depth, reverse=True)

    def done_count(self) -> int:
        return sum(1 for t in self.targets.values() if t.state == DONE)

    def summary(self) -> str:
        return (
            f'已收集 {self.done_count()}/{len(self.targets)}，'
            f'待补 {len(self.pending_targets())}，已探索 '
            f'{self._mapper.visited_count()}/{self._mapper.grid_size ** 2} 格'
        )

    # ---------------------------------------------------------- 返航路径规划

    def shortest_path(self, src: CellRC, dst: CellRC) -> List[CellRC]:
        """在已探明拓扑上做 BFS 最短路，返回含起点与终点的格序列。

        无通路时返回空列表。
        """
        if src == dst:
            return [src]

        queue: List[CellRC] = [src]
        came_from: Dict[CellRC, Optional[CellRC]] = {src: None}

        while queue:
            cur = queue.pop(0)
            for direction in self._mapper.known_neighbors(cur):
                nxt = self._mapper.neighbor_rc(cur, direction)
                if nxt is None or nxt in came_from:
                    continue
                came_from[nxt] = cur
                if nxt == dst:
                    return self._reconstruct(came_from, dst)
                queue.append(nxt)

        return []

    @staticmethod
    def _reconstruct(
        came_from: Dict[CellRC, Optional[CellRC]], dst: CellRC
    ) -> List[CellRC]:
        path: List[CellRC] = []
        cur: Optional[CellRC] = dst
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def path_directions(self, path: Sequence[CellRC]) -> List[str]:
        """把格序列转成方向序列，供运动控制逐步执行。"""
        return [
            self._direction_between(path[i], path[i + 1])
            for i in range(len(path) - 1)
        ]

    def return_home_directions(self) -> List[str]:
        """规划从当前位置返回出口的方向序列。"""
        path = self.shortest_path(self._mapper.robot_rc, self.exit_rc)
        return self.path_directions(path)
