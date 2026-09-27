"""7x7 网格拓扑建图。

场地 2.8m x 2.8m 恰好划分为 7x7 格、格心间距 0.4m（= 通道宽），因此不采用栅格
SLAM（小场地内麦轮打滑会让栅格图糊掉），而是直接维护"格—连通性"拓扑图，天然
匹配题目给出的树形迷宫抽象。

坐标约定
--------
网格 ``(r, c)`` 与世界的对应关系::

    x = c * cell_size
    y = -r * cell_size        （即 r 增大为"向南"，y 减小）

因此四个方向的世界朝向为::

    E (c+1) -> yaw   0
    N (r-1) -> yaw +90 度
    W (c-1) -> yaw 180 度
    S (r+1) -> yaw -90 度
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import rclpy
from rclpy.node import Node

CellRC = Tuple[int, int]

#: 网格方向 -> (dr, dc, 世界 yaw 弧度)
DIRECTIONS: Dict[str, Tuple[int, int, float]] = {
    'E': (0, +1, 0.0),
    'N': (-1, 0, math.pi / 2),
    'W': (0, -1, math.pi),
    'S': (+1, 0, -math.pi / 2),
}
OPPOSITE: Dict[str, str] = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
#: 左转 90 度后的朝向
LEFT: Dict[str, str] = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
#: 右转 90 度后的朝向
RIGHT: Dict[str, str] = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}


def angle_diff(a: float, b: float) -> float:
    """归一化角度差 ``a - b`` 到 [-pi, pi]。"""
    return math.atan2(math.sin(a - b), math.cos(a - b))


@dataclass
class Cell:
    """单个网格单元。"""

    r: int
    c: int
    #: 树深度（入口为 0），"先深后浅"策略的排序依据
    depth: int = 0
    #: 已确认开口的方向子集，元素取自 {"N","E","S","W"}
    open: Set[str] = field(default_factory=set)
    visited: bool = False
    #: 四个方向是否都已确认（无论开口与否），用于判断探索是否穷尽
    exhausted: bool = False

    @property
    def rc(self) -> CellRC:
        return self.r, self.c


class GridMapper:
    """纯算法拓扑图，不依赖 ROS。"""

    def __init__(
        self,
        grid_size: int = 7,
        cell_size: float = 0.40,
        opening_min_range: float = 0.55,
        origin_rc: CellRC = (0, 0),
    ) -> None:
        self.grid_size = int(grid_size)
        self.cell_size = float(cell_size)
        self.opening_min_range = float(opening_min_range)

        self.cells: Dict[CellRC, Cell] = {
            (r, c): Cell(r=r, c=c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        }
        #: 入口格（树根），深度为 0
        self.origin_rc: CellRC = origin_rc
        self.cell(*origin_rc).depth = 0
        self.cell(*origin_rc).visited = True

        #: 机器人当前报告位置（由上层在每次到位后更新）
        self.robot_rc: CellRC = origin_rc
        self.robot_heading: str = 'E'

    # -------------------------------------------------------------- 基础访问

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def cell(self, r: int, c: int) -> Cell:
        return self.cells[(r, c)]

    def neighbor_rc(self, rc: CellRC, direction: str) -> Optional[CellRC]:
        """沿 ``direction`` 的相邻格；越界返回 ``None``。"""
        dr, dc, _ = DIRECTIONS[direction]
        r, c = rc[0] + dr, rc[1] + dc
        return (r, c) if self.in_bounds(r, c) else None

    def open_dirs(self, rc: CellRC) -> Set[str]:
        """该格已确认的开口方向。"""
        return set(self.cells[rc].open)

    def known_neighbors(self, rc: CellRC) -> List[str]:
        """已确认可通行的方向。"""
        return [d for d in self.cells[rc].open if self.neighbor_rc(rc, d) is not None]

    def unvisited_openings(self, rc: CellRC) -> List[str]:
        """相邻且**尚未访问**的开口方向（DFS 的前进候选）。"""
        result = []
        for d in self.known_neighbors(rc):
            nrc = self.neighbor_rc(rc, d)
            if nrc is not None and not self.cells[nrc].visited:
                result.append(d)
        return result

    # ---------------------------------------------------------- 扫描结果更新

    def observe(
        self,
        rc: CellRC,
        heading: str,
        front_open: bool,
        left_open: bool,
        right_open: bool,
    ) -> None:
        """写入一次路口观测。

        :param rc: 观测所在的格。
        :param heading: 观测时车体朝向（网格方向）。
        :param front_open/left_open/right_open: 车体前/左/右是否有开口。
        :param rear 不参与——迷宫为树形连通，来路必然可通行，由调用方按需补。
        """
        cell = self.cells[rc]
        # 注意：不可用布尔值作字典键——True/False 会互相覆盖，导致只记录一个方向
        entries = (
            (front_open, heading),
            (left_open, LEFT[heading]),
            (right_open, RIGHT[heading]),
        )
        for is_open, direction in entries:
            if not is_open:
                continue
            nrc = self.neighbor_rc(rc, direction)
            if nrc is None:
                continue  # 边界外不存在邻格
            cell.open.add(direction)
            # 连通性也写回邻格，避免再次进入时重复探测
            self.cells[nrc].open.add(OPPOSITE[direction])

        # 边界方向视为"墙"，不算未探索；只有格内四个方向都有结论才算穷尽
        cell.exhausted = self._is_fully_examined(cell)

    def mark_arrival(self, rc: CellRC, heading: str) -> None:
        """标记抵达某格：设置深度（父深度 + 1）与已访问。"""
        cell = self.cells[rc]
        cell.visited = True
        self.robot_rc = rc
        self.robot_heading = heading

    def set_depth(self, rc: CellRC, parent_rc: CellRC) -> None:
        """按 BFS 式父子关系设置树深度。"""
        self.cells[rc].depth = self.cells[parent_rc].depth + 1

    def _is_fully_examined(self, cell: Cell) -> bool:
        """四个方向都已有结论（开口或确定是边界/墙）时视为穷尽。

        由于树形迷宫不会出现"已知为墙"的显式记录，这里以"开口集合 + 已探明
        的边界数"近似：只要四个方向的邻格都存在且都已访问过，或方向越界，
        就认为该格无剩余未知分支。
        """
        for direction in DIRECTIONS:
            nrc = self.neighbor_rc(cell.rc, direction)
            if nrc is None:
                continue  # 越界方向必为墙
            if direction not in cell.open and self.cells[nrc].visited:
                continue  # 已探明为墙
            if direction not in cell.open and not self.cells[nrc].visited:
                return False  # 仍有未知分支（可能是墙也可能是路）
        return True

    # ------------------------------------------------------------ 坐标换算

    def grid_to_world(self, rc: CellRC) -> Tuple[float, float]:
        r, c = rc
        return c * self.cell_size, -r * self.cell_size

    def world_to_grid(self, x: float, y: float) -> CellRC:
        return int(round(-y / self.cell_size)), int(round(x / self.cell_size))

    @staticmethod
    def heading_yaw(heading: str) -> float:
        return DIRECTIONS[heading][2]

    @staticmethod
    def yaw_to_heading(yaw: float) -> str:
        """把世界朝向角归到最接近的网格方向。"""
        return min(
            DIRECTIONS,
            key=lambda d: abs(angle_diff(DIRECTIONS[d][2], yaw)),
        )

    @staticmethod
    def turn_direction(current: str, target: str) -> int:
        """返回从 ``current`` 转到 ``target`` 所需的 90 度步数（-2..2）。

        正值表示左转（逆时针），负值表示右转。
        """
        order = ['E', 'N', 'W', 'S']  # 逆时针（yaw 递增）
        delta = (order.index(target) - order.index(current)) % 4
        if delta == 3:
            delta = -1
        return delta

    # ---------------------------------------------------------------- 统计

    def visited_count(self) -> int:
        return sum(1 for cell in self.cells.values() if cell.visited)

    def all_explored(self) -> bool:
        """是否所有格子都已访问（树形迷宫下等价于探索完成）。"""
        return self.visited_count() == self.grid_size * self.grid_size

    def dump(self) -> str:
        """输出便于日志查看的 ASCII 拓扑：每格用位掩码表示开口。"""
        arrows = {'N': '↑', 'E': '→', 'S': '↓', 'W': '←'}
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                cell = self.cells[(r, c)]
                marks = ''.join(arrows[d] for d in ('N', 'E', 'S', 'W') if d in cell.open)
                tag = '#' if cell.visited else '.'
                row.append(f'{tag}{marks or "-"}'.ljust(6))
            lines.append(''.join(row))
        return '\n'.join(lines)


class GridMapperNode(Node):
    """独立调试节点：打印当前拓扑快照。"""

    def __init__(self) -> None:
        super().__init__('grid_mapper')
        self.declare_parameter('grid_size', 7)
        self.declare_parameter('cell_size', 0.40)
        self.declare_parameter('opening_min_range', 0.55)
        self.declare_parameter('dump_period_sec', 10.0)

        self.mapper = GridMapper(
            grid_size=int(self.get_parameter('grid_size').value),
            cell_size=float(self.get_parameter('cell_size').value),
            opening_min_range=float(self.get_parameter('opening_min_range').value),
        )
        period = float(self.get_parameter('dump_period_sec').value)
        self.create_timer(period, self._dump)
        self.get_logger().info('GridMapper 就绪（仅拓扑，数据由上层填充）')

    def _dump(self) -> None:
        self.get_logger().info(
            f'已访问 {self.mapper.visited_count()}/{self.mapper.grid_size ** 2}\n'
            f'{self.mapper.dump()}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GridMapperNode()
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
