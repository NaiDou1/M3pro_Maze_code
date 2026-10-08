"""7x7 网格拓扑建图。

场地 2.8 m x 2.8 m 恰好划分为 7x7 格，格心间距 0.40 m 与通道宽相等，因此不采用
栅格 SLAM——小场地内麦轮打滑会让栅格图糊掉，而是直接维护格与连通性构成的拓扑图，
与题目的树形迷宫抽象天然一致。

坐标约定
--------
网格行号 row、列号 col 与世界坐标的换算::

    x = col * cell_size
    y = -row * cell_size

row 增大为向南，y 随之减小。四个方向对应的世界 yaw 为::

    E：col 增大，yaw 0
    N：row 减小，yaw 正 90 度
    W：col 减小，yaw 正负 180 度
    S：row 增大，yaw 负 90 度

纯算法模块，不依赖 ROS，可离线单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node

from maze_explorer._compat import StrEnum

#: 网格坐标二元组，row 在前 col 在后，取值区间 0 到 grid_size - 1
CellRC = Tuple[int, int]


class Heading(StrEnum):
    """网格方向，取值 N、E、S、W 四者之一，其他值构造时抛 ``ValueError``。

    成员与同名裸字符串在比较、字典键与集合成员上等价，调用方可互换使用。
    """

    NORTH = 'N'
    EAST = 'E'
    SOUTH = 'S'
    WEST = 'W'


#: 方向到格偏移 dr、dc 与世界 yaw 弧度的映射，键序为东、北、西、南
DIRECTIONS: Dict[Heading, Tuple[int, int, float]] = {
    Heading.EAST: (0, +1, 0.0),
    Heading.NORTH: (-1, 0, math.pi / 2),
    Heading.WEST: (0, -1, math.pi),
    Heading.SOUTH: (+1, 0, -math.pi / 2),
}

#: 反方向映射
OPPOSITE: Dict[Heading, Heading] = {
    Heading.NORTH: Heading.SOUTH,
    Heading.SOUTH: Heading.NORTH,
    Heading.EAST: Heading.WEST,
    Heading.WEST: Heading.EAST,
}

#: 左转 90 度后的方向
LEFT: Dict[Heading, Heading] = {
    Heading.NORTH: Heading.WEST,
    Heading.WEST: Heading.SOUTH,
    Heading.SOUTH: Heading.EAST,
    Heading.EAST: Heading.NORTH,
}

#: 右转 90 度后的方向
RIGHT: Dict[Heading, Heading] = {
    Heading.NORTH: Heading.EAST,
    Heading.EAST: Heading.SOUTH,
    Heading.SOUTH: Heading.WEST,
    Heading.WEST: Heading.NORTH,
}

#: 逆时针方向序，yaw 递增，用于计算转向步数
COUNTERCLOCKWISE: Tuple[Heading, ...] = (
    Heading.EAST,
    Heading.NORTH,
    Heading.WEST,
    Heading.SOUTH,
)


def angle_diff(a: float, b: float) -> float:
    """返回归一化到负 π 到正 π 区间的角度差 a 减 b，单位 rad。

    :param a: 被减角，单位 rad。
    :param b: 减角，单位 rad。
    :returns: 落在负 π 到正 π 闭区间内的差角，符号表示 a 相对 b 的转向。
    """
    return math.atan2(math.sin(a - b), math.cos(a - b))


@dataclass
class Cell:
    """单个网格单元。

    :ivar row: 行号，取值区间 0 到 grid_size - 1，增大方向为南。
    :ivar col: 列号，取值区间 0 到 grid_size - 1，增大方向为东。
    :ivar depth: 树深度，入口为 0，先深后浅策略的排序依据。
    :ivar open: 已确认开口的方向子集，元素取自 ``Heading``。
    :ivar visited: 是否已抵达过。
    :ivar exhausted: 四个方向是否都已有结论，用于判断探索是否穷尽。
    """

    row: int
    col: int
    #: 树深度，入口为 0
    depth: int = 0
    #: 已确认开口的方向子集
    open: Set[Heading] = field(default_factory=set)
    #: 是否已抵达过
    visited: bool = False
    #: 四个方向是否都已有结论，无论开口与否
    exhausted: bool = False

    @property
    def rc(self) -> CellRC:
        """返回行号与列号组成的网格坐标。"""
        return self.row, self.col


class GridMapper:
    """纯算法拓扑图，不依赖 ROS。

    持有 ``grid_size`` 平方个 :class:`Cell`，全部通过 :meth:`cell` 按坐标访问。
    """

    def __init__(
        self,
        grid_size: int = 7,
        cell_size: float = 0.40,
        opening_min_range: float = 0.55,
        origin_rc: CellRC = (0, 0),
    ) -> None:
        """创建全未访问的空拓扑，并把入口格标为树根。

        :param grid_size: 每边格数，场地 2.8 m 除以 0.40 m 得 7。
        :param cell_size: 格心间距，单位 m，默认与通道宽 0.40 m 相等。
        :param opening_min_range: 开口判定阈值，单位 m。激光在该方向的最近距离
            超过它即判为开口；侧墙实测在 0.40 m 附近，故该值须大于 0.40。
        :param origin_rc: 入口格坐标，深度 0，由 launch 的 ``origin_rc`` 注入。
        """
        self.grid_size = int(grid_size)
        self.cell_size = float(cell_size)
        self.opening_min_range = float(opening_min_range)

        self.cells: Dict[CellRC, Cell] = {
            (row, col): Cell(row=row, col=col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
        }
        #: 入口格坐标即树根，深度 0
        self.origin_rc: CellRC = origin_rc
        self.cell(*origin_rc).depth = 0
        self.cell(*origin_rc).visited = True

        #: 机器人当前报告位置，由上层在每次到位后更新
        self.robot_rc: CellRC = origin_rc
        #: 机器人当前报告朝向，同由上层在每次转向前更新
        self.robot_heading: Heading = Heading.EAST

    @property
    def robot_heading(self) -> Heading:
        """机器人当前报告朝向，取值见 ``Heading``。"""
        return self._robot_heading

    @robot_heading.setter
    def robot_heading(self, value: Heading) -> None:
        """写入朝向，按值构造为 ``Heading``，非法值抛 ``ValueError``。

        :param value: 朝向，可为 ``Heading`` 成员或同值裸字符串。
        """
        self._robot_heading = Heading(value)

    # -------------------------------------------------------------- 基础访问

    def in_bounds(self, row: int, col: int) -> bool:
        """判断网格坐标是否落在场地内。

        :param row: 行号。
        :param col: 列号。
        :returns: 两个坐标均满足不小于 0 且小于 ``grid_size`` 时为真。
        """
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def cell(self, row: int, col: int) -> Cell:
        """按网格坐标取单元格，坐标越界由字典键抛 ``KeyError``。

        :param row: 行号，取值区间 0 到 ``grid_size`` - 1。
        :param col: 列号，取值区间 0 到 ``grid_size`` - 1。
        """
        return self.cells[(row, col)]

    def neighbor_rc(self, rc: CellRC, direction: Heading) -> Optional[CellRC]:
        """返回沿 ``direction`` 的相邻格坐标，越界返回 ``None``。

        :param rc: 当前格坐标。
        :param direction: 网格方向，取值见 ``Heading``。
        :returns: 相邻格坐标；目标格在场地外时为 ``None``。
        """
        dr, dc, _ = DIRECTIONS[direction]
        row, col = rc[0] + dr, rc[1] + dc
        return (row, col) if self.in_bounds(row, col) else None

    def open_dirs(self, rc: CellRC) -> Set[Heading]:
        """返回该格已确认开口方向的副本，供调用方自由增删。

        :param rc: 目标格坐标。
        """
        return set(self.cells[rc].open)

    def known_neighbors(self, rc: CellRC) -> List[Heading]:
        """返回该格已确认且目标在场地内的通行方向。

        :param rc: 目标格坐标。
        """
        return [d for d in self.cells[rc].open if self.neighbor_rc(rc, d) is not None]

    def unvisited_openings(self, rc: CellRC) -> List[Heading]:
        """返回相邻且尚未访问的开口方向，即 DFS 的前进候选。

        :param rc: 目标格坐标。
        :returns: 方向列表，元素取自 ``Heading``，无候选时为空列表。
        """
        result = []
        for direction in self.known_neighbors(rc):
            nrc = self.neighbor_rc(rc, direction)
            if nrc is not None and not self.cells[nrc].visited:
                result.append(direction)
        return result

    # ---------------------------------------------------------- 扫描结果更新

    def observe(
        self,
        rc: CellRC,
        heading: Heading,
        front_open: bool,
        left_open: bool,
        right_open: bool,
    ) -> None:
        """写入一次路口观测，并双向回写邻格连通性。

        :param rc: 观测所在的格坐标。
        :param heading: 观测时车体朝向，取值见 ``Heading``。
        :param front_open: 车体正前方是否有开口。
        :param left_open: 车体左方是否有开口。
        :param right_open: 车体右方是否有开口。
        """
        cell = self.cells[rc]
        # 布尔值不能作字典键：True 与 False 会互相覆盖，只剩一个方向被记录
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
            # 连通性写回邻格，避免再次进入时重复探测
            self.cells[nrc].open.add(OPPOSITE[direction])

        # 边界方向视为墙，不算未探索；四个方向都有结论才算穷尽
        cell.exhausted = self._is_fully_examined(cell)

    def mark_arrival(self, rc: CellRC, heading: Heading) -> None:
        """标记抵达某格：置已访问、更新报告位置与朝向。

        :param rc: 抵达格坐标。
        :param heading: 抵达时机体朝向，取值见 ``Heading``。
        """
        cell = self.cells[rc]
        cell.visited = True
        self.robot_rc = rc
        self.robot_heading = heading

    def set_depth(self, rc: CellRC, parent_rc: CellRC) -> None:
        """按父子关系设置树深度，取父格深度加 1。

        :param rc: 子格坐标，须尚未设置深度。
        :param parent_rc: 父格坐标。
        """
        self.cells[rc].depth = self.cells[parent_rc].depth + 1

    def _is_fully_examined(self, cell: Cell) -> bool:
        """判断四个方向是否都已有结论。

        树形迷宫不记录显式墙，故以开口集合加已探明边界近似：方向越界即为墙，
        方向无开口且邻格已访问即探明为墙；仍存在无开口且邻格未访问的方向时
        返回假，表示还有未知分支。

        :param cell: 待判断单元格。
        :returns: 无剩余未知分支时为真。
        """
        for direction in DIRECTIONS:
            nrc = self.neighbor_rc(cell.rc, direction)
            if nrc is None:
                continue  # 越界方向必为墙
            if direction not in cell.open and self.cells[nrc].visited:
                continue  # 已探明为墙
            if direction not in cell.open and not self.cells[nrc].visited:
                return False  # 仍有未知分支，可能是墙也可能是路
        return True

    # ------------------------------------------------------------ 坐标换算

    def grid_to_world(self, rc: CellRC) -> Tuple[float, float]:
        """网格坐标换算到世界坐标，即 map 系下的格心位置。

        :param rc: 网格坐标。
        :returns: x 与 y 两个分量，单位 m。
        """
        row, col = rc
        return col * self.cell_size, -row * self.cell_size

    def world_to_grid(self, x: float, y: float) -> CellRC:
        """世界坐标换算到网格坐标，按最近格心取整。

        :param x: 世界横坐标，单位 m。
        :param y: 世界纵坐标，单位 m。
        :returns: 网格坐标；越出场地时返回的坐标不保证在界内，由调用方判界。
        """
        return int(round(-y / self.cell_size)), int(round(x / self.cell_size))

    @staticmethod
    def heading_yaw(heading: Heading) -> float:
        """返回网格方向对应的世界 yaw，单位 rad。

        :param heading: 网格方向，取值见 ``Heading``。
        :returns: 弧度值，取值区间负 π 到正 π。
        """
        return DIRECTIONS[heading][2]

    @staticmethod
    def yaw_to_heading(yaw: float) -> Heading:
        """把世界朝向角归到最接近的网格方向。

        :param yaw: 世界朝向角，单位 rad，超出负 π 到正 π 也能正确归类。
        :returns: 与该夹角最小的 ``Heading`` 成员。
        """
        return min(
            DIRECTIONS,
            key=lambda d: abs(angle_diff(DIRECTIONS[d][2], yaw)),
        )

    @staticmethod
    def turn_direction(current: Heading, target: Heading) -> int:
        """返回从 ``current`` 转到 ``target`` 所需的 90 度步数。

        :param current: 当前朝向，取值见 ``Heading``。
        :param target: 目标朝向，取值见 ``Heading``。
        :returns: 取值区间 -2 到 2；正值为左转即逆时针，负值为右转，0 为不转。
        """
        delta = (COUNTERCLOCKWISE.index(target) - COUNTERCLOCKWISE.index(current)) % 4
        if delta == 3:
            delta = -1
        return delta

    # ---------------------------------------------------------------- 统计

    def visited_count(self) -> int:
        """返回已访问格数，取值区间 0 到 ``grid_size`` 平方。"""
        return sum(1 for cell in self.cells.values() if cell.visited)

    def all_explored(self) -> bool:
        """判断全部格是否都已访问，树形迷宫下等价于探索完成。"""
        return self.visited_count() == self.grid_size * self.grid_size

    def dump(self) -> str:
        """输出便于日志查看的 ASCII 拓扑，每格以井号表示已访问加箭头表示开口。"""
        arrows = {
            Heading.NORTH: '↑',
            Heading.EAST: '→',
            Heading.SOUTH: '↓',
            Heading.WEST: '←',
        }
        lines = []
        for row in range(self.grid_size):
            cells_text = []
            for col in range(self.grid_size):
                cell = self.cells[(row, col)]
                marks = ''.join(arrows[d] for d in Heading if d in cell.open)
                tag = '#' if cell.visited else '.'
                cells_text.append(f'{tag}{marks or "-"}'.ljust(6))
            lines.append(''.join(cells_text))
        return '\n'.join(lines)


class GridMapperNode(Node):
    """独立调试节点，按周期打印拓扑快照，数据由上层填充。"""

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
        self.get_logger().info('GridMapper 就绪，仅拓扑，数据由上层填充')

    def _dump(self) -> None:
        """打印已访问格数与当前拓扑。"""
        self.get_logger().info(
            f'已访问 {self.mapper.visited_count()}/{self.mapper.grid_size ** 2}\n'
            f'{self.mapper.dump()}'
        )


def main(args: Optional[List[str]] = None) -> None:
    """调试节点入口：初始化、自旋、退出时按序关闭。

    :param args: 传给 ``rclpy.init`` 的命令行参数，取 ``None`` 时读进程参数。
    """
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
