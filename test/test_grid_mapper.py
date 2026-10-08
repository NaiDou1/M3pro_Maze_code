"""7x7 网格拓扑的单元测试，纯算法，不依赖 ROS 运行时。

运行::

    source /opt/ros/humble/setup.bash
    cd ~/yahboomcar_ws/src/maze_explorer
    python3 -m pytest test/test_grid_mapper.py -v
"""

import math

import pytest

from maze_explorer.grid_mapper import (
    DIRECTIONS,
    LEFT,
    OPPOSITE,
    RIGHT,
    GridMapper,
    angle_diff,
)


@pytest.fixture()
def mapper():
    """提供 7x7 网格，格宽 0.40 m，开口判定半宽 0.55 m。"""
    return GridMapper(grid_size=7, cell_size=0.40, opening_min_range=0.55)


# ------------------------------------------------------------ 坐标与朝向

@pytest.mark.parametrize('rc', [(0, 0), (3, 4), (6, 6), (2, 5)])
def test_world_grid_roundtrip(mapper, rc):
    """格坐标与世界坐标互转无损，覆盖四角与中间格。"""
    x, y = mapper.grid_to_world(rc)
    assert mapper.world_to_grid(x, y) == rc


def test_grid_to_world_origin_and_axes(mapper):
    """世界坐标原点在入口格中心，列增则 x 增，行增则 y 减。"""
    assert mapper.grid_to_world((0, 0)) == (0.0, 0.0)
    # c 增大 -> x 增大；r 增大 -> y 减小
    assert mapper.grid_to_world((0, 1)) == pytest.approx((0.4, 0.0))
    assert mapper.grid_to_world((1, 0)) == pytest.approx((0.0, -0.4))


def test_heading_yaw_values(mapper):
    """方向到 yaw 的取值固定：东 0、北正二分之 π、西 π、南负二分之 π。"""
    assert mapper.heading_yaw('E') == pytest.approx(0.0)
    assert mapper.heading_yaw('N') == pytest.approx(math.pi / 2)
    assert mapper.heading_yaw('W') == pytest.approx(math.pi)
    assert mapper.heading_yaw('S') == pytest.approx(-math.pi / 2)


@pytest.mark.parametrize('heading', ['N', 'E', 'S', 'W'])
def test_yaw_heading_roundtrip(mapper, heading):
    """四个方向下 yaw 与方向互转的结果都往返一致。"""
    assert mapper.yaw_to_heading(mapper.heading_yaw(heading)) == heading


def test_yaw_to_heading_tolerates_noise(mapper):
    """偏离正北 20 度仍归类为北，映射需容忍噪声。"""
    # 偏离正北 20 度仍应判为 N
    assert mapper.yaw_to_heading(math.pi / 2 + math.radians(20.0)) == 'N'


# ---------------------------------------------------------------- 转向关系

def test_left_right_tables_are_consistent():
    """左转表、右转表与对向表须两两自洽，否则转向步数算错。"""
    for heading in DIRECTIONS:
        assert LEFT[LEFT[heading]] == OPPOSITE[heading]
        assert RIGHT[RIGHT[heading]] == OPPOSITE[heading]
        assert LEFT[heading] == RIGHT[OPPOSITE[heading]]


@pytest.mark.parametrize(
    'cur,target,expected',
    [
        ('E', 'N', 1),    # 左转一步
        ('E', 'S', -1),   # 右转一步
        ('E', 'W', 2),    # 掉头
        ('E', 'E', 0),
        ('N', 'W', 1),
        ('S', 'N', 2),
    ],
)
def test_turn_direction_steps(mapper, cur, target, expected):
    """转向步数取最短方向，掉头为 2，同向为 0。"""
    assert mapper.turn_direction(cur, target) == expected


def test_angle_diff_wraps_at_pi():
    """角度差绕回负 π 到正 π，跨边界不判成接近一整圈。"""
    assert angle_diff(math.radians(179), math.radians(-179)) == pytest.approx(
        math.radians(-2), abs=1e-6
    )


# ------------------------------------------------------------------ 观测写入

def test_observe_records_direction_and_backlink(mapper):
    """一次观测同时写本格开口与邻格的反向开口。"""
    mapper.observe((3, 3), 'E', front_open=True, left_open=True, right_open=False)

    cell = mapper.cell(3, 3)
    assert cell.open == {'E', 'N'}                 # 前方为 E，左转一步为 N
    assert 'W' in mapper.cell(3, 4).open           # 邻格记录的是反向开口
    assert 'S' in mapper.cell(2, 3).open
    assert mapper.cell(3, 2).open == set()         # 右方未开口
    assert mapper.cell(4, 3).open == set()         # 后方未探测


def test_observe_ignores_out_of_bounds_openings(mapper):
    """界外方向不得写入开口，否则会规划出格。"""
    # 位于角落朝北，左侧为界外，不应产生开口
    mapper.observe((0, 0), 'N', front_open=True, left_open=True, right_open=True)

    cell = mapper.cell(0, 0)
    assert 'N' not in cell.open    # 越界
    assert 'W' not in cell.open    # 越界，北的左邻为 W 即界外
    assert 'E' in cell.open        # 北的右邻为 E


def test_observe_marks_exhausted_when_neighbors_known(mapper):
    """四邻已知且均无开口时本格标记为走尽。"""
    # 把 3 行 3 列四周都探明为墙：四邻格标记为已访问且无开口
    for direction in DIRECTIONS:
        nrc = mapper.neighbor_rc((3, 3), direction)
        mapper.cell(*nrc).visited = True

    mapper.observe((3, 3), 'E', front_open=False, left_open=False, right_open=False)
    assert mapper.cell(3, 3).exhausted


def test_not_exhausted_while_unknown_neighbor_remains(mapper):
    """仍有未知邻格时不判走尽，避免过早回溯。"""
    mapper.observe((3, 3), 'E', front_open=False, left_open=False, right_open=False)
    assert not mapper.cell(3, 3).exhausted


# -------------------------------------------------------------- DFS 相关查询

def test_unvisited_openings_reports_only_unvisited(mapper):
    """未访问开口查询须剔除已访问邻格，供 DFS 选路。"""
    mapper.observe((3, 3), 'E', front_open=True, left_open=True, right_open=False)
    assert set(mapper.unvisited_openings((3, 3))) == {'E', 'N'}

    mapper.cell(3, 4).visited = True
    assert mapper.unvisited_openings((3, 3)) == ['N']


def test_mark_arrival_and_depth(mapper):
    """到达即更新机器人位置与朝向，深度按父格加一。"""
    mapper.mark_arrival((0, 1), 'E')
    assert mapper.cell(0, 1).visited
    assert mapper.robot_rc == (0, 1)
    assert mapper.robot_heading == 'E'

    mapper.set_depth((0, 1), (0, 0))
    assert mapper.cell(0, 1).depth == 1


def test_all_explored_only_after_full_visit(mapper):
    """仅当全部格都访问过才判定探索完成。"""
    assert mapper.visited_count() == 1        # 入口默认已访问
    assert not mapper.all_explored()

    for rc in mapper.cells:
        mapper.cells[rc].visited = True
    assert mapper.all_explored()
