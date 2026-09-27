"""7x7 网格拓扑的单元测试（纯算法，不依赖 ROS 运行时）。

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
    return GridMapper(grid_size=7, cell_size=0.40, opening_min_range=0.55)


# ------------------------------------------------------------ 坐标与朝向

@pytest.mark.parametrize('rc', [(0, 0), (3, 4), (6, 6), (2, 5)])
def test_world_grid_roundtrip(mapper, rc):
    x, y = mapper.grid_to_world(rc)
    assert mapper.world_to_grid(x, y) == rc


def test_grid_to_world_origin_and_axes(mapper):
    assert mapper.grid_to_world((0, 0)) == (0.0, 0.0)
    # c 增大 -> x 增大；r 增大 -> y 减小
    assert mapper.grid_to_world((0, 1)) == pytest.approx((0.4, 0.0))
    assert mapper.grid_to_world((1, 0)) == pytest.approx((0.0, -0.4))


def test_heading_yaw_values(mapper):
    assert mapper.heading_yaw('E') == pytest.approx(0.0)
    assert mapper.heading_yaw('N') == pytest.approx(math.pi / 2)
    assert mapper.heading_yaw('W') == pytest.approx(math.pi)
    assert mapper.heading_yaw('S') == pytest.approx(-math.pi / 2)


@pytest.mark.parametrize('heading', ['N', 'E', 'S', 'W'])
def test_yaw_heading_roundtrip(mapper, heading):
    assert mapper.yaw_to_heading(mapper.heading_yaw(heading)) == heading


def test_yaw_to_heading_tolerates_noise(mapper):
    # 偏离正北 20 度仍应判为 N
    assert mapper.yaw_to_heading(math.pi / 2 + math.radians(20.0)) == 'N'


# ---------------------------------------------------------------- 转向关系

def test_left_right_tables_are_consistent():
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
    assert mapper.turn_direction(cur, target) == expected


def test_angle_diff_wraps_at_pi():
    assert angle_diff(math.radians(179), math.radians(-179)) == pytest.approx(
        math.radians(-2), abs=1e-6
    )


# ------------------------------------------------------------------ 观测写入

def test_observe_records_direction_and_backlink(mapper):
    mapper.observe((3, 3), 'E', front_open=True, left_open=True, right_open=False)

    cell = mapper.cell(3, 3)
    assert cell.open == {'E', 'N'}                 # 前=E，左=LEFT['E']='N'
    assert 'W' in mapper.cell(3, 4).open           # 邻格记录的是反向开口
    assert 'S' in mapper.cell(2, 3).open
    assert mapper.cell(3, 2).open == set()         # 右方未开口
    assert mapper.cell(4, 3).open == set()         # 后方未探测


def test_observe_ignores_out_of_bounds_openings(mapper):
    # 位于角落朝北，左侧为界外，不应产生开口
    mapper.observe((0, 0), 'N', front_open=True, left_open=True, right_open=True)

    cell = mapper.cell(0, 0)
    assert 'N' not in cell.open    # 越界
    assert 'W' not in cell.open    # 越界（LEFT['N']='W'）
    assert 'E' in cell.open        # RIGHT['N']='E'


def test_observe_marks_exhausted_when_neighbors_known(mapper):
    # 把 (3,3) 四周都探明为墙：四邻格标记为已访问且无开口
    for direction in DIRECTIONS:
        nrc = mapper.neighbor_rc((3, 3), direction)
        mapper.cell(*nrc).visited = True

    mapper.observe((3, 3), 'E', front_open=False, left_open=False, right_open=False)
    assert mapper.cell(3, 3).exhausted


def test_not_exhausted_while_unknown_neighbor_remains(mapper):
    mapper.observe((3, 3), 'E', front_open=False, left_open=False, right_open=False)
    assert not mapper.cell(3, 3).exhausted


# -------------------------------------------------------------- DFS 相关查询

def test_unvisited_openings_reports_only_unvisited(mapper):
    mapper.observe((3, 3), 'E', front_open=True, left_open=True, right_open=False)
    assert set(mapper.unvisited_openings((3, 3))) == {'E', 'N'}

    mapper.cell(3, 4).visited = True
    assert mapper.unvisited_openings((3, 3)) == ['N']


def test_mark_arrival_and_depth(mapper):
    mapper.mark_arrival((0, 1), 'E')
    assert mapper.cell(0, 1).visited
    assert mapper.robot_rc == (0, 1)
    assert mapper.robot_heading == 'E'

    mapper.set_depth((0, 1), (0, 0))
    assert mapper.cell(0, 1).depth == 1


def test_all_explored_only_after_full_visit(mapper):
    assert mapper.visited_count() == 1        # 入口默认已访问
    assert not mapper.all_explored()

    for rc in mapper.cells:
        mapper.cells[rc].visited = True
    assert mapper.all_explored()
