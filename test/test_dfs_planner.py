"""DFS 探索决策与目标调度的单元测试。"""

import pytest

from maze_explorer.dfs_planner import DfsPlanner
from maze_explorer.grid_mapper import GridMapper


@pytest.fixture()
def setup():
    mapper = GridMapper(grid_size=7, cell_size=0.40)
    planner = DfsPlanner(mapper, exit_rc=(0, 0))
    return mapper, planner


# ------------------------------------------------------------------ 决策

def test_finish_when_no_opening_known(setup):
    _, planner = setup
    decision = planner.decide()
    assert decision.kind == 'finish'


def test_advance_into_unvisited_opening(setup):
    mapper, planner = setup
    # 入口 (0,0) 朝东，前方与左方开口（左方越界会被忽略）
    mapper.observe((0, 0), 'E', front_open=True, left_open=True, right_open=False)

    decision = planner.decide()
    assert decision.kind == 'advance'
    assert decision.direction in ('E', 'N')
    assert decision.target_rc == mapper.neighbor_rc((0, 0), decision.direction)


def test_prefers_current_heading_to_avoid_turn(setup):
    mapper, planner = setup
    mapper.observe((3, 3), 'E', front_open=True, left_open=True, right_open=False)
    mapper.robot_rc = (3, 3)
    mapper.robot_heading = 'N'   # 当前朝北，北侧也开口
    planner.path = [(3, 3)]

    decision = planner.decide()
    assert decision.direction == 'N'   # 优先保持朝向


def test_commit_advance_pushes_path(setup):
    mapper, planner = setup
    mapper.observe((0, 0), 'E', front_open=True, left_open=False, right_open=False)
    decision = planner.decide()

    planner.commit(decision)
    assert planner.path == [(0, 0), (0, 1)]


def test_backtrack_when_branch_exhausted(setup):
    mapper, planner = setup
    # 走到 (0,1)，它没有其他开口
    mapper.cell(0, 0).open = {'E'}
    mapper.cell(0, 1).open = {'W'}
    mapper.cell(0, 1).visited = True
    mapper.robot_rc = (0, 1)
    planner.path = [(0, 0), (0, 1)]

    decision = planner.decide()
    assert decision.kind == 'backtrack'
    assert decision.direction == 'W'
    assert decision.target_rc == (0, 0)

    planner.commit(decision)
    assert planner.path == [(0, 0)]


def test_finish_at_root_when_everything_explored(setup):
    mapper, planner = setup
    mapper.cell(0, 0).open = {'E'}
    mapper.cell(0, 1).visited = True
    mapper.robot_rc = (0, 0)
    planner.path = [(0, 0)]

    assert planner.decide().kind == 'finish'


# -------------------------------------------------------------- 目标调度

def test_pending_targets_sorted_by_depth_desc(setup):
    mapper, planner = setup
    mapper.cell(0, 1).depth = 1
    mapper.cell(3, 3).depth = 5
    mapper.cell(2, 2).depth = 3

    planner.register_target('red', (0, 1))
    planner.register_target('blue', (3, 3))
    planner.register_target('green', (2, 2))

    depths = [t.depth for t in planner.pending_targets()]
    assert depths == [5, 3, 1]        # 深层优先，罚时最大


def test_mark_done_removes_from_pending(setup):
    _, planner = setup
    target = planner.register_target('red', (0, 1))

    planner.mark_done(target)
    assert planner.pending_targets() == []
    assert planner.done_count() == 1


def test_failed_target_stays_pending_for_retry(setup):
    _, planner = setup
    target = planner.register_target('red', (0, 1))

    planner.mark_failed(target)
    assert [t.color for t in planner.pending_targets()] == ['red']


# -------------------------------------------------------------- 路径规划

def test_shortest_path_bfs(setup):
    mapper, planner = setup
    # 构造 L 形通路: (0,0)-(0,1)-(0,2)-(1,2)
    mapper.cell(0, 0).open = {'E'}
    mapper.cell(0, 1).open = {'W', 'E'}
    mapper.cell(0, 2).open = {'W', 'S'}
    mapper.cell(1, 2).open = {'N'}

    path = planner.shortest_path((0, 0), (1, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2)]


def test_shortest_path_returns_empty_when_unreachable(setup):
    mapper, planner = setup
    mapper.cell(0, 0).open = {'E'}

    assert planner.shortest_path((0, 0), (5, 5)) == []


def test_path_directions(setup):
    _, planner = setup
    directions = planner.path_directions([(0, 0), (0, 1), (0, 2), (1, 2)])
    assert directions == ['E', 'E', 'S']


def test_same_cell_path_is_singleton(setup):
    _, planner = setup
    assert planner.shortest_path((2, 2), (2, 2)) == [(2, 2)]
