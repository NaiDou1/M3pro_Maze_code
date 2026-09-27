"""入口 B：现场标定。

三种标定项，**按依赖顺序**排列（也是现场推荐的操作顺序）：

====  ============  ================================================
编号   模式           作用
====  ============  ================================================
1     ``hsv``        黑线与四色 HSV 阈值。必须先做——否则巡线根本看不到线
2     ``line_pose``  机械臂巡线姿态，使相机同时看到地面黑线与前方通道
3     ``motion``     走格距离与 90 度转角系数，标定里程计缩放
====  ============  ================================================

用法::

    ros2 run maze_explorer maze_calib              # 打印菜单，交互选择
    ros2 run maze_explorer maze_calib hsv          # 直达某项
    ros2 run maze_explorer maze_calib motion

.. note::
   本模块是 ``calibration_tool`` 的薄壳：只负责「选哪一项」与「选完提示下一步」，
   具体标定流程完全复用已有实现，避免行为分叉。标定结果由它写回**源码目录**
   的 ``config/`` 下（而非 install 副本），否则下次 ``colcon build`` 会覆盖。
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: 菜单编号 -> (模式名, 说明)
CALIB_MENU: Dict[str, Tuple[str, str]] = {
    '1': ('hsv', '黑线与四色 HSV 阈值（最先做）'),
    '2': ('line_pose', '机械臂巡线姿态'),
    '3': ('motion', '走格距离与 90 度转角系数'),
}

#: 合法的模式名
MODES: Tuple[str, ...] = ('hsv', 'line_pose', 'motion')

#: 各模式标定完成后的下一步建议
NEXT_STEPS: Dict[str, str] = {
    'hsv': (
        '已写入 config/hsv_params.yaml\n'
        '    下一步建议：再跑一次 maze_calib，选择 2（巡线姿态）'
    ),
    'line_pose': (
        '已写入 config/arm_poses.yaml\n'
        '    下一步建议：再跑一次 maze_calib，选择 3（走格与转角系数）'
    ),
    'motion': (
        '请把上面打印的建议系数填进 config/maze_params.yaml 的\n'
        '    odom_linear_scale_correction 与 odom_angular_scale_correction，\n'
        '    然后运行 maze_run --check-only 确认硬件就绪后再正式开跑'
    ),
}


def print_menu() -> None:
    print()
    print('=' * 64)
    print('  迷宫探索系统 · 标定工具')
    print('=' * 64)
    print('  请选择标定项（建议按 1 → 2 → 3 的顺序，每项单独跑一次）：')
    print()
    for key, (mode, desc) in CALIB_MENU.items():
        print(f'    {key}) {desc}    [{mode}]')
    print()
    print('  提示：标定需要相机/机械臂已启动；HSV 与姿态标定需要图形界面。')
    print('=' * 64)


def resolve_mode(argv: Sequence[str]) -> Optional[str]:
    """解析出标定模式：支持命令行直达，或打印菜单交互选择。

    返回 ``None`` 表示用户取消。
    """
    for arg in argv:
        if arg in MODES:
            return arg

    print_menu()
    while True:
        try:
            choice = input('  请选择 [1/2/3，q 退出] > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice in ('q', 'quit', 'exit', ''):
            return None
        if choice in CALIB_MENU:
            return CALIB_MENU[choice][0]
        # 也允许直接输入模式名
        if choice in MODES:
            return choice
        print('  输入无效，请输入 1 / 2 / 3，或模式名 hsv / line_pose / motion')


def suggest_next(mode: str) -> str:
    """返回该模式标定完成后的下一步建议。"""
    return NEXT_STEPS.get(mode, '')


def main(argv: Optional[Sequence[str]] = None) -> int:
    args: List[str] = list(sys.argv[1:] if argv is None else argv)

    mode = resolve_mode(args)
    if mode is None:
        print('[maze_calib] 已取消')
        return 0

    # 复用 calibration_tool 的实现，仅以参数形式指定模式
    from maze_explorer.calibration_tool import main as calibration_main

    print(f'[maze_calib] 进入标定模式：{mode}（Ctrl-C 可随时退出）')
    calibration_main(['--ros-args', '-p', f'mode:={mode}'])

    print()
    print('=' * 64)
    print(f'  标定结束：{CALIB_MENU.get(mode, (mode, mode))[1]}')
    hint = suggest_next(mode)
    if hint:
        print(f'  {hint}')
    print('=' * 64)
    return 0


if __name__ == '__main__':
    sys.exit(main())
