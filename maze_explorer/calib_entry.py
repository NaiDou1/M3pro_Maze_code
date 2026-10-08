"""入口 B：现场标定。

四种标定项，按依赖顺序排列，这也是现场推荐的操作顺序：

====  ============  ================================================
编号   模式           作用
====  ============  ================================================
1     ``hsv``        黑线与四色 HSV 阈值。必须先做，否则巡线看不到线
2     ``line_pose``  机械臂巡线姿态，使相机同时看到地面黑线与前方通道
3     ``motion``     走格距离与 90 度转角系数，标定里程计缩放
4     ``follow``     纯巡线测试，不需要挡板，验证能否跟线
====  ============  ================================================

用法::

    ros2 run maze_explorer maze_calib              # 打印菜单，交互选择
    ros2 run maze_explorer maze_calib hsv          # 直达某项
    ros2 run maze_explorer maze_calib motion

.. note::
   本模块是 ``calibration_tool`` 的薄壳：只负责选哪一项与选完提示下一步，具体
   标定流程完全复用已有实现，避免行为分叉。标定结果由它写回源码目录的
   ``config/`` 而非 install 副本，否则下次 ``colcon build`` 会覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from maze_explorer.calibration_tool import CalibMode

#: 菜单编号 -> 模式与说明的二元组
CALIB_MENU: Dict[str, Tuple[CalibMode, str]] = {
    '1': (CalibMode.HSV, '黑线与四色 HSV 阈值，最先做'),
    '2': (CalibMode.LINE_POSE, '机械臂巡线姿态'),
    '3': (CalibMode.MOTION, '走格距离与 90 度转角系数'),
    '4': (CalibMode.FOLLOW, '纯巡线测试，不需要挡板，验证能否跟线'),
}

#: 合法的模式名，取值即 ``CalibMode`` 全部成员
MODES: Tuple[CalibMode, ...] = tuple(CalibMode)

#: 需要图形界面的模式，即依赖 cv2.imshow 的项；motion 与 follow 不需要
GUI_MODES: Tuple[CalibMode, ...] = (CalibMode.HSV, CalibMode.LINE_POSE)

#: 各模式标定完成后的下一步建议，键为标定模式
NEXT_STEPS: Dict[CalibMode, str] = {
    CalibMode.HSV: (
        '已写入 config/hsv_params.yaml\n'
        '    下一步建议：再跑一次 maze_calib，选择 2 即巡线姿态'
    ),
    CalibMode.LINE_POSE: (
        '已写入 config/arm_poses.yaml\n'
        '    下一步建议：再跑一次 maze_calib，选择 3 即走格与转角系数'
    ),
    CalibMode.MOTION: (
        '请把上面打印的建议系数填进 config/maze_params.yaml 的\n'
        '    odom_linear_scale_correction 与 odom_angular_scale_correction，\n'
        '    然后运行 maze_run --check-only 确认硬件就绪后再正式开跑'
    ),
    CalibMode.FOLLOW: (
        '巡线测试用于没有挡板时验证跟线：它只靠相机，不做激光拓扑。\n'
        '    注意正式任务 maze_run 依赖激光测墙判开口，没有挡板会乱转向'
    ),
}


def print_menu() -> None:
    """打印交互菜单，无返回值。"""
    print()
    print('=' * 64)
    print('  迷宫探索系统 · 标定工具')
    print('=' * 64)
    print('  请选择标定项，建议按 1 → 2 → 3 的顺序，每项单独跑一次：')
    print()
    for key, (mode, desc) in CALIB_MENU.items():
        print(f'    {key}) {desc}    [{mode}]')
    print()
    print('  提示：标定需要相机与机械臂已启动；HSV 与姿态标定需要图形界面。')
    print('=' * 64)


def resolve_mode(argv: Sequence[str]) -> Optional[CalibMode]:
    """解析出标定模式：支持命令行直达，或打印菜单交互选择。

    :param argv: 命令行参数，不含程序名。
    :returns: 命中的标定模式；用户取消或超时输入为 ``None``。
    """
    for arg in argv:
        if arg in MODES:
            return CalibMode(arg)

    print_menu()
    while True:
        try:
            choice = input('  请选择 1 到 3，q 退出 > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice in ('q', 'quit', 'exit', ''):
            return None
        if choice in CALIB_MENU:
            return CALIB_MENU[choice][0]
        # 也允许直接输入模式名
        if choice in MODES:
            return CalibMode(choice)
        print('  输入无效，请输入 1 / 2 / 3，或模式名 hsv / line_pose / motion')


def suggest_next(mode: CalibMode) -> str:
    """返回该模式标定完成后的下一步建议，无建议时为空串。

    :param mode: 刚完成的标定模式。
    :returns: 建议文本。
    """
    return NEXT_STEPS.get(mode, '')


def _share_config(filename: str) -> Optional[Path]:
    """返回 install/share 下的配置文件路径，找不到时返回 ``None``。

    :param filename: 配置文件名，不含目录。
    :returns: 已存在的配置文件路径。
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory('maze_explorer')) / 'config' / filename
        return path if path.is_file() else None
    except Exception:  # noqa: BLE001 - 未安装时退化为工具自身默认值
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析模式并调起标定工具，返回进程退出码。

    :param argv: 命令行参数，取 ``None`` 时读 ``sys.argv``。
    :returns: 正常完成与用户取消均为 0。
    """
    args: List[str] = list(sys.argv[1:] if argv is None else argv)

    mode = resolve_mode(args)
    if mode is None:
        print('[maze_calib] 已取消')
        return 0

    # 复用 calibration_tool 的实现，仅以参数形式指定模式
    from maze_explorer.calibration_tool import main as calibration_main

    # 把两份 yaml 一并传入：标定工具自己声明的默认值与 yaml 可能不一致，
    # 尤其 follow 模式必须用已标定的 line_hsv、line_pid 与 line_steer_sign。
    argv_tool = ['--ros-args', '-p', f'mode:={mode}']
    for name in ('maze_params.yaml', 'hsv_params.yaml'):
        path = _share_config(name)
        if path is not None:
            argv_tool += ['--params-file', str(path)]

    if mode in GUI_MODES:
        print('[maze_calib] 该模式需要图形界面即 cv2 窗口，请确认在桌面终端运行')
    print(f'[maze_calib] 进入标定模式：{mode}，Ctrl-C 可随时退出')
    calibration_main(argv_tool)

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
