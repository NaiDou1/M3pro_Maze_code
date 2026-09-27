"""maze_explorer: ROSMASTER M3 Pro 二维迷宫自主探索与彩色方块收集系统。

模块划分（分层架构）：
    硬件抽象层  base_driver / sensors / arm_controller
    感知层      line_detector / block_detector
    建图决策层  grid_mapper / dfs_planner
    控制层      motion_controller / grasp_fsm
    编排层      mission_manager
    工具        calibration_tool
"""

__version__ = '0.1.0'
