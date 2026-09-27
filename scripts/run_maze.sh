#!/usr/bin/env bash
#
# 迷宫探索与方块收集系统 —— 一键环境准备与启动
#
# 负责三件事：
#   1. 设置 ROS_DOMAIN_ID=30（本机必须，否则看不到任何话题）
#   2. source 三个工作区
#   3. 关闭手柄自启节点（其 JoyState 会让部分节点持续发零速）
# 然后启动自研包。
#
# 前提（按顺序先手动启动，见 AGENTS.md）：
#   sh /home/jetson/start_agent.sh
#   ros2 launch slam_mapping bringup.launch.py
#   ros2 launch orbbec_camera dabai_dcw2.launch.py
#
# 用法：
#   ./run_maze.sh
#   ./run_maze.sh origin_rc:="[0,0]" exit_rc:="[6,6]"

set -euo pipefail

export ROS_DOMAIN_ID=30

source /opt/ros/humble/setup.bash
source /home/jetson/yahboomcar_ws/install/setup.bash
source /home/jetson/M3Pro_ws/install/setup.bash 2>/dev/null || true

# 关闭手柄自启节点，避免 JoyState=True 触发其他节点的零速保护
if pgrep -f yahboom_joy_M3Pro >/dev/null 2>&1; then
    echo '[run_maze] 关闭手柄自启节点 yahboom_joy_M3Pro'
    pkill -f yahboom_joy_M3Pro || true
    sleep 1
fi

echo '[run_maze] 启动 maze_explorer（Ctrl-C 结束）'
exec ros2 launch maze_explorer maze_bringup.launch.py "$@"
