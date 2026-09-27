#!/usr/bin/env bash
#
# 迷宫探索与方块收集系统 —— 一键环境准备与启动
#
# 负责三件事：
#   1. 设置 ROS_DOMAIN_ID=30（本机必须，否则看不到任何话题）
#   2. source 三个工作区
#   3. 关闭手柄自启链路（start_joy_controller.py 等三个进程）
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

# 关闭手柄自启链路。joy.sh 会拉起 start_joy_controller.py，后者再启动
# yahboomcar_joy_launch.py 与 yahboom_joy_M3Pro；其中 autostart 节点会周期性
# 发布 /cmd_vel 零速、yahboom_joy_M3Pro 会发布 JoyState，两者都会与本体争抢
# 底盘控制权，必须整条链路一起关掉。
for pattern in start_joy_controller.py yahboomcar_joy_launch.py yahboom_joy_M3Pro; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        echo "[run_maze] 关闭手柄自启进程：$pattern"
        pkill -f "$pattern" || true
    fi
done
sleep 1

echo '[run_maze] 启动 maze_explorer（Ctrl-C 结束）'
exec ros2 launch maze_explorer maze_bringup.launch.py "$@"
