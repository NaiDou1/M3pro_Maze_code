"""一键启动迷宫探索与方块收集系统。

.. important::
   本 launch **只启动自研包**。以下外部依赖需按顺序先行启动，详见 AGENTS.md::

       sh /home/jetson/start_agent.sh                       # 下位机 micro-ROS
       ros2 launch slam_mapping bringup.launch.py           # 雷达链 + IMU + EKF
       ros2 launch orbbec_camera dabai_dcw2.launch.py       # 深度相机

   另外运行前须关闭手柄自启节点即 ``joy_control/joy.sh``，否则 ``JoyState``
   会触发部分节点的零速保护而干扰运动。

用法::

    ros2 launch maze_explorer maze_bringup.launch.py
    ros2 launch maze_explorer maze_bringup.launch.py origin_rc:=<入口 rc 数组> exit_rc:=<出口 rc 数组>
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_dir = os.path.join(get_package_share_directory('maze_explorer'), 'config')

    param_files = [
        os.path.join(config_dir, 'maze_params.yaml'),
        os.path.join(config_dir, 'hsv_params.yaml'),
        os.path.join(config_dir, 'arm_poses.yaml'),
    ]

    # 可由命令行覆盖的关键参数
    origin_rc = LaunchConfiguration('origin_rc')
    exit_rc = LaunchConfiguration('exit_rc')
    mount_calibrated = LaunchConfiguration('mount_calibrated')

    mission_manager = Node(
        package='maze_explorer',
        executable='mission_manager',
        name='mission_manager',
        output='screen',
        emulate_tty=True,
        parameters=param_files + [
            {
                'origin_rc': origin_rc,
                'exit_rc': exit_rc,
                'mount_calibrated': mount_calibrated,
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'origin_rc', default_value='[0, 0]',
            description='入口格 (行, 列)，作为 DFS 树根与深度 0',
        ),
        DeclareLaunchArgument(
            'exit_rc', default_value='[6, 6]',
            description='出口格 (行, 列)，返航目标',
        ),
        DeclareLaunchArgument(
            'mount_calibrated', default_value='false',
            description='相机外参是否已标定；false 时只使用相机系量',
        ),
        mission_manager,
    ])
