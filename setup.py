import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'maze_explorer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson-nx@example.com',
    description='ROSMASTER M3 Pro 二维迷宫自主探索与彩色方块收集系统',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 顶层任务编排
            'mission_manager = maze_explorer.mission_manager:main',
            # 标定工具
            'calibration_tool = maze_explorer.calibration_tool:main',
            # 感知层（可独立运行用于调试出图）
            'line_detector = maze_explorer.line_detector:main',
            'block_detector = maze_explorer.block_detector:main',
            # 建图与决策（可独立运行用于可视化拓扑）
            'grid_mapper = maze_explorer.grid_mapper:main',
            # 机械臂（可独立运行用于手动摆手）
            'arm_controller = maze_explorer.arm_controller:main',
        ],
    },
)
