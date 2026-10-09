"""Sphinx 配置：由 docstring 生成 maze_explorer 的 HTML API 文档。

用法：`python3 -m sphinx -b html docs docs/_build`。
autodoc 把 ROS 与 OpenCV 相关模块替换为占位对象，使文档在未安装 ROS 的机器上
同样可以构建；在 Jetson 上 source 两个工作区后构建则解析真实对象。
"""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'maze_explorer'
author = 'jetson'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

language = 'zh_CN'
exclude_patterns = ['_build']
html_theme = 'alabaster'

# 按源码顺序输出成员，便于对照阅读
autodoc_member_order = 'bysource'
# 类型标注以文字描述呈现，不生成悬空类型链接
autodoc_typehints = 'description'
# 第三方与 ROS 依赖的占位导入，缺包时 autodoc 不报错
autodoc_mock_imports = [
    'arm_interface',
    'arm_msgs',
    'cv2',
    'cv_bridge',
    'geometry_msgs',
    'message_filters',
    'nav_msgs',
    'numpy',
    'rclpy',
    'sensor_msgs',
    'std_msgs',
    'tf2_ros',
]
