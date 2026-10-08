maze_explorer API 文档
========================

按分层顺序列出全部模块。构建命令：`python3 -m sphinx -b html docs docs/_build`。

.. toctree::
   :maxdepth: 2

入口层
======

.. automodule:: maze_explorer.run_entry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer.calib_entry
   :members:
   :undoc-members:
   :show-inheritance:

编排层
======

.. automodule:: maze_explorer.mission_manager
   :members:
   :undoc-members:
   :show-inheritance:

控制层
======

.. automodule:: maze_explorer.motion_controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer.grasp_fsm
   :members:
   :undoc-members:
   :show-inheritance:

感知层
======

.. automodule:: maze_explorer.line_detector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer.block_detector
   :members:
   :undoc-members:
   :show-inheritance:

建图决策层
==========

.. automodule:: maze_explorer.grid_mapper
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer.dfs_planner
   :members:
   :undoc-members:
   :show-inheritance:

硬件抽象层
==========

.. automodule:: maze_explorer.base_driver
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer.sensors
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer.arm_controller
   :members:
   :undoc-members:
   :show-inheritance:

工具与兼容层
============

.. automodule:: maze_explorer.calibration_tool
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: maze_explorer._compat
   :members:
   :undoc-members:
   :show-inheritance:
