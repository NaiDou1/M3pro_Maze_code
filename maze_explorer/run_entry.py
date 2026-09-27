"""入口 A：一键启动迷宫探索与方块收集任务。

作为「薄壳」入口，本模块不重复实现任何业务逻辑，只负责四件事：

1. **逐项硬件自检**：按 ``/odom_raw``、``/scan``、相机、``get_kinemarics``
   逐项检查，缺哪项就报哪项，并给出可直接复制执行的中文提示；
   相比原先「等 15 秒后进 FAULT」的一句笼统报错，能直接消掉现场最常见的
   「到底哪个环节没起来」的往返排查。
2. **关闭手柄自启链路**：``joy.sh`` 拉起的三个进程会周期性发布 ``/cmd_vel``
   零速与 ``JoyState``，与本体争抢底盘控制权；
3. **启动任务**：自检通过后转为运行 ``maze_bringup.launch.py``，由它注入
   三份配置并启动 ``mission_manager``；
4. **安全退出**：``/cmd_vel`` 归零由 ``mission_manager`` 的退出路径保证，
   本入口只做兜底。

用法::

    ros2 run maze_explorer maze_run                 # 自检通过后开始探索
    ros2 run maze_explorer maze_run --check-only    # 只自检不启动（首次使用推荐）
    ros2 run maze_explorer maze_run origin_rc:="[0,0]" exit_rc:="[6,6]"

.. note::
   ``ROS_DOMAIN_ID=30`` 必须与其余节点一致，否则自检会全项失败——自检提示中
   会带上当前值，便于一眼看出问题。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import rclpy
from arm_interface.srv import ArmKinemarics
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

#: 单项自检的等待上限（秒）
DEFAULT_TIMEOUT = 4.0
#: 手柄自启链路涉及的可执行/进程名，需整条链路一起关闭
_JOY_PATTERNS = (
    'start_joy_controller.py',
    'yahboomcar_joy_launch.py',
    'yahboom_joy_M3Pro',
)


@dataclass
class HardwareCheck:
    """单项自检结果。"""

    name: str
    ok: bool
    #: 失败时给出的中文可执行提示
    hint: str = ''


class _Probe(Node):
    """自检探针：订阅关键话题并探测 IK 服务。"""

    def __init__(self) -> None:
        super().__init__('maze_run_probe')
        self.seen = {'odom': False, 'scan': False, 'camera': False}
        self.create_subscription(Odometry, '/odom_raw', self._on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self._on_image, 1)
        self.ik_client = self.create_client(ArmKinemarics, 'get_kinemarics')

    def _on_odom(self, _msg: Odometry) -> None:
        self.seen['odom'] = True

    def _on_scan(self, _msg: LaserScan) -> None:
        self.seen['scan'] = True

    def _on_image(self, _msg: Image) -> None:
        self.seen['camera'] = True


def run_self_check(probe: _Probe, timeout: float = DEFAULT_TIMEOUT) -> List[HardwareCheck]:
    """逐项检查硬件依赖，返回全部结果（不抛异常）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not all(probe.seen.values()):
        rclpy.spin_once(probe, timeout_sec=0.1)

    domain_id = os.environ.get('ROS_DOMAIN_ID', '<未设置>')
    domain_hint = f'（当前 ROS_DOMAIN_ID={domain_id}，须为 30）'

    return [
        HardwareCheck(
            name='/odom_raw',
            ok=probe.seen['odom'],
            hint=(
                '未收到里程计/IMU。请先启动下位机：'
                'sh /home/jetson/start_agent.sh' + domain_hint
            ),
        ),
        HardwareCheck(
            name='/scan',
            ok=probe.seen['scan'],
            hint=(
                '未收到激光数据。请启动雷达链：'
                'ros2 launch slam_mapping bringup.launch.py'
            ),
        ),
        HardwareCheck(
            name='/camera/color/image_raw',
            ok=probe.seen['camera'],
            hint=(
                '未收到相机图像。请启动深度相机：'
                'ros2 launch orbbec_camera dabai_dcw2.launch.py'
            ),
        ),
        HardwareCheck(
            name='get_kinemarics',
            ok=probe.ik_client.wait_for_service(timeout_sec=timeout),
            hint=(
                'IK/FK 服务不可用，机械臂抓取将完全无法工作。请启动 arm_kin，'
                '可参考：ros2 launch M3Pro_demo camera_arm_kin.launch.py'
            ),
        ),
    ]


def print_report(checks: Sequence[HardwareCheck]) -> bool:
    """打印自检报告，返回是否全部就绪。"""
    all_ok = True
    print()
    print('=' * 64)
    print('  迷宫探索与方块收集系统 · 启动前自检')
    print('=' * 64)
    for check in checks:
        mark = '[ OK ]' if check.ok else '[FAIL]'
        print(f'  {mark} {check.name}')
        if not check.ok:
            all_ok = False
            print(f'         → {check.hint}')
    print('=' * 64)
    print('  全部就绪，可以开始探索' if all_ok else '  存在未就绪项，请按上述提示处理后重试')
    print()
    return all_ok


def kill_joy_autostart() -> None:
    """关闭手柄自启链路，避免其与本体争抢 ``/cmd_vel``。"""
    for pattern in _JOY_PATTERNS:
        try:
            found = subprocess.run(
                ['pgrep', '-f', pattern], capture_output=True, check=False
            )
        except FileNotFoundError:
            return  # 系统无 pgrep/pkill，跳过（不阻塞启动）
        if found.returncode == 0:
            print(f'[maze_run] 关闭手柄自启进程：{pattern}')
            subprocess.run(['pkill', '-f', pattern], check=False)
    time.sleep(1.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """加载配置 → 自检 → 就绪则启动任务，否则打印缺失项并返回非零。"""
    args = list(sys.argv[1:] if argv is None else argv)
    check_only = any(a in ('--check-only', '-c') for a in args)

    rclpy.init()
    probe = _Probe()
    try:
        checks = run_self_check(probe)
    except KeyboardInterrupt:
        print('\n[maze_run] 自检被中断')
        return 130
    finally:
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not print_report(checks):
        return 1

    if check_only:
        print('[maze_run] --check-only：仅自检，不启动任务')
        return 0

    kill_joy_autostart()

    # 透传形如 key:=value 的 launch 参数
    passthrough = [a for a in args if ':=' in a and not a.startswith('-')]
    cmd = ['ros2', 'launch', 'maze_explorer', 'maze_bringup.launch.py'] + passthrough
    print(f'[maze_run] 启动：{" ".join(cmd)}')
    try:
        os.execvp('ros2', cmd)
    except FileNotFoundError:
        print('[maze_run] 未找到 ros2 命令，请先 source /opt/ros/humble/setup.bash')
        return 1
    return 0  # 理论上 exec 成功后不会走到这里


if __name__ == '__main__':
    sys.exit(main())
