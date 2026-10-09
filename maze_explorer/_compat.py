"""跨 Python 版本的字符串枚举基类。

Python 3.10 的 f-string 对字符串混合枚举输出成员值，Python 3.11 以上输出类名与
成员名。本模块统一为输出成员值，使日志、yaml 与 ROS 参数在任意 Python 版本下
拿到同一字符串。

用法：字符串取值域枚举继承 ``StrEnum``，见 AGENTS §6.5 的取值域选型表。
"""

from enum import Enum

__all__ = ['StrEnum']


class StrEnum(str, Enum):
    """字符串枚举基类，成员同时是 ``str`` 的实例。

    成员与原字符串在比较、字典键与哈希上等价，因此 ``DIRECTIONS`` 用
    ``Heading.NORTH`` 或 ``'N'`` 取到同一项。构造时传入未声明的值抛
    ``ValueError``，取值域由解释器在构造时刻强制。
    """

    def __str__(self) -> str:
        """返回成员值，固定跨版本的字符串化结果。"""
        return str.__str__(self.value)

    def __format__(self, format_spec: str) -> str:
        """按 ``str`` 协议格式化成员值，与 ``__str__`` 结果一致。

        :param format_spec: 传给 ``str.format`` 的格式说明符。
        :returns: 格式化后的成员值。
        """
        return str.__format__(self.value, format_spec)
