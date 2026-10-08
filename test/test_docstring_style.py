"""docstring 与注释的风格强制检查。

扫描包内全部 Python 文件，对三类机械可判的违规判失败：圆括号与方括号的半角及
全角形式、四位数字加横线的日期形式、emoji 码点。
因果性与时间性动词的规则无法机械判定，由代码评审把关。
用法：`pytest test -q`，或在仓库根目录执行 `python3 test/test_docstring_style.py`。
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Iterator, List, Tuple

#: 违禁字符集：半角圆括号、半角方括号、全角圆括号、全角方括号
FORBIDDEN_CHARS = frozenset('()[]（）【】')

#: 日期形式：四位数字、横线、两位数字、横线、两位数字
DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')

#: emoji 与装饰符号所在的码点区间
EMOJI_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0xFE0F, 0xFE0F),
)

#: 仓库根目录，测试文件位于根目录下的 test 目录
REPO_ROOT = Path(__file__).resolve().parent.parent

#: 待扫描目录，相对仓库根目录
SCAN_DIRS: Tuple[str, ...] = ('maze_explorer', 'test')


def _source_files() -> Iterator[Path]:
    """按目录名与文件名升序产出待扫描的 Python 文件路径。"""
    for dirname in SCAN_DIRS:
        yield from sorted((REPO_ROOT / dirname).glob('*.py'))


def _iter_docstring_lines(path: Path) -> Iterator[Tuple[int, str]]:
    """产出文件内全部 docstring 的行号与逐行文本。

    :param path: 待解析的 Python 文件路径。
    :returns: 行号与该行文本的二元组，行号相对文件首行为 1。
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        first_line = node.body[0].lineno
        for offset, text in enumerate(value.value.splitlines()):
            yield first_line + offset, text


def _iter_comment_lines(path: Path) -> Iterator[Tuple[int, str]]:
    """产出文件内全部行注释的行号与文本。

    :param path: 待读取的 Python 文件路径。
    :returns: 行号与该行文本的二元组，行号相对文件首行为 1。
    """
    source = path.read_text(encoding='utf-8')
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string


def _contains_emoji(text: str) -> bool:
    """判断文本是否含 emoji 或装饰符号码点。

    :param text: 待检查文本。
    :returns: 命中任一码点区间时为真。
    """
    return any(
        start <= ord(char) <= end
        for char in text
        for start, end in EMOJI_RANGES
    )


def _is_directive(text: str) -> bool:
    """判断行注释是否为类型检查或过检指令，指令需保持原始语法故豁免。

    :param text: 行注释全文，含开头的井号。
    :returns: 属于 `type:` 或 `noqa` 指令时为真。
    """
    body = text.lstrip('#').lstrip()
    return body.startswith('type:') or body.startswith('noqa')


def _find_violations(path: Path) -> List[str]:
    """扫描单个文件，返回全部违规描述行。

    :param path: 待检查的 Python 文件路径。
    :returns: 每项形如 `文件名:行号:原因` 的字符串。
    """
    docstrings = list(_iter_docstring_lines(path))
    comments = (entry for entry in _iter_comment_lines(path) if not _is_directive(entry[1]))
    entries = docstrings + list(comments)
    found: List[str] = []
    for line, text in entries:
        stripped = text.strip()
        for char in FORBIDDEN_CHARS:
            if char in text:
                found.append(f'{path.name}:{line}:存在括号字符:{stripped}')
                break
        if DATE_PATTERN.search(text):
            found.append(f'{path.name}:{line}:存在日期形式:{stripped}')
        if _contains_emoji(text):
            found.append(f'{path.name}:{line}:存在emoji符号:{stripped}')
    return found


def collect_violations() -> List[str]:
    """扫描全部受检文件，汇总违规描述行。

    :returns: 全部违规描述，按文件与行号升序。
    """
    found: List[str] = []
    for path in _source_files():
        found.extend(_find_violations(path))
    return found


def test_comments_and_docstrings_follow_style_rules() -> None:
    """注释与 docstring 不得含括号、日期与 emoji。"""
    violations = collect_violations()
    assert not violations, '违反注释规范：\n' + '\n'.join(violations)


def main() -> int:
    """直接执行入口：打印全部违规并返回进程退出码。

    :returns: 无违规为 0，存在违规为 1。
    """
    violations = collect_violations()
    if not violations:
        print('注释风格检查通过')
        return 0
    for item in violations:
        print(item)
    print(f'共 {len(violations)} 项违规')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
