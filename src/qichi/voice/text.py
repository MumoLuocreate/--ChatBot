"""把「她写的那句话」变成「可以念出来的那句话」。

剥离只作用于**送去合成的文本**，不改变她真正发出去的内容；被剥掉的部分原样留在
`SpeechText.residue` 里，由发送层决定是否另走文字通道。

规则（用户 2026-09-14 裁定「颜文字完全剥离」，见 `doc/TTS-实施计划-20260914.md` §2.35）：
- 剥掉 emoji 码位（含变体选择符、零宽连接符、肤色修饰、区域指示符等）；
- 剥掉「括号里既没有汉字也没有字母数字」的整组——即颜文字；
  反例（**必须保留**）：「（笑）」（含汉字）、「(TTS)」（含字母）、「（2026）」（含数字）。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Final


class SpeechTextError(ValueError):
    """这句话无法安全地转成语音。"""


# emoji 与它们的组合符。刻意不含 CJK 标点（U+3000–U+303F）与全角形式（U+FF00–U+FFEF），
# 因为颜文字里那些字符由下面的「括号组」规则统一处理。
_EMOJI_RANGES: Final = (
    (0x1F000, 0x1F0FF),  # 麻将/扑克/多米诺
    (0x1F100, 0x1F1FF),  # 带圈字母数字、区域指示符
    (0x1F200, 0x1F2FF),  # 带圈表意文字
    (0x1F300, 0x1F5FF),  # 天气、建筑、表情
    (0x1F600, 0x1F64F),  # 表情脸
    (0x1F650, 0x1F67F),  # 装饰符号
    (0x1F680, 0x1F6FF),  # 交通
    (0x1F700, 0x1F77F),  # 炼金符号
    (0x1F780, 0x1F7FF),  # 几何扩展
    (0x1F800, 0x1F8FF),  # 补充箭头-C
    (0x1F900, 0x1F9FF),  # 补充符号与象形
    (0x1FA00, 0x1FAFF),  # 扩展-A
    (0x2600, 0x26FF),    # 杂项符号（☀☺⚡等）
    (0x2700, 0x27BF),    # 装饰符号（✨✅❌等）
    (0x2B00, 0x2BFF),    # 杂项符号与箭头
    (0xFE00, 0xFE0F),    # 变体选择符
    (0x20E3, 0x20E3),    # 组合包围键帽
    (0x200D, 0x200D),    # 零宽连接符
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3299),
)
_EMOJI: Final = frozenset(
    chr(code) for start, end in _EMOJI_RANGES for code in range(start, end + 1)
)

# 单层括号组；不用贪婪或嵌套，避免把正常行文里的括号一并吞掉。
_PAREN_GROUP: Final = re.compile(r"[（(][^（()）]*[）)]")
_CJK: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ALNUM: Final = re.compile(r"[0-9A-Za-z\u3040-\u30ff\uac00-\ud7af]")
_SPACES: Final = re.compile(r"[ \t\u3000]{2,}")


@dataclass(frozen=True, slots=True)
class SpeechText:
    """送给合成的文本，以及被剥下来的东西。"""

    spoken: str
    residue: str

    def __post_init__(self) -> None:
        if not isinstance(self.spoken, str) or not isinstance(self.residue, str):
            raise TypeError("spoken and residue must be strings")

    @property
    def changed(self) -> bool:
        return bool(self.residue)


def _is_kaomoji(body: str) -> bool:
    """括号里的内容是否属于颜文字：既无汉字，也无字母数字。"""

    return not _CJK.search(body) and not _ALNUM.search(body)


def prepare_speech(text: str) -> SpeechText:
    """剥掉颜文字与 emoji，返回可念文本与被剥掉的部分。"""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    residue: list[str] = []

    def drop_group(match: re.Match[str]) -> str:
        group = match.group(0)
        if _is_kaomoji(group[1:-1]):
            residue.append(group)
            return ""
        return group

    without_groups = _PAREN_GROUP.sub(drop_group, text)

    kept: list[str] = []
    for char in without_groups:
        if char in _EMOJI:
            residue.append(char)
            continue
        if unicodedata.category(char) == "Cf":  # 其他不可见格式符
            residue.append(char)
            continue
        kept.append(char)

    spoken = _SPACES.sub(" ", "".join(kept)).strip()
    return SpeechText(spoken=spoken, residue="".join(residue))


def is_speakable(text: str, *, max_chars: int, min_units: int = 2) -> bool:
    """剥完之后还剩不剩能念的内容，以及是不是短到值得念。"""

    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if type(min_units) is not int or min_units < 1:
        raise ValueError("min_units must be a positive integer")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) > max_chars:
        return False
    return len(_CJK.findall(text)) + len(re.findall(r"[A-Za-z0-9]", text)) >= min_units
