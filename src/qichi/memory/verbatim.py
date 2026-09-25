"""逐字重合判据：用户把一整句话原样打回来。

冻结计划（doc/修复计划-20260912-记忆展开与识图.md §2.1 的 K3）允许一种额外的
指向方式：用户复述了某条明细的原文。这个判据只有在重合**足够长**时才成立——
2026-09-12 在 223 轮真机消息上复算过：同一段亲密关系里的中文对话天然共享
4-5 个字，所以 4 字门槛等于没设（108/223 开门），而巧合到不了 8 个字。

比对对象只有 exact_quote。normalized_detail 与 summary 是解析器的转述，
拿它们做比对正是「两个字就能打开成人原话」的成因（问题冻结 P1）。
"""

from __future__ import annotations

VERBATIM_MIN_CHARS = 8


def shares_run(text: str, candidate: str, *, min_chars: int = VERBATIM_MIN_CHARS) -> bool:
    """Do the two strings share one contiguous run of at least min_chars?"""

    if not isinstance(text, str) or not isinstance(candidate, str):
        raise TypeError("both sides of a verbatim comparison must be strings")
    if type(min_chars) is not int or min_chars < 1:
        raise ValueError("min_chars must be a positive integer")
    if len(text) < min_chars or len(candidate) < min_chars:
        return False
    windows = {text[index : index + min_chars] for index in range(len(text) - min_chars + 1)}
    return any(
        candidate[index : index + min_chars] in windows
        for index in range(len(candidate) - min_chars + 1)
    )
