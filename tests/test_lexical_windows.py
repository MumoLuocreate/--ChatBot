"""三字窗口的抽样：上限不许让句首独占（2026-09-17 真机缺陷）。"""

from __future__ import annotations

from qichi.memory.lexical import MAX_QUERY_FRAGMENTS, query_fragments


def test_a_long_turns_tail_still_takes_part_in_matching():
    """真机：76 字的句子里，关键词在第 70 字，旧抽样只到第 66 字。"""

    tail = "什么时候想要我"
    text = "你的小本本不给我看我也不知道是真的假的啦兔子你就让我看看呗我记得今天凌晨是聊了很多" + tail

    sample = query_fragments((text,))

    assert len(sample) <= MAX_QUERY_FRAGMENTS
    assert any(window in tail for window in sample), "句尾的关键词必须进样本"


def test_a_short_text_keeps_every_window_in_order():
    text = "今天凌晨聊了什么"

    sample = query_fragments((text,))

    assert sample == tuple(text[index : index + 3] for index in range(len(text) - 2))


def test_the_cap_still_bounds_the_sample():
    text = "啊" * 5000

    sample = query_fragments((text,))

    assert len(sample) <= MAX_QUERY_FRAGMENTS


def test_repeated_terms_do_not_duplicate_windows():
    text = "今天凌晨"

    assert query_fragments((text, text)) == query_fragments((text,))