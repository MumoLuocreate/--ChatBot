"""卡B④：常驻层优先级截断 + episode 年龄退场。

判据见 doc/方案-20260922-写入侧作用范围与移出可逆.md 第 5 节。
命中与不误判两侧都要锁住：年龄门只能退 episode，绝不能退偏好/约定/纠正
（AGENTS.md 冻结不变式：偏好、约定、纠正是常驻关系状态）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qichi.dialogue.context_builder import _fit_working_records
from qichi.domain.memory import working_set_resident

UTC = timezone.utc
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


class _FakeRecord:
    def __init__(self, name: str, type: str = "episode", age_days: float = 0.0) -> None:
        self.name = name
        self.type = type
        self.valid_from_utc = NOW - timedelta(days=age_days)


def _tokens(text: str) -> int:
    return len(text)


def _render(records) -> str:
    return "\n".join(record if isinstance(record, str) else record.name for record in records)


# ---------------------------------------------------------------- 年龄门（B4b）


@pytest.mark.parametrize(
    "age_days, expected",
    [
        (0.0, True),
        (3.0, True),
        (13.9, True),
        (14.0, True),   # 边界当天仍在场（含）
        (14.1, False),
        (20.0, False),
        (400.0, False),
    ],
)
def test_episode_ages_out_past_the_window(age_days, expected):
    record = _FakeRecord("old-episode", age_days=age_days)
    assert working_set_resident(record, NOW, episode_max_age_days=14) is expected


@pytest.mark.parametrize("kind", ["preference", "agreement", "correction", "self_expression"])
def test_only_episodes_age_out(kind):
    """冻结不变式：偏好、约定、纠正是常驻关系状态——再老也不退场。"""
    record = _FakeRecord("ancient", type=kind, age_days=3650.0)
    assert working_set_resident(record, NOW, episode_max_age_days=14) is True


def test_age_gate_can_be_disabled():
    record = _FakeRecord("old-episode", age_days=999.0)
    assert working_set_resident(record, NOW, episode_max_age_days=0) is True


def test_future_timestamp_is_not_aged_out():
    """时钟偏移不该把一条 episode 当成过期丢掉。"""
    record = _FakeRecord("skewed", age_days=-1.0)
    assert working_set_resident(record, NOW, episode_max_age_days=14) is True


# ------------------------------------------------------- 优先级截断（B4a）


def _records(count: int):
    # 调用方保证已按优先级排好序；越靠前越重要。
    return [f"r{i}" for i in range(count)]


def test_fit_keeps_everything_when_it_fits():
    records = _records(5)
    kept, dropped = _fit_working_records(records, _render, _tokens, 10_000)
    assert kept == tuple(records)
    assert dropped == 0


def test_fit_truncates_from_the_tail_not_the_head():
    """装不下时丢掉优先级最低的尾部，并如实报出丢弃条数。"""
    records = _records(6)  # "r0".."r5"，每行 2 字 + 换行
    budget = _tokens(_render(records[:3]))  # 只够前三条
    kept, dropped = _fit_working_records(records, _render, _tokens, budget)
    assert kept == ("r0", "r1", "r2")
    assert dropped == 3


def test_fit_keeps_at_least_one_when_any_fits():
    records = _records(4)
    kept, dropped = _fit_working_records(records, _render, _tokens, _tokens("r0"))
    assert kept == ("r0",)
    assert dropped == 3


def test_fit_drops_everything_when_nothing_fits():
    records = _records(3)
    kept, dropped = _fit_working_records(records, _render, _tokens, 0)
    assert kept == ()
    assert dropped == 3


def test_fit_never_exceeds_the_budget():
    records = _records(30)
    for budget in range(0, 200, 7):
        kept, dropped = _fit_working_records(records, _render, _tokens, budget)
        assert _tokens(_render(kept)) <= budget or kept == ()
        assert len(kept) + dropped == len(records)
        assert kept == tuple(records[: len(kept)])


def test_fit_of_nothing_is_nothing():
    assert _fit_working_records([], _render, _tokens, 100) == ((), 0)
