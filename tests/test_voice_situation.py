"""语音情境：写手看到的事实（她这一轮的全部内容、时间差、主动还是回应）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from qichi.voice.situation import SPOKEN_MARK, ago_label, build_situation

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class FakeEvent:
    actor: str
    text: str | None
    occurred_at_utc: datetime


def _at(hour: int, minute: int, *, day: int = 15) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=SHANGHAI).astimezone(timezone.utc)


def _situation(events, now, **kwargs):
    return build_situation(events=events, now=now, local_zone=SHANGHAI, **kwargs)


def test_the_writer_sees_every_part_of_this_turn_and_which_one_is_spoken():
    """2026-09-15：语音段常常接在她自己刚打出去的文字后面，只给一句就写不出承接。"""

    events = (
        FakeEvent("mumo", "在吗", _at(10, 0)),
        FakeEvent("qichi", "在呢", _at(10, 1)),
        FakeEvent("mumo", "说点好听的", _at(10, 30)),
    )

    text = _situation(
        events, _at(10, 35),
        spoken="想你了，想得有点过分。",
        parts=("敢啊，怎么不敢。", "想你了，想得有点过分。"),
        voice_index=2,
    )

    assert text == (
        "[最近往来]（括号里是那条消息的钟点与距现在多久）\n"
        "用户（10:00，35 分钟前）：在吗\n"
        "角色（10:01，34 分钟前）：在呢\n"
        "用户（10:30，5 分钟前）：说点好听的\n"
        "[角色这一轮要发的全部内容，按发送顺序]\n"
        "1. 敢啊，怎么不敢。\n"
        "2. 想你了，想得有点过分。" + SPOKEN_MARK
    )


def test_the_mark_lands_only_on_the_part_that_is_spoken():
    text = _situation(
        (FakeEvent("mumo", "在吗", _at(10, 0)),), _at(10, 1),
        spoken="第二段", parts=("第一段", "第二段", "第三段"), voice_index=2,
    )

    marked = [line for line in text.splitlines() if SPOKEN_MARK in line]
    assert marked == ["2. 第二段" + SPOKEN_MARK]


def test_without_a_voice_index_nothing_is_marked():
    """老行为：没有段号时不许凭空标一句。"""

    text = _situation((FakeEvent("mumo", "在吗", _at(10, 0)),), _at(10, 1), spoken="就这一句")

    assert SPOKEN_MARK not in text
    assert "1. 就这一句" in text


def test_an_initiative_turn_says_so_and_says_when_he_last_spoke():
    events = (FakeEvent("mumo", "我去上课了", _at(8, 0)),)

    text = _situation(events, _at(10, 0), spoken="在忙吗", initiative=True)

    assert "这一轮是角色自己先开口的" in text
    assert "用户上一条消息在 2 小时前" in text


def test_a_reply_turn_never_claims_to_be_an_initiative():
    events = (FakeEvent("mumo", "在吗", _at(9, 59)),)

    text = _situation(events, _at(10, 0), spoken="在呢")

    assert "自己先开口" not in text


def test_the_history_budget_drops_the_oldest_lines_first():
    events = tuple(
        FakeEvent("mumo" if index % 2 else "qichi", "很久以前第 %d 条" % index, _at(9, index))
        for index in range(12)
    )

    text = _situation(events, _at(10, 0), spoken="嗯", history_chars=120)

    assert "很久以前第 11 条" in text, "最新那一条永远留着"
    assert "很久以前第 0 条" not in text
    assert len(text) < 400


def test_empty_events_and_empty_parts_are_skipped():
    events = (
        FakeEvent("mumo", None, _at(9, 0)),
        FakeEvent("mumo", "   ", _at(9, 30)),
        FakeEvent("mumo", "真的在吗", _at(10, 0)),
    )

    text = _situation(events, _at(10, 1), spoken="在", parts=("", "在", "  "), voice_index=2)

    assert "真的在吗" in text
    marked = [line for line in text.splitlines() if SPOKEN_MARK in line]
    assert marked == ["2. 在" + SPOKEN_MARK], "段号跟她那一轮的原段对齐，空白段不占位"


def test_only_facts_go_in_never_an_emotion_label():
    """不误判：情境里不许出现替她判定的情绪词——那是写手自己的活。"""

    events = (FakeEvent("mumo", "在吗", _at(10, 0)),)

    text = _situation(events, _at(10, 1), spoken="在呢")

    for word in ("情绪", "开心", "生气", "难过", "撒娇", "害羞"):
        assert word not in text


def test_a_naive_clock_is_refused():
    with pytest.raises(ValueError):
        _situation((FakeEvent("mumo", "在吗", _at(10, 0)),), datetime(2026, 9, 15, 10, 1), spoken="在")


@pytest.mark.parametrize("delta,expected", [
    (timedelta(seconds=5), "刚刚"),
    (timedelta(minutes=3), "3 分钟前"),
    (timedelta(minutes=59), "59 分钟前"),
    (timedelta(hours=2), "2 小时前"),
    (timedelta(days=3), "3 天前"),
])
def test_ago_label_reads_like_a_clock_not_like_a_mood(delta, expected):
    assert ago_label(delta) == expected
