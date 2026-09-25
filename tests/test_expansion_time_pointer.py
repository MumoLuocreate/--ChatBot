"""卡D：时段/钟点指向哪一天——含纯钟点与跨日。

判据与真机证据见 doc/方案-20260922-展开时间指针.md。
真机缺口（2026-09-22 15:47 与 09-23 00:00）：用户只说「十一点左右」「早上」这类
时段/钟点、不提"今天"；旧规则要求出现日期词，于是指针落 none、展开从未被尝试。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qichi.memory.dates import referenced_past_segment  # noqa: E402

ZONE = ZoneInfo("Asia/Shanghai")
D22, D23 = date(2026, 9, 22), date(2026, 9, 23)


def _at(hour: int, minute: int = 0, day: int = 22) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=ZONE).astimezone(timezone.utc)


@pytest.mark.parametrize("text", [
    "是早上还是中午来着，应该在十点到十一点附近我找的你来着",
    "兔子你翻不到那时候吗？早上又或者是中午",
    "早上那段",
])
def test_past_band_points_at_today(text):
    """命中：时段指到今天已经过去的那一段（真机原话，23:48 说）。"""

    assert referenced_past_segment(text, now=_at(23, 48), local_zone=ZONE) == D22


@pytest.mark.parametrize("text", ["十一点左右我在上课", "十点到十一点", "十一点"])
def test_clock_only_points_at_today(text):
    """命中：**只有钟点、没有时段词**也要认（真机 16:00「十一点左右我在上课」）。"""

    assert referenced_past_segment(text, now=_at(16, 1), local_zone=ZONE) == D22


@pytest.mark.parametrize("text", ["早上", "十一点左右我在上课", "中午"])
def test_after_midnight_points_at_yesterday(text):
    """跨日：00:01 说的「早上/十一点/中午」指的是**昨天**那一段。"""

    assert referenced_past_segment(text, now=_at(0, 1, day=23), local_zone=ZONE) == D22


@pytest.mark.parametrize("text,hour", [
    ("早上好啊", 8),          # 时段刚到边界＝当下的招呼
    ("晚上吃什么", 10),       # 时段还没到
    ("中午吃了吗", 10),       # 还没到
    ("今天天气不错", 23),     # 只有日期词、无时段无钟点
    ("兔子你翻不到那时候吗", 23),
    ("", 23),
])
def test_non_pointing_text_returns_nothing(text, hour):
    """不误判：当前/未来的时段、纯招呼语、无时段无钟点——都不指向任何一段。"""

    assert referenced_past_segment(text, now=_at(hour), local_zone=ZONE) is None


def test_late_evening_band_is_not_a_past_segment():
    """晚上（17-24）在 23:48 是**正在过**的一段：不是"已经过去"。"""

    assert referenced_past_segment("晚上那会儿", now=_at(23, 48), local_zone=ZONE) is None
