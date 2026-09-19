from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qichi.dialogue.calendar_facts import render_calendar_facts


def at(year, month, day, hour=6):
    # 北京时间 14:00 的那一天；用 UTC 传进去，验证取的是北京那一天。
    return datetime(year, month, day, hour, 0, tzinfo=timezone.utc)


def test_plain_workday_carries_weekday_season_and_holiday_distance():
    lines = render_calendar_facts(at(2026, 9, 14))
    assert lines[0] == "今天: 2026-09-14 星期一 · 工作日 · 秋季"
    assert lines[1] == "下一个节日: 中秋节 2026-09-25（星期五）放假 3 天，还有 11 天"


def test_mid_autumn_day_is_a_rest_day_and_names_the_next_holiday():
    lines = render_calendar_facts(at(2026, 9, 25))
    assert "休息日" in lines[0]
    assert "今天节日: 中秋节（放假至 2026-09-27）" in lines
    assert any(line.startswith("下一个节日: 国庆节") for line in lines)


def test_makeup_workday_beats_the_weekend():
    # 10 月 10 日是周六，但国办通知点名上班。
    lines = render_calendar_facts(at(2026, 10, 10))
    assert lines == ("今天: 2026-10-10 星期六 · 工作日 · 秋季",)


def test_sunday_makeup_workday_before_national_day():
    lines = render_calendar_facts(at(2026, 9, 20))
    assert "工作日" in lines[0]
    # 最近的那个节日是中秋（9/25），不是国庆——只报最近的一个。
    assert any("中秋节" in line for line in lines)


def test_far_away_holiday_is_not_mentioned_every_day():
    lines = render_calendar_facts(at(2026, 6, 1))
    assert len(lines) == 1
    assert "节日" not in lines[0]


def test_unknown_year_falls_back_to_weekday_only():
    lines = render_calendar_facts(at(2027, 6, 1))
    assert lines == ("今天: 2027-06-01 星期二 · 工作日 · 夏季",)


def test_local_date_comes_from_beijing_not_utc():
    # UTC 还是 14 日，北京已经是 15 日。
    lines = render_calendar_facts(datetime(2026, 9, 14, 17, 30, tzinfo=timezone.utc))
    assert lines[0].startswith("今天: 2026-09-15 星期二")


def test_seasons():
    assert "春季" in render_calendar_facts(at(2026, 4, 1))[0]
    assert "夏季" in render_calendar_facts(at(2026, 7, 1))[0]
    assert "秋季" in render_calendar_facts(at(2026, 11, 1))[0]
    assert "冬季" in render_calendar_facts(at(2026, 1, 20))[0]


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        render_calendar_facts(datetime(2026, 9, 14))