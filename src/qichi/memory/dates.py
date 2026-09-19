
"""Date references in the user's own words, resolved as calendar facts.

"九号那天" is a statement about the calendar, not a judgement about meaning, so
the code resolves it.  Before this module the only clue the search had was word
overlap, and a newer conversation that merely *talks about* the ninth outranked
the episode that actually happened on the ninth -- the model then read back its
own past sentence "I have nothing from the ninth" as the content of the request.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Sequence

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_MONTH_DAY = re.compile(
    r"([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*月\s*([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*[号日]"
)
_DAY = re.compile(r"([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*[号日]")
_RELATIVE = (("大前天", 3), ("前天", 2), ("昨天", 1), ("今天", 0))
# 「今天」单独出现时是日常口语（"今天天气不错"），不是指向某一段经历；用户真要指
# 今天的那一段时，会顺口说出**哪一段**——"今天中午"、"今天下午"。2026-09-12 在
# 223 轮真机消息上复算：9 次点到今天的消息全部带时段词，没有一次是光秃秃的今天。
# 「昨天／前天」不需要这一层：样本里没人把它们当语气词用，说了就是在往回看。
# 一天怎么分段，只有这一处定义：索引行用它做标签，用户点「凌晨/晚上」时也用它
# 决定先取哪一段（2026-09-12 T10）。边界与索引的显示保持一致。
BAND_EDGES = (5, 8, 11, 13, 17, 24)
BAND_LABELS = ("凌晨", "早上", "上午", "中午", "下午", "晚上")


def band_label(hour: int) -> str:
    """The label the index prints for that hour."""

    for edge, label in zip(BAND_EDGES, BAND_LABELS):
        if hour < edge:
            return label
    return BAND_LABELS[-1]


# 用户嘴里的时段词 → 小时区间。同义的说法指同一段；「白天」是跨段的粗说法。
_TIME_OF_DAY_HOURS: dict[str, tuple[int, int]] = {
    "凌晨": (0, 5), "半夜": (0, 5),
    "早上": (5, 8), "早晨": (5, 8), "一大早": (5, 8), "今早": (5, 8), "今晨": (5, 8),
    "上午": (8, 11),
    "中午": (11, 13),
    "下午": (13, 17),
    "傍晚": (17, 19),
    "晚上": (17, 24), "今晚": (17, 24), "夜里": (17, 24),
    "白天": (8, 17),
}
_TIME_OF_DAY = tuple(_TIME_OF_DAY_HOURS)
_MAX_DAYS = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def _value(token: str) -> int | None:
    """Read either an Arabic number or a written-out number up to thirty-one."""

    if token.isdigit():
        return int(token)
    if not token or any(character not in _DIGITS and character != "十" for character in token):
        return None
    if token == "十":
        return 10
    if "十" not in token:
        return _DIGITS[token] if len(token) == 1 else None
    tens, _, ones = token.partition("十")
    high = _DIGITS.get(tens, 1) if tens else 1
    if ones:
        if ones not in _DIGITS:
            return None
        return high * 10 + _DIGITS[ones]
    return high * 10


def _past_date(year: int, month: int, day: int, today: date) -> date | None:
    """The most recent occurrence of a month/day that is not in the future."""

    for offset in range(0, 25):
        total = (year * 12 + (month - 1)) - offset
        candidate_year, candidate_month = divmod(total, 12)
        candidate_month += 1
        if day > _MAX_DAYS[candidate_month]:
            continue
        candidate = date(candidate_year, candidate_month, day)
        if candidate <= today:
            return candidate
    return None


# 「几点」：阿拉伯数字或汉字数字 + 点/点钟。汉字的「一点」也是「稍微」的意思
# （"我今天有点失落"），所以要靠上下文把钟点认出来，否则会把一句日常话读成凌晨一点。
_CLOCK = re.compile(r"([0-9]{1,2}|[零〇一二三四五六七八九十]{1,3})\s*(?:点|点钟)")
_CLOCK_AFTER = ("钟", "半", "多", "整", "左右", "过后", "以后", "之后", "以前", "之前", "前后")
_CLOCK_BEFORE = (
    "凌晨", "早上", "早晨", "一大早", "上午", "中午", "下午", "傍晚", "晚上", "半夜", "夜里",
    "今晚", "今早", "今天", "昨天", "前天", "大前天", "当天", "第二天", "次日",
)
_CLOCK_NEVER_BEFORE = ("有", "差", "不", "这么", "那么", "一")


def time_of_day_anchors(text: str) -> tuple[int, ...]:
    """The clock hours the message names ("零点", "凌晨一点多", "23点"), if any.

    Only used to rank that day's episodes: the one nearest the hour the user named
    comes first, so "零点过后" no longer loses to a later fragment on the same day.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    found: list[int] = []
    for match in _CLOCK.finditer(text):
        hour = _value(match.group(1))
        if hour is None or not 0 <= hour <= 23:
            continue
        before = text[: match.start()]
        after = text[match.end() : match.end() + 3]
        if before.endswith(_CLOCK_NEVER_BEFORE):
            continue
        spoken = (
            len(match.group(1)) > 1
            or after.startswith(_CLOCK_AFTER)
            or before.endswith(_CLOCK_BEFORE)
        )
        if spoken and hour not in found:
            found.append(hour)
    return tuple(found)


def time_of_day_hours(text: str) -> tuple[tuple[int, int], ...]:
    """The hour windows the message points at, in the order it mentions them.

    "今天的凌晨" points at 00:00-05:00; that has to decide *which* of the day's
    episodes come first, not merely whether "today" counts (2026-09-12 T10:
    asking about 凌晨 handed over the afternoon and morning instead).
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    windows: list[tuple[int, int]] = []
    for index in range(len(text)):
        for word, window in _TIME_OF_DAY_HOURS.items():
            if text.startswith(word, index) and window not in windows:
                windows.append(window)
    return tuple(windows)


def names_a_time_of_day(text: str) -> bool:
    """Does the message say *which part* of the day it means?

    Only used to keep a bare "今天" from being read as a pointer at today's
    stored episode: the word doubles as ordinary filler in daily chat, while
    "今天中午" is a place in the timeline.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return any(word in text for word in _TIME_OF_DAY)


def referenced_dates(terms: Sequence[str], *, now: datetime, local_zone) -> tuple[date, ...]:
    """Return the local dates the user named, most recent first."""

    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        raise TypeError("terms must be a sequence of strings")
    today = now.astimezone(local_zone).date()
    found: list[date] = []
    for term in terms:
        if not isinstance(term, str) or not term:
            continue
        for day_offset in (offset for label, offset in _RELATIVE if label in term):
            moment = today - timedelta(days=day_offset)
            if moment not in found:
                found.append(moment)
        for month_token, day_token in _MONTH_DAY.findall(term):
            month, day = _value(month_token), _value(day_token)
            if month is None or day is None or not 1 <= month <= 12 or not 1 <= day <= 31:
                continue
            resolved = _past_date(today.year, month, day, today)
            if resolved is not None and resolved not in found:
                found.append(resolved)
        for day_token in _DAY.findall(term):
            day = _value(day_token)
            if day is None or not 1 <= day <= 31:
                continue
            resolved = _past_date(today.year, today.month, day, today)
            if resolved is not None and resolved not in found:
                found.append(resolved)
    found.sort(reverse=True)
    return tuple(found)
