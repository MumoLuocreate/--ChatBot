
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
_RELATIVE = (
    ("大前天", 3),
    ("前天", 2),
    ("昨天", 1),
    ("昨夜", 1),
    ("昨晚", 1),
    ("今天", 0),
    ("今晨", 0),
    ("今早", 0),
    ("今晚", 0),
)
# 「今天」单独出现时是日常口语（"今天天气不错"），不是指向某一段经历；用户真要指
# 今天的那一段时，会顺口说出**哪一段**——"今天中午"、"今天下午"。2026-09-12 在
# 223 轮真机消息上复算：9 次点到今天的消息全部带时段词，没有一次是光秃秃的今天。
# 「昨天／前天」不需要这一层：样本里没人把它们当语气词用，说了就是在往回看。
# 2026-09-22 补：**合体写法此前一个都没被认出来**。「昨晚」（= 昨天 + 晚上）不在任何
# 词表里——_RELATIVE 只有「昨天」，时段表只有「今晚／今早」，_CLOCK_BEFORE 也只有
# 「昨天」——于是用户每次用「昨晚」指前一夜，referenced_dates 解析不出任何日期，
# 指针落到 none，**展开永不发生**（真机 2026-09-22 早上连续 5 轮，她只能说
# 「我这边没摊开细节」）。这里把合体形式按「哪一天 + 哪个时段」补回去：
# 昨夜／昨晚 → 昨天那天；今晨／今早／今晚 → 今天那天。
# 它们与既有的「今天中午」同类（今天 + 明确的时段词），所以不需要额外的时段门槛。
# 「今晚」是其中唯一带未来可能的说法（"今晚我要加班"），风险等级与既有的
# 「今天下午」相同，不额外开一档。
# 一天怎么分段，只有这一处定义：索引行用它做标签，用户点「凌晨/晚上」时也用它
# 决定先取哪一段（2026-09-12 T10）。边界与索引的显示保持一致。
BAND_EDGES = (5, 8, 11, 13, 17, 24)
# 卡D：凌晨（0-5 点）说的「早上/中午/十一点」指的是**昨天**那一段——00:01 说「早上」
# 不可能指今天早上（还没到）。这条线只在"回顾"语境里用，不影响 referenced_dates。
_NEW_DAY_HOURS = 5
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
    "昨晚": (17, 24), "昨夜": (17, 24),
    "白天": (8, 17),
}
_TIME_OF_DAY = tuple(_TIME_OF_DAY_HOURS)

# 卡C（2026-09-22）：把「时间指示词」与「命名实体」分开。指向一件事用的是命名实体
# （团建／校区／文档／TTS），而时间词指的是时间——**未来的时间词更不可能指向一件已经
# 发生过的事**。这里复用本模块既有词表，只补上未来几天：此前 dates.py 完全不管
# 明天／后天（它们不是回顾目标），所以下面这张未来词表是本文件里新增的唯一一处。
_FUTURE_RELATIVE = ("明天", "明日", "后天", "大后天")
_RELATIVE_TIME_WORDS = tuple(word for word, _offset in _RELATIVE)


def referenced_past_segment(text: str, *, now: datetime, local_zone) -> date | None:
    """文本里的时段词是否指向**今天已经过去**的那一段（卡D）。

    真机证据（2026-09-22 15:47-15:48）：用户说「是早上还是中午来着，应该在十点到十一点
    附近我找的你来着」「兔子你翻不到那时候吗？早上又或者是中午」——**只指时段与钟点、
    一个字没提"今天"**。而指针的 date 分支要求 referenced_dates 里出现今天，于是
    today_named=False、指针落 none、**展开从未被尝试**（memory_detail_count=0,
    expanded=false），她连续三轮只能说"真翻不到"。库里那段材料是有的。

    当年支撑「今天 + 时段」那条设计的实测是「9 次点到今天的消息全部**带时段词**」——
    证据说的是"带时段词"，代码却额外要求"带『今天』二字"，**实现比自己的证据更严**。

    这里的边界是「**已经过去**的那一段」：晚上（17-24）在 23:48 是正在过、不是已过，
    所以「晚上那会儿」不算；「早上好啊」在 08:00 是当前时段，也不算——招呼语不开门。
    """

    local = now.astimezone(local_zone)
    today, local_hour = local.date(), local.hour
    # 钟点（十一点/十点到十一点）：已经过去的钟点就是今天的。
    # 2026-09-22 真机缺口：用户只说「十一点左右」而没有时段词，旧规则接不住。
    for anchor in time_of_day_anchors(text):
        if anchor < local_hour:
            return today
        if local_hour < _NEW_DAY_HOURS:
            return today - timedelta(days=1)
    for start, end in time_of_day_hours(text):
        # 严格"已经结束"（end < 当前小时）：08:00 说「早上好啊」时早上（5-8）刚走到边界，
        # 那是**当下**的招呼，不算"已经过去的那一段"。
        if end < local_hour:
            return today
        # 跨日：凌晨（0-5 点）说的"早上/中午"指的是**昨天**那一段——00:01 说「早上」
        # 不可能指今天早上（还没到）。真机 2026-09-23 00:00 他点的正是昨天上午。
        if local_hour < _NEW_DAY_HOURS:
            return today - timedelta(days=1)
    return None


def names_a_time_term(value: str) -> bool:
    """这个窗口是不是时间指示（而不是命名实体）。

    卡C 用它把时间词排除在「指向」之外：2026-09-22 实测「明天下午两点提醒我开会」
    把 3 条退场经历拉进了提示，泄漏词是「下午」与「明天」；而词频分不开它们
    （「明天」在 episode 里出现 2 次，与三个真目标完全相同）——差别是**语言类别**。
    """

    text = value.strip()
    if not text:
        return False
    if any(word in text for word in _FUTURE_RELATIVE):
        return True
    if any(word in text for word in _RELATIVE_TIME_WORDS):
        return True
    return any(word in text for word in _TIME_OF_DAY)
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
