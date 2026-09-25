"""对话用的日历事实：星期、休息日/工作日、季节，以及离下一个节日几天。

2026-09-14 用户裁定做「口子一」：只放代码能确定的时间事实——不联网、不猜天气、不放假消息。
数据来源：国务院办公厅关于 2026 年部分节假日安排的通知（国办发明电〔2025〕7 号），含调休上班日。
农历节日无法用公历推算，所以这张表按年维护：**表里没有的年份就不出节日行**——宁可不说，也不猜。
到时候补下一年即可，别把猜测写进表里。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

_ZONE = ZoneInfo("Asia/Shanghai")
_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
# 只在快到时才提，免得每天都念一遍「离国庆还有 200 天」。
_HORIZON_DAYS = 14


@dataclass(frozen=True, slots=True)
class Holiday:
    """一段法定假期。makeup_workdays 是国务院通知里点名的调休上班日（周末但要上班）。"""

    name: str
    start: date
    end: date
    makeup_workdays: tuple[date, ...] = ()

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


# 2026 年（国办发明电〔2025〕7 号）。补下一年的表之前，节日行在 2027 年会自然消失。
_HOLIDAYS: tuple[Holiday, ...] = (
    Holiday("元旦", date(2026, 1, 1), date(2026, 1, 3), (date(2026, 1, 4),)),
    Holiday("春节", date(2026, 2, 15), date(2026, 2, 23), (date(2026, 2, 14), date(2026, 2, 28))),
    Holiday("清明节", date(2026, 4, 4), date(2026, 4, 6)),
    Holiday("劳动节", date(2026, 5, 1), date(2026, 5, 5), (date(2026, 5, 9),)),
    Holiday("端午节", date(2026, 6, 19), date(2026, 6, 21)),
    Holiday("中秋节", date(2026, 9, 25), date(2026, 9, 27)),
    Holiday("国庆节", date(2026, 10, 1), date(2026, 10, 7), (date(2026, 9, 20), date(2026, 10, 10))),
)


def _is_rest_day(day: date) -> bool:
    """调休上班日优先于周末；假期区间内一定是休息日。"""
    if any(holiday.start <= day <= holiday.end for holiday in _HOLIDAYS):
        return True
    if any(day in holiday.makeup_workdays for holiday in _HOLIDAYS):
        return False
    return day.weekday() >= 5


def _season(day: date) -> str:
    return {1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏",
            7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋", 12: "冬"}[day.month]


def _containing(day: date) -> Holiday | None:
    for holiday in _HOLIDAYS:
        if holiday.start <= day <= holiday.end:
            return holiday
    return None


def _upcoming(day: date) -> Holiday | None:
    for holiday in sorted(_HOLIDAYS, key=lambda item: item.start):
        gap = (holiday.start - day).days
        if 0 < gap <= _HORIZON_DAYS:
            return holiday
    return None


def render_calendar_facts(now_utc: datetime) -> tuple[str, ...]:
    """这一天的事实行。约定用北京时间的那一天，不是 UTC 的那一天。"""
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise ValueError("now_utc must be an aware datetime")
    day = now_utc.astimezone(_ZONE).date()
    kind = "休息日" if _is_rest_day(day) else "工作日"
    lines = [f"今天: {day.isoformat()} {_WEEKDAYS[day.weekday()]} · {kind} · {_season(day)}季"]
    today = _containing(day)
    if today is not None:
        suffix = f"（放假至 {today.end.isoformat()}）" if today.end > day else ""
        lines.append(f"今天节日: {today.name}{suffix}")
    upcoming = _upcoming(day)
    if upcoming is not None:
        weekday = _WEEKDAYS[upcoming.start.weekday()]
        lines.append(
            f"下一个节日: {upcoming.name} {upcoming.start.isoformat()}（{weekday}）"
            f"放假 {upcoming.days} 天，还有 {(upcoming.start - day).days} 天"
        )
    return tuple(lines)