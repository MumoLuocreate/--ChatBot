from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import re
from zoneinfo import ZoneInfo


_LOCAL_ZONE = ZoneInfo("Asia/Shanghai")
_QUIET_HOURS = re.compile(
    r"^(?P<start_hour>[01][0-9]|2[0-3]):(?P<start_minute>[0-5][0-9])-"
    r"(?P<end_hour>[01][0-9]|2[0-3]):(?P<end_minute>[0-5][0-9])$"
)


@dataclass(frozen=True)
class InitiativePolicy:
    enabled: bool = False
    idle_attempt_minutes: int = 60
    max_unanswered_attempts: int = 2
    reset_on_user_message: bool = True
    use_same_dialogue_engine: bool = True
    allow_model_to_skip: bool = True
    daily_send_limit: int | None = None
    quiet_hours_local: str | None = None
    cancel_if_conversation_changed: bool = True
    unknown_delivery_retry: bool = False
    _quiet_start: time | None = field(init=False, repr=False, compare=False)
    _quiet_end: time | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if type(self.idle_attempt_minutes) is not int or self.idle_attempt_minutes != 60:
            raise ValueError("idle_attempt_minutes must be 60")
        if (
            type(self.max_unanswered_attempts) is not int
            or self.max_unanswered_attempts < 1
        ):
            raise ValueError("max_unanswered_attempts must be a positive integer")
        for name in (
            "reset_on_user_message",
            "use_same_dialogue_engine",
            "allow_model_to_skip",
            "cancel_if_conversation_changed",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be true")
        if self.daily_send_limit is not None:
            raise ValueError("daily_send_limit must remain unset")
        if self.quiet_hours_local is None:
            quiet_start = quiet_end = None
        else:
            if type(self.quiet_hours_local) is not str:
                raise TypeError("quiet_hours_local must be a string or None")
            match = _QUIET_HOURS.fullmatch(self.quiet_hours_local)
            if match is None:
                raise ValueError("quiet_hours_local must use HH:MM-HH:MM")
            quiet_start = time(
                int(match.group("start_hour")),
                int(match.group("start_minute")),
            )
            quiet_end = time(
                int(match.group("end_hour")),
                int(match.group("end_minute")),
            )
            if quiet_start == quiet_end:
                raise ValueError("quiet_hours_local start and end must differ")
        object.__setattr__(self, "_quiet_start", quiet_start)
        object.__setattr__(self, "_quiet_end", quiet_end)
        if type(self.unknown_delivery_retry) is not bool or self.unknown_delivery_retry:
            raise ValueError("unknown_delivery_retry must be false")

    @property
    def idle_delta(self) -> timedelta:
        return timedelta(minutes=self.idle_attempt_minutes)

    def is_quiet_at(self, value: datetime) -> bool:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quiet-hours clock must be an aware datetime")
        if self._quiet_start is None or self._quiet_end is None:
            return False
        local = value.astimezone(_LOCAL_ZONE).time().replace(tzinfo=None)
        if self._quiet_start < self._quiet_end:
            return self._quiet_start <= local < self._quiet_end
        return local >= self._quiet_start or local < self._quiet_end

    def allows_initiative_at(self, value: datetime) -> bool:
        return not self.is_quiet_at(value)
