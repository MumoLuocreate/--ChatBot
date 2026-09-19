from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Any, Literal, Mapping, TypeAlias


JSONPrimitive: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
Direction: TypeAlias = Literal["inbound", "outbound", "internal"]
Actor: TypeAlias = Literal["mumo", "qichi", "platform"]


class _FrozenDict(dict[str, Any]):
    """A JSON-serializable dict that rejects mutation after construction."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze_json(value: Any, path: str = "value") -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen = _FrozenDict()
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            dict.__setitem__(frozen, key, _freeze_json(child, f"{path}.{key}"))
        return frozen
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child, f"{path}[]") for child in value)  # type: ignore[return-value]
    raise TypeError(f"{path} is not JSON serializable")


def _thaw_json(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}  # type: ignore[return-value]
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]  # type: ignore[return-value]
    return value


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class MessageSegment:
    type: str
    data: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        _non_empty_text(self.type, "type")
        if not isinstance(self.data, Mapping):
            raise TypeError("data must be a mapping")
        object.__setattr__(self, "data", _freeze_json(self.data, "data"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {"type": self.type, "data": _thaw_json(self.data)}  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MessageSegment":
        if not isinstance(value, Mapping):
            raise TypeError("message segment must be a mapping")
        return cls(type=value.get("type"), data=value.get("data"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "MessageSegment":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ConversationEvent:
    event_id: str
    platform_event_id: str | None
    platform_message_id: str | None
    conversation_id: str
    sequence: int
    direction: Direction
    actor: Actor
    kind: str
    text: str | None
    message_segments: tuple[MessageSegment, ...]
    reply_to_event_id: str | None
    reply_to_platform_message_id: str | None
    occurred_at_utc: datetime
    received_at_utc: datetime
    status: str
    metadata: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        _non_empty_text(self.event_id, "event_id")
        _optional_text(self.platform_event_id, "platform_event_id")
        _optional_text(self.platform_message_id, "platform_message_id")
        _non_empty_text(self.conversation_id, "conversation_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if self.direction not in {"inbound", "outbound", "internal"}:
            raise ValueError("direction must be inbound, outbound, or internal")
        if self.actor not in {"mumo", "qichi", "platform"}:
            raise ValueError("actor must be mumo, qichi, or platform")
        _non_empty_text(self.kind, "kind")
        _optional_text(self.text, "text")
        if not isinstance(self.message_segments, tuple) or not all(
            isinstance(segment, MessageSegment) for segment in self.message_segments
        ):
            raise TypeError("message_segments must be a tuple of MessageSegment")
        _optional_text(self.reply_to_event_id, "reply_to_event_id")
        _optional_text(self.reply_to_platform_message_id, "reply_to_platform_message_id")
        object.__setattr__(self, "occurred_at_utc", _aware_utc(self.occurred_at_utc, "occurred_at_utc"))
        object.__setattr__(self, "received_at_utc", _aware_utc(self.received_at_utc, "received_at_utc"))
        _non_empty_text(self.status, "status")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @property
    def visible_handle(self) -> str | None:
        if self.direction == "internal":
            return None
        if self.actor == "mumo":
            return f"M{self.sequence}"
        if self.actor == "qichi":
            return f"Q{self.sequence}"
        return None

    @property
    def handle(self) -> str | None:
        return self.visible_handle

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "event_id": self.event_id,
            "platform_event_id": self.platform_event_id,
            "platform_message_id": self.platform_message_id,
            "conversation_id": self.conversation_id,
            "sequence": self.sequence,
            "direction": self.direction,
            "actor": self.actor,
            "kind": self.kind,
            "text": self.text,
            "message_segments": [segment.to_dict() for segment in self.message_segments],
            "reply_to_event_id": self.reply_to_event_id,
            "reply_to_platform_message_id": self.reply_to_platform_message_id,
            "occurred_at_utc": self.occurred_at_utc.isoformat(),
            "received_at_utc": self.received_at_utc.isoformat(),
            "status": self.status,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConversationEvent":
        if not isinstance(value, Mapping):
            raise TypeError("conversation event must be a mapping")
        segments = value.get("message_segments")
        if not isinstance(segments, (list, tuple)):
            raise TypeError("message_segments must be a list")
        return cls(
            event_id=value.get("event_id"),
            platform_event_id=value.get("platform_event_id"),
            platform_message_id=value.get("platform_message_id"),
            conversation_id=value.get("conversation_id"),
            sequence=value.get("sequence"),
            direction=value.get("direction"),
            actor=value.get("actor"),
            kind=value.get("kind"),
            text=value.get("text"),
            message_segments=tuple(MessageSegment.from_dict(item) for item in segments),
            reply_to_event_id=value.get("reply_to_event_id"),
            reply_to_platform_message_id=value.get("reply_to_platform_message_id"),
            occurred_at_utc=datetime.fromisoformat(value.get("occurred_at_utc")),
            received_at_utc=datetime.fromisoformat(value.get("received_at_utc")),
            status=value.get("status"),
            metadata=value.get("metadata"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "ConversationEvent":
        return cls.from_dict(json.loads(value))
