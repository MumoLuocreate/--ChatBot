from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from uuid import uuid4

from qichi.domain.events import ConversationEvent, MessageSegment


class NormalizationError(ValueError):
    """A malformed OneBot frame that claims to be on the owner private channel."""


def _decimal_id(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise NormalizationError(f"{field} must be a decimal identifier")
    normalized = str(value)
    if not normalized or not normalized.isdecimal():
        raise NormalizationError(f"{field} must be a decimal identifier")
    return normalized


def _received_at(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise NormalizationError("received_at_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


def _occurred_at(value: object) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise NormalizationError("time must be a non-negative Unix timestamp")
    try:
        return datetime.fromtimestamp(value, timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise NormalizationError("time is outside the supported range") from error


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NormalizationError(f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise NormalizationError(f"{field} keys must be strings")
    return value


def _event_id(factory: Callable[[], str]) -> str:
    value = factory()
    if not isinstance(value, str) or not value:
        raise NormalizationError("event_id_factory must return a non-empty string")
    return value


def _segments(value: object) -> tuple[tuple[MessageSegment, ...], str | None, str | None]:
    if not isinstance(value, list):
        raise NormalizationError("message must be a list of segments")
    result: list[MessageSegment] = []
    text_parts: list[str] = []
    reply_target: str | None = None
    for index, item in enumerate(value):
        segment = _mapping(item, f"message[{index}]")
        segment_type = segment.get("type")
        if not isinstance(segment_type, str) or not segment_type:
            raise NormalizationError(f"message[{index}].type must be non-empty text")
        data = _mapping(segment.get("data"), f"message[{index}].data")
        try:
            normalized = MessageSegment(segment_type, dict(data))
        except (TypeError, ValueError) as error:
            raise NormalizationError(f"message[{index}] is not JSON-safe") from error
        if segment_type == "text":
            text = data.get("text")
            if not isinstance(text, str):
                raise NormalizationError(f"message[{index}].data.text must be a string")
            text_parts.append(text)
        if segment_type == "reply":
            target = _decimal_id(data.get("id"), f"message[{index}].data.id")
            if reply_target is not None and reply_target != target:
                raise NormalizationError("multiple contradictory reply targets")
            if reply_target is not None:
                raise NormalizationError("multiple reply segments are unsupported")
            reply_target = target
        result.append(normalized)
    text = "".join(text_parts)
    return tuple(result), (text if text else None), reply_target


def _message_event(
    raw: Mapping[str, object],
    *,
    bot_qq: str,
    owner_qq: str,
    received_at_utc: datetime,
    event_id_factory: Callable[[], str],
    outbound: bool,
) -> ConversationEvent | None:
    message_type = raw.get("message_type")
    if message_type == "group":
        return None
    if message_type != "private":
        raise NormalizationError("message_type must be private for a message event")
    self_id = _decimal_id(raw.get("self_id"), "self_id")
    if self_id != bot_qq:
        raise NormalizationError("self_id does not match configured bot")
    user_id = _decimal_id(raw.get("user_id"), "user_id")
    target_id = _decimal_id(raw.get("target_id"), "target_id")
    if outbound:
        if user_id == bot_qq and target_id != owner_qq:
            return None
        if target_id == owner_qq and user_id != bot_qq:
            raise NormalizationError("outbound user_id does not match configured bot")
        if user_id != bot_qq or target_id != owner_qq:
            return None
    elif user_id != owner_qq:
        return None
    elif target_id != owner_qq:
        raise NormalizationError("inbound target_id does not match owner conversation")
    sender = _mapping(raw.get("sender"), "sender")
    if _decimal_id(sender.get("user_id"), "sender.user_id") != user_id:
        raise NormalizationError("sender.user_id does not match user_id")
    message_id = _decimal_id(raw.get("message_id"), "message_id")
    occurred_at_utc = _occurred_at(raw.get("time"))
    message_segments, text, reply_target = _segments(raw.get("message"))
    return ConversationEvent(
        event_id=_event_id(event_id_factory),
        platform_event_id=f"message:{self_id}:{message_id}",
        platform_message_id=message_id,
        conversation_id=owner_qq,
        sequence=0,
        direction="outbound" if outbound else "inbound",
        actor="qichi" if outbound else "mumo",
        kind="text",
        text=text,
        message_segments=message_segments,
        reply_to_event_id=None,
        reply_to_platform_message_id=reply_target,
        occurred_at_utc=occurred_at_utc,
        received_at_utc=received_at_utc,
        status="sent" if outbound else "received",
        metadata={},
    )


def _notice_event(
    raw: Mapping[str, object],
    *,
    bot_qq: str,
    owner_qq: str,
    received_at_utc: datetime,
    event_id_factory: Callable[[], str],
) -> ConversationEvent | None:
    notice_type = raw.get("notice_type")
    if notice_type == "group_msg_emoji_like":
        return None
    if notice_type != "notify" or raw.get("sub_type") != "poke":
        return None
    self_id = _decimal_id(raw.get("self_id"), "self_id")
    if self_id != bot_qq:
        raise NormalizationError("self_id does not match configured bot")
    user_id = _decimal_id(raw.get("user_id"), "user_id")
    target_id = _decimal_id(raw.get("target_id"), "target_id")
    if user_id != owner_qq:
        return None
    sender_id = _decimal_id(raw.get("sender_id"), "sender_id")
    if sender_id == owner_qq and target_id == bot_qq:
        inbound = True
    elif sender_id == bot_qq and target_id == owner_qq:
        inbound = False
    else:
        raise NormalizationError("poke sender and target are inconsistent")
    occurred_at_utc = _occurred_at(raw.get("time"))
    kind = "poke"
    platform_message_id = None
    metadata: dict[str, object] = {
        "notice_type": "notify", "sub_type": "poke", "sender_id": sender_id, "target_id": target_id,
    }
    return ConversationEvent(
        event_id=_event_id(event_id_factory),
        platform_event_id=None,
        platform_message_id=None,
        conversation_id=owner_qq,
        sequence=0,
        direction="inbound" if inbound else "outbound",
        actor="mumo" if inbound else "qichi",
        kind=kind,
        text=None,
        message_segments=(),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=occurred_at_utc,
        received_at_utc=received_at_utc,
        status="sent" if not inbound else "received",
        metadata=metadata,
    )


def normalize_event(
    raw: object,
    *,
    bot_qq: object,
    owner_qq: object,
    received_at_utc: datetime,
    event_id_factory: Callable[[], str] | None = None,
) -> ConversationEvent | None:
    """Normalize the supported owner-private OneBot events without persistence."""
    payload = _mapping(raw, "raw")
    bot_id = _decimal_id(bot_qq, "bot_qq")
    owner_id = _decimal_id(owner_qq, "owner_qq")
    if bot_id == owner_id:
        raise NormalizationError("bot_qq and owner_qq must differ")
    received = _received_at(received_at_utc)
    if event_id_factory is not None and not callable(event_id_factory):
        raise NormalizationError("event_id_factory must be callable")
    factory = event_id_factory or (lambda: str(uuid4()))
    post_type = payload.get("post_type")
    if post_type == "message":
        return _message_event(payload, bot_qq=bot_id, owner_qq=owner_id, received_at_utc=received, event_id_factory=factory, outbound=False)
    if post_type == "message_sent":
        return _message_event(payload, bot_qq=bot_id, owner_qq=owner_id, received_at_utc=received, event_id_factory=factory, outbound=True)
    if post_type == "notice":
        return _notice_event(payload, bot_qq=bot_id, owner_qq=owner_id, received_at_utc=received, event_id_factory=factory)
    return None
