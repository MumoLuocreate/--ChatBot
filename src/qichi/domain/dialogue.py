from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Literal, Mapping, TypeAlias

from .events import Actor, JSONValue, MessageSegment, _freeze_json, _thaw_json, _aware_utc


ModelRole: TypeAlias = Literal["system", "user", "assistant"]
DialogueSource: TypeAlias = Literal["dialogue", "interaction", "initiative"]
ModelRoute: TypeAlias = Literal["primary"]


def _text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{field_name} must be a {qualifier}")
    return value


def _tuple_text(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{field_name} must be a tuple of non-empty strings")
    return value


def _route(value: Any) -> ModelRoute:
    if value != "primary":
        raise ValueError("model_route must be primary")
    return value


def _version(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("context_version must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ModelImage:
    """One picture carried by a single model message.

    Only the user's current turn carries pictures: history replay stays text,
    because re-sending old images would grow the window for no gain.  The path
    points at a file the media layer already landed; nothing here reads it --
    the provider boundary turns it into a data url.
    """

    path: str
    content_type: str

    def __post_init__(self) -> None:
        _text(self.path, "path")
        _text(self.content_type, "content_type")
        if not self.content_type.startswith("image/"):
            raise ValueError("content_type must be an image type")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"path": self.path, "content_type": self.content_type}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelImage":
        if not isinstance(value, Mapping):
            raise TypeError("model image must be a mapping")
        return cls(path=value.get("path"), content_type=value.get("content_type"))


@dataclass(frozen=True)
class ModelMessage:
    role: ModelRole
    content: str
    # Pictures belong to the user's current turn only, and a system message may
    # never carry one: a role file or a memory block must not be able to smuggle
    # an image into the prompt.
    images: tuple[ModelImage, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant")
        _text(self.content, "content", allow_empty=True)
        if not isinstance(self.images, tuple) or not all(
            isinstance(image, ModelImage) for image in self.images
        ):
            raise TypeError("images must be a tuple of ModelImage")
        if self.images and self.role != "user":
            raise ValueError("only a user message may carry images")

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {"role": self.role, "content": self.content}
        if self.images:
            payload["images"] = [image.to_dict() for image in self.images]
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelMessage":
        if not isinstance(value, Mapping):
            raise TypeError("model message must be a mapping")
        raw = value.get("images", ())
        if not isinstance(raw, (list, tuple)):
            raise TypeError("images must be a list")
        return cls(
            role=value.get("role"),
            content=value.get("content"),
            images=tuple(ModelImage.from_dict(item) for item in raw),
        )


@dataclass(frozen=True)
class CapabilityManifest:
    capabilities: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, Mapping):
            raise TypeError("capabilities must be a mapping")
        object.__setattr__(self, "capabilities", _freeze_json(self.capabilities, "capabilities"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {"capabilities": _thaw_json(self.capabilities)}  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityManifest":
        if not isinstance(value, Mapping):
            raise TypeError("capability manifest must be a mapping")
        return cls(capabilities=value.get("capabilities", {}))


@dataclass(frozen=True)
class QuotedTarget:
    conversation_id: str
    event_id: str
    platform_message_id: str | None
    handle: str
    actor: Actor
    text: str | None
    message_segments: tuple[MessageSegment, ...]
    occurred_at_utc: datetime

    def __post_init__(self) -> None:
        _text(self.conversation_id, "conversation_id")
        _text(self.event_id, "event_id")
        if self.platform_message_id is not None:
            _text(self.platform_message_id, "platform_message_id")
        _text(self.handle, "handle")
        if self.actor not in {"mumo", "qichi", "platform"}:
            raise ValueError("actor must be mumo, qichi, or platform")
        if self.text is not None:
            _text(self.text, "text", allow_empty=True)
        if not isinstance(self.message_segments, tuple) or not all(
            isinstance(segment, MessageSegment) for segment in self.message_segments
        ):
            raise TypeError("message_segments must be a tuple of MessageSegment")
        object.__setattr__(self, "occurred_at_utc", _aware_utc(self.occurred_at_utc, "occurred_at_utc"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "conversation_id": self.conversation_id,
            "event_id": self.event_id,
            "platform_message_id": self.platform_message_id,
            "handle": self.handle,
            "actor": self.actor,
            "text": self.text,
            "message_segments": [segment.to_dict() for segment in self.message_segments],
            "occurred_at_utc": self.occurred_at_utc.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuotedTarget":
        if not isinstance(value, Mapping):
            raise TypeError("quoted target must be a mapping")
        segments = value.get("message_segments")
        if not isinstance(segments, (list, tuple)):
            raise TypeError("message_segments must be a list")
        return cls(
            conversation_id=value.get("conversation_id"),
            event_id=value.get("event_id"),
            platform_message_id=value.get("platform_message_id"),
            handle=value.get("handle"),
            actor=value.get("actor"),
            text=value.get("text"),
            message_segments=tuple(MessageSegment.from_dict(item) for item in segments),
            occurred_at_utc=datetime.fromisoformat(value.get("occurred_at_utc")),
        )


@dataclass(frozen=True)
class ExpressionIntent:
    kind: Literal["face", "reaction"]
    key: str
    target_event_handle: str | None

    def __post_init__(self) -> None:
        if self.kind not in {"face", "reaction"}:
            raise ValueError("expression intent kind must be face or reaction")
        _text(self.key, "key")
        if self.target_event_handle is not None:
            _text(self.target_event_handle, "target_event_handle")
        if self.kind == "face" and self.target_event_handle is not None:
            raise ValueError("face expression intent must not have a target")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kind": self.kind,
            "key": self.key,
            "target_event_handle": self.target_event_handle,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpressionIntent":
        if not isinstance(value, Mapping):
            raise TypeError("expression intent must be a mapping")
        return cls(
            kind=value.get("kind"),
            key=value.get("key"),
            target_event_handle=value.get("target_event_handle"),
        )


@dataclass(frozen=True)
class DialogueInput:
    conversation_id: str
    trigger_event_ids: tuple[str, ...]
    current_event_handle: str
    quoted_target: QuotedTarget | None
    context_version: int
    role_messages: tuple[ModelMessage, ...]
    current_time: datetime
    platform_capabilities: CapabilityManifest
    source: DialogueSource

    def __post_init__(self) -> None:
        _text(self.conversation_id, "conversation_id")
        _tuple_text(self.trigger_event_ids, "trigger_event_ids")
        _text(self.current_event_handle, "current_event_handle")
        if self.quoted_target is not None and not isinstance(self.quoted_target, QuotedTarget):
            raise TypeError("quoted_target must be QuotedTarget or None")
        _version(self.context_version)
        if not isinstance(self.role_messages, tuple) or not all(
            isinstance(message, ModelMessage) for message in self.role_messages
        ):
            raise TypeError("role_messages must be a tuple of ModelMessage")
        object.__setattr__(self, "current_time", _aware_utc(self.current_time, "current_time"))
        if not isinstance(self.platform_capabilities, CapabilityManifest):
            raise TypeError("platform_capabilities must be CapabilityManifest")
        if self.source not in {"dialogue", "interaction", "initiative"}:
            raise ValueError("source must be dialogue, interaction, or initiative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "conversation_id": self.conversation_id,
            "trigger_event_ids": list(self.trigger_event_ids),
            "current_event_handle": self.current_event_handle,
            "quoted_target": self.quoted_target.to_dict() if self.quoted_target is not None else None,
            "context_version": self.context_version,
            "role_messages": [message.to_dict() for message in self.role_messages],
            "current_time": self.current_time.isoformat(),
            "platform_capabilities": self.platform_capabilities.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DialogueInput":
        if not isinstance(value, Mapping):
            raise TypeError("dialogue input must be a mapping")
        trigger_ids = value.get("trigger_event_ids")
        messages = value.get("role_messages")
        if not isinstance(trigger_ids, (list, tuple)) or not isinstance(messages, (list, tuple)):
            raise TypeError("trigger_event_ids and role_messages must be lists")
        quoted = value.get("quoted_target")
        return cls(
            conversation_id=value.get("conversation_id"),
            trigger_event_ids=tuple(trigger_ids),
            current_event_handle=value.get("current_event_handle"),
            quoted_target=QuotedTarget.from_dict(quoted) if quoted is not None else None,
            context_version=value.get("context_version"),
            role_messages=tuple(ModelMessage.from_dict(item) for item in messages),
            current_time=datetime.fromisoformat(value.get("current_time")),
            platform_capabilities=CapabilityManifest.from_dict(value.get("platform_capabilities")),
            source=value.get("source"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "DialogueInput":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class DialogueResult:
    text: str
    reply_target: str | None
    expression_intent: ExpressionIntent | None
    model_route: ModelRoute
    context_version: int
    message_parts: tuple[str, ...] | None = None
    # 2026-09-14：哪一段用语音说（1 起）。语音是**载体**，不是表达——
    # 台词与语气仍由这一次生成决定，这里只记"第几段改用音频"。
    voice_part_index: int | None = None
    # 2026-09-20：她在这一轮里标记的场景状态（"on"/"off"）。这只是**状态**，
    # 不是表达：正文里不会出现它，代码只把它记下来、下一轮当作事实陈述。
    scene_mark: str | None = None

    def __post_init__(self) -> None:
        _text(self.text, "text")
        if self.reply_target is not None:
            _text(self.reply_target, "reply_target")
        if self.expression_intent is not None and not isinstance(self.expression_intent, ExpressionIntent):
            raise TypeError("expression_intent must be ExpressionIntent or None")
        _route(self.model_route)
        _version(self.context_version)
        parts = (self.text,) if self.message_parts is None else _tuple_text(
            self.message_parts, "message_parts"
        )
        if not parts or any(not part.strip() for part in parts):
            raise ValueError("message_parts must contain non-blank text")
        cursor = 0
        for part in parts:
            position = self.text.find(part, cursor)
            if position < 0 or self.text[cursor:position].strip():
                raise ValueError("message_parts must preserve text order")
            cursor = position + len(part)
        if self.text[cursor:].strip():
            raise ValueError("message_parts must cover all non-whitespace text")
        object.__setattr__(self, "message_parts", parts)
        index = self.voice_part_index
        if index is not None:
            if type(index) is not int:
                raise TypeError("voice_part_index must be an int or None")
            if not 1 <= index <= len(parts):
                raise ValueError("voice_part_index must point at an existing message part")
        if self.scene_mark is not None and self.scene_mark not in {"on", "off"}:
            raise ValueError("scene_mark must be on, off, or None")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "text": self.text,
            "reply_target": self.reply_target,
            "expression_intent": self.expression_intent.to_dict() if self.expression_intent else None,
            "model_route": self.model_route,
            "context_version": self.context_version,
            "message_parts": list(self.message_parts),
            "voice_part_index": self.voice_part_index,
            "scene_mark": self.scene_mark,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DialogueResult":
        if not isinstance(value, Mapping):
            raise TypeError("dialogue result must be a mapping")
        expression = value.get("expression_intent")
        raw_parts = value.get("message_parts")
        if raw_parts is not None and not isinstance(raw_parts, (list, tuple)):
            raise TypeError("message_parts must be a list")
        return cls(
            text=value.get("text"),
            reply_target=value.get("reply_target"),
            expression_intent=ExpressionIntent.from_dict(expression) if expression is not None else None,
            model_route=value.get("model_route"),
            context_version=value.get("context_version"),
            message_parts=(
                None
                if raw_parts is None
                else tuple(raw_parts)
            ),
            voice_part_index=value.get("voice_part_index"),
            scene_mark=value.get("scene_mark"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "DialogueResult":
        return cls.from_dict(json.loads(value))


def split_voice_part(result: DialogueResult) -> tuple["DialogueResult | None", str]:
    """把要念的那一段从这一轮里摘出来。

    返回（剩下的纯文字结果或 None, 语音段的原文）。
    摘出去之后剩下的是一个完整的 DialogueResult，可以原样走今天那条发送链路 ——
    语音段因此成为独立的投递单元，不会把文字组的语义搅乱。

    剩下的文字段为空时返回 (None, 原文)：这一轮只有语音，调用方需要单独处理。
    """

    if not isinstance(result, DialogueResult):
        raise TypeError("result must be a DialogueResult")
    index = result.voice_part_index
    if index is None:
        return result, ""
    parts = result.message_parts or (result.text,)
    spoken = parts[index - 1]
    remaining = tuple(part for position, part in enumerate(parts) if position != index - 1)
    if not remaining:
        return None, spoken
    return (
        DialogueResult(
            text="\n\n".join(remaining),
            reply_target=result.reply_target,
            expression_intent=result.expression_intent,
            model_route=result.model_route,
            context_version=result.context_version,
            message_parts=remaining,
            # 场景标记属于这一轮，不属于被摘出去的那一段：摘剩的文字结果照旧带着它。
            scene_mark=result.scene_mark,
        ),
        spoken,
    )


@dataclass(frozen=True)
class DialogueSkip:
    model_route: ModelRoute
    context_version: int

    def __post_init__(self) -> None:
        _route(self.model_route)
        _version(self.context_version)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"model_route": self.model_route, "context_version": self.context_version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DialogueSkip":
        if not isinstance(value, Mapping):
            raise TypeError("dialogue skip must be a mapping")
        return cls(model_route=value.get("model_route"), context_version=value.get("context_version"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "DialogueSkip":
        return cls.from_dict(json.loads(value))


DialogueOutcome: TypeAlias = DialogueResult | DialogueSkip
