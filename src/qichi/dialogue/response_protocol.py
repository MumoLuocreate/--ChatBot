"""Parse the deliberately small text protocol emitted by the primary model."""

from __future__ import annotations

from collections.abc import Iterable
import logging
import re
from typing import Final

from qichi.domain.dialogue import (
    DialogueOutcome,
    DialogueResult,
    DialogueSkip,
    ExpressionIntent,
)


class ResponseProtocolError(ValueError):
    """The model output does not satisfy the response wire format."""


_LOGGER = logging.getLogger(__name__)


_KEY: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HANDLE: Final = re.compile(r"^[MQ](?:0|[1-9][0-9]*)$")
_MARKER_PREFIX: Final = re.compile(r"\[\[(?:qq|qichi):")
_QQ_LINE: Final = re.compile(r"^\[\[qq:(reply|face|react|voice):([^:\[\]\r\n]+)\]\]$")
# 语音只认"第几段"（1 起）。裸 [[qq:voice]] 没有冒号，会走上面的 invalid 分支 fail closed。
_VOICE_INDEX: Final = re.compile(r"^[1-9][0-9]?$")
_MAX_NATIVE_ACTIONS: Final = 3
_QQ_PREFIX_LINE: Final = re.compile(r"^\[\[qq:")
_QICHI_PREFIX_LINE: Final = re.compile(r"^\[\[qichi:")
_SKIP_MARKER: Final = "[[qichi:skip]]"
_MESSAGE_BOUNDARY: Final = re.compile(r"(?:\r\n|\r|\n)[ \t]*(?:\r\n|\r|\n)+")
_MAX_MESSAGE_PARTS: Final = 6


def _keys(value: Iterable[str], name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of keys")
    try:
        result = frozenset(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an iterable of keys") from exc
    if any(type(key) is not str or _KEY.fullmatch(key) is None for key in result):
        raise ValueError(f"{name} contains an invalid key")
    return result


def _without_control_regions(lines: list[tuple[str, bool]]) -> str:
    """Remove control lines without adding the whitespace on both sides.

    A control line is transport metadata, so the blank-line runs surrounding a
    contiguous metadata region describe the same body boundary.  Preserve the
    largest one of those runs instead of concatenating all of them.  Blank
    lines elsewhere in the model's body remain byte-for-byte unchanged.
    """

    body: list[str] = []
    index = 0
    while index < len(lines):
        line, is_control = lines[index]
        bare = line.rstrip("\r\n")
        if not is_control and bare.strip():
            body.append(line)
            index += 1
            continue

        region_start = index
        has_control = False
        blank_runs: list[list[str]] = []
        current_blank_run: list[str] = []
        while index < len(lines):
            region_line, region_is_control = lines[index]
            region_bare = region_line.rstrip("\r\n")
            if not region_is_control and region_bare.strip():
                break
            if region_is_control:
                has_control = True
                if current_blank_run:
                    blank_runs.append(current_blank_run)
                    current_blank_run = []
            else:
                current_blank_run.append(region_line)
            index += 1
        if current_blank_run:
            blank_runs.append(current_blank_run)

        if not has_control:
            body.extend(line for line, _ in lines[region_start:index])
            continue

        at_start = region_start == 0
        at_end = index == len(lines)
        if at_start or at_end:
            if at_end and body:
                body[-1] = re.sub(r"(?:\r\n|\r|\n)$", "", body[-1], count=1)
            continue

        if blank_runs:
            body.extend(max(blank_runs, key=len))

    return "".join(body)


def _message_parts(text: str) -> tuple[str, ...]:
    parts = tuple(part for part in _MESSAGE_BOUNDARY.split(text) if part.strip())
    if not parts:
        raise ResponseProtocolError("response body must not be empty")
    if len(parts) > _MAX_MESSAGE_PARTS:
        raise ResponseProtocolError("too many message parts")
    return parts


class ResponseProtocol:
    def __init__(self, *, face_keys: Iterable[str], reaction_keys: Iterable[str]) -> None:
        self.face_keys = _keys(face_keys, "face_keys")
        self.reaction_keys = _keys(reaction_keys, "reaction_keys")

    def parse(
        self,
        raw_output: str,
        *,
        source: str,
        current_event_handle: str,
        context_version: int,
    ) -> DialogueOutcome:
        if type(raw_output) is not str:
            raise TypeError("raw_output must be a string")
        if source not in {"dialogue", "interaction", "initiative"}:
            raise ValueError("source must be dialogue, interaction, or initiative")
        internal_initiative = (
            source == "initiative"
            and isinstance(current_event_handle, str)
            and re.fullmatch(r"I[0-9]+", current_event_handle) is not None
        )
        if type(current_event_handle) is not str or (
            _HANDLE.fullmatch(current_event_handle) is None and not internal_initiative
        ):
            raise ValueError("current_event_handle must be a valid message handle")
        if type(context_version) is not int or context_version < 0:
            raise ValueError("context_version must be a non-negative integer")

        # A provider may append transport line framing to an otherwise exact
        # skip marker.  Only CR/LF is ignored here; spaces or any body text
        # keep the marker invalid and fail closed below.
        if raw_output.rstrip("\r\n") == _SKIP_MARKER:
            if source != "initiative":
                raise ResponseProtocolError("skip is only valid for initiative")
            return DialogueSkip("primary", context_version)

        # With no protocol prefix, the model's text is already the payload.
        if not _MARKER_PREFIX.search(raw_output):
            if not raw_output.strip():
                raise ResponseProtocolError("response body must not be empty")
            return DialogueResult(
                raw_output,
                None,
                None,
                "primary",
                context_version,
                _message_parts(raw_output),
            )

        # Terminal CR/LF belongs to transport framing, not to the body.  A
        # control marker is metadata whenever it occupies an exact standalone
        # line; its position relative to natural text carries no meaning.
        content = raw_output.rstrip("\r\n")
        lines = content.splitlines(keepends=True)
        if not lines:
            raise ResponseProtocolError("response must not be empty")

        controls: list[tuple[str, str]] = []
        classified_lines: list[tuple[str, bool]] = []
        for line in lines:
            bare = line.rstrip("\r\n")
            if _QQ_PREFIX_LINE.match(bare):
                marker = _QQ_LINE.fullmatch(bare)
                if marker is None:
                    raise ResponseProtocolError("invalid qq control marker")
                controls.append((marker.group(1), marker.group(2)))
                classified_lines.append((line, True))
                continue
            if _QICHI_PREFIX_LINE.match(bare) or _MARKER_PREFIX.search(bare):
                raise ResponseProtocolError("project control marker is not standalone")
            classified_lines.append((line, False))

        # Models can repeat an already selected action while finishing a
        # response.  The repeated wire operation is idempotent when its
        # complete marker is identical; collapse it before checking the
        # one-reply/one-expression cardinality.  Conflicting targets or keys
        # remain an ambiguity and are still rejected below.
        controls = list(dict.fromkeys(controls))

        if len(controls) > _MAX_NATIVE_ACTIONS:
            raise ResponseProtocolError("too many control markers")

        replies = [(kind, value) for kind, value in controls if kind == "reply"]
        expressions = [(kind, value) for kind, value in controls if kind in {"face", "react"}]
        voices = [(kind, value) for kind, value in controls if kind == "voice"]
        if len(replies) > 1 or len(expressions) > 1 or len(voices) > 1:
            raise ResponseProtocolError("duplicate control action")

        text = _without_control_regions(classified_lines)
        if not text.strip():
            raise ResponseProtocolError("response body must not be empty")

        reply_target = None
        if replies and _HANDLE.fullmatch(replies[0][1]):
            reply_target = replies[0][1]

        expression = None
        if expressions:
            kind, key = expressions[0]
            if (kind == "face" and key in self.face_keys) or (
                kind == "react" and key in self.reaction_keys
            ):
                expression = ExpressionIntent(
                    "face" if kind == "face" else "reaction",
                    key,
                    None if kind == "face" else current_event_handle,
                )

        parts = _message_parts(text)
        voice_part_index = None
        if voices:
            value = voices[0][1]
            if _VOICE_INDEX.fullmatch(value) is None:
                raise ResponseProtocolError("invalid voice control marker")
            # 越界不是协议错误：与她引用了一个解析不到的句柄一样，只放弃这个动作，
            # 正文照发（2026-09-14 裁定，见历史TTS计划 §3.3）。
            if int(value) <= len(parts):
                voice_part_index = int(value)
            else:
                # 2026-09-14 深夜：这条丢弃原本一点痕迹都不留，于是「她试了但被丢掉」和
                # 「她这次没选语音」在账本上完全一样。行为不动，只让它可见。
                _LOGGER.warning(
                    "voice_part_out_of_range (requested=%s, parts=%d); action dropped, body kept",
                    value,
                    len(parts),
                )

        return DialogueResult(
            text,
            reply_target,
            expression,
            "primary",
            context_version,
            parts,
            voice_part_index,
        )
