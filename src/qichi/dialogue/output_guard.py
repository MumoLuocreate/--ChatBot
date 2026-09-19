"""Narrow structural validation for primary-model text output."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Protocol

from qichi.domain.dialogue import ModelMessage


class OutputGuardError(ValueError):
    """Output failed a structural check; instances never contain model text."""


class OutputGuardInfrastructureError(RuntimeError):
    """The exact tokenizer dependency failed or returned an invalid count."""


class _TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


# These exact tokens come from the hash-verified V4 tokenizer artifact.
V4_INTERNAL_SENTINELS = frozenset(
    {
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜begin▁sys｜>",
        "<｜end▁sys｜>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁outputs▁begin｜>",
        "<｜tool▁outputs▁end｜>",
        "<｜tool▁output▁begin｜>",
        "<｜tool▁output▁end｜>",
        "<｜tool▁sep｜>",
    }
)


class OutputGuard:
    def __init__(
        self,
        token_counter: _TokenCounter,
        max_output_tokens: int,
        *,
        reject_empty: bool = True,
        reject_internal_prompt_leak: bool = True,
        reject_raw_protocol_payload: bool = True,
    ) -> None:
        if not callable(getattr(token_counter, "count_text", None)):
            raise TypeError("token_counter must provide count_text")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        for name, value in (
            ("reject_empty", reject_empty),
            ("reject_internal_prompt_leak", reject_internal_prompt_leak),
            ("reject_raw_protocol_payload", reject_raw_protocol_payload),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")
        self._counter = token_counter
        self._max = max_output_tokens
        self._reject_empty = reject_empty
        self._reject_internal = reject_internal_prompt_leak
        self._reject_json = reject_raw_protocol_payload

    def validate(self, output: str, *, role_messages: Iterable[ModelMessage] = ()) -> None:
        try:
            messages = tuple(role_messages)
        except TypeError:
            messages = None
        if messages is None:
            raise TypeError("role_messages must be iterable")
        if any(not isinstance(message, ModelMessage) for message in messages):
            raise TypeError("role_messages must contain only ModelMessage")
        if type(output) is not str:
            raise OutputGuardError("output must be text")
        stripped = output.strip()
        if self._reject_empty and not stripped:
            raise OutputGuardError("empty output")
        count_failed = False
        try:
            count = self._counter.count_text(output)
        except Exception:
            count_failed = True
        if count_failed:
            raise OutputGuardInfrastructureError("token counting failed")
        if type(count) is not int or count < 0:
            raise OutputGuardInfrastructureError("token counter returned an invalid count") from None
        if count > self._max:
            raise OutputGuardError("output exceeds token limit")
        if self._reject_json and stripped:
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                raise OutputGuardError("raw protocol payload")
        if self._reject_internal:
            if any(token in output for token in V4_INTERNAL_SENTINELS):
                raise OutputGuardError("internal sentinel")
            for message in messages:
                if message.role == "system" and message.content and message.content in stripped:
                    raise OutputGuardError("system prompt leak")
