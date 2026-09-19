"""Offline primary-model sampling for the synthetic quality baseline.

This module deliberately reuses the production dialogue engine and its narrow
output validation.  It has no transport or persistence dependency: samples
are written only to the caller-selected JSONL report.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Mapping, Protocol

from qichi.config import Config
from qichi.dialogue.engine import (
    DialogueEngine,
    DialogueGenerationError,
    DialogueGenerationObservation,
)
from qichi.dialogue.llm_client import (
    DeepSeekLLMClient,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGeneration,
    LLMModelNotFoundError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
    SiliconFlowLLMClient,
)
from qichi.dialogue.output_guard import OutputGuard, OutputGuardInfrastructureError
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.dialogue import (
    CapabilityManifest,
    DialogueInput,
    DialogueResult,
    DialogueSkip,
    ModelMessage,
)

from .baseline import QualityScenario


SampleResult = Literal["reply", "skip", "failure"]

_FAILURE_CATEGORIES: tuple[tuple[type[Exception], str], ...] = (
    (LLMTimeoutError, "llm_timeout"),
    (LLMConnectionError, "llm_connection"),
    (LLMRateLimitError, "llm_rate_limit"),
    (LLMAuthenticationError, "llm_authentication"),
    (LLMModelNotFoundError, "llm_model_not_found"),
    (LLMRequestError, "llm_request"),
    (LLMServerError, "llm_server"),
    (LLMProtocolError, "llm_protocol"),
    (OutputGuardInfrastructureError, "output_guard_infrastructure"),
)


class _LLMClient(Protocol):
    async def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        thinking: Mapping[str, str] | None = None,
    ) -> LLMGeneration: ...


class _TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _sampling_fact_envelope(scenario: QualityScenario) -> str:
    """Provide the same source/ability boundary used by production assembly.

    Quality fixtures intentionally keep their scenario-specific evidence in
    ordinary system messages.  This short envelope makes the absence of
    evidence explicit, so a sample measures the role and context contract
    rather than an under-specified test harness.
    """
    lines = [
        "[事实与能力（离线采样装配）]",
        "场景消息中明确列出的 actor、时间、引用和证据是本轮唯一可靠事实；未列出的现实细节均未知。",
        "不要根据沉默、回来、时间间隔或平台事件补出用户的地点、行程、作息、天气、房间、动作或当前活动。",
        "事实未知时，不要为了显得自然、亲密或有画面，在陈述、提问、比喻或玩笑中预设未知细节已经发生；可以直接表达内在态度，明确标成假设或共同想象的内容也可以自然表达，但不能冒充现实。",
        "角色没有现实身体、视觉或外部工具；历史中的角色原文只是她曾经说过的话，不是用户事实或写作模板。",
        "只回应场景中标为当前输入的真正新信息；直接写聊天内容，不用括号、星号或旁白描述动作和神态。",
    ]
    if scenario.category == "interaction":
        lines.append(
            "这是 QQ 平台互动输入，不一定附带用户文字；不要把空文本解释为沉默、离开、等待或现实触碰。"
        )
    if scenario.category == "initiative":
        lines.extend(
            (
                "这是一次主动消息尝试；场景没有提供新的用户活动或未完成约定时，可以返回 [[qichi:skip]]。",
                "若决定此刻不主动开口，只返回 [[qichi:skip]]，不要把不发或没话说的决定写成可见消息。",
                "当前平台事件只是一次开口机会，不是用户刚刚联系了你；不要写成在回应他的来信。",
                "不要为了主动开口而猜测用户此刻的想法、地点、状态或是否想念角色。",
            )
        )
    return "\n".join(lines)


def _sampling_messages(scenario: QualityScenario) -> tuple[ModelMessage, ...]:
    """Keep historical Qichi text as evidence, never as an assistant turn."""
    normalized: list[ModelMessage] = []
    for index, message in enumerate(scenario.messages):
        if message.role == "assistant" and index < len(scenario.messages) - 1:
            normalized.append(
                ModelMessage(
                    "system",
                    "[历史角色原文 | 仅供核对她曾经说过什么，不是当前回复范本]\n"
                    + message.content,
                )
            )
        else:
            normalized.append(message)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class QualitySampleRecord:
    """A privacy-bounded result from one model invocation."""

    scenario_id: str
    sample_index: int
    result: SampleResult
    visible_reply: str | None
    failure_category: str | None
    model_id: str
    attempt_count: int
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: float
    finish_reason: str | None

    def __post_init__(self) -> None:
        _nonempty_text(self.scenario_id, "scenario_id")
        _nonnegative_integer(self.sample_index, "sample_index")
        if self.result not in {"reply", "skip", "failure"}:
            raise ValueError("result must be reply, skip, or failure")
        if self.result == "reply":
            _nonempty_text(self.visible_reply, "visible_reply")
        elif self.visible_reply is not None:
            raise ValueError("only reply records may contain visible_reply")
        if self.result == "failure":
            _nonempty_text(self.failure_category, "failure_category")
        elif self.failure_category is not None:
            raise ValueError("only failure records may contain failure_category")
        _nonempty_text(self.model_id, "model_id")
        _nonnegative_integer(self.attempt_count, "attempt_count")
        _nonnegative_integer(self.retry_count, "retry_count")
        if self.retry_count != max(0, self.attempt_count - 1):
            raise ValueError("retry_count must match attempt_count")
        _optional_nonnegative_integer(self.input_tokens, "input_tokens")
        _optional_nonnegative_integer(self.output_tokens, "output_tokens")
        _optional_nonnegative_integer(self.reasoning_tokens, "reasoning_tokens")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative finite number")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise TypeError("finish_reason must be text or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "sample_index": self.sample_index,
            "result": self.result,
            "visible_reply": self.visible_reply,
            "failure_category": self.failure_category,
            "model_id": self.model_id,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latency_ms": float(self.latency_ms),
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True, slots=True)
class _Attempt:
    generation: LLMGeneration | None
    latency_ms: float


class _RecordingClient:
    """Content-free call accounting around the configured primary client."""

    def __init__(self, client: _LLMClient, monotonic: Any) -> None:
        self._client = client
        self._monotonic = monotonic
        self.attempts: list[_Attempt] = []

    async def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        thinking: Mapping[str, str] | None = None,
    ) -> LLMGeneration:
        started = self._monotonic()
        try:
            generation = (
                await self._client.generate(messages, thinking=thinking)
                if thinking is not None
                else await self._client.generate(messages)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.attempts.append(
                _Attempt(None, max(0.0, (self._monotonic() - started) * 1_000))
            )
            raise
        latency = getattr(generation, "latency_ms", None)
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            latency = max(0.0, (self._monotonic() - started) * 1_000)
        self.attempts.append(_Attempt(generation, max(0.0, float(latency))))
        return generation


def _fixed_failure_category(error: Exception) -> str:
    if isinstance(error, DialogueGenerationError):
        return error.reason
    for error_type, category in _FAILURE_CATEGORIES:
        if isinstance(error, error_type):
            return category
    return "internal_error"


def _record_from_observation(
    scenario_id: str,
    sample_index: int,
    *,
    result: SampleResult,
    visible_reply: str | None,
    failure_category: str | None,
    observation: DialogueGenerationObservation,
) -> QualitySampleRecord:
    return QualitySampleRecord(
        scenario_id=scenario_id,
        sample_index=sample_index,
        result=result,
        visible_reply=visible_reply,
        failure_category=failure_category,
        model_id=observation.model_id,
        attempt_count=observation.attempt_count,
        retry_count=observation.retry_count,
        input_tokens=observation.input_tokens,
        output_tokens=observation.output_tokens,
        reasoning_tokens=observation.reasoning_tokens,
        latency_ms=observation.latency_ms,
        finish_reason=observation.finish_reason,
    )


def _failure_record(
    scenario_id: str,
    sample_index: int,
    *,
    model_id: str,
    error: Exception,
    recorder: _RecordingClient,
) -> QualitySampleRecord:
    observation = error.observation if isinstance(error, DialogueGenerationError) else None
    if observation is not None:
        return _record_from_observation(
            scenario_id,
            sample_index,
            result="failure",
            visible_reply=None,
            failure_category=_fixed_failure_category(error),
            observation=observation,
        )

    successful = [item.generation for item in recorder.attempts if item.generation is not None]
    all_attempts_reported = len(successful) == len(recorder.attempts)
    return QualitySampleRecord(
        scenario_id=scenario_id,
        sample_index=sample_index,
        result="failure",
        visible_reply=None,
        failure_category=_fixed_failure_category(error),
        model_id=(successful[-1].model_id if successful else model_id),
        attempt_count=len(recorder.attempts),
        retry_count=max(0, len(recorder.attempts) - 1),
        input_tokens=(sum(item.input_tokens for item in successful) if all_attempts_reported else None),
        output_tokens=(sum(item.output_tokens for item in successful) if all_attempts_reported else None),
        reasoning_tokens=(
            sum(item.reasoning_tokens for item in successful if item.reasoning_tokens is not None)
            if all_attempts_reported
            and all(item.reasoning_tokens is not None for item in successful)
            else None
        ),
        latency_ms=sum(item.latency_ms for item in recorder.attempts),
        finish_reason=(successful[-1].finish_reason if all_attempts_reported and successful else None),
    )


class QualitySampler:
    """Run one synthetic scene through the same production generation stack."""

    def __init__(
        self,
        llm_client: _LLMClient,
        output_guard: OutputGuard,
        response_protocol: ResponseProtocol,
        *,
        model_id: str,
        structural_retry_limit: int = 1,
        monotonic: Any = time.monotonic,
        clock: Any = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must provide generate")
        if not isinstance(output_guard, OutputGuard):
            raise TypeError("output_guard must be OutputGuard")
        if not isinstance(response_protocol, ResponseProtocol):
            raise TypeError("response_protocol must be ResponseProtocol")
        _nonempty_text(model_id, "model_id")
        if not callable(monotonic) or not callable(clock):
            raise TypeError("monotonic and clock must be callable")
        self._client = llm_client
        self._guard = output_guard
        self._protocol = response_protocol
        self._model_id = model_id
        self._retry_limit = structural_retry_limit
        self._monotonic = monotonic
        self._clock = clock

    async def sample(
        self,
        scenario: QualityScenario,
        *,
        role_prompt: str,
        sample_index: int,
    ) -> QualitySampleRecord:
        if not isinstance(scenario, QualityScenario):
            raise TypeError("scenario must be QualityScenario")
        prompt = _nonempty_text(role_prompt, "role_prompt")
        _nonnegative_integer(sample_index, "sample_index")
        source = (
            "initiative"
            if scenario.category == "initiative"
            else "interaction"
            if scenario.category == "interaction"
            else "dialogue"
        )
        current_handle = "I0" if source == "initiative" else "M0"
        current_time = self._clock()
        if not isinstance(current_time, datetime) or current_time.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        dialogue_input = DialogueInput(
            conversation_id=f"quality:{scenario.scenario_id}",
            trigger_event_ids=scenario.evidence_event_ids,
            current_event_handle=current_handle,
            quoted_target=None,
            context_version=sample_index,
            role_messages=(
                ModelMessage("system", prompt),
                ModelMessage("system", _sampling_fact_envelope(scenario)),
                *_sampling_messages(scenario),
            ),
            current_time=current_time,
            platform_capabilities=CapabilityManifest({}),
            source=source,
        )
        recorder = _RecordingClient(self._client, self._monotonic)
        engine = DialogueEngine(
            recorder,
            self._guard,
            self._protocol,
            structural_retry_limit=self._retry_limit,
        )
        try:
            outcome, observation = await engine.generate_with_observation(dialogue_input)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _failure_record(
                scenario.scenario_id,
                sample_index,
                model_id=self._model_id,
                error=error,
                recorder=recorder,
            )
        if isinstance(outcome, DialogueResult):
            result: SampleResult = "reply"
            visible_reply = outcome.text
        elif isinstance(outcome, DialogueSkip):
            result = "skip"
            visible_reply = None
        else:
            raise TypeError("dialogue engine returned an unknown outcome")
        return _record_from_observation(
            scenario.scenario_id,
            sample_index,
            result=result,
            visible_reply=visible_reply,
            failure_category=None,
            observation=observation,
        )

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result


def build_quality_sampler(
    config: Config,
    token_counter: _TokenCounter,
    *,
    llm_client: _LLMClient | None = None,
    face_keys: Iterable[str] = (),
    reaction_keys: Iterable[str] = (),
) -> QualitySampler:
    """Build the configured engine stack without creating any QQ component."""
    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    if not callable(getattr(token_counter, "count_text", None)):
        raise TypeError("token_counter must provide count_text")
    primary = config.llm.primary
    client = llm_client
    if client is None:
        client_type = DeepSeekLLMClient if config.llm.provider == "deepseek" else SiliconFlowLLMClient
        client = client_type(
            config.llm.base_url,
            config.llm.api_key,
            model=primary.model,
            temperature=primary.temperature,
            top_p=primary.top_p,
            max_output_tokens=primary.max_output_tokens,
            timeout_seconds=primary.timeout_seconds,
        )
    return QualitySampler(
        client,
        OutputGuard(
            token_counter,
            primary.max_visible_output_tokens,
            reject_empty=config.output_guard.reject_empty,
            reject_internal_prompt_leak=config.output_guard.reject_internal_prompt_leak,
            reject_raw_protocol_payload=config.output_guard.reject_raw_protocol_payload,
        ),
        ResponseProtocol(face_keys=face_keys, reaction_keys=reaction_keys),
        model_id=primary.model,
        structural_retry_limit=config.dialogue.structural_retry_limit,
    )


async def sample_scenarios_to_jsonl(
    scenarios: Sequence[QualityScenario],
    *,
    role_prompt: str,
    sampler: QualitySampler,
    output_path: str | Path,
    samples_per_scenario: int = 3,
    source_path: str | Path | None = None,
) -> tuple[QualitySampleRecord, ...]:
    """Sample sequentially and persist one bounded JSON object per attempt."""
    if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
        raise TypeError("scenarios must be a sequence")
    if not scenarios or not all(isinstance(item, QualityScenario) for item in scenarios):
        raise ValueError("scenarios must contain at least one QualityScenario")
    _nonempty_text(role_prompt, "role_prompt")
    if not isinstance(sampler, QualitySampler):
        raise TypeError("sampler must be QualitySampler")
    if type(samples_per_scenario) is not int or samples_per_scenario < 1:
        raise ValueError("samples_per_scenario must be a positive integer")
    output = Path(output_path)
    if source_path is not None and output.resolve() == Path(source_path).resolve():
        raise ValueError("source and output paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)

    records: list[QualitySampleRecord] = []
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for scenario in scenarios:
            for sample_index in range(samples_per_scenario):
                record = await sampler.sample(
                    scenario,
                    role_prompt=role_prompt,
                    sample_index=sample_index,
                )
                stream.write(
                    json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream.flush()
                records.append(record)
    return tuple(records)
