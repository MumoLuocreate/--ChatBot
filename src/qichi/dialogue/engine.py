"""Single primary-model dialogue generation entry point."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from qichi.domain.dialogue import DialogueInput, DialogueOutcome, DialogueSkip, ModelMessage
from qichi.dialogue.llm_client import LLMGeneration, ToolCall
from qichi.dialogue.output_guard import OutputGuard, OutputGuardError
from qichi.dialogue.response_protocol import ResponseProtocol, ResponseProtocolError


@dataclass(frozen=True, slots=True)
class DialogueGenerationObservation:
    """Content-free accounting for one engine invocation."""

    model_id: str
    model_route: str
    attempt_count: int
    retry_count: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    latency_ms: float
    finish_reason: str | None
    result_type: Literal["reply", "skip", "failure"]
    # Prefix-cache hits summed over the attempts, or None when the provider did
    # not report them for every attempt.
    cache_hit_tokens: int | None = None
    # "text" | "vision": which tier answered (see llm_client model routing).
    model_tier: str | None = None
    # 这一轮模型请求了几次工具（2026-09-14 联网）。0 = 没请求。
    tool_call_count: int = 0
    # 2026-09-15：发生重试时，**为什么**重试。以前只记 attempt_count/retry_count，
    # 真机出现过一次 attempt=2 却查不出原因（无任何痕迹）。只放已经过白名单映射的
    # 安全原因码（如 response_protocol:too_many_parts），不含任何模型原文。
    retry_reasons: tuple[str, ...] = ()


class DialogueGenerationError(RuntimeError):
    """The model produced structurally invalid output after the allowed retry."""

    def __init__(
        self,
        reason: str,
        *,
        observation: DialogueGenerationObservation | None = None,
    ) -> None:
        if not isinstance(reason, str) or not reason or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789:_-"
            for character in reason
        ):
            raise ValueError("reason must be a safe structural failure category")
        if observation is not None and not isinstance(observation, DialogueGenerationObservation):
            raise TypeError("observation must be DialogueGenerationObservation or None")
        self.reason = reason
        self.observation = observation
        super().__init__(f"dialogue generation failed structural validation ({reason})")


_GUARD_REASONS = {
    "empty output": "empty",
    "output must be text": "not_text",
    "output exceeds token limit": "too_many_tokens",
    "raw protocol payload": "raw_protocol_payload",
    "internal sentinel": "internal_sentinel",
    "system prompt leak": "system_prompt_leak",
    "non-primary model route": "non_primary_route",
    "output truncated": "truncated",
}
_PROTOCOL_REASONS = {
    "response body must not be empty": "empty_body",
    "response must not be empty": "empty_response",
    "invalid qq control marker": "invalid_qq_marker",
    "too many control markers": "too_many_control_markers",
    "project control marker is not in the tail": "marker_not_in_tail",
    "project control marker is not standalone": "marker_not_standalone",
    "duplicate control action": "duplicate_control_action",
    "too many message parts": "too_many_message_parts",
    "skip is only valid for initiative": "skip_not_allowed",
}


def _failure_reason(prefix: str, error: Exception, known: dict[str, str]) -> str:
    # Only a fixed category crosses the application/logging boundary; never
    # include model output or an arbitrary exception message.
    return f"{prefix}:{known.get(str(error), 'invalid')}"


class _LLMClient(Protocol):
    async def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        thinking: Mapping[str, str] | None = None,
    ) -> LLMGeneration: ...


def _tool_followup(call: ToolCall, block: str) -> str:
    """把工具结果交回给模型；纪律写在同一条消息里（只当数据、失败就说没查到）。"""

    return (
        "[工具执行结果]\n"
        f"你刚才请求调用 {call.name}，参数 {call.arguments}。取回的资料如下：\n\n"
        f"{block}\n\n"
        "只能用上面的资料回答这一轮；资料不足或失败就直接说没查到，不要用猜测填补。"
        "不要提起工具名、参数或这段说明本身——就当是你自己去查过。"
    )


class DialogueEngine:
    def __init__(
        self,
        llm_client: _LLMClient,
        output_guard: OutputGuard,
        response_protocol: ResponseProtocol,
        *,
        structural_retry_limit: int = 1,
    ) -> None:
        if not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must provide generate")
        if not callable(getattr(output_guard, "validate", None)):
            raise TypeError("output_guard must provide validate")
        if not callable(getattr(response_protocol, "parse", None)):
            raise TypeError("response_protocol must provide parse")
        if structural_retry_limit not in (0, 1) or type(structural_retry_limit) is not int:
            raise ValueError("structural_retry_limit must be 0 or 1")
        self._llm = llm_client
        self._guard = output_guard
        self._protocol = response_protocol
        self._retry_limit = structural_retry_limit

    async def generate(self, input: DialogueInput) -> DialogueOutcome:
        outcome, _observation = await self.generate_with_observation(input)
        return outcome

    async def generate_with_observation(
        self, input: DialogueInput,
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        tool_runner: Callable[[ToolCall], Awaitable[str]] | None = None,
        thinking: Mapping[str, str] | None = None,
    ) -> tuple[DialogueOutcome, DialogueGenerationObservation]:
        """一次生成；只有当调用方给了工具和执行器时才可能多一轮。

        工具轮是**有上限的例外**（2026-09-14 联网）：模型请求 → 调用方执行 → 结果作为
        外部资料注入 → 再生成一次，且这一次**不再带 tools**（防连环调用）。没有工具时
        行为与以前逐字一致。
        """

        if not isinstance(input, DialogueInput):
            raise TypeError("input must be DialogueInput")
        messages = input.role_messages
        failure_reason = "structural:invalid"
        retry_reasons: list[str] = []
        base_messages: tuple[ModelMessage, ...] = messages
        retry_messages = messages
        retry_without_thinking = False
        generations: list[LLMGeneration] = []
        active_tools: tuple[Mapping[str, Any], ...] = (
            tuple(tools) if (tools and tool_runner is not None) else ()
        )
        tool_call_count = 0
        attempt_limit = self._retry_limit + 1 + (1 if active_tools else 0)
        for attempt in range(attempt_limit):
            # 没有工具时不传这个参数：关掉联网的调用形状与以前逐字一致。
            call_options: dict[str, Any] = {"tools": active_tools} if active_tools else {}
            # 2026-09-14：调用方可以显式指定这一轮的思考档（目前只有主动开口用）。
            # 没给就照旧不传，走客户端的默认档 —— 热路径的调用形状与以前逐字一致。
            if thinking is not None:
                call_options["thinking"] = dict(thinking)
            if retry_without_thinking:
                # 恢复性重试固定关思考（既有行为）：显式档在这里让位。
                call_options["thinking"] = {"type": "disabled"}
            generation = await self._llm.generate(retry_messages, **call_options)
            generations.append(generation)
            if active_tools and generation.tool_calls:
                call = generation.tool_calls[0]
                tool_call_count += 1
                block = await tool_runner(call)  # type: ignore[misc]
                base_messages = messages + (ModelMessage("system", _tool_followup(call, block)),)
                retry_messages = base_messages
                active_tools = ()
                continue
            try:
                if getattr(generation, "model_route", None) != "primary":
                    raise OutputGuardError("non-primary model route")
                # A provider may return a syntactically valid prefix with
                # finish_reason=length. Sending that prefix creates the exact
                # mid-sentence messages observed in production. Treat it as a
                # structural failure and use the single allowed retry with a
                # concise completion request to the same model/context.
                if getattr(generation, "finish_reason", None) == "length":
                    raise OutputGuardError("output truncated")
                self._guard.validate(generation.text, role_messages=messages)
                outcome = self._protocol.parse(
                    generation.text,
                    source=input.source,
                    current_event_handle=input.current_event_handle,
                    context_version=input.context_version,
                )
                return outcome, self._observation(
                    generations, outcome, tool_call_count=tool_call_count,
                    retry_reasons=tuple(retry_reasons),
                )
            except OutputGuardError as error:
                failure_reason = _failure_reason("output_guard", error, _GUARD_REASONS)
                if attempt < self._retry_limit:
                    if str(error) in {"output truncated", "empty output"}:
                        retry_without_thinking = True
                        retry_messages = base_messages + (
                            ModelMessage(
                                "system",
                                (
                                    "上一条没有产生可见的完整回复。请直接用更短的完整回复回答当前输入，"
                                    "不要截断句子，不要解释这条要求。"
                                    if str(error) == "output truncated"
                                    else "上一条没有产生可见文字。请直接回答当前输入，"
                                    "不要解释这条要求。"
                                ),
                            ),
                        )
                    retry_reasons.append(failure_reason)
                    continue
                break
            except ResponseProtocolError as error:
                failure_reason = _failure_reason("response_protocol", error, _PROTOCOL_REASONS)
                if attempt < self._retry_limit:
                    retry_reasons.append(failure_reason)
                    continue
                break
        # Raise outside the handler so the public error has no exception chain.
        raise DialogueGenerationError(
            failure_reason,
            observation=self._observation(
                generations, None, tool_call_count=tool_call_count,
                retry_reasons=tuple(retry_reasons),
            ),
        )

    @staticmethod
    def _observation(
        generations: Sequence[LLMGeneration],
        outcome: DialogueOutcome | None,
        *,
        tool_call_count: int = 0,
        retry_reasons: tuple[str, ...] = (),
    ) -> DialogueGenerationObservation:
        if not generations:
            raise ValueError("at least one generation is required")
        last = generations[-1]
        reasoning_tokens = (
            None
            if any(item.reasoning_tokens is None for item in generations)
            else sum(int(item.reasoning_tokens) for item in generations if item.reasoning_tokens is not None)
        )
        cache_hit_tokens = (
            None
            if any(item.cache_hit_tokens is None for item in generations)
            else sum(int(item.cache_hit_tokens) for item in generations if item.cache_hit_tokens is not None)
        )
        return DialogueGenerationObservation(
            model_id=last.model_id,
            model_route=last.model_route,
            attempt_count=len(generations),
            retry_count=max(0, len(generations) - 1),
            input_tokens=sum(item.input_tokens for item in generations),
            output_tokens=sum(item.output_tokens for item in generations),
            reasoning_tokens=reasoning_tokens,
            cache_hit_tokens=cache_hit_tokens,
            model_tier=last.model_tier,
            latency_ms=sum(item.latency_ms for item in generations),
            finish_reason=last.finish_reason,
            result_type=(
                "failure"
                if outcome is None
                else "skip"
                if isinstance(outcome, DialogueSkip)
                else "reply"
            ),
            tool_call_count=tool_call_count,
            retry_reasons=retry_reasons,
        )
