from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import traceback

import pytest

from qichi.dialogue.engine import DialogueEngine, DialogueGenerationError
from qichi.dialogue.output_guard import OutputGuard, OutputGuardError, OutputGuardInfrastructureError
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.dialogue import CapabilityManifest, DialogueInput, DialogueResult, DialogueSkip, ModelMessage
from qichi.dialogue.llm_client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGeneration,
    LLMModelNotFoundError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
)


class Counter:
    def __init__(self, value=1):
        self.value = value

    def count_text(self, text):
        return self.value


class FakeLLM:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def generate(self, messages):
        self.calls.append(messages)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return LLMGeneration(value, "primary", "test", 1, 1, 1.0)


@pytest.mark.asyncio
async def test_length_terminated_prefix_is_retried_as_concise_complete_reply():
    class TruncatingLLM:
        def __init__(self):
            self.calls = []

        async def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                return LLMGeneration("未完成", "primary", "test", 1, 1, 1.0, finish_reason="length")
            return LLMGeneration("完整回复", "primary", "test", 1, 1, 1.0, finish_reason="stop")

    llm = TruncatingLLM()
    result = await DialogueEngine(
        llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())
    ).generate(inp("dialogue"))
    assert result.text == "完整回复"
    assert len(llm.calls) == 2
    assert llm.calls[0][1] == {}
    assert llm.calls[1][0][-1].role == "system"
    assert llm.calls[1][1] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_generation_observation_reports_usage_and_structural_retry_without_content():
    class ObservedLLM:
        def __init__(self):
            self.calls = 0

        async def generate(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMGeneration(
                    "未完成的私聊正文", "primary", "test-v4f", 11, 7, 2.5,
                    finish_reason="length", reasoning_tokens=3, cache_hit_tokens=4,
                )
            return LLMGeneration(
                "完整回复", "primary", "test-v4f", 13, 5, 3.5,
                finish_reason="stop", reasoning_tokens=2, cache_hit_tokens=6,
            )

    outcome, observation = await DialogueEngine(
        ObservedLLM(), guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())
    ).generate_with_observation(inp())

    assert outcome.text == "完整回复"
    assert observation.model_id == "test-v4f"
    assert observation.model_route == "primary"
    assert observation.attempt_count == 2
    assert observation.retry_count == 1
    assert observation.input_tokens == 24
    assert observation.output_tokens == 12
    assert observation.reasoning_tokens == 5
    assert observation.cache_hit_tokens == 10
    assert observation.latency_ms == 6.0
    assert observation.finish_reason == "stop"
    assert observation.result_type == "reply"
    # 2026-09-15：重试过就要说清为什么（真机出现过 attempt=2 却查不出原因）。
    assert observation.retry_reasons == ("output_guard:truncated",)
    assert "未完成" not in " ".join(observation.retry_reasons), "原因码里不许有模型原文"
    assert "未完成的私聊正文" not in repr(observation)
    assert "完整回复" not in repr(observation)


@pytest.mark.asyncio
async def test_empty_output_is_retried_with_a_completion_instruction():
    class EmptyThenCompleteLLM(FakeLLM):
        async def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            value = self.values.pop(0)
            return LLMGeneration(value, "primary", "test", 1, 1, 1.0)

    llm = EmptyThenCompleteLLM(["", "有，我记得。"])
    result = await DialogueEngine(
        llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())
    ).generate(inp())
    assert result.text == "有，我记得。"
    assert len(llm.calls) == 2
    assert llm.calls[0][1] == {}
    assert "没有产生可见文字" in llm.calls[1][0][-1].content
    assert llm.calls[1][1] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_normal_success_does_not_override_model_thinking():
    class CapturingLLM:
        def __init__(self):
            self.kwargs = []

        async def generate(self, messages, **kwargs):
            self.kwargs.append(kwargs)
            return LLMGeneration("正常回复", "primary", "test", 1, 1, 1.0)

    llm = CapturingLLM()
    result = await DialogueEngine(
        llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())
    ).generate(inp())
    assert result.text == "正常回复"
    assert llm.kwargs == [{}]


def exception_graph_text(error):
    parts = []
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        parts.extend((str(current), repr(current)))
        parts.extend(traceback.format_exception(type(current), current, current.__traceback__))
        pending.extend((current.__cause__, current.__context__))
    return "".join(parts)


def inp(source="dialogue", prompt="secret system prompt"):
    return DialogueInput(
        "c", ("e",), "M1", None, 3,
        (ModelMessage("system", prompt), ModelMessage("user", "hi")),
        datetime.now(timezone.utc), CapabilityManifest({}), source,
    )


def guard(**kw):
    return OutputGuard(Counter(), 10, **kw)


def test_guard_allows_normal_unicode_brackets_intimacy_and_json_code():
    texts = [
        "你好 🙂（抱一下）[普通方括号]",
        "system prompt 系统提示 规则 JSON 亲密（括号）",
        "普通文本 <|begin_of_sentence|>",
        "```json\n{\"a\": 1}\n```",
    ]
    for text in texts:
        guard().validate(text, role_messages=(ModelMessage("system", "unrelated"),))


@pytest.mark.parametrize("text", [
    "", "  ", json.dumps({"text": "x"}), "<｜begin▁of▁sentence｜>",
    "<｜begin▁sys｜>", "<｜tool▁calls▁begin｜>",
])
def test_guard_rejects_structural_payloads(text):
    with pytest.raises(OutputGuardError):
        guard().validate(text)


def test_guard_rejects_prompt_only_when_exact_system_content_is_present():
    with pytest.raises(OutputGuardError): guard().validate("secret system prompt", role_messages=inp().role_messages)


def test_invalid_counter_fails_closed():
    for value in (-1, "bad", True):
        with pytest.raises(OutputGuardInfrastructureError):
            OutputGuard(Counter(value), 10).validate("x")


def test_guard_rejects_exact_token_limit_and_counter_failure():
    with pytest.raises(OutputGuardError):
        OutputGuard(Counter(2), 1).validate("x")

    class BrokenCounter:
        def count_text(self, text):
            raise RuntimeError("sensitive completion")

    with pytest.raises(OutputGuardInfrastructureError) as error:
        OutputGuard(BrokenCounter(), 10).validate("x")
    assert "sensitive" not in repr(error.value)


def test_counter_exception_graph_is_empty_of_sensitive_values():
    counter_error = "counter-sensitive-value"
    prompt = "prompt-sensitive-value"
    completion = "completion-sensitive-value"

    class BrokenCounter:
        def count_text(self, text):
            raise RuntimeError(counter_error)

    with pytest.raises(OutputGuardInfrastructureError) as error:
        OutputGuard(BrokenCounter(), 10).validate(completion)
    graph = exception_graph_text(error.value)
    assert counter_error not in graph
    assert prompt not in graph
    assert completion not in graph
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_role_message_iterator_exception_graph_is_empty_of_sensitive_values():
    iterator_error = "iterator-sensitive-value"

    class BrokenMessages:
        def __iter__(self):
            raise TypeError(iterator_error)

    with pytest.raises(TypeError) as error:
        guard().validate("ordinary", role_messages=BrokenMessages())
    graph = exception_graph_text(error.value)
    assert iterator_error not in graph
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_guard_json_only_rejects_dict_and_list():
    for text in ['{"text": "x"}', '[1, 2]']:
        with pytest.raises(OutputGuardError):
            guard().validate(text)
    for text in ['"scalar"', "123", "true", "null", "解释：{\"text\": \"x\"}", "```json\n{\"x\": 1}\n```"]:
        guard().validate(text)


def test_guard_constructor_and_role_message_contracts_are_strict():
    with pytest.raises(TypeError):
        OutputGuard(object(), 10)
    with pytest.raises((TypeError, ValueError)):
        OutputGuard(Counter(), 0)
    for option in ("reject_empty", "reject_internal_prompt_leak", "reject_raw_protocol_payload"):
        with pytest.raises(TypeError):
            OutputGuard(Counter(), 10, **{option: 1})
    with pytest.raises(TypeError):
        guard().validate("x", role_messages=("not a message",))

    with pytest.raises((TypeError, ValueError)):
        DialogueEngine(FakeLLM(["ok"]), guard(), ResponseProtocol(face_keys=set(), reaction_keys=set()), structural_retry_limit=True)


@pytest.mark.asyncio
async def test_success_calls_once_and_binds_context():
    llm = FakeLLM(["好呀"])
    result = await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp())
    assert result == DialogueResult("好呀", None, None, "primary", 3)
    assert len(llm.calls) == 1
    assert llm.calls[0] == inp().role_messages


@pytest.mark.asyncio
async def test_interaction_returns_normal_result():
    llm = FakeLLM(["戳回来了"])
    result = await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp("interaction"))
    assert isinstance(result, DialogueResult)
    assert result.text == "戳回来了"


@pytest.mark.asyncio
async def test_structural_retry_recovers_and_reuses_identical_messages():
    llm = FakeLLM(["{}", "恢复了"])
    value = inp()
    result = await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(value)
    assert result.text == "恢复了"
    assert len(llm.calls) == 2
    assert llm.calls[0] is value.role_messages
    assert llm.calls[1] is value.role_messages


@pytest.mark.asyncio
async def test_protocol_failure_retry_and_second_failure_is_nonsensitive():
    prompt = "prompt-sensitive-value"
    completion = "completion-sensitive-value [[qq:face:x]]"
    llm = FakeLLM([completion, completion])
    with pytest.raises(DialogueGenerationError) as error:
        await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp(prompt=prompt))
    graph = exception_graph_text(error.value)
    assert prompt not in str(error.value)
    assert completion not in repr(error.value)
    assert prompt not in graph
    assert completion not in graph
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion", "reason"),
    [
        ("{}", "output_guard:raw_protocol_payload"),
        ("x\n[[qq:reply:M1]]\n[[qq:reply:Q2]]", "response_protocol:duplicate_control_action"),
    ],
)
async def test_structural_failure_exposes_only_safe_reason(completion, reason):
    llm = FakeLLM([completion, completion])
    with pytest.raises(DialogueGenerationError) as error:
        await DialogueEngine(
            llm,
            guard(),
            ResponseProtocol(face_keys=set(), reaction_keys=set()),
        ).generate(inp())
    assert error.value.reason == reason
    assert str(error.value) == f"dialogue generation failed structural validation ({reason})"
    assert completion not in repr(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("counter_result", [-1, True, "invalid"])
async def test_counter_contract_errors_are_not_retried(counter_result):
    class BadCounter:
        def count_text(self, text):
            return counter_result

    prompt = "prompt-sensitive-value"
    completion = "completion-sensitive-value"
    llm = FakeLLM([completion, "ok"])
    with pytest.raises(OutputGuardInfrastructureError) as error:
        await DialogueEngine(
            llm,
            OutputGuard(BadCounter(), 10),
            ResponseProtocol(face_keys=set(), reaction_keys=set()),
        ).generate(inp(prompt=prompt))
    assert len(llm.calls) == 1
    graph = exception_graph_text(error.value)
    assert prompt not in graph
    assert completion not in graph


@pytest.mark.asyncio
async def test_counter_exception_is_not_retried_and_is_nonsensitive():
    class BrokenCounter:
        def count_text(self, text):
            raise RuntimeError(counter_error)

    counter_error = "counter-sensitive-value"
    prompt = "prompt-sensitive-value"
    completion = "completion-sensitive-value"
    llm = FakeLLM([completion, "ok"])
    with pytest.raises(OutputGuardInfrastructureError) as error:
        await DialogueEngine(
            llm,
            OutputGuard(BrokenCounter(), 10),
            ResponseProtocol(face_keys=set(), reaction_keys=set()),
        ).generate(inp(prompt=prompt))
    assert len(llm.calls) == 1
    graph = exception_graph_text(error.value)
    assert counter_error not in graph


@pytest.mark.asyncio
async def test_non_primary_route_is_a_structural_retry():
    class NonPrimaryLLM(FakeLLM):
        async def generate(self, messages):
            self.calls.append(messages)
            value = self.values.pop(0)
            return LLMGeneration(value, "fallback", "test", 1, 1, 1.0)

    llm = NonPrimaryLLM(["bad", "bad"])
    with pytest.raises(DialogueGenerationError):
        await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp())
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TypeError("contract"), ValueError("contract")])
async def test_contract_errors_are_not_structural_retries(error):
    class BrokenGuard:
        def validate(self, output, *, role_messages):
            raise error

    llm = FakeLLM(["ok", "ok"])
    with pytest.raises(type(error), match="contract"):
        await DialogueEngine(llm, BrokenGuard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp())
    assert len(llm.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TypeError("contract"), ValueError("contract")])
async def test_protocol_contract_errors_are_not_structural_retries(error):
    class BrokenProtocol:
        def parse(self, *args, **kwargs):
            raise error

    llm = FakeLLM(["ok", "ok"])
    with pytest.raises(type(error), match="contract"):
        await DialogueEngine(llm, guard(), BrokenProtocol()).generate(inp())
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_retry_limit_zero_and_llm_errors_are_not_retried():
    llm = FakeLLM(["{}", "ok"])
    with pytest.raises(DialogueGenerationError):
        await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set()), structural_retry_limit=0).generate(inp())
    assert len(llm.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    LLMTimeoutError("x"), LLMConnectionError("x"), LLMRateLimitError("x"),
    LLMAuthenticationError("x"), LLMModelNotFoundError("x"), LLMRequestError("x"),
    LLMServerError("x"), LLMProtocolError("x"),
])
async def test_all_llm_service_errors_are_propagated_without_retry(error):
    llm = FakeLLM([error, "ok"])
    with pytest.raises(type(error)):
        await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp())
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_initiative_skip_is_valid_but_other_sources_fail():
    protocol = ResponseProtocol(face_keys=set(), reaction_keys=set())
    llm = FakeLLM(["[[qichi:skip]]"])
    result = await DialogueEngine(llm, guard(), protocol).generate(inp("initiative"))
    assert result == DialogueSkip("primary", 3)
    llm = FakeLLM(["[[qichi:skip]]", "[[qichi:skip]]"])
    with pytest.raises(DialogueGenerationError):
        await DialogueEngine(llm, guard(), protocol).generate(inp("dialogue"))

    llm = FakeLLM(["[[qichi:skip]]", "[[qichi:skip]]"])
    with pytest.raises(DialogueGenerationError):
        await DialogueEngine(llm, guard(), protocol).generate(inp("interaction"))
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [
    "[[qq:reply:M1]]\n[[qq:face:shy]]",
    "[[qq:reply:M1]]\n[[qq:react:heart]]",
])
async def test_valid_protocol_tail_is_parsed_by_engine(marker):
    llm = FakeLLM(["收到\n" + marker])
    protocol = ResponseProtocol(face_keys={"shy"}, reaction_keys={"heart"})
    result = await DialogueEngine(llm, guard(), protocol).generate(inp())
    assert isinstance(result, DialogueResult)
    assert result.reply_target == "M1"
    assert result.expression_intent is not None


@pytest.mark.asyncio
async def test_cancelled_error_propagates():
    llm = FakeLLM([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await DialogueEngine(llm, guard(), ResponseProtocol(face_keys=set(), reaction_keys=set())).generate(inp())
    assert len(llm.calls) == 1
