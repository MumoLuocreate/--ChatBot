from __future__ import annotations

import json
from pathlib import Path

import pytest

from qichi.dialogue.llm_client import LLMAuthenticationError, LLMGeneration
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.dialogue import ModelMessage
from qichi.quality import QualityScenario
from qichi.config import load_config
from qichi.quality.sampler import (
    QualitySampler,
    build_quality_sampler,
    sample_scenarios_to_jsonl,
)


MODEL = "deepseek-v4-flash"


class CharacterCounter:
    def count_text(self, text: str) -> int:
        return len(text)


class FakeClient:
    def __init__(self, results: list[LLMGeneration | BaseException]) -> None:
        self.results = list(results)
        self.calls: list[tuple[ModelMessage, ...]] = []
        self.call_kwargs: list[dict[str, object]] = []

    async def generate(self, messages: tuple[ModelMessage, ...], **kwargs) -> LLMGeneration:
        self.calls.append(tuple(messages))
        self.call_kwargs.append(dict(kwargs))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def generation(
    text: str,
    *,
    input_tokens: int = 11,
    output_tokens: int = 3,
    reasoning_tokens: int | None = 2,
    latency_ms: float = 7.5,
    finish_reason: str | None = "stop",
) -> LLMGeneration:
    return LLMGeneration(
        text=text,
        model_route="primary",
        model_id=MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )


def sampler(client: FakeClient) -> QualitySampler:
    return QualitySampler(
        client,
        OutputGuard(CharacterCounter(), 1_200),
        ResponseProtocol(face_keys=(), reaction_keys=()),
        model_id=MODEL,
        structural_retry_limit=1,
    )


def test_configured_sampler_separates_provider_and_visible_output_budgets(
    config_path, complete_environment
):
    config = load_config(config_path, environ=complete_environment)
    client = FakeClient([])
    configured = build_quality_sampler(config, CharacterCounter(), llm_client=client)

    assert config.llm.primary.max_output_tokens == 6144
    assert configured._guard._max == 1200


@pytest.mark.asyncio
async def test_sampler_prepends_thin_prompt_and_runs_three_samples_per_scenario(tmp_path):
    dialogue = QualityScenario(
        "casual",
        "casual",
        (ModelMessage("system", "场景事实"), ModelMessage("user", "忙完啦")),
        ("event-1",),
    )
    initiative = QualityScenario(
        "initiative",
        "initiative",
        (ModelMessage("user", "[initiative]"),),
        (),
    )
    client = FakeClient(
        [generation(f"可见回复 {index}") for index in range(3)]
        + [generation("[[qichi:skip]]") for _ in range(3)]
    )
    output = tmp_path / "samples.jsonl"

    records = await sample_scenarios_to_jsonl(
        (dialogue, initiative),
        role_prompt="当前薄角色核心",
        sampler=sampler(client),
        output_path=output,
    )

    assert len(records) == 6
    assert [record.sample_index for record in records] == [0, 1, 2, 0, 1, 2]
    assert all(call[0] == ModelMessage("system", "当前薄角色核心") for call in client.calls)
    assert all(call[1].role == "system" and "事实与能力" in call[1].content for call in client.calls)
    assert all("不要为了显得自然、亲密或有画面" in call[1].content for call in client.calls)
    assert all("提问、比喻或玩笑中预设未知细节已经发生" in call[1].content for call in client.calls)
    assert all(call[2:] == dialogue.messages for call in client.calls[:3])
    assert all(call[2:] == initiative.messages for call in client.calls[3:])
    payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [item["result"] for item in payloads] == ["reply"] * 3 + ["skip"] * 3
    assert set(payloads[0]) == {
        "scenario_id",
        "sample_index",
        "result",
        "visible_reply",
        "failure_category",
        "model_id",
        "attempt_count",
        "retry_count",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "latency_ms",
        "finish_reason",
    }
    assert payloads[0]["visible_reply"] == "可见回复 0"
    assert payloads[-1]["visible_reply"] is None
    assert "当前薄角色核心" not in output.read_text(encoding="utf-8")
    assert "messages" not in payloads[0]


@pytest.mark.asyncio
async def test_interaction_scenario_uses_interaction_source(tmp_path):
    class SourceRecordingProtocol(ResponseProtocol):
        def __init__(self) -> None:
            super().__init__(face_keys=(), reaction_keys=())
            self.sources: list[str] = []

        def parse(self, raw_output: str, *, source: str, current_event_handle: str, context_version: int):
            self.sources.append(source)
            return super().parse(
                raw_output,
                source=source,
                current_event_handle=current_event_handle,
                context_version=context_version,
            )

    protocol = SourceRecordingProtocol()
    client = FakeClient([generation("收到你戳我了。")])
    quality_sampler = QualitySampler(
        client,
        OutputGuard(CharacterCounter(), 1_200),
        protocol,
        model_id=MODEL,
        structural_retry_limit=1,
    )
    scenario = QualityScenario(
        "poke",
        "interaction",
        (ModelMessage("system", "[当前输入 | actor=mumo; handle=M0; kind=poke]"), ModelMessage("user", "")),
        ("poke-1",),
    )

    await sample_scenarios_to_jsonl(
        (scenario,),
        role_prompt="薄角色核心",
        sampler=quality_sampler,
        output_path=tmp_path / "samples.jsonl",
        samples_per_scenario=1,
    )

    assert protocol.sources == ["interaction"]


@pytest.mark.asyncio
async def test_sampler_marks_historical_qichi_text_as_evidence_not_assistant_turn(tmp_path):
    scenario = QualityScenario(
        "history-shape",
        "correction",
        (
            ModelMessage("assistant", "（动作旁白）旧回复"),
            ModelMessage("user", "我没有这么说"),
        ),
        (),
    )
    client = FakeClient([generation("收到。")])

    await sample_scenarios_to_jsonl(
        (scenario,),
        role_prompt="薄角色核心",
        sampler=sampler(client),
        output_path=tmp_path / "samples.jsonl",
        samples_per_scenario=1,
    )

    historical = client.calls[0][2]
    assert historical.role == "system"
    assert "历史角色原文" in historical.content
    assert client.calls[0][-1] == ModelMessage("user", "我没有这么说")


@pytest.mark.asyncio
async def test_sampler_preserves_engine_structural_retry_observation(tmp_path):
    scenario = QualityScenario(
        "retry",
        "casual",
        (ModelMessage("user", "在吗"),),
        (),
    )
    client = FakeClient(
        [
            generation("", input_tokens=10, output_tokens=0, reasoning_tokens=1, latency_ms=4),
            generation("在。", input_tokens=12, output_tokens=2, reasoning_tokens=3, latency_ms=6),
        ]
    )

    records = await sample_scenarios_to_jsonl(
        (scenario,),
        role_prompt="薄角色核心",
        sampler=sampler(client),
        output_path=tmp_path / "samples.jsonl",
        samples_per_scenario=1,
    )

    record = records[0]
    assert record.result == "reply"
    assert record.attempt_count == 2
    assert record.retry_count == 1
    assert record.input_tokens == 22
    assert record.output_tokens == 2
    assert record.reasoning_tokens == 4
    assert record.latency_ms == 10
    assert record.finish_reason == "stop"
    assert client.call_kwargs == [{}, {"thinking": {"type": "disabled"}}]


@pytest.mark.asyncio
async def test_sampler_redacts_failures_and_continues_later_samples(tmp_path):
    scenario = QualityScenario(
        "failure",
        "casual",
        (ModelMessage("user", "在吗"),),
        (),
    )
    client = FakeClient(
        [
            LLMAuthenticationError("supplier says leaked-secret"),
            RuntimeError("raw response with another-secret"),
            generation("还在。"),
        ]
    )
    output = tmp_path / "samples.jsonl"

    records = await sample_scenarios_to_jsonl(
        (scenario,),
        role_prompt="薄角色核心",
        sampler=sampler(client),
        output_path=output,
    )

    assert [item.result for item in records] == ["failure", "failure", "reply"]
    assert [item.failure_category for item in records] == [
        "llm_authentication",
        "internal_error",
        None,
    ]
    assert records[0].attempt_count == 1
    assert records[0].retry_count == 0
    raw = output.read_text(encoding="utf-8")
    assert "leaked-secret" not in raw
    assert "another-secret" not in raw
    assert "RuntimeError" not in raw


@pytest.mark.asyncio
async def test_sampler_does_not_accept_initiative_skip_for_ordinary_dialogue(tmp_path):
    scenario = QualityScenario(
        "ordinary",
        "casual",
        (ModelMessage("user", "今天不想说话"),),
        (),
    )
    client = FakeClient([generation("[[qichi:skip]]"), generation("[[qichi:skip]]")])

    records = await sample_scenarios_to_jsonl(
        (scenario,),
        role_prompt="薄角色核心",
        sampler=sampler(client),
        output_path=tmp_path / "samples.jsonl",
        samples_per_scenario=1,
    )

    assert records[0].result == "failure"
    assert records[0].failure_category == "response_protocol:skip_not_allowed"
    assert records[0].attempt_count == 2


@pytest.mark.asyncio
async def test_sampler_rejects_input_output_collision_without_modifying_baseline(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    original = '{"scenario_id":"kept"}\n'
    baseline.write_text(original, encoding="utf-8")
    scenario = QualityScenario("one", "casual", (ModelMessage("user", "hi"),), ())

    with pytest.raises(ValueError, match="must differ"):
        await sample_scenarios_to_jsonl(
            (scenario,),
            role_prompt="薄角色核心",
            sampler=sampler(FakeClient([generation("hi")])),
            output_path=baseline,
            source_path=baseline,
            samples_per_scenario=1,
        )

    assert baseline.read_text(encoding="utf-8") == original
