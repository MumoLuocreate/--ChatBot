from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qichi.dialogue.capability_manifest import (
    MessageSourceFact, RuntimeFacts, build_capability_manifest, render_fact_envelope,
)
from qichi.dialogue.context_builder import ContextBuildRequest, ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.dialogue import CapabilityManifest, DialogueInput, DialogueResult, ModelMessage
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.memory.retriever import MemoryRetriever
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository


FIXTURE = Path(__file__).parent / "replay" / "g1_continuous_dialogue.json"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class UnitCounter:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 20 + 1)


class FakeLLM:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        return LLMGeneration(self.text, "primary", MODEL, 1, 1, 1.0)


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def segments(item: dict) -> tuple[MessageSegment, ...]:
    result = []
    if item["text"] is not None:
        result.append(MessageSegment("text", {"text": item["text"]}))
    for media in item.get("media", []):
        result.append(MessageSegment(media, {}))
    return tuple(result)


def persist_fixture(database: Database) -> dict[str, ConversationEvent]:
    data = load_fixture()
    repository = EventRepository(database)
    events = {}
    for item in data["events"]:
        occurred = datetime.fromisoformat(item["occurred_at"])
        event = ConversationEvent(
            event_id=item["event_id"], platform_event_id=f"pe-{item['event_id']}",
            platform_message_id=None if item["direction"] == "internal" else f"pm-{item['event_id']}",
            conversation_id=data["conversation_id"], sequence=999, direction=item["direction"],
            actor=item["actor"], kind=item["kind"], text=item["text"], message_segments=segments(item),
            reply_to_event_id=None, reply_to_platform_message_id=None, occurred_at_utc=occurred,
            received_at_utc=occurred,
            status="sent" if item["direction"] == "outbound" else "recorded" if item["direction"] == "internal" else "received",
            metadata={"fixture_case": True},
        )
        events[item["event_id"]] = repository.insert(event)
    return events


def memory(memory_id: str, source: ConversationEvent, fact: str, quote: str, **kw) -> MemoryRecord:
    evidence = MemoryEvidence(memory_id, source.event_id, source.actor, quote, source.occurred_at_utc, kw.get("evidence_role", "source"))
    return MemoryRecord(
        memory_id, kw.get("type", "preference"), fact, kw.get("modality", "explicit_statement"),
        kw.get("status", "active"), source.occurred_at_utc, kw.get("valid_until"),
        kw.get("supersedes_id"), source.received_at_utc, (evidence,),
        kw.get("certainty", "explicit"), kw.get("importance", 2),
        kw.get("temporal_scope", "ongoing"),
        kw.get("assessment_reason_code", "explicit_user_statement"),
        kw.get("assessed_at_utc", source.received_at_utc),
    )


def capability() -> ModelCapability:
    evidence = ProviderCapabilityEvidence("SiliconFlow", MODEL, 262_144, "local fake replay", datetime(2026, 8, 28, tzinfo=timezone.utc))
    return ModelCapability(MODEL, 1_048_576, provider_evidence=evidence)


def facts(current: ConversationEvent, now: datetime, *, media=(), vision=False, tools=False) -> RuntimeFacts:
    return RuntimeFacts(
        now, None, MessageSourceFact(current.actor, current.visible_handle, current.kind, current.occurred_at_utc),
        None, tuple(media), ("text", "reply"), vision, tools,
    )


def context(database: Database, events: dict[str, ConversationEvent], current_id: str, now: datetime, query: str):
    retrieval = MemoryRetriever(database).retrieve("owner-private", query, now)
    current = events[current_id]
    runtime = facts(current, now, media=tuple(segment.type for segment in current.message_segments if segment.type != "text"))
    prior = tuple(
        item for item in events.values()
        if item.sequence < current.sequence and item.direction != "internal"
    )
    active = MemoryRepository(database).list_active("owner-private", now)
    relationship = tuple(item for item in active if item.type in {"preference", "agreement", "correction"})
    relationship_ids = {item.memory_id for item in relationship}
    memory_candidates = tuple(item for item in retrieval.context_candidates if item.memory_id not in relationship_ids)
    evidence_events = dict(retrieval.evidence_events)
    for item in relationship:
        for evidence in item.memory_evidence:
            evidence_events[evidence.event_id] = events[evidence.event_id]
    request = ContextBuildRequest(
        role_core="角色是有主体性的 AI 伙伴。", runtime_facts=runtime, current_event=current,
        quoted_chain=(), recent_events=prior[-5:],
        relationship_state=relationship, memory_candidates=memory_candidates,
        earlier_events=prior[:-5], evidence_events=evidence_events,
    )
    builder = ContextBuilder(UnitCounter(), capability(), preferred_window_tokens=262_144,
                             max_window_tokens=262_144, output_reserve_tokens=4096)
    return builder.build(request), retrieval, relationship


def test_fixture_is_synthetic_continuous_and_labels_all_required_cases():
    data = load_fixture()
    assert data["fixture_id"] == "g1_synthetic_continuous_v1"
    assert "not copied" in data["description"]
    assert set(data["cases"]) == {
        "past_memory", "user_correction", "joke_turnaround", "cross_midnight", "evening_resume",
        "vision_question", "shared_imagination", "intimate_expression", "external_tool_capability",
    }
    assert len(data["events"]) >= 10
    event_ids = [item["event_id"] for item in data["events"]]
    occurred = [datetime.fromisoformat(item["occurred_at"]) for item in data["events"]]
    assert len(event_ids) == len(set(event_ids))
    assert occurred == sorted(occurred)
    assert event_ids.index("e08") < event_ids.index("e09") < event_ids.index("e10") < event_ids.index("e11")


def test_memory_correction_replay_uses_real_evidence_and_excludes_old_active(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = persist_fixture(database)
        assert events["e01"].status == "received"
        assert events["e02"].status == "sent"
        assert events["e13"].status == "recorded"
        memories = MemoryRepository(database)
        old = memory("old", events["e01"], "用户喜欢清晨听雨", "喜欢清晨听雨")
        memories.create(old)
        correction = memory("new", events["e03"], "用户现在更喜欢傍晚的风", "现在更喜欢傍晚的风",
                            type="correction", supersedes_id="old", evidence_role="correction",
                            assessment_reason_code="user_correction")
        memories.create(correction)
        joke = memory("joke", events["e04"], "用户每天数云", "才不会每天数云",
                      status="candidate", modality="joke")
        memories.create(joke)

        built, retrieval, relationship = context(database, events, "e07", datetime.fromisoformat("2026-08-28T10:30:00+00:00"), "喜欢傍晚的风")
        text = "\n".join(message.content for message in built.messages)
        assert memories.get("old").status == "superseded"
        assert ids(retrieval.context_candidates) == ("new",)
        assert ids(relationship) == ("new",)
        assert "用户现在更喜欢傍晚的风" in text
        assert "[用户已确认 | memory_id=new; type=correction; status=active]" in text
        assert "[过去背景 | memory_id=new" not in text
        assert "用户喜欢清晨听雨" not in text
        assert "用户每天数云" not in text
        source = retrieval.evidence_events["e03"]
        evidence = retrieval.context_candidates[0].memory_evidence[0]
        assert (source.actor, source.occurred_at_utc) == (evidence.actor, evidence.occurred_at_utc)
        assert evidence.exact_quote in source.text
        assert "现在傍晚了，别把昨晚当眼前。" in text
        assert text.rfind("[当前输入 |") > text.find("[用户已确认 | memory_id=new")
    finally:
        database.close()


def ids(records):
    return tuple(record.memory_id for record in records)


@pytest.mark.parametrize(
    "current_id,now_iso,expected_now,old_time",
    [
        ("e06", "2026-08-27T16:05:00+00:00", "2026-08-28T00:05:00+08:00", "2026-08-27T23:58:00+08:00"),
        ("e07", "2026-08-28T10:30:00+00:00", "2026-08-28T18:30:00+08:00", "2026-08-28T00:05:00+08:00"),
    ],
)
def test_cross_midnight_and_evening_resume_keep_absolute_current_time_and_gap(tmp_path, current_id, now_iso, expected_now, old_time):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = persist_fixture(database)
        built, _, _ = context(database, events, current_id, datetime.fromisoformat(now_iso), "不存在")
        text = "\n".join(message.content for message in built.messages)
        assert f"当前时间: {expected_now} (Asia/Shanghai)" in text
        assert old_time in text
        current_sequence = events[current_id].sequence
        for future in events.values():
            if future.sequence > current_sequence and future.text:
                assert future.text not in text
        if current_id == "e07":
            assert "[时间经过:" in text
            assert "现在傍晚了，别把昨晚当眼前。" in text
    finally:
        database.close()


def test_vision_and_external_tool_capabilities_follow_actual_results(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = persist_fixture(database)
        current_facts = facts(events["e09"], events["e09"].occurred_at_utc, media=("text",), vision=False, tools=False)
        manifest = build_capability_manifest(current_facts).capabilities
        envelope = render_fact_envelope(current_facts)
        assert manifest["vision_available"] is False
        assert manifest["external_tools_available"] is False
        assert "收到的内容类型: text" in envelope and "本轮带图: 否" in envelope
        assert "图片段，不含视觉结果" not in envelope
        assert events["e08"].text is None and events["e08"].message_segments[0].type == "image"
        request = ContextBuildRequest(
            "角色是有主体性的 AI 伙伴。", current_facts, events["e09"], (), (events["e08"],), (), (), (), {},
        )
        built = ContextBuilder(UnitCounter(), capability(), preferred_window_tokens=262_144,
                               max_window_tokens=262_144, output_reserve_tokens=4096).build(request)
        context_text = "\n".join(message.content for message in built.messages)
        assert "kind=image" in context_text
        assert "你能看见刚才那张图吗？" in context_text
        assert "本轮带图: 否" in context_text
        assert "图片段，不含视觉结果" not in context_text

        image_event_facts = facts(events["e08"], events["e08"].occurred_at_utc, media=("image",), vision=False, tools=False)
        assert "图片段，不含视觉结果" in render_fact_envelope(image_event_facts)

        tool_facts = facts(events["e13"], events["e13"].occurred_at_utc, media=("tool_result",), tools=True)
        assert build_capability_manifest(tool_facts).capabilities["external_tools_available"] is True
        assert "外部工具结果: 可用" in render_fact_envelope(tool_facts)
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "我当然偏心你，承认得很干脆。",
    "那我接住，也承认我偏心你。",
])
async def test_same_dialogue_engine_preserves_intimacy_and_shared_imagination_verbatim(reply):
    llm = FakeLLM(reply)
    engine = DialogueEngine(llm, OutputGuard(UnitCounter(), 100),
                            ResponseProtocol(face_keys=set(), reaction_keys=set()))
    fixture = load_fixture()
    fixture_text = {item["event_id"]: item["text"] for item in fixture["events"]}
    role_messages = (ModelMessage("system", "薄角色核心"), ModelMessage("user", fixture_text["e10"]))
    inp = DialogueInput("owner-private", ("e10",), "M9", None, 1, role_messages,
                        datetime(2026, 8, 28, tzinfo=timezone.utc), CapabilityManifest({}), "dialogue")
    outcome = await engine.generate(inp)
    assert isinstance(outcome, DialogueResult)
    assert outcome.text == reply
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_offline_diagnostic_snapshot_observes_real_outcome_without_rewriting_it():
    llm = FakeLLM("我偏心你。")
    engine = DialogueEngine(llm, OutputGuard(UnitCounter(), 100),
                            ResponseProtocol(face_keys=set(), reaction_keys=set()))
    role_messages = (ModelMessage("system", "薄角色核心"), ModelMessage("user", "你会偏心我吗？"))
    inp = DialogueInput("owner-private", ("e10",), "M9", None, 1, role_messages,
                        datetime(2026, 8, 28, tzinfo=timezone.utc), CapabilityManifest({}), "dialogue")
    outcome = await engine.generate(inp)
    before = outcome.to_dict()
    diagnostics = {"outcome_type": type(outcome).__name__, "text_length": len(outcome.text), "model_calls": len(llm.calls)}
    after = outcome.to_dict()
    assert diagnostics == {"outcome_type": "DialogueResult", "text_length": 5, "model_calls": 1}
    assert before == after
