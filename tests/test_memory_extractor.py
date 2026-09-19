from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.dialogue.llm_client import LLMConnectionError, LLMTimeoutError
from qichi.memory.extractor import MemoryExtractor


NOW = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)


def event(
    event_id: str,
    text: str,
    *,
    actor: str = "mumo",
    direction: str = "inbound",
    conversation_id: str = "conversation-a",
    sequence: int = 0,
    at: datetime = NOW,
    metadata: dict[str, object] | None = None,
) -> ConversationEvent:
    return ConversationEvent(
        event_id=event_id, platform_event_id=f"pe-{event_id}", platform_message_id=f"pm-{event_id}",
        conversation_id=conversation_id, sequence=sequence, direction=direction, actor=actor, kind="text",
        text=text, message_segments=(MessageSegment("text", {"text": text}),), reply_to_event_id=None,
        reply_to_platform_message_id=None, occurred_at_utc=at, received_at_utc=at,
        status="sent" if actor == "qichi" else "received",
        metadata=metadata or ({"generation_metadata": {"source": "dialogue"}} if actor == "qichi" else {}),
    )


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[ConversationEvent, ...]] = []

    async def generate(self, events):
        self.calls.append(tuple(events))
        return self.response


def candidate(**overrides: object) -> dict[str, object]:
    candidate = {
        "type": "preference", "normalized_fact": "用户喜欢雨声", "modality": "explicit_statement",
        "certainty": "explicit", "importance": 2, "temporal_scope": "ongoing",
        "assessment_reason_code": "explicit_user_statement",
        "valid_from_utc": "2026-08-28T15:00:00+00:00", "valid_until_utc": None,
        "supersedes_id": None,
        "evidence": [
            {"event_id": "source", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "source"}
        ],
    }
    candidate.update(overrides)
    return candidate


def review(**overrides: object) -> dict[str, object]:
    item = {
        "memory_id": "existing-memory", "action": "confirm", "certainty": "confirmed",
        "importance": 2, "temporal_scope": "ongoing",
        "assessment_reason_code": "later_user_confirmation",
        "evidence": [
            {"event_id": "source", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "confirmation"}
        ],
    }
    item.update(overrides)
    return item


def payload(*, candidates=None, reviews=None, outcome=None) -> str:
    candidate_items = [candidate()] if candidates is None else candidates
    review_items = [] if reviews is None else reviews
    if outcome is None:
        outcome = {
            "kind": "memory_found" if candidate_items or review_items else "no_persistent_memory",
            "reason_code": "explicit_user_preference" if candidate_items else (
                "existing_memory_review" if review_items else "nothing_new"
            ),
        }
    return json.dumps(
        {
            "outcome": outcome,
            "candidates": candidate_items,
            "reviews": review_items,
        },
        ensure_ascii=False,
    )


class TruncatedLLM:
    """模型返回了一个 fragment_type 非法的片段，且响应是截断的。"""

    def __init__(self, text, finish_reason):
        self.text = text
        self.last_finish_reason = finish_reason

    async def generate(self, events, *, evidence_event_ids=None):
        return self.text


@pytest.mark.asyncio
async def test_parse_failure_reports_which_field_and_whether_it_was_truncated():
    source = event("source", "我喜欢雨声")
    body = json.dumps(
        {
            "outcome": {"kind": "no_persistent_memory", "reason_code": "nothing_new"},
            "candidates": [],
            "reviews": [],
            "fragment": {
                "fragment_type": "conversation",
                "reality_scope": "conversation",
                "summary": "一段日常",
                "privacy_class": "ordinary",
                "recall_policy": "daily_safe",
                "closed": True,
            },
        },
        ensure_ascii=False,
    )
    result = await MemoryExtractor(TruncatedLLM(body, "length")).extract((source,))

    assert not result.ok
    assert result.failure.kind == "parse_error"
    assert result.failure.details["parse_error_code"] == "response_schema"
    assert result.failure.details["schema_field"] == "fragment_type"
    assert result.failure.details["finish_reason"] == "length", (
        "截断与格式错必须能区分，否则无法判断要不要缩小窗口重试"
    )


@pytest.mark.asyncio
async def test_missing_fragment_is_derived_from_the_parsed_items():
    """提示词说 fragment 可选，所以缺了它不该毁掉整段会话。"""

    source = event("source", "我喜欢雨声")
    result = await MemoryExtractor(FakeLLM(payload(candidates=[candidate()]))).extract((source,))

    assert result.ok
    assert result.fragment is not None, "缺 fragment 时按已解析内容推导，而不是整段失败"
    assert result.fragment.fragment_type == "daily"
    assert result.fragment.privacy_class == "ordinary"
    assert result.fragment.recall_policy == "daily_safe"
    assert result.fragment.reality_scope == "conversation"
    assert result.fragment.closed is True
    assert result.fragment.summary


@pytest.mark.asyncio
async def test_derived_fragment_follows_the_strictest_privacy_present():
    source = event("source", "我喜欢雨声")
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(privacy_class="adult", recall_policy="explicit_request_only")]))
    ).extract((source,))

    assert result.fragment is not None
    assert result.fragment.fragment_type == "adult"
    assert result.fragment.privacy_class == "adult"
    assert result.fragment.recall_policy == "explicit_request_only", "adult 不得落回 daily_safe"


@pytest.mark.asyncio
async def test_model_provided_fragment_still_wins():
    source = event("source", "我喜欢雨声")
    provided = json.dumps(
        {
            "outcome": {"kind": "memory_found", "reason_code": "explicit_user_preference"},
            "candidates": [candidate()],
            "reviews": [],
            "fragment": {
                "fragment_type": "mixed",
                "reality_scope": "conversation",
                "summary": "模型自己写的那句话",
                "privacy_class": "ordinary",
                "recall_policy": "daily_safe",
                "closed": True,
            },
        },
        ensure_ascii=False,
    )
    result = await MemoryExtractor(FakeLLM(provided)).extract((source,))

    assert result.fragment is not None
    assert result.fragment.fragment_type == "mixed"
    assert result.fragment.summary == "模型自己写的那句话"


@pytest.mark.asyncio
async def test_expire_review_with_the_wrong_certainty_is_dropped_not_fatal():
    """提取器与仓储必须同口径：放过去的那条会在落库时炸掉整段会话。"""

    source = event("source", "我喜欢雨声")
    body = payload(
        candidates=[candidate()],
        reviews=[review(
            action="expire", certainty="confirmed", importance=0,
            assessment_reason_code="expired_or_completed",
            evidence=[{"event_id": "source", "actor": "mumo", "exact_quote": "喜欢雨声",
                       "role": "counterevidence"}],
        )],
    )
    result = await MemoryExtractor(FakeLLM(body)).extract((source,))

    assert result.ok, "坏 review 只应被丢弃并计数，不该让整段失败"
    assert len(result.candidates) == 1
    assert result.reviews == ()
    assert result.diagnostics.get("dropped_review_count") == "1"


@pytest.mark.asyncio
async def test_extracts_candidate_with_exact_event_evidence_and_preserves_structure():
    source = event("source", "我喜欢雨声")
    llm = FakeLLM(payload(candidates=[candidate(valid_until_utc="2026-08-29T15:00:00+00:00", temporal_scope="bounded")]))
    result = await MemoryExtractor(llm).extract((source,))
    assert result.ok
    assert len(result.candidates) == 1
    record = result.candidates[0]
    assert record.status == "candidate"
    assert record.normalized_fact == "用户喜欢雨声"
    assert record.certainty == "explicit"
    assert record.importance == 2
    assert record.temporal_scope == "bounded"
    assert record.memory_evidence[0].event_id == "source"
    assert record.memory_evidence[0].exact_quote == "喜欢雨声"
    assert record.memory_evidence[0].evidence_role == "source"
    assert result.reviews == ()
    assert llm.calls == [(source,)]


@pytest.mark.asyncio
async def test_adult_episode_is_persistable_when_evidence_is_explicit():
    source = event("source", "我们昨晚明确玩过强势的成人互动，具体是由我提出支配、你接受")
    item = candidate(
        type="episode",
        normalized_fact="双方过去明确进行过一次强势成人互动，其中包含支配与接受",
        temporal_scope="historical",
        assessment_reason_code="historical_event",
        privacy_class="adult",
        recall_policy="explicit_request_only",
        evidence=[{
            "event_id": "source", "actor": "mumo",
            "exact_quote": "我们昨晚明确玩过强势的成人互动，具体是由我提出支配、你接受",
            "role": "source",
        }],
    )
    result = await MemoryExtractor(FakeLLM(payload(candidates=[item]))).extract((source,))
    assert result.ok
    assert result.outcome is not None and result.outcome.kind == "memory_found"
    record = result.candidates[0]
    assert record.privacy_class == "adult"
    assert record.recall_policy == "explicit_request_only"
    assert record.temporal_scope == "historical"


@pytest.mark.asyncio
async def test_sensitive_candidate_does_not_require_sensitive_reason_code():
    source = event("source", "我长期偏好强势一点，但每次都要重新确认")
    item = candidate(
        normalized_fact="用户偏好强势表达且每次需要重新确认",
        privacy_class="adult",
        recall_policy="topic_only",
        evidence=[{
            "event_id": "source", "actor": "mumo",
            "exact_quote": "我长期偏好强势一点，但每次都要重新确认",
            "role": "source",
        }],
    )
    result = await MemoryExtractor(FakeLLM(payload(candidates=[item]))).extract((source,))
    assert result.ok
    assert result.candidates[0].privacy_class == "adult"
    assert result.candidates[0].recall_policy == "topic_only"


@pytest.mark.asyncio
async def test_temporary_roleplay_label_does_not_drop_a_completed_bilateral_episode():
    events = (
        event("m1", "我提出了一个成人互动的方式", sequence=1),
        event("q1", "我接住并继续了这个共同想象", actor="qichi", direction="outbound", sequence=2),
        event("m2", "后来按这个方式继续，最后我们停下了", sequence=3),
        event("q2", "好，这段到这里结束", actor="qichi", direction="outbound", sequence=4),
    )
    no_memory = payload(
        candidates=[],
        outcome={"kind": "no_persistent_memory", "reason_code": "temporary_scene_or_roleplay"},
    )
    result = await MemoryExtractor(FakeLLM(no_memory)).extract(events)
    assert result.ok
    assert result.outcome is not None
    assert result.outcome.kind == "memory_found"
    assert result.outcome.reason_code == "historical_episode"
    assert len(result.candidates) == 1
    record = result.candidates[0]
    assert record.type == "episode"
    assert record.privacy_class == "adult"
    assert record.recall_policy == "explicit_request_only"
    assert {item.event_id for item in record.evidence} == {"m1", "m2"}


@pytest.mark.asyncio
async def test_temporary_roleplay_label_stays_empty_for_unilateral_fragment():
    events = (
        event("m1", "我单方面写了一句成人场景台词", sequence=1),
        event("q1", "我没有继续这个场景", actor="qichi", direction="outbound", sequence=2),
    )
    no_memory = payload(
        candidates=[],
        outcome={"kind": "no_persistent_memory", "reason_code": "temporary_scene_or_roleplay"},
    )
    result = await MemoryExtractor(FakeLLM(no_memory)).extract(events)
    assert result.ok
    assert result.outcome is not None
    assert result.outcome.kind == "no_persistent_memory"
    assert result.candidates == ()


@pytest.mark.asyncio
async def test_invalid_candidate_is_dropped_without_discarding_valid_candidate():
    source = event("source", "我喜欢雨声")
    wrong_actor = candidate(evidence=[{
        "event_id": "source", "actor": "qichi", "exact_quote": "喜欢雨声", "role": "source",
    }])
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[wrong_actor, candidate()]))
    ).extract((source,))

    assert result.ok
    assert len(result.candidates) == 1
    assert result.candidates[0].memory_evidence[0].actor == "mumo"
    assert result.diagnostics == {
        "dropped_candidate_count": "1",
        "candidate_error_codes": "candidate_evidence",
    }


@pytest.mark.asyncio
async def test_all_invalid_items_still_fail_with_safe_parse_code():
    source = event("source", "我喜欢雨声")
    wrong_actor = candidate(evidence=[{
        "event_id": "source", "actor": "qichi", "exact_quote": "喜欢雨声", "role": "source",
    }])
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[wrong_actor], reviews=[]))
    ).extract((source,))

    assert not result.ok
    assert result.candidates == () and result.reviews == ()
    assert result.failure is not None
    assert result.failure.details == {
        "parse_error_code": "all_items_invalid",
        "dropped_candidate_count": "1",
        "candidate_error_codes": "candidate_evidence",
    }


@pytest.mark.asyncio
async def test_an_invalid_candidate_no_longer_throws_away_the_timeline():
    """2026-09-17 用户裁定后收窄：候选项全被丢弃时，已解析好的明细必须留下。

    真机：模型的一条候选项把 actor/role 写反 → 整窗（上百条事件）的原文与明细一起作废；
    同一窗口切到 10 条、原样重试 3 次都过不去（错误类别稳定）。明细才是回忆入口，候选项只是提议。
    """

    source = event("source", "我喜欢雨声")
    raw = json.dumps({
        "outcome": {"kind": "memory_found", "reason_code": "explicit_user_preference"},
        "candidates": [candidate(evidence=[{
            "event_id": "source", "actor": "qichi", "exact_quote": "喜欢雨声", "role": "source",
        }])],
        "reviews": [],
        "details": [{
            "ordinal": 0, "detail_kind": "message", "actor": "mumo", "reality_scope": "conversation",
            "normalized_detail": "用户说喜欢雨声", "exact_quote": "我喜欢雨声",
            "source_event_id": "source", "certainty": "explicit", "temporal_scope": "historical",
            "status": "active", "privacy_class": "ordinary", "recall_policy": "daily_safe",
            "evidence": [{"event_id": "source", "role": "source"}],
        }],
    }, ensure_ascii=False)

    result = await MemoryExtractor(FakeLLM(raw)).extract((source,))

    assert result.ok, "明细已经解析好了，不该因为候选项被丢弃而整窗作废"
    assert result.candidates == () and result.reviews == ()
    assert len(result.details) == 1, "时间线必须留下"


@pytest.mark.asyncio
async def test_explicit_empty_v2_result_is_not_treated_as_invalid_items():
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=[]))
    ).extract((event("source", "只是闲聊"),))

    assert result.ok
    assert result.candidates == () and result.reviews == ()
    assert result.diagnostics == {}
    assert result.outcome is not None
    assert result.outcome.kind == "no_persistent_memory"
    assert result.outcome.reason_code == "nothing_new"


@pytest.mark.asyncio
async def test_empty_result_without_public_outcome_is_a_protocol_failure():
    result = await MemoryExtractor(
        FakeLLM('{"candidates":[],"reviews":[]}')
    ).extract((event("source", "只是闲聊"),))

    assert not result.ok
    assert result.failure is not None
    assert result.failure.kind == "parse_error"
    assert "outcome" in result.failure.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, reason",
    [
        ({"kind": "memory_found", "reason_code": "explicit_user_preference"}, "requires a candidate"),
        ({"kind": "no_persistent_memory", "reason_code": "nothing_new"}, "cannot contain candidates"),
    ],
)
async def test_outcome_must_agree_with_the_structured_items(outcome, reason):
    items = [candidate()] if outcome["kind"] == "no_persistent_memory" else []
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=items, reviews=[], outcome=outcome))
    ).extract((event("source", "我喜欢雨声"),))

    assert not result.ok
    assert result.failure is not None
    assert reason in result.failure.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,reason",
    [
        ('{"candidates":[{"type":"preference"}],"reviews":[]}', "missing field"),
        (json.dumps({"candidates": [{**candidate(), "unknown": 1}], "reviews": []}), "unknown field"),
        ('{"candidates":[', "invalid JSON"),
        ('{"candidates":"not-a-list","reviews":[]}', "candidates must be a list"),
        ('{"candidates":[],"reviews":[],"reasoning":"hidden"}', "outcome, candidates, and reviews"),
    ],
)
async def test_malformed_or_unknown_candidate_fails_closed(response, reason):
    result = await MemoryExtractor(FakeLLM(response)).extract((event("source", "原话"),))
    assert not result.ok
    assert result.candidates == ()
    assert reason in result.failure.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,reason",
    [
        (payload(candidates=[candidate(evidence=[{"event_id": "missing", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "source"}])]), "event does not exist"),
        (payload(candidates=[candidate(evidence=[{"event_id": "source", "actor": "qichi", "exact_quote": "喜欢雨声", "role": "source"}])]), "mumo evidence"),
        (payload(candidates=[candidate(evidence=[{"event_id": "source", "actor": "mumo", "exact_quote": "不在原文", "role": "source"}])]), "exact quote"),
        (payload(candidates=[candidate(status="active")]), "unknown field"),
        (payload(candidates=[candidate(type="correction")]), "supersedes_id"),
    ],
)
async def test_evidence_and_lifecycle_ownership_are_fail_closed(response, reason):
    source = event("source", "我喜欢雨声")
    result = await MemoryExtractor(FakeLLM(response)).extract((source,))
    assert not result.ok
    assert result.candidates == ()
    assert reason in result.failure.reason


@pytest.mark.asyncio
async def test_qichi_self_expression_is_allowed_but_remains_candidate():
    source = event("source", "我很在意用户", actor="qichi", direction="outbound")
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(
            type="self_expression", normalized_fact="角色表达过在意用户",
            evidence=[{"event_id": "source", "actor": "qichi", "exact_quote": "在意用户", "role": "source"}],
        )]))
    ).extract((source,))
    assert result.ok
    assert result.candidates[0].status == "candidate"
    assert result.candidates[0].memory_evidence[0].actor == "qichi"


@pytest.mark.asyncio
async def test_self_expression_requires_qichi_evidence():
    source = event("source", "我表达过在意", actor="mumo")
    result = await MemoryExtractor(
        FakeLLM(
            payload(candidates=[candidate(
                type="self_expression",
                normalized_fact="角色表达过在意用户",
                evidence=[{"event_id": "source", "actor": "mumo", "exact_quote": "表达过在意", "role": "source"}],
            )])
        )
    ).extract((source,))
    assert not result.ok
    assert result.candidates == ()
    assert "qichi evidence" in result.failure.reason


@pytest.mark.asyncio
async def test_preference_cannot_mix_qichi_boundary_into_user_evidence():
    user_event = event("user", "我喜欢雨声", sequence=1)
    qichi_event = event(
        "qichi", "我也有自己的边界", actor="qichi", direction="outbound", sequence=2
    )
    item = candidate(
        evidence=[
            {"event_id": "user", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "source"},
            {"event_id": "qichi", "actor": "qichi", "exact_quote": "自己的边界", "role": "source"},
        ]
    )
    result = await MemoryExtractor(FakeLLM(payload(candidates=[item]))).extract(
        (user_event, qichi_event)
    )
    assert not result.ok
    assert "candidate evidence role actor" in result.failure.reason


@pytest.mark.asyncio
async def test_agreement_requires_cross_actor_proposal_and_acceptance():
    first = event(
        "first", "我提议认真时不要拆台", actor="qichi", direction="outbound", sequence=1
    )
    second = event("second", "我接受认真时不要拆台", sequence=2)
    invalid = candidate(
        type="agreement",
        normalized_fact="认真时不拆台",
        certainty="confirmed",
        importance=3,
        assessment_reason_code="bilateral_agreement",
        evidence=[
            {"event_id": "first", "actor": "qichi", "exact_quote": "提议认真时不要拆台", "role": "proposal"},
            {"event_id": "second", "actor": "mumo", "exact_quote": "接受认真时不要拆台", "role": "acceptance"},
        ],
    )
    result = await MemoryExtractor(FakeLLM(payload(candidates=[invalid]))).extract(
        (first, second)
    )
    assert not result.ok
    assert "cross-actor proposal and acceptance" in result.failure.reason


@pytest.mark.asyncio
async def test_agreement_with_cross_actor_roles_is_accepted():
    first = event("first", "我提议认真时不要拆台", sequence=1)
    second = event(
        "second", "好，认真时我不拆台", actor="qichi", direction="outbound", sequence=2
    )
    valid = candidate(
        type="agreement",
        normalized_fact="认真时不拆台",
        certainty="confirmed",
        importance=3,
        assessment_reason_code="bilateral_agreement",
        evidence=[
            {"event_id": "first", "actor": "mumo", "exact_quote": "提议认真时不要拆台", "role": "proposal"},
            {"event_id": "second", "actor": "qichi", "exact_quote": "认真时我不拆台", "role": "acceptance"},
        ],
    )
    result = await MemoryExtractor(FakeLLM(payload(candidates=[valid]))).extract(
        (first, second)
    )
    assert result.ok
    assert result.candidates[0].type == "agreement"


@pytest.mark.asyncio
async def test_fragment_must_be_one_conversation_without_duplicate_events():
    first = event("first", "第一条")
    other = event("other", "另一条", conversation_id="conversation-b")
    first_payload = payload(candidates=[candidate(evidence=[
        {"event_id": "first", "actor": "mumo", "exact_quote": "第一条", "role": "source"}
    ])])
    cross = await MemoryExtractor(FakeLLM(first_payload)).extract(
        (first, other)
    )
    assert not cross.ok
    assert "conversation" in cross.failure.reason

    duplicate = await MemoryExtractor(FakeLLM(first_payload)).extract(
        (first, first)
    )
    assert not duplicate.ok
    assert "duplicate" in duplicate.failure.reason


@pytest.mark.asyncio
async def test_explicit_evidence_scope_keeps_context_but_rejects_context_only_evidence():
    evidence_source = event("allowed", "我喜欢雨声", sequence=1)
    context_only = event(
        "context", "那我记得了", actor="qichi", direction="outbound", sequence=2
    )

    class EvidenceAwareLLM:
        def __init__(self, response):
            self.response = response
            self.calls = []

        async def generate(self, events, *, evidence_event_ids):
            self.calls.append((tuple(events), evidence_event_ids))
            return self.response

    allowed_payload = payload(candidates=[candidate(evidence=[{
        "event_id": "allowed", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "source",
    }])])
    allowed_llm = EvidenceAwareLLM(allowed_payload)
    allowed = await MemoryExtractor(allowed_llm).extract(
        (evidence_source, context_only), evidence_event_ids=frozenset({"allowed"})
    )
    assert allowed.ok
    assert allowed_llm.calls == [
        ((evidence_source, context_only), frozenset({"allowed"}))
    ]

    context_payload = payload(candidates=[candidate(
        type="self_expression", normalized_fact="角色说她会记得",
        evidence=[{
            "event_id": "context", "actor": "qichi", "exact_quote": "记得了", "role": "source",
        }],
    )])
    rejected = await MemoryExtractor(EvidenceAwareLLM(context_payload)).extract(
        (evidence_source, context_only), evidence_event_ids=frozenset({"allowed"})
    )
    assert not rejected.ok
    assert rejected.candidates == () and rejected.reviews == ()
    assert "context-only" in rejected.failure.reason


@pytest.mark.asyncio
async def test_evidence_scope_must_be_a_fragment_subset_before_model_call():
    llm = FakeLLM(payload(candidates=[]))
    result = await MemoryExtractor(llm).extract(
        (event("source", "原话"),), evidence_event_ids=frozenset({"outside"})
    )
    assert not result.ok
    assert result.failure is not None and result.failure.kind == "input_error"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_ambiguous_joke_and_correction_modality_is_not_promoted():
    source = event("source", "开玩笑，我喜欢雨声")
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(
            modality="joke", certainty="unsupported", importance=0,
            temporal_scope="unclassified", assessment_reason_code="unsupported_or_transient",
            evidence=[{"event_id": "source", "actor": "mumo", "exact_quote": "开玩笑，我喜欢雨声", "role": "source"}],
        )]))
    ).extract((source,))
    assert result.ok
    assert result.candidates[0].status == "candidate"
    assert result.candidates[0].modality == "joke"


@pytest.mark.asyncio
async def test_single_json_markdown_fence_is_transport_only():
    source = event("source", "我喜欢雨声")
    response = "```json\n" + payload() + "\n```"
    result = await MemoryExtractor(FakeLLM(response)).extract((source,))
    assert result.ok
    assert result.candidates[0].memory_evidence[0].exact_quote == "喜欢雨声"


@pytest.mark.asyncio
async def test_json_fence_with_narration_stays_a_parse_failure():
    source = event("source", "我喜欢雨声")
    response = "这是结果：\n```json\n" + payload() + "\n```"
    result = await MemoryExtractor(FakeLLM(response)).extract((source,))
    assert not result.ok
    assert result.failure is not None and result.failure.kind == "parse_error"


@pytest.mark.asyncio
async def test_complete_fragment_can_withdraw_an_earlier_claim_without_partial_memory():
    first = event("first", "我喜欢雨声")
    reply = event("reply", "这个我会记着", actor="qichi", direction="outbound")
    correction = event("correction", "其实是开玩笑，不是那个意思")
    llm = FakeLLM(payload(candidates=[], reviews=[]))
    result = await MemoryExtractor(llm).extract((first, reply, correction))
    assert result.ok
    assert result.candidates == ()
    assert llm.calls == [(first, reply, correction)]


@pytest.mark.asyncio
async def test_legacy_nonempty_candidate_shape_is_not_accepted_as_v2():
    response = json.dumps({
        "candidates": [{
            "type": "preference", "normalized_fact": "用户喜欢雨声",
            "modality": "explicit_statement", "status": "candidate",
            "event_id": "source", "actor": "mumo", "exact_quote": "喜欢雨声",
        }]
    }, ensure_ascii=False)
    result = await MemoryExtractor(FakeLLM(response)).extract((event("source", "我喜欢雨声"),))
    assert not result.ok
    assert result.candidates == () and result.reviews == ()
    assert "legacy empty result" in result.failure.reason


@pytest.mark.asyncio
async def test_schema_failure_does_not_copy_untrusted_field_names():
    response = json.dumps(
        {"candidates": [{**candidate(normalized_fact="x", evidence=[
            {"event_id": "source", "actor": "mumo", "exact_quote": "x", "role": "source"}
        ]), "secret-api-key": 1}], "reviews": []}
    )
    result = await MemoryExtractor(FakeLLM(response)).extract((event("source", "x"),))
    assert not result.ok
    assert "secret-api-key" not in result.failure.reason


@pytest.mark.asyncio
async def test_candidate_identity_includes_all_immutable_semantic_fields():
    source = event("source", "我喜欢雨声")
    explicit = await MemoryExtractor(FakeLLM(payload())).extract((source,))
    joke = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(modality="joke")]))
    ).extract((source,))
    assert explicit.ok and joke.ok
    assert explicit.candidates[0].memory_id != joke.candidates[0].memory_id


@pytest.mark.asyncio
async def test_v2_reviews_are_strict_and_derive_evidence_from_the_event_ledger():
    source = event("source", "对，我确实喜欢雨声")
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=[review(exact_unknown="nope")]))
    ).extract((source,))
    assert not result.ok
    assert "unknown field" in result.failure.reason

    accepted = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=[review()]))
    ).extract((source,))
    assert accepted.ok
    assert accepted.candidates == ()
    assert len(accepted.reviews) == 1
    assert accepted.reviews[0].action == "confirm"
    assert accepted.reviews[0].evidence[0].occurred_at_utc == source.occurred_at_utc
    assert accepted.reviews[0].evidence[0].evidence_role == "confirmation"


@pytest.mark.asyncio
async def test_malformed_review_is_isolated_without_discarding_valid_candidate():
    source = event("source", "我喜欢雨声")
    malformed = review(
        certainty="explicit",
        assessment_reason_code="explicit_user_statement",
        evidence=[{
            "event_id": "source", "actor": "mumo",
            "exact_quote": "喜欢雨声", "role": "source",
        }],
    )
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate()], reviews=[malformed]))
    ).extract((source,))

    assert result.ok
    assert len(result.candidates) == 1
    assert result.reviews == ()
    assert result.diagnostics == {
        "dropped_review_count": "1",
        "review_error_codes": "review_invalid",
    }


@pytest.mark.asyncio
async def test_all_malformed_review_items_remain_a_parse_failure():
    source = event("source", "我喜欢雨声")
    malformed = review(evidence=[{
        "event_id": "source", "actor": "mumo",
        "exact_quote": "喜欢雨声", "role": "source",
    }])
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=[malformed]))
    ).extract((source,))

    assert not result.ok
    assert result.failure is not None
    assert result.failure.details == {
        "parse_error_code": "all_items_invalid",
        "dropped_review_count": "1",
        "review_error_codes": "review_invalid",
    }


@pytest.mark.asyncio
async def test_evidence_is_sorted_by_event_time_then_sequence_and_identity_ignores_grades_and_later_support():
    later = event("z-later", "后来我又说喜欢雨声", sequence=2, at=NOW + timedelta(minutes=1))
    first = event("a-first", "我喜欢雨声", sequence=1)
    evidence = [
        {"event_id": "z-later", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "confirmation"},
        {"event_id": "a-first", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "source"},
    ]
    original = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(evidence=evidence)]))
    ).extract((later, first))
    regraded = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(
            certainty="confirmed", importance=3,
            assessment_reason_code="later_user_confirmation",
            evidence=list(reversed(evidence)),
        )]))
    ).extract((later, first))
    without_later_support = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(evidence=[{
            "event_id": "a-first", "actor": "mumo", "exact_quote": "喜欢雨声", "role": "source",
        }])]))
    ).extract((first,))

    assert original.ok and regraded.ok and without_later_support.ok
    assert [item.event_id for item in original.candidates[0].memory_evidence] == [
        "a-first", "z-later"
    ]
    assert original.candidates[0].memory_id == regraded.candidates[0].memory_id
    assert original.candidates[0].memory_id == without_later_support.candidates[0].memory_id


@pytest.mark.asyncio
async def test_v2_limits_accept_the_boundary_and_reject_one_over_without_partial_output():
    source = event("source", "我喜欢雨声")
    twelve = [candidate(normalized_fact=f"用户喜欢雨声 {index}") for index in range(12)]
    accepted = await MemoryExtractor(FakeLLM(payload(candidates=twelve))).extract((source,))
    assert accepted.ok and len(accepted.candidates) == 12

    too_many = await MemoryExtractor(
        FakeLLM(payload(candidates=twelve + [candidate(normalized_fact="第十三条")]))
    ).extract((source,))
    assert not too_many.ok
    assert too_many.candidates == () and too_many.reviews == ()
    assert "at most 12" in too_many.failure.reason

    too_much_evidence = candidate(evidence=[candidate()["evidence"][0]] * 5)
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[too_much_evidence]))
    ).extract((source,))
    assert not result.ok
    assert "at most 4" in result.failure.reason

    twelve_reviews = [review(memory_id=f"memory-{index}") for index in range(12)]
    accepted_reviews = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=twelve_reviews))
    ).extract((source,))
    assert accepted_reviews.ok and len(accepted_reviews.reviews) == 12

    too_many_reviews = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=twelve_reviews + [review(memory_id="memory-12")]))
    ).extract((source,))
    assert not too_many_reviews.ok
    assert too_many_reviews.candidates == () and too_many_reviews.reviews == ()
    assert "at most 12" in too_many_reviews.failure.reason

    review_with_unknown_evidence = review(evidence=[{
        "event_id": "source", "actor": "mumo", "exact_quote": "喜欢雨声",
        "role": "confirmation", "reasoning": "private",
    }])
    unknown = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=[review_with_unknown_evidence]))
    ).extract((source,))
    assert not unknown.ok
    assert "unknown field in evidence" in unknown.failure.reason

    too_much_review_evidence = review(evidence=[review()["evidence"][0]] * 5)
    over = await MemoryExtractor(
        FakeLLM(payload(candidates=[], reviews=[too_much_review_evidence]))
    ).extract((source,))
    assert not over.ok
    assert "at most 4" in over.failure.reason


@pytest.mark.asyncio
async def test_explicit_intimate_preference_is_not_rejected_by_topic_words():
    source = event("source", "我长期偏好直接谈成人亲密内容，但旧场景不代表现在同意")
    item = candidate(
        normalized_fact="用户长期偏好直接表达成人亲密内容，且旧场景不延续当前同意",
        evidence=[{
            "event_id": "source", "actor": "mumo",
            "exact_quote": "我长期偏好直接谈成人亲密内容，但旧场景不代表现在同意",
            "role": "source",
        }],
    )
    result = await MemoryExtractor(FakeLLM(payload(candidates=[item]))).extract((source,))
    assert result.ok
    assert result.candidates[0].certainty == "explicit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extreme",
    ["0001-01-01T00:00:00+14:00", "9999-12-31T23:59:59-14:00"],
)
async def test_extreme_offset_time_is_an_auditable_parse_failure(extreme):
    source = event("source", "我喜欢雨声")
    result = await MemoryExtractor(
        FakeLLM(payload(candidates=[candidate(valid_from_utc=extreme)]))
    ).extract((source,))
    assert not result.ok
    assert result.candidates == () and result.reviews == ()
    assert result.failure is not None and result.failure.kind == "parse_error"
    assert "valid_from_utc is invalid" in result.failure.reason


@pytest.mark.asyncio
async def test_llm_failure_is_auditable_and_has_no_candidate():
    class Broken:
        async def generate(self, events):
            raise RuntimeError("provider unavailable secret-api-key")

    result = await MemoryExtractor(Broken()).extract((event("source", "原话"),))
    assert not result.ok
    assert result.candidates == ()
    assert result.failure.kind == "llm_error"
    assert result.failure.event_ids == ("source",)
    assert "secret-api-key" not in result.failure.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,kind,provider_error",
    [
        (LLMTimeoutError("secret timeout"), "timeout", "timeout"),
        (LLMConnectionError("secret connection"), "network_error", "connection"),
        (RuntimeError("secret unknown"), "llm_error", "unknown"),
    ],
)
async def test_provider_failure_has_safe_operational_category(error, kind, provider_error):
    class Broken:
        async def generate(self, events):
            raise error

    result = await MemoryExtractor(Broken()).extract((event("source", "原话"),))
    assert not result.ok
    assert result.failure.kind == kind
    assert result.failure.details == {"provider_error": provider_error}
    assert "secret" not in json.dumps(dict(result.failure.details), ensure_ascii=False)
