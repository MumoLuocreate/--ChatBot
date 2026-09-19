from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from qichi.dialogue.capability_manifest import MessageSourceFact, RuntimeFacts
from qichi.dialogue.context_builder import (
    IMAGE_HISTORY_MARKER,
    ContextBudgetError,
    ContextBuildRequest,
    ContextBuilder,
    ContextValidationError,
)
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence, ProviderCapabilityUnverifiedError
from qichi.domain.dialogue import ModelImage
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import Agreement, Correction, MemoryEvidence, MemoryRecord
from qichi.domain.memory_details import MemoryDetailEvidence, MemoryDetailRecord


UTC = timezone.utc
BASE = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)


class UnitTokenCounter:
    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError
        return 0 if not text else 1


class CharacterTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text)


class LargeRecentTokenCounter:
    def count_text(self, text: str) -> int:
        return 20 if "TOO-LARGE-RECENT" in text else 1


class LargeWorkingSetTokenCounter:
    def count_text(self, text: str) -> int:
        return 20 if "[关系记忆工作集 |" in text else 1


def capability(tokens: int) -> ModelCapability:
    return ModelCapability(
        "test/model",
        tokens,
        provider_evidence=ProviderCapabilityEvidence(
            "SiliconFlow", "test/model", tokens, "test-evidence", BASE
        ),
    )


def event(sequence: int, actor: str, text: str | None, *, at: datetime | None = None, kind: str = "text", conversation="owner-1"):
    direction = "inbound" if actor == "mumo" else "outbound" if actor == "qichi" else "internal"
    segments = () if text is None else (MessageSegment("text", {"text": text}),)
    occurred = at or BASE + timedelta(minutes=sequence)
    return ConversationEvent(
        event_id=f"event-{sequence}-{actor}",
        platform_event_id=None,
        platform_message_id=f"platform-{sequence}-{actor}" if actor != "platform" else None,
        conversation_id=conversation,
        sequence=sequence,
        direction=direction,
        actor=actor,
        kind=kind,
        text=text,
        message_segments=segments,
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=occurred,
        received_at_utc=occurred,
        status="received",
        metadata={},
    )


def evidence(memory_id: str, source: ConversationEvent, quote: str | None = None) -> MemoryEvidence:
    return MemoryEvidence(
        memory_id,
        source.event_id,
        source.actor,
        quote if quote is not None else source.text,
        source.occurred_at_utc,
    )


def memory(
    memory_id: str,
    source: ConversationEvent,
    *,
    type="episodic",
    status="active",
    fact=None,
    certainty="explicit",
    importance=1,
    temporal_scope="ongoing",
    assessment_reason_code="explicit_user_statement",
    privacy_class="ordinary",
    recall_policy="daily_safe",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        type=type,
        normalized_fact=fact or f"fact-{memory_id}",
        modality="explicit_statement",
        status=status,
        valid_from_utc=source.occurred_at_utc,
        valid_until_utc=None,
        supersedes_id=None,
        created_at_utc=source.occurred_at_utc,
        memory_evidence=(evidence(memory_id, source),),
        certainty=certainty,
        importance=importance,
        temporal_scope=temporal_scope,
        assessment_reason_code=assessment_reason_code,
        assessed_at_utc=source.occurred_at_utc,
        privacy_class=privacy_class,
        recall_policy=recall_policy,
    )


def runtime_facts(current: ConversationEvent, quote: ConversationEvent | None = None, *, now: datetime | None = None):
    return RuntimeFacts(
        current_time=now or current.occurred_at_utc + timedelta(seconds=5),
        seconds_since_last_message=5,
        current_source=MessageSourceFact(current.actor, current.visible_handle, current.kind, current.occurred_at_utc),
        quoted_source=(
            MessageSourceFact(quote.actor, quote.visible_handle, quote.kind, quote.occurred_at_utc) if quote is not None else None
        ),
        received_media=("text",) if current.text is not None else ("poke",),
        available_actions=("text", "reply"),
        vision_available=False,
        external_tools_available=False,
    )


def request(**overrides) -> ContextBuildRequest:
    current = overrides.pop("current_event", event(20, "mumo", "current"))
    quote_chain = overrides.pop("quoted_chain", ())
    values = {
        "role_core": "role-core",
        "runtime_facts": runtime_facts(current, quote_chain[0] if quote_chain else None),
        "current_event": current,
        "quoted_chain": quote_chain,
        "recent_events": (),
        "relationship_state": (),
        "memory_candidates": (),
        "memory_working_set": (),
        "earlier_events": (),
        "evidence_events": {},
    }
    values.update(overrides)
    return ContextBuildRequest(**values)


def test_adult_memory_is_hidden_from_daily_context_but_available_when_topic_is_in_scope():
    source = event(19, "mumo", "昨晚我们明确进入过那个场景，具体方式以这条原话为准")
    current = event(20, "mumo", "今天陪我聊聊工作")
    sensitive = memory(
        "adult-episode",
        source,
        type="episode",
        fact="双方过去明确进行过一次强势成人互动",
        temporal_scope="historical",
        importance=2,
        assessment_reason_code="historical_event",
        privacy_class="adult",
        recall_policy="explicit_request_only",
    )
    daily = request(
        current_event=current,
        memory_working_set=(sensitive,),
        evidence_events={source.event_id: source},
    )
    daily_result = ContextBuilder(UnitTokenCounter(), capability(262144)).build(daily)
    assert "adult-episode" not in "\n".join(message.content for message in daily_result.messages)

    related = request(
        current_event=current,
        memory_working_set=(sensitive,),
        memory_candidates=(sensitive,),
        evidence_events={source.event_id: source},
        allow_sensitive_memory=True,
    )
    related_result = ContextBuilder(UnitTokenCounter(), capability(262144)).build(related)
    rendered = "\n".join(message.content for message in related_result.messages)
    assert "adult-episode" in rendered
    assert "双方过去明确进行过一次强势成人互动" in rendered


def test_detailed_adult_timeline_is_hidden_daily_and_recalled_in_order_when_explicit():
    source = event(19, "mumo", "片段中的原话")
    current = event(20, "mumo", "请回顾刚才那段具体经过")
    detail = MemoryDetailRecord(
        detail_id="detail-1",
        fragment_id="fragment-1",
        ordinal=0,
        detail_kind="message",
        actor="mumo",
        reality_scope="shared_imagination",
        normalized_detail="用户在历史想象片段中说出了一句原话",
        exact_quote="片段中的原话",
        source_event_id=source.event_id,
        occurred_at_utc=source.occurred_at_utc,
        certainty="explicit",
        temporal_scope="historical",
        status="candidate",
        privacy_class="adult",
        recall_policy="explicit_request_only",
        evidence=(MemoryDetailEvidence(source.event_id),),
    )
    daily = builder().build(
        request(
            current_event=event(20, "mumo", "今天聊工作"),
            memory_details=(detail,),
            evidence_events={source.event_id: source},
        )
    )
    assert "片段中的原话" not in joined(daily)

    explicit = builder().build(
        request(
            current_event=current,
            memory_details=(detail,),
            evidence_events={source.event_id: source},
            allow_sensitive_memory=True,
        )
    )
    rendered = joined(explicit)
    assert "详细时间线证据" in rendered
    assert "片段中的原话" in rendered
    assert "scope=shared_imagination" in rendered


def builder(*, preferred=20, maximum=30, reserve=2, provider_tokens=30, counter=None):
    return ContextBuilder(
        counter or UnitTokenCounter(),
        capability(provider_tokens),
        preferred_window_tokens=preferred,
        max_window_tokens=maximum,
        output_reserve_tokens=reserve,
    )


def joined(result) -> str:
    return "\n".join(message.content for message in result.messages)


def test_only_internal_platform_initiative_may_be_a_platform_current_event():
    valid = event(20, "platform", None, kind="initiative")
    result = builder().build(request(current_event=valid, runtime_facts=runtime_facts(valid)))
    assert "actor=platform" in joined(result)
    assert "kind=initiative" in joined(result)

    invalid = (
        (replace(valid, direction="inbound"), None),
        (replace(valid, direction="outbound", actor="qichi"), None),
        (replace(valid, kind="text"), None),
    )
    for current, source in invalid:
        facts = (
            runtime_facts(current)
            if source is None
            else replace(runtime_facts(valid), current_source=source)
        )
        with pytest.raises(ContextValidationError, match="initiative platform event"):
            builder().build(request(current_event=current, runtime_facts=facts))


def test_builds_role_facts_history_quote_and_current_with_exact_sources():
    old = event(1, "mumo", "old-user", at=BASE)
    reply = event(2, "qichi", "old-assistant", at=BASE + timedelta(minutes=2))
    quote = event(3, "qichi", "quoted", at=BASE + timedelta(minutes=3))
    current = event(4, "mumo", "current", at=BASE + timedelta(minutes=4))

    result = builder().build(
        request(current_event=current, quoted_chain=(quote,), recent_events=(old, reply))
    )
    text = joined(result)

    assert result.messages[0].role == "system"
    assert result.messages[-1].role == "user"
    assert "role-core" in result.messages[0].content
    assert "[事实与能力]" in text
    assert "[历史原文 | actor=mumo; handle=M1; time=2026-08-27T18:00:00+08:00; kind=text]" in text
    assert "[历史原文 | actor=qichi; handle=Q2; time=2026-08-27T18:02:00+08:00; kind=text]" in text
    assert "[直接引用 | actor=qichi; handle=Q3; time=2026-08-27T18:03:00+08:00; kind=text]" in text
    assert "[当前输入 | actor=mumo; handle=M4; time=2026-08-27T18:04:00+08:00; kind=text]" in text
    assert text.index("old-user") < text.index("old-assistant") < text.index("quoted") < text.index("current")
    assert result.metrics.input_tokens == len(result.messages)
    assert result.metrics.window_tokens == 20
    assert result.metrics.expanded is False


def test_dialogue_roles_contain_only_original_text_and_context_headers_stay_system_owned():
    old_user = event(1, "mumo", "old-user", at=BASE)
    old_assistant = event(2, "qichi", "old-assistant", at=BASE + timedelta(minutes=2))
    current = event(3, "mumo", "current-user", at=BASE + timedelta(minutes=3))

    result = builder().build(
        request(current_event=current, recent_events=(old_user, old_assistant))
    )

    user_contents = [message.content for message in result.messages if message.role == "user"]
    assistant_contents = [
        message.content for message in result.messages if message.role == "assistant"
    ]
    system_text = "\n".join(
        message.content for message in result.messages if message.role == "system"
    )
    history_blocks = [
        message
        for message in result.messages
        if message.role == "system" and message.content.startswith("[历史对话证据 |")
    ]

    assert user_contents == ["current-user"]
    assert assistant_contents == []
    assert len(history_blocks) == 1
    assert "[历史原文 | actor=mumo; handle=M1;" in system_text
    assert "[历史原文 | actor=qichi; handle=Q2;" in system_text
    assert "old-assistant" in system_text
    assert "old-user" in system_text
    assert "[当前输入 | actor=mumo; handle=M3;" in system_text
    assert all("[历史原文 |" not in content for content in user_contents + assistant_contents)
    assert all("[当前输入 |" not in content for content in user_contents + assistant_contents)


def test_historical_qichi_text_is_a_system_record_not_an_assistant_turn():
    old_user = event(1, "mumo", "old-user", at=BASE)
    old_assistant = event(2, "qichi", "（曾经的动作旁白）旧台词", at=BASE + timedelta(minutes=2))
    current = event(3, "mumo", "当前输入", at=BASE + timedelta(minutes=3))

    result = builder().build(
        request(current_event=current, recent_events=(old_user, old_assistant))
    )
    system_text = "\n".join(message.content for message in result.messages if message.role == "system")
    assistant_contents = [message.content for message in result.messages if message.role == "assistant"]
    history_blocks = [
        message
        for message in result.messages
        if message.role == "system" and message.content.startswith("[历史对话证据 |")
    ]

    assert "（曾经的动作旁白）旧台词" in system_text
    assert assistant_contents == []
    assert len(history_blocks) == 1
    assert [message.content for message in result.messages if message.role == "user"] == ["当前输入"]


def test_priority_keeps_pins_then_newest_recent_events_and_omits_lower_layers():
    recent = tuple(event(index, "mumo" if index % 2 else "qichi", f"recent-{index}") for index in range(1, 5))
    earlier = (event(0, "mumo", "earlier"),)
    quote = event(10, "qichi", "quoted")
    current = event(11, "mumo", "current")
    source = event(12, "mumo", "memory-source", at=BASE + timedelta(minutes=5))
    preference = memory("pref", source, type="preference")
    candidate = memory("episode", source)

    # 11 -> 12：2026-09-12 起「本轮事实」也是一条必带 system 片段（缓存版式），
    # 多占一个单位 token；槽位整体 +1，断言的分层取舍不变。
    result = builder(preferred=12, maximum=12, reserve=1, provider_tokens=12).build(
        request(
            current_event=current,
            quoted_chain=(quote,),
            recent_events=recent,
            relationship_state=(preference,),
            memory_candidates=(candidate,),
            earlier_events=earlier,
            evidence_events={source.event_id: source},
        )
    )
    text = joined(result)

    for pinned in ("role-core", "[事实与能力]", "fact-pref", "quoted", "current"):
        assert pinned in text
    assert "recent-4" in text
    assert "recent-3" in text
    assert "recent-2" in text
    assert "recent-1" not in text
    assert "fact-episode" not in text
    assert "earlier" not in text
    assert result.metrics.omitted_counts["recent_history"] == 1
    assert result.metrics.omitted_counts["memory_evidence"] == 1
    assert result.metrics.omitted_counts["earlier_history"] == 1


def test_earlier_history_never_jumps_over_an_omitted_recent_event():
    recent = event(5, "mumo", "TOO-LARGE-RECENT")
    earlier = event(1, "mumo", "small-earlier")
    result = builder(
        preferred=8,
        maximum=8,
        reserve=1,
        provider_tokens=8,
        counter=LargeRecentTokenCounter(),
    ).build(request(recent_events=(recent,), earlier_events=(earlier,)))

    assert "TOO-LARGE-RECENT" not in joined(result)
    assert "small-earlier" not in joined(result)
    assert result.metrics.omitted_counts["recent_history"] == 1
    assert result.metrics.omitted_counts["earlier_history"] == 1


def test_recent_history_soft_budget_keeps_newest_continuity_without_filling_window_with_old_dialogue():
    recent = tuple(event(index, "mumo", f"recent-{index}") for index in range(1, 5))
    # The default production budget is much larger than this test's normal
    # unit context, so use a dedicated builder with an intentionally tiny
    # continuity budget to exercise the structural boundary.
    bounded = ContextBuilder(
        UnitTokenCounter(), capability(100), preferred_window_tokens=100,
        max_window_tokens=100, output_reserve_tokens=1,
        recent_history_budget_tokens=2,
    ).build(request(recent_events=recent))
    bounded_text = joined(bounded)
    assert "recent-4" in bounded_text
    assert "recent-3" not in bounded_text
    assert "recent-2" not in bounded_text
    assert "recent-1" not in bounded_text
    assert bounded.metrics.omitted_counts["recent_history"] == 3


def test_recent_history_budget_rejects_negative_or_non_integer_values():
    with pytest.raises(ValueError, match="must not be negative"):
        ContextBuilder(UnitTokenCounter(), capability(20), recent_history_budget_tokens=-1)
    with pytest.raises(TypeError, match="must be an int"):
        ContextBuilder(UnitTokenCounter(), capability(20), recent_history_budget_tokens=True)


def test_expands_only_when_real_content_was_omitted_and_max_is_verified():
    recent = tuple(event(index, "mumo", f"recent-{index}") for index in range(1, 6))
    expanded = builder(preferred=6, maximum=16, reserve=1, provider_tokens=16).build(
        request(recent_events=recent)
    )
    short = builder(preferred=9, maximum=16, reserve=1, provider_tokens=16).build(request())

    assert expanded.metrics.expanded is True
    assert expanded.metrics.window_tokens == 16
    assert expanded.metrics.omitted_counts["recent_history"] == 0
    assert short.metrics.expanded is False
    assert short.metrics.window_tokens == 9
    assert short.metrics.input_tokens < short.metrics.input_budget_tokens


def test_does_not_expand_when_provider_only_supports_preferred_window():
    recent = tuple(event(index, "mumo", f"recent-{index}") for index in range(1, 6))
    result = builder(preferred=6, maximum=10, reserve=1, provider_tokens=6).build(request(recent_events=recent))

    assert result.metrics.expanded is False
    assert result.metrics.window_tokens == 6
    assert result.metrics.omitted_counts["recent_history"] > 0


def test_unverified_preferred_window_fails_closed():
    context_builder = ContextBuilder(
        UnitTokenCounter(),
        ModelCapability("test/model", 30),
        preferred_window_tokens=20,
        max_window_tokens=30,
        output_reserve_tokens=2,
    )

    with pytest.raises(ProviderCapabilityUnverifiedError, match="UNVERIFIED"):
        context_builder.build(request())


def test_mandatory_current_quote_correction_and_pending_agreement_are_never_trimmed():
    quote = event(1, "qichi", "quoted")
    current = event(2, "mumo", "current")
    correction_source = event(3, "mumo", "不是那个意思", at=BASE)
    agreement_source = event(4, "mumo", "九点再聊", at=BASE + timedelta(minutes=1))
    correction = Correction(
        "correction-1",
        "old-memory",
        "不是旧说法",
        correction_source.occurred_at_utc,
        (evidence("correction-memory", correction_source),),
    )
    agreement = Agreement(
        "agreement-1",
        "九点再聊",
        "pending",
        agreement_source.occurred_at_utc,
        None,
        agreement_source.occurred_at_utc,
        (evidence("agreement-memory", agreement_source),),
    )
    build_request = request(
        current_event=current,
        quoted_chain=(quote,),
        relationship_state=(correction, agreement),
        evidence_events={correction_source.event_id: correction_source, agreement_source.event_id: agreement_source},
    )

    with pytest.raises(ContextBudgetError, match="mandatory"):
        builder(preferred=5, maximum=6, reserve=1, provider_tokens=6).build(build_request)

    result = builder(preferred=9, maximum=9, reserve=1, provider_tokens=9).build(build_request)
    text = joined(result)
    assert "不是旧说法" in text
    assert "九点再聊" in text


def test_memory_candidates_are_capped_at_twelve_and_require_active_evidence():
    source_events = tuple(event(index, "mumo", f"source-{index}") for index in range(1, 15))
    candidates = tuple(memory(f"memory-{index}", source) for index, source in enumerate(source_events, 1))
    result = builder(preferred=30, maximum=30, reserve=1, provider_tokens=30).build(
        request(memory_candidates=candidates, evidence_events={item.event_id: item for item in source_events})
    )

    assert len(result.metrics.selected_memory_ids) == 12
    assert result.metrics.omitted_counts["memory_evidence"] == 2
    assert "fact-memory-12" in joined(result)
    assert "fact-memory-13" not in joined(result)

    inactive = memory("inactive", source_events[0], status="candidate")
    with pytest.raises(ContextValidationError, match="active"):
        builder().build(request(memory_candidates=(inactive,), evidence_events={source_events[0].event_id: source_events[0]}))


def test_active_memory_working_set_is_compact_ranked_background_not_detailed_evidence():
    low_source = event(1, "mumo", "偶尔想听两句好听的")
    high_source = event(2, "mumo", "不要在我还想聊的时候让我先去睡")
    low = replace(
        memory("low", low_source, type="preference", fact="用户偶尔想听两句好听的"),
        certainty="explicit",
        importance=2,
        temporal_scope="ongoing",
        assessment_reason_code="explicit_user_statement",
        assessed_at_utc=low_source.occurred_at_utc,
    )
    high = replace(
        memory("high", high_source, type="agreement", fact="用户还想聊的时候不要让他先去睡"),
        certainty="explicit",
        importance=3,
        temporal_scope="ongoing",
        assessment_reason_code="explicit_user_statement",
        assessed_at_utc=high_source.occurred_at_utc,
    )

    result = builder(preferred=30, maximum=30, reserve=1, provider_tokens=30).build(
        request(
            memory_working_set=(low, high),
            evidence_events={
                low_source.event_id: low_source,
                high_source.event_id: high_source,
            },
        )
    )
    text = joined(result)

    assert text.count("[关系记忆工作集 |") == 1
    assert text.index("memory_id=high") < text.index("memory_id=low")
    assert "importance=3" in text and "importance=2" in text
    assert "certainty=explicit" in text and "temporal_scope=ongoing" in text
    assert "用户还想聊的时候不要让他先去睡" in text
    assert "用户偶尔想听两句好听的" in text
    assert "exact_quote=" not in text
    assert "[过去背景 | memory_id=low" not in text
    assert result.metrics.selected_working_memory_ids == ("high", "low")
    assert result.metrics.selected_memory_ids == ()


def test_memory_working_set_rejects_inactive_or_unresolved_records_and_never_blocks_current():
    source = event(1, "mumo", "source")
    inactive = memory("inactive", source, status="candidate")
    with pytest.raises(ContextValidationError, match="working set memory must be active"):
        builder().build(
            request(
                memory_working_set=(inactive,),
                evidence_events={source.event_id: source},
            )
        )

    active = replace(
        memory("active", source, fact="X" * 50),
        certainty="explicit",
        importance=1,
        temporal_scope="historical",
        assessment_reason_code="historical_event",
        assessed_at_utc=source.occurred_at_utc,
    )
    with pytest.raises(ContextValidationError, match="evidence event"):
        builder().build(request(memory_working_set=(active,)))

    result = builder(
        preferred=8,
        maximum=8,
        reserve=1,
        provider_tokens=8,
        counter=LargeWorkingSetTokenCounter(),
    ).build(
        request(
            memory_working_set=(active,),
            evidence_events={source.event_id: source},
        )
    )
    assert "current" in joined(result)
    assert "X" * 50 not in joined(result)
    assert result.metrics.selected_working_memory_ids == ()
    assert result.metrics.omitted_counts["memory_working_set"] == 1


def test_evidence_must_resolve_to_matching_actor_time_and_exact_quote():
    source = event(1, "mumo", "原话证据")
    record = memory("memory-1", source)

    with pytest.raises(ContextValidationError, match="evidence event"):
        builder().build(request(memory_candidates=(record,)))

    wrong_text = event(1, "mumo", "不同原文")
    with pytest.raises(ContextValidationError, match="exact quote"):
        builder().build(request(memory_candidates=(record,), evidence_events={source.event_id: wrong_text}))


def test_qichi_evidence_is_not_labeled_as_a_confirmed_mumo_fact():
    source = event(1, "qichi", "我以前这样表达过")
    record = memory("qichi-memory", source, type="self_expression")

    result = builder().build(
        request(relationship_state=(record,), evidence_events={source.event_id: source})
    )
    text = joined(result)

    assert "[关系状态 | memory_id=qichi-memory" in text
    assert "[用户已确认 | memory_id=qichi-memory" not in text


def test_an_episode_is_rendered_as_a_plain_remembered_line():
    """2026-09-18 呈现层：常驻的 episode 只写一行归一事实，不带台账字段，并附边界说明。"""

    source = event(1, "mumo", "那块板子能接屏幕")
    record = memory("episode-1", source, type="episode")

    result = builder().build(
        request(relationship_state=(record,), evidence_events={source.event_id: source})
    )
    text = joined(result)

    assert "- fact-episode-1" in text, "一行归一事实"
    assert "memory_id=episode-1" not in text, "不带 memory_id（不是台账）"
    assert "证据:" not in text, "不带证据块"
    assert "这些是你本来就知道的事，不必当作清单逐条回应" in text, "边界说明要在"


def test_a_non_episode_keeps_the_labeled_relationship_rendering():
    """不误判：preference/agreement 照旧走带标签的关系状态渲染，不套常驻层的措辞。"""

    source = event(1, "mumo", "我喜欢这样")
    record = memory("pref-1", source, type="preference")

    result = builder().build(
        request(relationship_state=(record,), evidence_events={source.event_id: source})
    )
    text = joined(result)

    assert "memory_id=pref-1" in text, "非 episode 仍带标签"
    assert "这些是你本来就知道的事" not in text, "边界说明只属于常驻层"


def test_future_relationship_memory_is_not_injected_before_it_becomes_valid():
    current = event(2, "mumo", "current")
    source = event(3, "mumo", "以后才生效", at=current.occurred_at_utc + timedelta(minutes=2))
    record = memory("future-memory", source, type="preference")

    result = builder().build(
        request(
            current_event=current,
            relationship_state=(record,),
            evidence_events={source.event_id: source},
        )
    )

    assert "fact-future-memory" not in joined(result)


def test_confirmation_candidates_are_validated_and_labeled_separately():
    source = event(1, "mumo", "confirmation-source")
    candidate = replace(
        memory("confirmation", source, status="candidate"),
        certainty="ambiguous", importance=2, temporal_scope="unclassified",
        assessment_reason_code="ambiguous_scope", assessed_at_utc=source.occurred_at_utc,
    )
    result = builder().build(request(
        confirmation_candidates=(candidate,),
        evidence_events={source.event_id: source},
    ))
    text = joined(result)
    assert "待确认" in text and "不是事实" in text
    with pytest.raises(ValueError):
        request(confirmation_candidates=(candidate, candidate))


def test_long_gap_inserts_time_fact_without_forcing_a_new_opening():
    old = event(1, "mumo", "早上的话", at=BASE)
    current = event(2, "mumo", "傍晚继续", at=BASE + timedelta(hours=4, minutes=18))
    result = builder().build(
        request(current_event=current, recent_events=(old,), runtime_facts=runtime_facts(current))
    )
    text = joined(result)

    assert "[时间经过: 4小时18分; 从 2026-08-27T18:00:00+08:00 到 2026-08-27T22:18:00+08:00 (Asia/Shanghai)]" in text
    assert "重新开场" not in text
    assert result.metrics.category_tokens["time_gaps"] == 1


def test_repeated_recent_emoji_is_exposed_as_a_neutral_fact_without_selecting_a_replacement():
    old_reply = event(1, "qichi", "这句先收下。😌")
    second_reply = event(2, "qichi", "这句也先收下。😌")
    current = event(3, "mumo", "继续", at=BASE + timedelta(minutes=3))

    result = builder().build(request(current_event=current, recent_events=(old_reply, second_reply)))

    text = joined(result)
    assert "近期表情使用统计" in text
    assert "😌=2" in text
    assert "不代表当前表达选择" in text
    # 历史架构设计 18.3：这条统计要提醒「可以省略、不必机械复用」；但仍然不指定替代。
    assert "表情可以省略" in text
    assert "同一个也不必机械复用" in text
    assert "😊" not in text and "换一个" not in text, "统计不指定替代表情"


def test_a_repeated_kaomoji_is_counted_as_one_expression_token():
    """2026-09-14 真机：她整晚只用 (￣▽￣) 41 次，而统计块看不见它（￣ 和 ▽ 是普通字符）。"""

    first = event(1, "qichi", "行啊，就按你说的(￣▽￣)")
    second = event(2, "qichi", "这点我不认(￣▽￣)")
    third = event(3, "qichi", "那就这样(￣▽￣)")
    current = event(4, "mumo", "继续", at=BASE + timedelta(minutes=4))

    result = builder().build(
        request(current_event=current, recent_events=(first, second, third))
    )

    text = joined(result)
    assert "近期表情使用统计" in text
    assert "(￣▽￣)=3" in text
    assert "不代表当前表达选择" in text
    assert "表情可以省略" in text
    assert "(｡･ω･｡)" not in text, "统计不给出替代表情"


def test_one_kaomoji_alone_is_ordinary_wording():
    current = event(2, "mumo", "继续", at=BASE + timedelta(minutes=2))

    result = builder().build(
        request(current_event=current, recent_events=(event(1, "qichi", "好呀(￣▽￣)"),))
    )

    assert "近期表情使用统计" not in joined(result)


def test_parentheses_with_words_are_not_expression_tokens():
    """（纸鸢）这种普通括号不是表情；混了汉字的括号也不算。"""

    events = (
        event(1, "qichi", "我形象是一只纸鸢（纸鸢）"),
        event(2, "qichi", "（这就是那只 ￣▽￣ 纸鸢）"),
    )
    current = event(3, "mumo", "继续", at=BASE + timedelta(minutes=3))

    result = builder().build(request(current_event=current, recent_events=events))

    assert "近期表情使用统计" not in joined(result)


def test_emoji_and_kaomoji_share_one_statistic_block():
    events = (
        event(1, "qichi", "好呀(￣▽￣)😌"),
        event(2, "qichi", "那就这样(￣▽￣)😌"),
    )
    current = event(3, "mumo", "继续", at=BASE + timedelta(minutes=3))

    result = builder().build(request(current_event=current, recent_events=events))

    text = joined(result)
    assert "(￣▽￣)=2" in text and "😌=2" in text


def test_single_recent_emoji_is_not_promoted_to_an_expression_instruction():
    old_reply = event(1, "qichi", "这句先收下。🙂")
    current = event(2, "mumo", "继续", at=BASE + timedelta(minutes=2))

    result = builder().build(request(current_event=current, recent_events=(old_reply,)))

    text = joined(result)
    assert "近期表情使用统计" not in text
    assert "🙂=1" not in text


def test_history_evidence_does_not_present_old_qichi_style_as_a_template():
    old_reply = event(1, "qichi", "哎，在呢。😌")
    current = event(2, "mumo", "继续", at=BASE + timedelta(minutes=2))

    result = builder().build(request(current_event=current, recent_events=(old_reply,)))

    text = joined(result)
    assert "不是当前回复的固定写作模板" in text
    assert "不是当前回复的固定写作模板" in text
    assert "默认不要续用其中的口头禅" not in text


def test_duplicate_current_and_quote_events_are_not_repeated_as_history():
    quote = event(1, "qichi", "quoted")
    current = event(2, "mumo", "current")
    result = builder().build(
        request(current_event=current, quoted_chain=(quote,), recent_events=(quote, current))
    )
    text = joined(result)

    assert text.count("quoted") == 1
    assert text.count("current") == 1


def test_platform_reply_and_media_identifiers_do_not_leak_into_model_context():
    quote = event(1, "qichi", "quoted")
    current = replace(
        event(2, "mumo", "current"),
        reply_to_platform_message_id="RAW-PLATFORM-MESSAGE-ID",
        message_segments=(
            MessageSegment("reply", {"id": "RAW-PLATFORM-MESSAGE-ID"}),
            MessageSegment("image", {"file": "PRIVATE-IMAGE-LOCATOR"}),
            MessageSegment("face", {"id": "14"}),
            MessageSegment("text", {"text": "current"}),
        ),
    )
    result = builder().build(
        request(current_event=current, quoted_chain=(quote,))
    )
    text = joined(result)

    assert "RAW-PLATFORM-MESSAGE-ID" not in text
    assert "PRIVATE-IMAGE-LOCATOR" not in text
    # 2026-09-16：image 段的类型行按设计不再出现（图片有自己的标记，见上），
    # 但其它非文本段的类型必须照实保留——这里同时守住「不泄露定位符」与「不吞掉事实」。
    assert '"type":"image"' not in text
    assert '"type":"face","id":"14"' in text
    assert "[直接引用 | actor=qichi; handle=Q1;" in text


def test_cross_conversation_and_runtime_source_mismatches_fail_closed():
    current = event(2, "mumo", "current")
    foreign = event(1, "qichi", "foreign", conversation="other")
    with pytest.raises(ContextValidationError, match="conversation"):
        builder().build(request(current_event=current, quoted_chain=(foreign,)))

    mismatched_facts = runtime_facts(current)
    object.__setattr__(mismatched_facts, "current_source", MessageSourceFact("mumo", "M99", "text", current.occurred_at_utc))
    with pytest.raises(ContextValidationError, match="current source"):
        builder().build(request(current_event=current, runtime_facts=mismatched_facts))


def test_metrics_contain_counts_and_ids_but_not_prompt_text():
    history = event(1, "mumo", "PRIVATE_HISTORY_SENTINEL")
    result = builder().build(request(recent_events=(history,)))
    rendered_metrics = repr(result.metrics)

    assert "PRIVATE_HISTORY_SENTINEL" not in rendered_metrics
    assert result.metrics.selected_history_event_ids == (history.event_id,)
    assert all(type(value) is int for value in result.metrics.category_tokens.values())


def test_real_character_counter_never_exceeds_input_budget():
    current = event(2, "mumo", "当前消息")
    context_builder = builder(
        preferred=1000,
        maximum=1000,
        reserve=50,
        provider_tokens=1000,
        counter=CharacterTokenCounter(),
    )
    result = context_builder.build(request(current_event=current))

    assert result.metrics.input_tokens == sum(len(message.content) for message in result.messages)
    assert result.metrics.input_tokens <= 950


def vision_builder(**overrides):
    return builder(preferred=5000, maximum=6000, reserve=50, provider_tokens=6000, **overrides)


def test_a_replayed_image_message_keeps_the_fact_but_not_the_pixels():
    """2026-09-11：历史里那条图片消息渲染成空，她便否认了自己正确的观察。"""

    picture = replace(
        event(1, "mumo", None),
        message_segments=(MessageSegment("image", {"file": "x.jpg"}),),
    )
    current = event(2, "mumo", "你看到了啥兔子")

    result = builder(preferred=2000, maximum=3000, reserve=50, provider_tokens=3000).build(
        request(current_event=current, recent_events=(picture,))
    )
    rendered = joined(result)

    assert IMAGE_HISTORY_MARKER in rendered
    assert "不是新收到的图" in rendered
    assert "[无文本内容]" not in rendered
    # 2026-09-16：图片段不再额外列一行「非文本消息段类型」，那行会被读成「这里又有一张图」。
    assert "非文本消息段类型" not in rendered


def test_only_image_segments_lose_the_generic_segment_line():
    """不误判：face 段与未知段（如 marketface）仍然要照实列出来。"""

    face = replace(
        event(1, "mumo", None),
        message_segments=(MessageSegment("face", {"id": "289"}),),
    )
    market = replace(
        event(3, "mumo", None, at=BASE + timedelta(minutes=6)),
        message_segments=(MessageSegment("marketface", {"id": "5"}),),
    )
    current = event(4, "mumo", "你看到了啥兔子")

    result = builder(preferred=2000, maximum=3000, reserve=50, provider_tokens=3000).build(
        request(current_event=current, recent_events=(face, market))
    )
    rendered = joined(result)

    assert '"type":"face","id":"289"' in rendered
    assert '"type":"marketface"' in rendered
    assert IMAGE_HISTORY_MARKER not in rendered


def test_the_current_turn_never_gets_the_history_marker():
    image = ModelImage(path="C:/media/a.png", content_type="image/png")
    current = event(2, "mumo", "看看这张")

    result = builder(preferred=5000, maximum=6000, reserve=50, provider_tokens=6000).build(
        request(current_event=current, current_images=(image,))
    )

    assert IMAGE_HISTORY_MARKER not in joined(result), "当轮带的是真图，不再需要标记"
    assert any(message.images for message in result.messages)


def test_only_the_current_user_turn_carries_pictures():
    old = event(1, "mumo", "老消息")
    current = event(2, "mumo", "看这张")
    image = ModelImage(path="C:/media/2026-09/event-2-mumo-0.png", content_type="image/png")

    result = vision_builder().build(
        request(current_event=current, recent_events=(old,), current_images=(image,))
    )

    carriers = [message for message in result.messages if message.images]
    assert len(carriers) == 1
    assert carriers[0].role == "user"
    assert carriers[0].images == (image,)
    assert "看这张" in carriers[0].content
    assert all(not message.images for message in result.messages if message.role != "user")
    assert "老消息" in joined(result), "历史原文仍然照常注入，只是不带图"


def test_an_image_only_turn_still_reaches_the_model_as_a_user_turn():
    current = event(2, "mumo", None)
    image = ModelImage(path="C:/media/2026-09/event-2-mumo-0.png", content_type="image/png")

    result = vision_builder().build(request(current_event=current, current_images=(image,)))

    carriers = [message for message in result.messages if message.images]
    assert len(carriers) == 1
    assert carriers[0].content == ""
    assert carriers[0].images == (image,)


def test_pictures_are_counted_with_a_reserve_estimate():
    current = event(2, "mumo", "看这张")
    image = ModelImage(path="C:/media/a.png", content_type="image/png")
    plain = builder(preferred=5000, maximum=6000, reserve=50, provider_tokens=6000).build(
        request(current_event=current)
    )
    with_image = vision_builder().build(request(current_event=current, current_images=(image,)))

    assert with_image.metrics.input_tokens - plain.metrics.input_tokens == 700


def test_duplicate_picture_paths_and_platform_carriers_are_refused():
    current = event(2, "mumo", "看这张")
    image = ModelImage(path="C:/media/a.png", content_type="image/png")

    with pytest.raises(ContextValidationError, match="duplicate current image"):
        vision_builder().build(request(current_event=current, current_images=(image, image)))

    platform_event = event(3, "platform", None, kind="initiative")
    with pytest.raises(ContextValidationError, match="platform message cannot carry images"):
        vision_builder().build(
            request(
                current_event=platform_event,
                runtime_facts=runtime_facts(platform_event),
                current_images=(image,),
            )
        )


def test_the_detail_block_names_the_episode_it_belongs_to():
    """明细块必须带上与索引同一套说法的窗口标签（2026-09-11 九号那次）。"""

    source = event(19, "mumo", "等到我要利息的那天")
    detail = MemoryDetailRecord(
        detail_id="detail-9",
        fragment_id="fragment-9",
        ordinal=0,
        detail_kind="message",
        actor="mumo",
        reality_scope="conversation",
        normalized_detail="用户在九号中午说过一句原话",
        exact_quote="等到我要利息的那天",
        source_event_id=source.event_id,
        occurred_at_utc=source.occurred_at_utc,
        certainty="explicit",
        temporal_scope="historical",
        status="candidate",
        privacy_class="adult",
        recall_policy="explicit_request_only",
        evidence=(MemoryDetailEvidence(source.event_id),),
    )

    labelled = ContextBuilder._render_memory_details(
        (detail,), "", {"fragment-9": "09-09 中午 · 含成人内容"}
    )
    unnamed = ContextBuilder._render_memory_details((detail,))

    assert "[片段 09-09 中午 · 含成人内容 · 1 条]" in labelled
    assert "exact_quote=" in labelled
    assert "[片段 fragment-9 · 1 条]" in unnamed, "没有标签时退回片段 id，仍然成组"


def test_a_note_about_the_evidence_follows_it():
    """2026-09-12 T7：B2 的「先确认再讲」整条路已经删掉（aff2478 去掉猜最近一段），
    它的 lead 参数没有调用者了。这里只留下仍然成立的一半：说明跟在原文之后。"""

    source = event(19, "mumo", "这是一句原话")
    detail = MemoryDetailRecord(
        detail_id="detail-note",
        fragment_id="fragment-note",
        ordinal=0,
        detail_kind="message",
        actor="mumo",
        reality_scope="conversation",
        normalized_detail="用户说过一句原话",
        exact_quote="这是一句原话",
        source_event_id=source.event_id,
        occurred_at_utc=source.occurred_at_utc,
        certainty="explicit",
        temporal_scope="historical",
        status="candidate",
        privacy_class="ordinary",
        recall_policy="daily_safe",
        evidence=(MemoryDetailEvidence(source.event_id),),
    )
    note = "目标为推定：这是最近一次留存片段"

    rendered = ContextBuilder._render_memory_details((detail,), note)

    assert rendered.splitlines()[0].startswith("[详细时间线证据"), "说明跟在标题之后"
    assert note in rendered
