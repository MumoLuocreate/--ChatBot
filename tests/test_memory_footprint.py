"""最近原文足迹 + 工作集原话：让她「找得到原文、复述贴原文」，且不越隐私门。

背景（2026-09-12 真机）：她在被转述式追问往事时回答「原话我翻不到，不给你编」——
写入层其实 100% 逐字（199/199 evidence、449/449 明细全过），但她常驻上下文里只有
改写层（工作集只有 normalized_fact、索引只有存在性），逐字原话只在钥匙开门时才注入。
这两块把「最近的逐字原话」变成常驻事实，并给工作集补一条证据原话。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qichi.dialogue.capability_manifest import MessageSourceFact, RuntimeFacts
from qichi.dialogue.context_builder import ContextBuildRequest, ContextBuilder
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.domain.memory_details import MemoryDetailEvidence, MemoryDetailRecord


UTC = timezone.utc
BASE = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)


class UnitTokenCounter:
    def count_text(self, text: str) -> int:
        return 0 if not text else 1


def capability(tokens: int = 200_000) -> ModelCapability:
    return ModelCapability(
        "test/model", tokens,
        provider_evidence=ProviderCapabilityEvidence("SiliconFlow", "test/model", tokens, "ev", BASE),
    )


def event(sequence: int, actor: str, text: str, *, at: datetime | None = None) -> ConversationEvent:
    direction = "inbound" if actor == "mumo" else "outbound"
    occurred = at or BASE + timedelta(minutes=sequence)
    return ConversationEvent(
        event_id=f"event-{sequence}-{actor}",
        platform_event_id=None,
        platform_message_id=f"platform-{sequence}",
        conversation_id="owner-1",
        sequence=sequence,
        direction=direction,
        actor=actor,
        kind="text",
        text=text,
        message_segments=(MessageSegment("text", {"text": text}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=occurred,
        received_at_utc=occurred,
        status="received",
        metadata={},
    )


def detail(
    ordinal: int,
    source: ConversationEvent,
    quote: str,
    *,
    fragment: str = "fragment-a",
    privacy: str = "ordinary",
    recall: str = "daily_safe",
    status: str = "candidate",
) -> MemoryDetailRecord:
    return MemoryDetailRecord(
        detail_id=f"{fragment}-{ordinal}",
        fragment_id=fragment,
        ordinal=ordinal,
        detail_kind="message",
        actor=source.actor,
        reality_scope="conversation",
        normalized_detail=f"{source.actor} 说了 {quote[:12]}",
        exact_quote=quote,
        source_event_id=source.event_id,
        occurred_at_utc=source.occurred_at_utc,
        certainty="explicit",
        temporal_scope="historical",
        status=status,
        privacy_class=privacy,
        recall_policy=recall,
        evidence=(MemoryDetailEvidence(source.event_id),),
    )


def memory_record(memory_id: str, source: ConversationEvent, quote: str, *, privacy="ordinary", recall="daily_safe") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        type="preference",
        normalized_fact=f"用户表示 {quote[:16]}",
        modality="explicit_statement",
        status="active",
        valid_from_utc=source.occurred_at_utc,
        valid_until_utc=None,
        supersedes_id=None,
        created_at_utc=source.occurred_at_utc,
        memory_evidence=(MemoryEvidence(memory_id, source.event_id, source.actor, quote, source.occurred_at_utc),),
        certainty="explicit",
        importance=2,
        temporal_scope="ongoing",
        assessment_reason_code="explicit_user_statement",
        assessed_at_utc=source.occurred_at_utc,
        privacy_class=privacy,
        recall_policy=recall,
    )


def facts(current: ConversationEvent) -> RuntimeFacts:
    return RuntimeFacts(
        current_time=current.occurred_at_utc + timedelta(seconds=5),
        seconds_since_last_message=5,
        current_source=MessageSourceFact(current.actor, current.visible_handle, current.kind, current.occurred_at_utc),
        quoted_source=None,
        received_media=("text",),
        available_actions=("text", "reply"),
        vision_available=False,
        external_tools_available=False,
    )


def request(**overrides) -> ContextBuildRequest:
    current = overrides.pop("current_event", event(900, "mumo", "在吗"))
    values = {
        "role_core": "role-core",
        "runtime_facts": facts(current),
        "current_event": current,
        "quoted_chain": (),
        "recent_events": (),
        "relationship_state": (),
        "memory_candidates": (),
        "memory_working_set": (),
        "earlier_events": (),
        "evidence_events": {},
    }
    values.update(overrides)
    return ContextBuildRequest(**values)


def builder() -> ContextBuilder:
    return ContextBuilder(
        UnitTokenCounter(), capability(),
        preferred_window_tokens=400, max_window_tokens=800, output_reserve_tokens=2,
    )


def joined(result) -> str:
    return "\n".join(message.content for message in result.messages)


# ---------------------------------------------------------------- 甲：最近原文足迹

def test_footprint_renders_recent_quotes_verbatim():
    first = event(1, "mumo", "把接下来的都交给我吧")
    second = event(2, "qichi", "行，交给你了。我不攥着了，你带吧")
    result = builder().build(
        request(
            memory_footprint=(detail(0, first, first.text), detail(1, second, second.text)),
            memory_footprint_labels={"fragment-a": "09-12 上午"},
        )
    )
    text = joined(result)

    assert "[最近原文足迹" in text
    assert "把接下来的都交给我吧" in text          # 逐字，不是改写
    assert "行，交给你了。我不攥着了，你带吧" in text
    assert "09-12 上午" in text                     # 片段标签带上
    assert "用户表示" not in text                   # 改写层不得冒充原话


def test_footprint_never_carries_adult_or_non_daily_quotes():
    adult = event(3, "mumo", "成人内容原话不该进日常")
    ordinary = event(4, "mumo", "普通原话可以进")
    result = builder().build(
        request(
            memory_footprint=(
                detail(0, adult, adult.text, privacy="adult", recall="explicit_request_only"),
                detail(1, ordinary, ordinary.text),
            ),
        )
    )
    text = joined(result)

    assert "普通原话可以进" in text
    assert "成人内容原话不该进日常" not in text


def test_a_quote_longer_than_the_cap_is_dropped_whole_never_cut():
    """超长引文整条不进足迹——切一半的「原话」比没有更糟。"""

    long_quote = "很长的一句原话" * 60
    source = event(5, "mumo", long_quote)
    short = event(6, "mumo", "短句")
    result = builder().build(
        request(
            memory_footprint=(detail(0, source, long_quote), detail(1, short, short.text)),
        )
    )
    text = joined(result)

    assert "短句" in text
    assert "很长的一句原话" not in text


def test_no_fragments_means_no_footprint_block():
    result = builder().build(request())
    assert "[最近原文足迹" not in joined(result)


# ---------------------------------------------------------------- 乙：工作集补原话

def test_working_set_carries_one_verbatim_evidence_quote():
    source = event(7, "mumo", "只要兔子你不把所有都忘了，我不会骂你的")
    record = memory_record("pref-1", source, source.text)
    result = builder().build(
        request(memory_working_set=(record,), evidence_events={source.event_id: source})
    )
    text = joined(result)

    assert "[关系记忆工作集" in text
    assert "只要兔子你不把所有都忘了，我不会骂你的" in text


def test_working_set_keeps_the_placeholder_for_adult_records_without_quoting_them():
    source = event(8, "mumo", "成人相关的原话")
    record = memory_record("pref-2", source, source.text, privacy="adult", recall="explicit_request_only")
    result = builder().build(
        request(
            memory_working_set=(record,),
            allow_sensitive_memory=True,
            evidence_events={source.event_id: source},
        )
    )
    text = joined(result)

    assert "[关系记忆工作集" in text
    assert "成人相关的原话" not in text
    # 卡②不误判：成人 explicit_request_only 仍必须被替换成占位句，不得照抄原话。
    assert "已有成人相关历史证据" in text


def test_daily_chat_context_is_unchanged_without_memories():
    result = builder().build(request())
    text = joined(result)

    assert "[关系记忆工作集" not in text
    assert "[最近原文足迹" not in text
