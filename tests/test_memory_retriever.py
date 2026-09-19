from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.memory.retriever import MemoryRetrievalResult, MemoryRetriever
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository


NOW = datetime(2026, 8, 28, 14, 0, tzinfo=timezone.utc)


def event(event_id: str, text: str, **overrides: object) -> ConversationEvent:
    values: dict[str, object] = {
        "event_id": event_id,
        "platform_event_id": f"pe-{event_id}",
        "platform_message_id": f"pm-{event_id}",
        "conversation_id": "conversation-a",
        "sequence": 999,
        "direction": "inbound",
        "actor": "mumo",
        "kind": "text",
        "text": text,
        "message_segments": (MessageSegment("text", {"text": text}),),
        "reply_to_event_id": None,
        "reply_to_platform_message_id": None,
        "occurred_at_utc": NOW,
        "received_at_utc": NOW,
        "status": "received",
        "metadata": {},
    }
    values.update(overrides)
    return ConversationEvent(**values)  # type: ignore[arg-type]


def memory(
    memory_id: str,
    source: ConversationEvent,
    *,
    fact: str,
    status: str = "active",
    type: str = "preference",
    valid_from: datetime = NOW,
    valid_until: datetime | None = None,
    evidence: tuple[MemoryEvidence, ...] | None = None,
    certainty: str = "explicit",
    importance: int = 2,
    temporal_scope: str = "ongoing",
    assessment_reason_code: str = "explicit_user_statement",
    privacy_class: str = "ordinary",
    recall_policy: str = "daily_safe",
) -> MemoryRecord:
    items = evidence or (
        MemoryEvidence(
            memory_id, source.event_id, source.actor, source.text or "", source.occurred_at_utc,
            "source",
        ),
    )
    return MemoryRecord(
        memory_id=memory_id,
        type=type,
        normalized_fact=fact,
        modality="explicit_statement",
        status=status,  # type: ignore[arg-type]
        valid_from_utc=valid_from,
        valid_until_utc=valid_until,
        supersedes_id=None,
        created_at_utc=NOW,
        memory_evidence=items,
        certainty=certainty,
        importance=importance,
        temporal_scope=temporal_scope,
        assessment_reason_code=assessment_reason_code,
        assessed_at_utc=NOW,
        privacy_class=privacy_class,
        recall_policy=recall_policy,
    )


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        yield database, EventRepository(database), MemoryRepository(database)
    finally:
        database.close()


def ids(records: tuple[MemoryRecord, ...]) -> tuple[str, ...]:
    return tuple(record.memory_id for record in records)


def temp_fts_tables(database: Database) -> tuple[str, ...]:
    rows = database.connection.execute(
        "SELECT name FROM sqlite_temp_master WHERE name LIKE 'qichi_memory_fts_%' ORDER BY name"
    ).fetchall()
    return tuple(row[0] for row in rows)


def test_result_and_constructor_contracts(store):
    database, _, _ = store
    retriever = MemoryRetriever(database)
    result = retriever.retrieve("conversation-a", "   ", NOW)
    assert isinstance(result, MemoryRetrievalResult)
    assert result.candidates == result.context_candidates == ()
    assert dict(result.evidence_events) == {}
    assert result.search_mode == "like_short_query"
    assert result.degraded_reason is None
    with pytest.raises(TypeError):
        result.evidence_events["x"] = object()  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        result.search_mode = "changed"  # type: ignore[misc]
    for candidate_limit, context_limit in ((0, 1), (25, 12), (24, 0), (24, 13), (5, 6)):
        with pytest.raises(ValueError):
            MemoryRetriever(database, candidate_limit=candidate_limit, context_limit=context_limit)


def test_trigram_uses_or_and_searches_fact_and_exact_quote(store):
    database, events, memories = store
    fact_source = events.insert(event("fact-source", "这是明确偏好"))
    quote_source = events.insert(event("quote-source", "我很喜欢窗外细碎雨声"))
    decoy_source = events.insert(event("decoy-source", "我喜欢热牛奶"))
    memories.create(memory("fact-hit", fact_source, fact="用户偏爱下雨天气"))
    memories.create(memory("quote-hit", quote_source, fact="用户的天气偏好"))
    memories.create(memory("decoy", decoy_source, fact="用户喜欢热牛奶"))

    fact_result = MemoryRetriever(database).retrieve(
        "conversation-a", "完全无关的开头偏爱下雨但后面也不匹配", NOW
    )
    assert fact_result.search_mode == "fts5_trigram"
    assert ids(fact_result.candidates) == ("fact-hit",)

    quote_result = MemoryRetriever(database).retrieve(
        "conversation-a", "另一个开头窗外细碎雨声以及无关结尾", NOW
    )
    assert ids(quote_result.candidates) == ("quote-hit",)
    assert ids(quote_result.context_candidates) == ("quote-hit",)
    assert set(quote_result.evidence_events) == {quote_source.event_id}


def test_short_query_and_no_overlap_do_not_backfill(store):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢雨声"))
    memories.create(memory("rain", source, fact="用户喜欢雨声"))
    short = MemoryRetriever(database).retrieve("conversation-a", "雨声", NOW)
    assert short.search_mode == "like_short_query"
    assert ids(short.candidates) == ("rain",)
    assert MemoryRetriever(database).retrieve("conversation-a", "海边散步", NOW).candidates == ()


def test_candidates_are_ranked_by_explainable_lexical_hits(store):
    database, events, memories = store
    exact_source = events.insert(event("exact", "我喜欢窗外细碎雨声"))
    weak_source = events.insert(event("weak", "我喜欢雨声"))
    memories.create(memory("weak", weak_source, fact="用户有天气偏好"))
    memories.create(memory("exact", exact_source, fact="用户喜欢窗外细碎雨声"))

    result = MemoryRetriever(database).retrieve(
        "conversation-a", "词面查询", NOW, query_terms=("窗外", "雨声")
    )

    assert ids(result.candidates) == ("exact", "weak")
    assert result.scores["exact"] > result.scores["weak"]
    assert "normalized_fact" in result.reasons["exact"]
    assert "exact_quote" in result.reasons["exact"]


def test_trigram_overlap_scores_a_quote_with_a_missing_sentence_prefix(store):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢窗外细碎雨声"))
    evidence = MemoryEvidence(
        "rain", source.event_id, source.actor, "喜欢窗外细碎雨声", source.occurred_at_utc
    )
    memories.create(
        memory(
            "rain",
            source,
            fact="用户的天气偏好",
            evidence=(evidence,),
        )
    )

    result = MemoryRetriever(database).retrieve(
        "conversation-a",
        "我喜欢窗外细碎雨声",
        NOW,
        query_terms=("我喜欢窗外细碎雨声",),
    )

    assert ids(result.candidates) == ("rain",)
    assert result.scores["rain"] > 0
    assert "trigram_overlap" in result.reasons["rain"]


def test_trigram_budget_is_shared_across_current_and_quoted_sources(store):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢窗外细碎雨声"))
    memories.create(memory("rain", source, fact="用户喜欢窗外细碎雨声"))
    long_current = "".join(chr(0x4E00 + index) for index in range(100))

    result = MemoryRetriever(database).retrieve(
        "conversation-a",
        long_current,
        NOW,
        query_terms=(long_current, "我喜欢窗外细碎雨声"),
    )

    assert ids(result.candidates) == ("rain",)
    assert result.scores["rain"] > 0


def test_short_current_source_does_not_disable_quoted_trigram_retrieval(store):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢窗外细碎雨声"))
    evidence = MemoryEvidence(
        "rain", source.event_id, source.actor, "喜欢窗外细碎雨声", source.occurred_at_utc
    )
    memories.create(
        memory("rain", source, fact="用户的天气偏好", evidence=(evidence,))
    )

    result = MemoryRetriever(database).retrieve(
        "conversation-a",
        "呢\n我喜欢窗外细碎雨声",
        NOW,
        query_terms=("呢", "我喜欢窗外细碎雨声"),
    )

    assert result.search_mode == "fts5_trigram"
    assert ids(result.candidates) == ("rain",)
    assert result.scores["rain"] > 0


def test_candidate_without_a_lexical_hit_is_not_returned(store, monkeypatch):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢热牛奶"))
    memories.create(memory("milk", source, fact="用户喜欢热牛奶"))
    retriever = MemoryRetriever(database)
    monkeypatch.setattr(retriever, "_supports_fts5_trigram", lambda: True)
    monkeypatch.setattr(retriever, "_fts_candidate_ids", lambda *_args: ("milk",))

    result = retriever.retrieve("conversation-a", "窗外细碎雨声", NOW)

    assert result.candidates == ()
    assert dict(result.scores) == {}
    assert dict(result.reasons) == {}


def test_one_incidental_trigram_does_not_expand_unrelated_evidence(store):
    database, events, memories = store
    source = events.insert(event("source", "我在考虑换一种说法，偶尔想听你柔软一点"))
    memories.create(memory("soft-tone", source, fact="用户偶尔想听角色柔软一点"))

    result = MemoryRetriever(database).retrieve(
        "conversation-a", "键盘空格键有点涩，我考虑换个键帽", NOW
    )

    assert result.candidates == ()
    assert dict(result.scores) == {}
    assert dict(result.reasons) == {}


def test_multiple_trigram_overlap_still_expands_indirect_evidence(store):
    database, events, memories = store
    source = events.insert(event("source", "你还想聊一会儿时，我不会赶你睡觉"))
    memories.create(memory("keep-talking", source, fact="用户还想聊天时不要赶他睡觉"))

    result = MemoryRetriever(database).retrieve(
        "conversation-a", "今天还想聊一会儿", NOW
    )

    assert ids(result.candidates) == ("keep-talking",)
    assert "trigram_overlap" in result.reasons["keep-talking"]


def test_supplied_query_terms_are_bounded_and_do_not_backfill(store):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢窗外细碎雨声"))
    memories.create(memory("rain", source, fact="用户喜欢窗外细碎雨声"))

    matched = MemoryRetriever(database).retrieve(
        "conversation-a", "无关长文本", NOW, query_terms=("窗外", "雨声")
    )
    missing = MemoryRetriever(database).retrieve(
        "conversation-a", "窗外", NOW, query_terms=("海边",)
    )

    assert ids(matched.candidates) == ("rain",)
    assert missing.candidates == ()


def test_like_short_query_escapes_wildcards(store):
    database, events, memories = store
    literal = events.insert(event("literal", "进度是100%_完成\\路径"))
    decoy = events.insert(event("decoy", "完全不相关"))
    memories.create(memory("literal", literal, fact="进度100%_完成\\路径"))
    memories.create(memory("decoy", decoy, fact="普通记录"))
    retriever = MemoryRetriever(database)
    assert ids(retriever.retrieve("conversation-a", "%", NOW).candidates) == ("literal",)
    assert ids(retriever.retrieve("conversation-a", "_", NOW).candidates) == ("literal",)
    assert ids(retriever.retrieve("conversation-a", "\\", NOW).candidates) == ("literal",)


def test_conversation_status_and_validity_are_fail_closed(store):
    database, events, memories = store
    sources = {
        name: events.insert(
            event(name, f"共同检索词 {name}", conversation_id="conversation-b" if name == "other" else "conversation-a")
        )
        for name in ("active", "candidate", "rejected", "expired", "future", "until", "other")
    }
    memories.create(memory("active", sources["active"], fact="共同检索词 active"))
    memories.create(memory("candidate", sources["candidate"], fact="共同检索词 candidate", status="candidate"))
    memories.create(memory("rejected", sources["rejected"], fact="共同检索词 rejected", status="candidate"))
    memories.reject("rejected")
    memories.create(memory("expired", sources["expired"], fact="共同检索词 expired"))
    memories.expire("expired")
    memories.create(memory("future", sources["future"], fact="共同检索词 future", valid_from=NOW + timedelta(seconds=1)))
    memories.create(memory("until", sources["until"], fact="共同检索词 until", valid_until=NOW))
    memories.create(memory("other", sources["other"], fact="共同检索词 other"))

    at_boundary = MemoryRetriever(database).retrieve("conversation-a", "共同检索词", NOW)
    assert ids(at_boundary.candidates) == ("active", "until")
    after_boundary = MemoryRetriever(database).retrieve(
        "conversation-a", "共同检索词", NOW + timedelta(microseconds=1)
    )
    assert ids(after_boundary.candidates) == ("active",)


def test_limits_deduplicate_memories_and_evidence_mapping_is_context_only(store):
    database, events, memories = store
    shared = events.insert(event("shared", "共同检索词"))
    extra = events.insert(event("extra", "另有共同检索词证据"))
    for index in range(30):
        memory_id = f"memory-{index:02d}"
        evidence = None
        if index == 0:
            evidence = (
                MemoryEvidence(memory_id, shared.event_id, "mumo", "共同检索词", NOW),
                MemoryEvidence(memory_id, extra.event_id, "mumo", "共同检索词", NOW),
            )
        memories.create(memory(memory_id, shared, fact=f"共同检索词 {index:02d}", evidence=evidence))

    result = MemoryRetriever(database).retrieve("conversation-a", "共同检索词", NOW)
    assert len(result.candidates) == 24
    assert len(set(ids(result.candidates))) == 24
    assert len(result.context_candidates) == 12
    assert ids(result.context_candidates) == ids(result.candidates[:12])
    expected_event_ids = {
        evidence.event_id
        for record in result.context_candidates
        for evidence in record.memory_evidence
    }
    assert set(result.evidence_events) == expected_event_ids


def test_importance_breaks_equal_relevance_ties_without_overriding_relevance(store):
    database, events, memories = store
    shared = events.insert(event("shared", "共同检索词 精确主题"))
    low = memory(
        "low",
        shared,
        fact="共同检索词",
        importance=1,
    )
    high = memory("high", shared, fact="共同检索词", importance=3)
    exact = memory("exact", shared, fact="共同检索词 精确主题", importance=1)
    memories.create(low)
    memories.create(high)
    memories.create(exact)

    result = MemoryRetriever(database).retrieve(
        "conversation-a",
        "共同检索词 精确主题",
        NOW,
        query_terms=("共同检索词", "精确主题"),
    )

    assert ids(result.candidates) == ("exact", "high", "low")
    assert result.scores["exact"] > result.scores["high"] == result.scores["low"]


def test_qichi_self_expression_keeps_qichi_actor(store):
    database, events, memories = store
    source = events.insert(
        event(
            "qichi", "我确实很在意用户", actor="qichi", direction="outbound",
            status="sent", metadata={"generation_metadata": {"source": "dialogue"}},
        )
    )
    memories.create(
        memory("self", source, fact="角色表达过很在意用户", type="self_expression")
    )
    result = MemoryRetriever(database).retrieve("conversation-a", "表达过很在意", NOW)
    assert ids(result.context_candidates) == ("self",)
    assert result.context_candidates[0].memory_evidence[0].actor == "qichi"
    assert result.evidence_events[source.event_id].actor == "qichi"


def test_degraded_mode_is_explicit_and_uses_local_or_fallback(store, monkeypatch):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢窗外细碎雨声"))
    memories.create(memory("rain", source, fact="用户的天气偏好"))
    retriever = MemoryRetriever(database)
    monkeypatch.setattr(retriever, "_supports_fts5_trigram", lambda: False)
    result = retriever.retrieve("conversation-a", "无关开头窗外细碎雨声无关结尾", NOW)
    assert ids(result.candidates) == ("rain",)
    assert result.search_mode == "like_degraded"
    assert result.degraded_reason
    assert temp_fts_tables(database) == ()


def test_non_capability_fts_error_fails_closed(store, monkeypatch):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢雨声"))
    memories.create(memory("rain", source, fact="用户喜欢雨声"))
    retriever = MemoryRetriever(database)
    monkeypatch.setattr(
        retriever,
        "_create_fts_probe",
        lambda _name: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        retriever.retrieve("conversation-a", "喜欢雨声", NOW)


def test_corrupted_evidence_fails_closed_and_temp_table_is_cleaned(store):
    database, events, memories = store
    source = events.insert(event("source", "我喜欢雨声"))
    memories.create(memory("rain", source, fact="用户喜欢雨声"))
    database.connection.execute(
        "UPDATE memory_evidence SET exact_quote = '损坏证据内容' WHERE memory_id = 'rain'"
    )
    with pytest.raises(ValueError, match="exact quote is absent"):
        MemoryRetriever(database).retrieve("conversation-a", "损坏证据内容", NOW)
    assert temp_fts_tables(database) == ()


def test_temp_fts_cleanup_and_restart_results_are_identical(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    source = EventRepository(database).insert(event("source", "我喜欢窗外细碎雨声"))
    MemoryRepository(database).create(memory("rain", source, fact="用户喜欢雨天"))
    before = MemoryRetriever(database).retrieve("conversation-a", "窗外细碎雨声", NOW)
    assert temp_fts_tables(database) == ()
    database.close()
    reopened = Database(path)
    try:
        after = MemoryRetriever(reopened).retrieve("conversation-a", "窗外细碎雨声", NOW)
        assert ids(after.candidates) == ids(before.candidates)
        assert dict(after.evidence_events) == dict(before.evidence_events)
        assert temp_fts_tables(reopened) == ()
        persistent = reopened.connection.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'qichi_memory_fts_%'"
        ).fetchall()
        assert persistent == []
    finally:
        reopened.close()


def test_confirmation_candidates_are_separate_and_limited_to_one(store):
    database, events, memories = store
    source = events.insert(event("confirmation-source", "窗边雨声"))
    memories.create(memory("active", source, fact="用户喜欢窗边雨声"))
    memories.create(memory(
        "candidate", source, fact="可能喜欢窗边雨声", status="candidate",
        certainty="ambiguous", importance=2, temporal_scope="unclassified",
        assessment_reason_code="ambiguous_scope",
    ))
    result = MemoryRetriever(database).retrieve(
        "conversation-a", "窗边雨声", NOW, include_confirmation=True
    )
    assert ids(result.context_candidates) == ("active",)
    assert ids(result.confirmation_candidates) == ("candidate",)


def test_confirmation_retrieval_never_returns_more_than_one_candidate(store):
    database, events, memories = store
    source = events.insert(event("confirmation-many", "窗边雨声"))
    for index in range(3):
        memories.create(memory(
            f"candidate-{index}", source, fact=f"可能偏好窗边雨声 {index}", status="candidate",
            certainty="ambiguous", importance=2, temporal_scope="unclassified",
            assessment_reason_code="ambiguous_scope",
        ))
    result = MemoryRetriever(database).retrieve(
        "conversation-a", "窗边雨声", NOW, include_confirmation=True
    )
    assert len(result.confirmation_candidates) == 1


def test_sensitive_retrieval_is_opt_in_and_lexically_scoped(store):
    database, events, memories = store
    source = events.insert(event("adult-source", "我们过去明确有过强势成人互动"))
    memories.create(memory(
        "adult-history", source, fact="双方过去明确有过强势成人互动",
        type="episode", temporal_scope="historical", importance=2,
        assessment_reason_code="historical_event", privacy_class="adult",
        recall_policy="explicit_request_only",
    ))
    hidden = MemoryRetriever(database).retrieve("conversation-a", "成人互动", NOW)
    assert hidden.candidates == ()
    related = MemoryRetriever(database).retrieve(
        "conversation-a", "成人互动", NOW, include_sensitive=True
    )
    assert ids(related.candidates) == ("adult-history",)
    unrelated = MemoryRetriever(database).retrieve(
        "conversation-a", "今天聊工作", NOW, include_sensitive=True
    )
    assert unrelated.candidates == ()
