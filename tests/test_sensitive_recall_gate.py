from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from qichi.app import sensitive_detail_allowed, transitive_quote_chain
from qichi.domain.events import ConversationEvent

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def event(sequence, *, actor="mumo", reply_to=None, conversation="10001", text="x"):
    return ConversationEvent(
        event_id=f"e{sequence}",
        platform_event_id=None,
        platform_message_id=None,
        conversation_id=conversation,
        sequence=sequence,
        direction="inbound" if actor == "mumo" else "outbound",
        actor=actor,
        kind="text",
        text=text,
        message_segments=(),
        reply_to_event_id=reply_to,
        reply_to_platform_message_id=None,
        occurred_at_utc=NOW,
        received_at_utc=NOW,
        status="received",
        metadata={},
    )


# 2026-09-11 用户裁定删掉固定词表：是否展开由「这条消息有没有指向某一段」决定，
# 词面判据已不存在，因此原先按词表断言的两个用例一并移除（新契约在
# tests/test_detail_targeting.py 里覆盖）。

@pytest.mark.parametrize(("privacy", "policy", "explicit", "expected"), [
    ("ordinary", "daily_safe", False, True),
    ("intimate", "topic_only", False, True),
    # Adult preferences and agreements default to topic_only, so a real topic
    # hit keeps them usable; only explicit_request_only waits for a request.
    ("adult", "topic_only", False, True),
    ("adult", "topic_only", True, True),
    ("adult", "explicit_request_only", False, False),
    ("adult", "explicit_request_only", True, True),
    ("intimate", "explicit_request_only", False, False),
    ("intimate", "explicit_request_only", True, True),
    ("ordinary", "explicit_request_only", True, True),
    ("adult", "daily_safe", False, False),
    ("ordinary", "unknown_policy", False, False),
    ("alien", "daily_safe", False, False),
])
def test_recall_matrix_is_executed_per_record(privacy, policy, explicit, expected):
    assert sensitive_detail_allowed(
        privacy_class=privacy, recall_policy=policy, explicit_recall=explicit
    ) is expected


def test_quote_chain_follows_up_to_three_hops():
    e4, e3, e2, e1 = event(4), event(3, reply_to="e4"), event(2, reply_to="e3"), event(1, reply_to="e2")
    table = {e.event_id: e for e in (e1, e2, e3, e4)}

    chain = transitive_quote_chain(e1, table.get)

    assert [item.event_id for item in chain] == ["e1", "e2", "e3"]


def test_quote_chain_stops_on_a_cycle():
    a, b = event(1, reply_to="e2"), event(2, reply_to="e1")
    table = {"e1": a, "e2": b}

    assert [item.event_id for item in transitive_quote_chain(a, table.get)] == ["e1", "e2"]


def test_quote_chain_rejects_a_cross_conversation_hop():
    same = event(1)
    foreign = event(2, reply_to="e1", conversation="99999")

    assert [item.event_id for item in transitive_quote_chain(foreign, {same.event_id: same}.get)] == ["e2"]


def test_quote_chain_handles_missing_targets_and_empty_input():
    only = event(1, reply_to="gone")

    assert [item.event_id for item in transitive_quote_chain(only, {}.get)] == ["e1"]
    assert transitive_quote_chain(None, {}.get) == ()

# ---------------------------------------------------------------------------
# Pipeline checks.  Every non-misjudgment case first proves that the adult
# record really was retrieved for that phrasing -- otherwise the assertion
# would pass for the wrong reason and prove nothing about the gate.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import NOW, OWNER, FakeLLM, FakeNapCat, app, seed_event  # noqa: E402

from qichi.domain.events import ConversationEvent, MessageSegment  # noqa: E402
from qichi.domain.memory_details import MemoryDetailDraft  # noqa: E402
from qichi.storage.memory_detail_repository import MemoryDetailRepository  # noqa: E402
from qichi.domain.memory import MemoryEvidence, MemoryRecord  # noqa: E402
from qichi.storage.database import Database  # noqa: E402
from qichi.storage.event_repository import EventRepository  # noqa: E402
from qichi.storage.memory_repository import MemoryRepository  # noqa: E402

# 10 个字：刚好越过「逐字原话」那道 8 字门槛，复述它是**有钥匙**的——所以下面那
# 条负向用例证明的是「复述只开普通明细」，而不是「什么都没发生」。
ADULT_QUOTE = "成人原话在此不要外传"
ADULT_FACT = "便利店那晚的成人经历"
ORDINARY_QUOTE = "普通原话在此"
EVERYDAY_THAT_RETRIEVES_ADULT = (
    "便利店那晚到底怎么了",
    "便利店那晚我们说的那个方案",
    "便利店那晚的事你还提吗",
)
# 2026-09-12（T7）：这一组原来断言「引它自己的原话 → 成人证据被注入」。那条断言是
# 假绿——原话就写在用户消息里，随便注入点什么它都成立（审计 P5）。现在拆成两侧：
# 逐字复述只开普通明细，点名那一天才把成人内容交出来。
REPEAT_THAT_RETRIEVES_ADULT = (
    f"你那天说“{ADULT_QUOTE}”的时候是什么意思",
    f"关于“{ADULT_QUOTE}”那句，你还记得吗",
)


def _seed(database, *, memory_id, privacy, policy, fact, quote, source_text):
    events, memories = EventRepository(database), MemoryRepository(database)
    source = seed_event(events, memory_id + "-source", source_text, at=NOW - timedelta(days=3))
    evidence = MemoryEvidence(memory_id, source.event_id, source.actor, quote, source.occurred_at_utc, "source")
    memories.create(MemoryRecord(
        memory_id, "episode", fact, "explicit_statement", "active", source.occurred_at_utc,
        None, None, source.received_at_utc, (evidence,), "explicit", 3, "historical",
        "explicit_user_statement", source.received_at_utc, privacy_class=privacy, recall_policy=policy,
    ))
    return source


def _prepare(tmp_path, name="gate.sqlite3"):
    database = Database(tmp_path / name)
    _seed(database, memory_id="ordinary-memory", privacy="ordinary", policy="daily_safe",
          fact="喜欢便利店的关东煮", quote=ORDINARY_QUOTE, source_text="便利店 " + ORDINARY_QUOTE)
    _seed(database, memory_id="adult-memory", privacy="adult", policy="explicit_request_only",
          fact="便利店那晚的成人经历", quote=ADULT_QUOTE, source_text="便利店那晚 " + ADULT_QUOTE)
    return database, EventRepository(database)


def _retrieved(application, text):
    terms = application._memory_query_terms(text, ())
    result = application.memory_retriever.retrieve(
        OWNER, "\n".join(terms), NOW, query_terms=terms or None,
        include_confirmation=True, include_sensitive=True,
    )
    return {record.memory_id for record in result.candidates}


def _turn(application, events, text, event_id):
    event = events.insert(ConversationEvent(
        event_id, "pe-" + event_id, "pm-" + event_id, OWNER, 999, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    ))
    built = asyncio.run(application._build_input(event, 1))
    return " ".join(str(message) for message in built.role_messages)


def test_pipeline_still_injects_an_ordinary_memory_on_a_topic_hit(tmp_path):
    database, events = _prepare(tmp_path, "control.sqlite3")
    try:
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        assert "ordinary-memory" in _retrieved(application, "便利店的关东煮好吃吗")
        assert ORDINARY_QUOTE in _turn(application, events, "便利店的关东煮好吃吗", "c1")
    finally:
        database.close()


@pytest.mark.parametrize("text", EVERYDAY_THAT_RETRIEVES_ADULT)
def test_pipeline_never_injects_adult_evidence_without_a_real_request(tmp_path, text):
    database, events = _prepare(tmp_path, "everyday.sqlite3")
    try:
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        assert "adult-memory" in _retrieved(application, text)  # the gate is what blocks it
        rendered = _turn(application, events, text, "c2")
        assert ADULT_QUOTE not in rendered
        assert "便利店那晚的成人经历" not in rendered
    finally:
        database.close()


@pytest.mark.parametrize("text", REPEAT_THAT_RETRIEVES_ADULT)
def test_a_verbatim_repeat_opens_ordinary_details_only(tmp_path, text):
    database, events = _prepare(tmp_path, "repeat.sqlite3")
    try:
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        assert "adult-memory" in _retrieved(application, text), "检索照样能命中成人记录"
        rendered = _turn(application, events, text, "c3")
        assert ADULT_FACT not in rendered, "复述一句话不是打开成人内容的钥匙"
    finally:
        database.close()


def test_adult_content_opens_when_the_user_names_the_day(tmp_path):
    """2026-09-12 T7：这是原来那条假绿缺的真覆盖——明确回顾真的把成人内容交出来。"""

    database = Database(tmp_path / "adult-day.sqlite3")
    try:
        events = EventRepository(database)
        _seed(database, memory_id="ordinary-memory", privacy="ordinary", policy="daily_safe",
              fact="喜欢便利店的关东煮", quote=ORDINARY_QUOTE, source_text="便利店 " + ORDINARY_QUOTE)
        source = _seed(database, memory_id="adult-memory", privacy="adult", policy="explicit_request_only",
                       fact=ADULT_FACT, quote=ADULT_QUOTE, source_text="便利店那晚 " + ADULT_QUOTE)
        repository = MemoryDetailRepository(database)
        fragment = repository.build_fragment((source,), "adult-episode", None, created_at_utc=NOW)
        drafts = (MemoryDetailDraft(
            ordinal=0, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="记录一条原文。", exact_quote=ADULT_QUOTE, source_event_id=source.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class="adult", recall_policy="explicit_request_only",
            evidence=((source.event_id, "source"),),
        ),)
        details = repository.build_details(fragment, drafts, (source,))
        with database.transaction() as connection:
            repository.store_in_transaction(connection, fragment=fragment, events=(source,), details=details)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        day = source.occurred_at_utc.astimezone(application.local_zone).day
        text = f"便利店那晚到底怎么了，{day}号那天"

        assert "adult-memory" in _retrieved(application, text)
        rendered = _turn(application, events, text, "c5")

        assert ADULT_FACT in rendered, "点名那天，成人证据要被准入"
        assert ADULT_QUOTE in rendered, "成人明细本身也要交出来"
    finally:
        database.close()


def test_a_quote_admits_only_the_record_it_points_at(tmp_path):
    database, _ = _prepare(tmp_path, "quote.sqlite3")
    try:
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        memories = MemoryRepository(database)
        adult, control = memories.get("adult-memory"), memories.get("ordinary-memory")
        adult_event = memories.get("adult-memory").memory_evidence[0].event_id
        control_event = control.memory_evidence[0].event_id

        assert application._sensitive_admitted(adult, {control_event}, False) is False
        assert application._sensitive_admitted(adult, {adult_event}, False) is True
        assert application._sensitive_admitted(control, {control_event}, False) is True
        assert application._sensitive_admitted(adult, set(), True) is True
        assert application._sensitive_admitted(adult, set(), False) is False
    finally:
        database.close()

ADULT_TOPIC_ONLY_QUOTE = "成人的偏好原话"


def test_adult_topic_only_is_usable_on_a_topic_hit_but_not_otherwise(tmp_path):
    database = Database(tmp_path / "adult-topic.sqlite3")
    try:
        _seed(database, memory_id="adult-preference", privacy="adult", policy="topic_only",
              fact="亲密时刻喜欢强势与掌控感", quote=ADULT_TOPIC_ONLY_QUOTE,
              source_text="便利店那晚 " + ADULT_TOPIC_ONLY_QUOTE)
        events = EventRepository(database)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        assert "adult-preference" in _retrieved(application, "亲密时刻你喜欢怎样")
        assert ADULT_TOPIC_ONLY_QUOTE in _turn(application, events, "亲密时刻你喜欢怎样", "t1")

        assert "adult-preference" not in _retrieved(application, "今晚吃什么")
        assert ADULT_TOPIC_ONLY_QUOTE not in _turn(application, events, "今晚吃什么", "t2")
    finally:
        database.close()

INTIMATE_QUOTE = "亲密的偏好原话"
CANDIDATE_FACT = "便利店那晚的成人候选事实"
CANDIDATE_QUOTE = "成人候选原话"


def test_an_admitted_record_does_not_smuggle_an_unadmitted_candidate(tmp_path):
    """Admitting one sensitive record must not open the door for another.

    The everyday phrasing below really does retrieve both records, and the
    intimate one is admitted because its topic matched, so allow_sensitive_memory
    is true for the turn.  The adult candidate is not admitted and must stay out
    of the assembled context anyway -- neither as a confirmation candidate nor
    through the rebuilt evidence pool.
    """
    database = Database(tmp_path / "smuggle.sqlite3")
    try:
        _seed(database, memory_id="intimate-topic", privacy="intimate", policy="topic_only",
              fact="便利店那晚的亲密偏好", quote=INTIMATE_QUOTE,
              source_text="便利店那晚 " + INTIMATE_QUOTE)
        events, memories = EventRepository(database), MemoryRepository(database)
        source = seed_event(events, "adult-candidate-source", "便利店那晚 " + CANDIDATE_QUOTE,
                            at=NOW - timedelta(days=3))
        evidence = MemoryEvidence("adult-candidate", source.event_id, source.actor, CANDIDATE_QUOTE,
                                  source.occurred_at_utc, "source")
        memories.create(MemoryRecord(
            "adult-candidate", "episode", CANDIDATE_FACT, "explicit_statement", "candidate",
            source.occurred_at_utc, None, None, source.received_at_utc, (evidence,), "ambiguous", 2,
            "historical", "explicit_user_statement", source.received_at_utc,
            privacy_class="adult", recall_policy="explicit_request_only",
        ))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        # Non-vacuity: both records really are retrieved for this phrasing --
        # the ambiguous adult one arrives as a confirmation candidate.
        terms = application._memory_query_terms("便利店那晚到底怎么了", ())
        result = application.memory_retriever.retrieve(
            OWNER, "\n".join(terms), NOW, query_terms=terms or None,
            include_confirmation=True, include_sensitive=True,
        )
        retrieved = {r.memory_id for r in (*result.candidates, *result.confirmation_candidates)}
        assert {"intimate-topic", "adult-candidate"} <= retrieved
        rendered = _turn(application, EventRepository(database), "便利店那晚到底怎么了", "smuggle")
        assert INTIMATE_QUOTE in rendered
        assert CANDIDATE_FACT not in rendered
        assert CANDIDATE_QUOTE not in rendered
    finally:
        database.close()
