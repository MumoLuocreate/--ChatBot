from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord, MemoryReview
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


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
    quote: str | None = None,
    status: str = "candidate",
    type: str = "preference",
    fact: str = "用户喜欢雨声",
    valid_until: datetime | None = None,
    supersedes_id: str | None = None,
    evidence: tuple[MemoryEvidence, ...] | None = None,
    modality: str = "explicit_statement",
    certainty: str = "explicit",
    importance: int = 2,
    temporal_scope: str = "ongoing",
    assessment_reason_code: str | None = None,
) -> MemoryRecord:
    if assessment_reason_code is None:
        assessment_reason_code = {
            "correction": "user_correction",
            "episode": "historical_event",
        }.get(type, "explicit_user_statement")
    evidence_role = "correction" if type == "correction" else "source"
    items = evidence or (
        MemoryEvidence(
            memory_id=memory_id,
            event_id=source.event_id,
            actor=source.actor,
            exact_quote=quote or source.text or "missing",
            occurred_at_utc=source.occurred_at_utc,
            evidence_role=evidence_role,
        ),
    )
    return MemoryRecord(
        memory_id=memory_id,
        type=type,
        normalized_fact=fact,
        modality=modality,
        status=status,  # type: ignore[arg-type]
        valid_from_utc=NOW,
        valid_until_utc=valid_until,
        supersedes_id=supersedes_id,
        created_at_utc=NOW,
        memory_evidence=items,
        certainty=certainty,
        importance=importance,
        temporal_scope=temporal_scope,
        assessment_reason_code=assessment_reason_code,
        assessed_at_utc=NOW,
    )


@pytest.fixture
def repositories(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        yield database, EventRepository(database), MemoryRepository(database)
    finally:
        database.close()


def test_candidate_creation_is_idempotent_and_survives_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    source = EventRepository(database).insert(event("source", "我喜欢雨声"))
    record = memory("memory-1", source, quote="喜欢雨声")
    repository = MemoryRepository(database)
    assert repository.create(record) == repository.create(record)
    assert repository.get(record.memory_id).status == "candidate"
    database.close()

    reopened = Database(path)
    try:
        assert MemoryRepository(reopened).get(record.memory_id) == record
    finally:
        reopened.close()


def test_create_cannot_implicitly_activate_an_existing_candidate(repositories):
    _, events, memories = repositories
    source = events.insert(event("source", "我喜欢雨声"))
    candidate = memory("memory-1", source)
    memories.create(candidate)

    with pytest.raises(ValueError, match=r"status changes must use activate\(\)"):
        memories.create(memory("memory-1", source, status="active"))
    assert memories.get("memory-1").status == "candidate"


def test_creation_replays_are_idempotent_without_reversing_lifecycle(repositories):
    _, events, memories = repositories
    source = events.insert(event("source", "我喜欢雨声"))

    candidate = memory("candidate-first", source)
    assert memories.create(candidate).status == "candidate"
    assert memories.create(candidate).status == "candidate"
    memories.activate(candidate.memory_id)
    assert memories.create(candidate).status == "active"
    memories.expire(candidate.memory_id)
    assert memories.create(candidate).status == "expired"

    active = memory("active-first", source, status="active")
    assert memories.create(active).status == "active"
    assert memories.create(active).status == "active"
    memories.expire(active.memory_id)
    assert memories.create(active).status == "expired"


@pytest.mark.parametrize(
    "change,match",
    [
        ({"event_id": "missing"}, "evidence event does not exist"),
        ({"actor": "qichi"}, "actor does not match"),
        ({"occurred_at_utc": NOW + timedelta(seconds=1)}, "time does not match"),
        ({"exact_quote": "我喜欢晴天"}, "exact quote is absent"),
    ],
)
def test_evidence_must_match_real_event(repositories, change, match):
    _, events, memories = repositories
    source = events.insert(event("source", "我喜欢雨声"))
    fields = {
        "memory_id": "invalid",
        "event_id": source.event_id,
        "actor": source.actor,
        "exact_quote": "喜欢雨声",
        "occurred_at_utc": source.occurred_at_utc,
    }
    fields.update(change)
    record = memory("invalid", source, evidence=(MemoryEvidence(**fields),))
    with pytest.raises(ValueError, match=match):
        memories.create(record)
    assert memories.count() == 0


def test_all_evidence_must_belong_to_one_conversation(repositories):
    _, events, memories = repositories
    first = events.insert(event("first", "我喜欢雨声"))
    second = events.insert(event("second", "我也喜欢晚风", conversation_id="conversation-b"))
    evidence = (
        MemoryEvidence("cross", first.event_id, "mumo", "雨声", first.occurred_at_utc),
        MemoryEvidence("cross", second.event_id, "mumo", "晚风", second.occurred_at_utc),
    )
    with pytest.raises(ValueError, match="same conversation"):
        memories.create(memory("cross", first, evidence=evidence))
    assert memories.count() == 0


def test_whitespace_quote_is_rejected_but_exact_single_character_is_valid(repositories):
    _, events, memories = repositories
    source = events.insert(event("source", "我 喜欢雨声"))
    whitespace = memory("whitespace", source, quote=" ")
    with pytest.raises(ValueError, match="non-whitespace"):
        memories.create(whitespace)
    assert memories.create(memory("short", source, quote="雨")).status == "candidate"


def test_user_fact_requires_mumo_evidence_but_self_expression_may_use_qichi(repositories):
    _, events, memories = repositories
    qichi = events.insert(
        event("qichi", "我很在意用户", actor="qichi", direction="outbound", kind="text", status="sent", metadata={"generation_metadata": {"source": "dialogue"}})
    )
    user_fact = memory("bad-user-fact", qichi, status="active", fact="用户喜欢雨声")
    with pytest.raises(ValueError, match="active matrix"):
        memories.create(user_fact)

    self_expression = memory(
        "self-expression",
        qichi,
        status="active",
        type="self_expression",
        fact="角色表达过在意用户",
    )
    assert memories.create(self_expression).status == "active"
    assert [item.memory_id for item in memories.list_active("conversation-a", NOW)] == ["self-expression"]


def test_activation_rechecks_evidence_and_user_fact_ownership(repositories):
    database, events, memories = repositories
    qichi = events.insert(event("qichi", "我猜用户喜欢雨声", actor="qichi", direction="outbound", kind="text", status="sent", metadata={"generation_metadata": {"source": "dialogue"}}))
    memories.create(memory("candidate", qichi))
    with pytest.raises(ValueError, match="active matrix"):
        memories.activate("candidate")
    assert memories.get("candidate").status == "candidate"

    source = events.insert(event("source", "我喜欢晚风"))
    memories.create(memory("tampered", source, quote="喜欢晚风"))
    database.connection.execute(
        "UPDATE memory_evidence SET exact_quote = '不存在的原话' WHERE memory_id = 'tampered'"
    )
    with pytest.raises(ValueError, match="exact quote is absent"):
        memories.activate("tampered")
    assert database.connection.execute(
        "SELECT status FROM memory_records WHERE memory_id = 'tampered'"
    ).fetchone()[0] == "candidate"


def test_active_query_is_conversation_scoped_and_respects_validity(repositories):
    _, events, memories = repositories
    source_a = events.insert(event("a", "我今天在实验室"))
    source_b = events.insert(event("b", "我喜欢晚风", conversation_id="conversation-b"))
    memories.create(memory("short", source_a, status="active", valid_until=NOW + timedelta(hours=1)))
    memories.create(memory("other", source_b, status="active"))
    rejected = memories.create(memory("rejected", source_a))
    memories.reject(rejected.memory_id)
    expired = memories.create(memory("expired", source_a))
    memories.expire(expired.memory_id)

    assert [item.memory_id for item in memories.list_active("conversation-a", NOW)] == ["short"]
    assert [item.memory_id for item in memories.list_active("conversation-a", NOW + timedelta(hours=1))] == ["short"]
    assert memories.list_active("conversation-a", NOW + timedelta(hours=1, microseconds=1)) == ()
    assert [item.memory_id for item in memories.list_active("conversation-b", NOW)] == ["other"]


def test_reviewable_query_includes_past_deadlines_for_expiry_review_in_stable_order(repositories):
    _, events, memories = repositories
    source_a = events.insert(event("reviewable-a", "我喜欢雨声"))
    source_b = events.insert(event("reviewable-b", "我喜欢晚风", conversation_id="conversation-b"))
    memories.create(memory("z-active", source_a, status="active"))
    memories.create(memory("a-candidate", source_a))
    terminal = memories.create(memory("m-terminal", source_a))
    memories.reject(terminal.memory_id)
    memories.create(memory(
        "b-past-deadline", source_a, valid_until=NOW + timedelta(minutes=1),
        temporal_scope="bounded",
    ))
    memories.create(memory("c-other-conversation", source_b, status="active"))

    assert [
        item.memory_id
        for item in memories.list_reviewable(
            "conversation-a", NOW + timedelta(minutes=2)
        )
    ] == ["a-candidate", "b-past-deadline", "z-active"]


def test_correction_atomically_supersedes_old_active_record(repositories):
    database, events, memories = repositories
    old_source = events.insert(event("old-source", "我喜欢雨声"))
    correction_source = events.insert(event("correction-source", "其实我更喜欢晚风"))
    memories.create(memory("old", old_source, status="active"))
    correction = memory(
        "correction",
        correction_source,
        status="active",
        type="correction",
        fact="用户更喜欢晚风",
        supersedes_id="old",
    )
    assert memories.create(correction).status == "active"
    assert memories.get("old").status == "superseded"
    assert [item.memory_id for item in memories.list_active("conversation-a", NOW)] == ["correction"]
    assert database.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 2


def test_candidate_correction_activation_is_atomic_on_failure(repositories):
    database, events, memories = repositories
    old_source = events.insert(event("old-source", "我喜欢雨声"))
    correction_source = events.insert(event("new-source", "其实我更喜欢晚风"))
    memories.create(memory("old", old_source, status="active"))
    memories.create(
        memory("new", correction_source, type="correction", supersedes_id="old")
    )
    database.connection.execute(
        "CREATE TRIGGER fail_old_supersede BEFORE UPDATE ON memory_records "
        "WHEN OLD.memory_id = 'old' BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        memories.activate("new")
    assert memories.get("old").status == "active"
    assert memories.get("new").status == "candidate"


def test_identity_conflict_and_illegal_transitions_fail_closed(repositories):
    _, events, memories = repositories
    source = events.insert(event("source", "我喜欢雨声"))
    original = memory("same", source)
    memories.create(original)
    with pytest.raises(ValueError, match="identity conflict"):
        memories.create(memory("same", source, fact="用户喜欢晴天"))
    memories.reject("same")
    with pytest.raises(ValueError, match="cannot activate from rejected"):
        memories.activate("same")
    with pytest.raises(ValueError, match="cannot reject from rejected"):
        memories.reject("same")
    with pytest.raises(ValueError, match="cannot expire from rejected"):
        memories.expire("same")


@pytest.mark.parametrize("status", ["superseded", "rejected", "expired"])
def test_terminal_memory_cannot_be_created_directly(repositories, status):
    _, events, memories = repositories
    source = events.insert(event(f"source-{status}", "我喜欢雨声"))
    with pytest.raises(ValueError, match="candidate or active"):
        memories.create(memory(f"memory-{status}", source, status=status))
    assert memories.count() == 0


def test_wrong_supersedes_target_and_conversation_are_rejected(repositories):
    _, events, memories = repositories
    source_a = events.insert(event("a", "我喜欢雨声"))
    source_b = events.insert(event("b", "其实我喜欢晚风", conversation_id="conversation-b"))
    memories.create(memory("old", source_a, status="active"))
    with pytest.raises(KeyError):
        memories.create(memory("missing-old", source_a, status="active", type="correction", supersedes_id="absent"))
    with pytest.raises(ValueError, match="same conversation"):
        memories.create(memory("cross", source_b, status="active", type="correction", supersedes_id="old"))
    assert memories.get("old").status == "active"
    assert memories.count() == 1


@pytest.mark.parametrize(
    "statement,match",
    [
        ("UPDATE memory_records SET status = 'broken' WHERE memory_id = 'good'", "persisted memory status"),
        ("UPDATE memory_records SET valid_from_utc = 'not-a-time' WHERE memory_id = 'good'", "valid_from_utc"),
        ("UPDATE memory_evidence SET exact_quote = 'not in event' WHERE memory_id = 'good'", "exact quote is absent"),
    ],
)
def test_corrupted_persisted_rows_fail_closed(repositories, statement, match):
    database, events, memories = repositories
    source = events.insert(event("source", "我喜欢雨声"))
    memories.create(memory("good", source))
    if "status = 'broken'" in statement:
        database.connection.execute("PRAGMA ignore_check_constraints = ON")
    database.connection.execute(statement)
    with pytest.raises((TypeError, ValueError), match=match):
        memories.get("good")


def test_evidence_insert_failure_rolls_back_memory_row(repositories):
    database, events, memories = repositories
    source = events.insert(event("source", "我喜欢雨声"))
    database.connection.execute(
        "CREATE TRIGGER fail_evidence BEFORE INSERT ON memory_evidence "
        "BEGIN SELECT RAISE(ABORT, 'injected evidence failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected evidence failure"):
        memories.create(memory("half", source))
    assert database.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 0


def test_concurrent_corrections_leave_one_active_successor(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    setup = Database(path)
    events = EventRepository(setup)
    old_source = events.insert(event("old-source", "我喜欢雨声"))
    first_source = events.insert(event("first-source", "其实我喜欢晚风"))
    second_source = events.insert(event("second-source", "不对，我喜欢海浪"))
    repository = MemoryRepository(setup)
    repository.create(memory("old", old_source, status="active"))
    repository.create(memory("first", first_source, type="correction", supersedes_id="old"))
    repository.create(memory("second", second_source, type="correction", supersedes_id="old"))
    setup.close()

    barrier = threading.Barrier(2)
    successes: list[str] = []
    errors: list[BaseException] = []

    def activate(memory_id: str) -> None:
        database = Database(path)
        try:
            barrier.wait(timeout=5)
            MemoryRepository(database).activate(memory_id)
            successes.append(memory_id)
        except BaseException as error:
            errors.append(error)
        finally:
            database.close()

    threads = [threading.Thread(target=activate, args=(item,)) for item in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)

    check = Database(path)
    try:
        records = MemoryRepository(check)
        assert records.get("old").status == "superseded"
        assert [item.memory_id for item in records.list_active("conversation-a", NOW)] == successes
        loser = ({"first", "second"} - set(successes)).pop()
        assert records.get(loser).status == "candidate"
    finally:
        check.close()


def test_consolidation_uses_callers_transaction_and_writes_audit_atomically(repositories):
    database, events, memories = repositories
    source = events.insert(event("batch-source", "我喜欢雨声"))
    candidate_record = memory("batch-memory", source)

    with pytest.raises(ValueError, match="active transaction"):
        memories.apply_consolidation_in_transaction(
            database.connection,
            conversation_id="conversation-a",
            session_job_id="job-1",
            candidates=(candidate_record,),
            reviews=(),
            assessed_at_utc=NOW + timedelta(minutes=1),
        )

    with pytest.raises(RuntimeError, match="caller rollback"):
        with database.transaction() as connection:
            outcome = memories.apply_consolidation_in_transaction(
                connection,
                conversation_id="conversation-a",
                session_job_id="job-1",
                candidates=(candidate_record,),
                reviews=(),
                assessed_at_utc=NOW + timedelta(minutes=1),
            )
            assert outcome.created_memory_ids == ("batch-memory",)
            assert outcome.activated_memory_ids == ("batch-memory",)
            raise RuntimeError("caller rollback")

    assert database.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM memory_audit_events").fetchone()[0] == 0

    with database.transaction() as connection:
        memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id="job-1",
            candidates=(candidate_record,),
            reviews=(),
            assessed_at_utc=NOW + timedelta(minutes=1),
        )
    assert memories.get("batch-memory").status == "active"
    restored = memories.get("batch-memory")
    assert restored.assessed_at_utc == NOW + timedelta(minutes=1)
    audit = database.connection.execute(
        "SELECT session_job_id, action, before_status, after_status, "
        "before_certainty, after_certainty, before_importance, after_importance, "
        "before_temporal_scope, after_temporal_scope, assessment_reason_code, occurred_at_utc "
        "FROM memory_audit_events ORDER BY rowid"
    ).fetchall()
    assert [tuple(row) for row in audit] == [
        (
            "job-1", "create", None, "candidate", None, "explicit", None, 2,
            None, "ongoing", "explicit_user_statement",
            (NOW + timedelta(minutes=1)).isoformat(),
        ),
        (
            "job-1", "activate", "candidate", "active", "explicit", "explicit", 2, 2,
            "ongoing", "ongoing", "explicit_user_statement",
            (NOW + timedelta(minutes=1)).isoformat(),
        ),
    ]


def test_consolidation_audit_failure_rolls_back_memory_and_evidence(repositories):
    database, events, memories = repositories
    source = events.insert(event("audit-source", "我喜欢雨声"))
    database.connection.execute(
        "CREATE TRIGGER fail_memory_audit BEFORE INSERT ON memory_audit_events "
        "BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected audit failure"):
        with database.transaction() as connection:
            memories.apply_consolidation_in_transaction(
                connection,
                conversation_id="conversation-a",
                session_job_id="job-audit",
                candidates=(memory("audit-memory", source),),
                reviews=(),
                assessed_at_utc=NOW + timedelta(minutes=1),
            )
    assert database.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM memory_audit_events").fetchone()[0] == 0


def test_evidence_reads_in_real_event_order_not_uuid_order(repositories):
    _, events, memories = repositories
    same_time_first = events.insert(event("z-tie-first", "a雨声 z雨声"))
    same_time_second = events.insert(event("a-tie-second", "后来同一时刻确认雨声"))
    later_sequence = events.insert(event("a-late", "稍后再次确认喜欢雨声", occurred_at_utc=NOW + timedelta(minutes=1), received_at_utc=NOW + timedelta(minutes=1)))
    earlier_time = events.insert(event("z-early", "最早说喜欢雨声", occurred_at_utc=NOW - timedelta(minutes=1), received_at_utc=NOW - timedelta(minutes=1)))
    evidence = (
        MemoryEvidence("ordered", same_time_second.event_id, "mumo", "确认雨声", NOW, "confirmation"),
        MemoryEvidence("ordered", same_time_first.event_id, "mumo", "z雨声", NOW, "confirmation"),
        MemoryEvidence("ordered", later_sequence.event_id, "mumo", "喜欢雨声", NOW + timedelta(minutes=1), "confirmation"),
        MemoryEvidence("ordered", earlier_time.event_id, "mumo", "喜欢雨声", NOW - timedelta(minutes=1), "source"),
        MemoryEvidence("ordered", same_time_first.event_id, "mumo", "a雨声", NOW, "confirmation"),
    )
    memories.create(memory("ordered", earlier_time, evidence=evidence))
    assert [
        (item.event_id, item.exact_quote)
        for item in memories.get("ordered").memory_evidence
    ] == [
        ("z-early", "喜欢雨声"),
        ("z-tie-first", "a雨声"),
        ("z-tie-first", "z雨声"),
        ("a-tie-second", "确认雨声"),
        ("a-late", "喜欢雨声"),
    ]


@pytest.mark.parametrize(
    "overrides,actor,expected_status",
    [
        ({}, "mumo", "active"),
        ({"certainty": "ambiguous", "importance": 3, "assessment_reason_code": "ambiguous_scope"}, "mumo", "candidate"),
        ({"importance": 0}, "mumo", "candidate"),
        ({"modality": "scene_instruction"}, "mumo", "candidate"),
        ({"type": "episode", "temporal_scope": "historical"}, "mumo", "active"),
        ({"type": "episode", "temporal_scope": "ongoing"}, "mumo", "candidate"),
        ({"type": "self_expression", "certainty": "confirmed", "importance": 3, "assessment_reason_code": "later_user_confirmation"}, "qichi", "candidate"),
        ({"temporal_scope": "bounded", "valid_until": NOW + timedelta(minutes=2)}, "mumo", "active"),
        ({"temporal_scope": "bounded", "valid_until": NOW + timedelta(seconds=1)}, "mumo", "candidate"),
    ],
)
def test_auto_activation_matrix_has_paired_hits_and_non_matches(
    repositories, overrides, actor, expected_status
):
    database, events, memories = repositories
    source = events.insert(event(
        "matrix-source", "明确原话", actor=actor,
        direction="outbound" if actor == "qichi" else "inbound",
        kind="text",
        status="sent" if actor == "qichi" else "received",
        metadata={"generation_metadata": {"source": "dialogue"}} if actor == "qichi" else {},
    ))
    record = memory("matrix-memory", source, **overrides)
    with database.transaction() as connection:
        memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id="matrix-job",
            candidates=(record,),
            reviews=(),
            assessed_at_utc=NOW + timedelta(minutes=1),
        )
    assert memories.get("matrix-memory").status == expected_status


def test_bilateral_agreement_requires_distinct_proposal_and_acceptance(repositories):
    database, events, memories = repositories
    proposal = events.insert(event("proposal", "明天九点我来找你"))
    acceptance = events.insert(event(
        "acceptance", "好，九点等你", actor="qichi", direction="outbound", kind="text", status="sent", metadata={"generation_metadata": {"source": "dialogue"}}
    ))
    bilateral_evidence = (
        MemoryEvidence("bilateral", proposal.event_id, "mumo", "明天九点我来找你", NOW, "proposal"),
        MemoryEvidence("bilateral", acceptance.event_id, "qichi", "好，九点等你", NOW, "acceptance"),
    )
    bilateral = memory(
        "bilateral", proposal, type="agreement", certainty="confirmed", importance=1,
        assessment_reason_code="bilateral_agreement", evidence=bilateral_evidence,
    )
    unilateral = memory(
        "unilateral", proposal, type="agreement", certainty="confirmed", importance=1,
        assessment_reason_code="bilateral_agreement",
        evidence=(MemoryEvidence("unilateral", proposal.event_id, "mumo", "明天九点我来找你", NOW, "proposal"),),
    )
    with database.transaction() as connection:
        outcome = memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id="agreement-job",
            candidates=(bilateral, unilateral),
            reviews=(),
            assessed_at_utc=NOW + timedelta(minutes=1),
        )
    assert outcome.activated_memory_ids == ("bilateral",)
    assert memories.get("bilateral").status == "active"
    assert memories.get("unilateral").status == "candidate"


def test_bounded_bilateral_agreement_is_active_only_through_its_deadline(repositories):
    database, events, memories = repositories
    proposal = events.insert(event("bounded-proposal", "今晚睡前一起看一集"))
    acceptance = events.insert(event(
        "bounded-acceptance", "好，今晚等你", actor="qichi", direction="outbound",
        kind="text", status="sent",
        metadata={"generation_metadata": {"source": "dialogue"}},
    ))
    deadline = NOW + timedelta(hours=10)
    evidence = (
        MemoryEvidence(
            "bounded-agreement", proposal.event_id, "mumo",
            "今晚睡前一起看一集", NOW, "proposal",
        ),
        MemoryEvidence(
            "bounded-agreement", acceptance.event_id, "qichi",
            "好，今晚等你", NOW, "acceptance",
        ),
    )
    agreement = memory(
        "bounded-agreement", proposal, type="agreement", certainty="confirmed",
        importance=1, temporal_scope="bounded", valid_until=deadline,
        assessment_reason_code="bilateral_agreement", evidence=evidence,
    )
    with database.transaction() as connection:
        outcome = memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id="bounded-agreement-job",
            candidates=(agreement,),
            reviews=(),
            assessed_at_utc=NOW + timedelta(minutes=1),
        )

    assert outcome.activated_memory_ids == ("bounded-agreement",)
    assert [
        item.memory_id for item in memories.list_active("conversation-a", deadline)
    ] == ["bounded-agreement"]
    assert memories.list_active(
        "conversation-a", deadline + timedelta(microseconds=1)
    ) == ()


def test_confirm_review_adds_evidence_and_activates_candidate(repositories):
    database, events, memories = repositories
    first = events.insert(event("first-claim", "我可能喜欢雨声"))
    candidate_record = memory(
        "reviewed", first, certainty="ambiguous", importance=2,
        temporal_scope="unclassified", assessment_reason_code="ambiguous_scope",
    )
    memories.create(candidate_record)
    confirmation = events.insert(event(
        "later-confirmation", "对，我确实一直喜欢雨声",
        occurred_at_utc=NOW + timedelta(minutes=2), received_at_utc=NOW + timedelta(minutes=2),
    ))
    review_item = MemoryReview(
        memory_id="reviewed", action="confirm", certainty="confirmed", importance=2,
        temporal_scope="ongoing", assessment_reason_code="later_user_confirmation",
        evidence=(MemoryEvidence(
            "reviewed", confirmation.event_id, "mumo", "确实一直喜欢雨声",
            confirmation.occurred_at_utc, "confirmation",
        ),),
    )
    with database.transaction() as connection:
        outcome = memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id="review-job",
            candidates=(),
            reviews=(review_item,),
            assessed_at_utc=NOW + timedelta(minutes=3),
        )
    assert outcome.activated_memory_ids == ("reviewed",)
    restored = memories.get("reviewed")
    assert restored.status == "active" and restored.certainty == "confirmed"
    assert [item.evidence_role for item in restored.memory_evidence] == ["source", "confirmation"]
    assert [row[0] for row in database.connection.execute(
        "SELECT action FROM memory_audit_events WHERE session_job_id='review-job' ORDER BY rowid"
    )] == ["confirm", "activate"]


def test_support_review_regrades_from_new_source_evidence_then_runs_activation_matrix(repositories):
    database, events, memories = repositories
    first = events.insert(event("support-first", "我也许喜欢雨声"))
    memories.create(memory(
        "supported", first, certainty="ambiguous", importance=2,
        temporal_scope="unclassified", assessment_reason_code="ambiguous_scope",
    ))
    later = events.insert(event(
        "support-later", "我明确说过，我喜欢雨声",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    support = MemoryReview(
        "supported", "support", "explicit", 2, "ongoing",
        "explicit_user_statement",
        (MemoryEvidence(
            "supported", later.event_id, "mumo", "我喜欢雨声",
            later.occurred_at_utc, "source",
        ),),
    )
    with database.transaction() as connection:
        outcome = memories.apply_consolidation_in_transaction(
            connection, conversation_id="conversation-a", session_job_id="support-job",
            candidates=(), reviews=(support,), assessed_at_utc=NOW + timedelta(minutes=2),
        )
    assert outcome.reviewed_memory_ids == ("supported",)
    assert outcome.activated_memory_ids == ("supported",)
    assert memories.get("supported").status == "active"
    assert [row[0] for row in database.connection.execute(
        "SELECT action FROM memory_audit_events WHERE session_job_id='support-job' ORDER BY rowid"
    )] == ["support", "activate"]


@pytest.mark.parametrize(
    "action,certainty,reason,role",
    [
        ("confirm", "explicit", "later_user_confirmation", "confirmation"),
        ("confirm", "confirmed", "explicit_user_statement", "confirmation"),
        ("confirm", "confirmed", "later_user_confirmation", "source"),
        ("reject", "unsupported", "expired_or_completed", "counterevidence"),
        ("reject", "unsupported", "contradicted_by_user", "source"),
        ("expire", "explicit", "contradicted_by_user", "counterevidence"),
        ("support", "explicit", "explicit_user_statement", "counterevidence"),
    ],
)
def test_review_action_semantics_fail_closed(
    repositories, action, certainty, reason, role
):
    database, events, memories = repositories
    original = events.insert(event("review-original", "我可能喜欢雨声"))
    memories.create(memory(
        "review-target", original, certainty="ambiguous", importance=2,
        temporal_scope="unclassified", assessment_reason_code="ambiguous_scope",
    ))
    evidence_event = events.insert(event(
        "review-evidence", "这是新的证据",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    item = MemoryReview(
        "review-target", action, certainty, 2, "ongoing", reason,
        (MemoryEvidence(
            "review-target", evidence_event.event_id, "mumo", "新的证据",
            evidence_event.occurred_at_utc, role,
        ),),
    )
    with pytest.raises(ValueError, match="review"):
        with database.transaction() as connection:
            memories.apply_consolidation_in_transaction(
                connection, conversation_id="conversation-a", session_job_id="bad-review",
                candidates=(), reviews=(item,), assessed_at_utc=NOW + timedelta(minutes=2),
            )
    assert memories.get("review-target").status == "candidate"
    assert database.connection.execute(
        "SELECT COUNT(*) FROM memory_audit_events WHERE session_job_id='bad-review'"
    ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "action,certainty,reason,role,expected",
    [
        ("reject", "unsupported", "contradicted_by_user", "counterevidence", "rejected"),
        ("expire", "unsupported", "expired_or_completed", "counterevidence", "expired"),
    ],
)
def test_terminal_reviews_are_evidence_backed(
    repositories, action, certainty, reason, role, expected
):
    database, events, memories = repositories
    original = events.insert(event("terminal-original", "我喜欢雨声"))
    memories.create(memory("terminal-memory", original, status="active"))
    changed = events.insert(event(
        "terminal-change", "这条已经不成立了",
        occurred_at_utc=NOW + timedelta(minutes=1), received_at_utc=NOW + timedelta(minutes=1),
    ))
    item = MemoryReview(
        "terminal-memory", action, certainty, 0, "unclassified", reason,
        (MemoryEvidence(
            "terminal-memory", changed.event_id, "mumo", "已经不成立了",
            changed.occurred_at_utc, role,
        ),),
    )
    with database.transaction() as connection:
        memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id=f"{action}-job",
            candidates=(), reviews=(item,), assessed_at_utc=NOW + timedelta(minutes=2),
        )
    assert memories.get("terminal-memory").status == expected


def test_invalid_second_candidate_rolls_back_the_entire_batch(repositories):
    database, events, memories = repositories
    good_source = events.insert(event("good-source", "我喜欢雨声"))
    qichi_source = events.insert(event(
        "qichi-source", "我猜用户喜欢晴天", actor="qichi", direction="outbound", kind="text", status="sent", metadata={"generation_metadata": {"source": "dialogue"}}
    ))
    bad = memory("bad", qichi_source, fact="用户喜欢晴天")
    with pytest.raises(ValueError, match="mumo evidence"):
        with database.transaction() as connection:
            memories.apply_consolidation_in_transaction(
                connection,
                conversation_id="conversation-a",
                session_job_id="all-or-nothing",
                candidates=(memory("good", good_source), bad),
                reviews=(), assessed_at_utc=NOW + timedelta(minutes=1),
            )
    assert memories.count() == 0
    assert database.connection.execute("SELECT COUNT(*) FROM memory_audit_events").fetchone()[0] == 0


def test_batch_correction_activates_and_supersedes_with_one_audit_chain(repositories):
    database, events, memories = repositories
    old_source = events.insert(event("batch-old", "我喜欢雨声"))
    memories.create(memory("batch-old-memory", old_source, status="active"))
    correction_source = events.insert(event(
        "batch-correction", "其实我更喜欢海浪声",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    correction = memory(
        "batch-new-memory", correction_source, type="correction",
        fact="用户更喜欢海浪声", supersedes_id="batch-old-memory",
    )
    with database.transaction() as connection:
        outcome = memories.apply_consolidation_in_transaction(
            connection, conversation_id="conversation-a", session_job_id="correction-job",
            candidates=(correction,), reviews=(), assessed_at_utc=NOW + timedelta(minutes=2),
        )
    assert outcome.activated_memory_ids == ("batch-new-memory",)
    assert memories.get("batch-new-memory").status == "active"
    assert memories.get("batch-old-memory").status == "superseded"
    rows = database.connection.execute(
        "SELECT memory_id, action FROM memory_audit_events "
        "WHERE session_job_id='correction-job' ORDER BY rowid"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [
        ("batch-new-memory", "create"),
        ("batch-new-memory", "activate"),
        ("batch-old-memory", "supersede"),
    ]


def test_confirmation_presentation_rechecks_fragment_and_seven_day_cooldown(repositories):
    database, events, memories = repositories
    source = events.insert(event("candidate-source", "我可能喜欢雨声"))
    trigger = events.insert(event("trigger", "最近又下雨了"))
    candidate_record = memory(
        "candidate-memory", source, certainty="ambiguous", importance=2,
        temporal_scope="unclassified", assessment_reason_code="ambiguous_scope",
    )
    other_record = memory(
        "other-candidate", source, certainty="ambiguous", importance=3,
        temporal_scope="unclassified", assessment_reason_code="ambiguous_scope",
    )
    memories.create(candidate_record)
    memories.create(other_record)
    database.connection.execute(
        "INSERT INTO conversation_cursors (conversation_id, context_version) VALUES (?, ?)",
        ("conversation-a", 2),
    )

    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="candidate-memory",
        fragment_key="fragment-1", trigger_event_id=trigger.event_id, context_version=2,
        presented_at_utc=NOW,
    ) is True
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="other-candidate",
        fragment_key="fragment-1", trigger_event_id=trigger.event_id, context_version=2,
        presented_at_utc=NOW + timedelta(minutes=1),
    ) is False
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="candidate-memory",
        fragment_key="stale-context", trigger_event_id=trigger.event_id, context_version=3,
        presented_at_utc=NOW + timedelta(days=8),
    ) is False
    database.connection.execute(
        "UPDATE conversation_cursors SET context_version = 3 WHERE conversation_id = ?",
        ("conversation-a",),
    )
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="candidate-memory",
        fragment_key="fragment-2", trigger_event_id=trigger.event_id, context_version=3,
        presented_at_utc=NOW + timedelta(days=7) - timedelta(microseconds=1),
    ) is False
    database.connection.execute(
        "UPDATE conversation_cursors SET context_version = 4 WHERE conversation_id = ?",
        ("conversation-a",),
    )
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="candidate-memory",
        fragment_key="fragment-3", trigger_event_id=trigger.event_id, context_version=4,
        presented_at_utc=NOW + timedelta(days=7),
    ) is True
    assert database.connection.execute(
        "SELECT COUNT(*) FROM memory_confirmation_presentations"
    ).fetchone()[0] == 2


def test_legacy_public_lifecycle_is_explicitly_audited(repositories):
    database, events, memories = repositories
    source = events.insert(event("legacy-source", "我喜欢雨声"))
    memories.create(memory("legacy-candidate", source))
    memories.activate("legacy-candidate")

    rejected_source = events.insert(event("legacy-rejected", "我也许喜欢风"))
    memories.create(memory("legacy-rejected-memory", rejected_source))
    memories.reject("legacy-rejected-memory")

    expired_source = events.insert(event("legacy-expired", "我暂时喜欢云"))
    memories.create(memory("legacy-expired-memory", expired_source))
    memories.expire("legacy-expired-memory")

    rows = database.connection.execute(
        "SELECT memory_id, action, session_job_id, assessment_reason_code "
        "FROM memory_audit_events ORDER BY rowid"
    ).fetchall()
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
        ("legacy-candidate", "create", "legacy-public-api", "legacy_manual_review"),
        ("legacy-candidate", "activate", "legacy-public-api", "legacy_manual_review"),
        ("legacy-rejected-memory", "create", "legacy-public-api", "legacy_manual_review"),
        ("legacy-rejected-memory", "reject", "legacy-public-api", "legacy_manual_review"),
        ("legacy-expired-memory", "create", "legacy-public-api", "legacy_manual_review"),
        ("legacy-expired-memory", "expire", "legacy-public-api", "legacy_manual_review"),
    ]


def test_batch_rejects_candidate_with_incoherent_evidence_roles(repositories):
    database, events, memories = repositories
    source = events.insert(event("role-source", "我喜欢雨声"))
    later = events.insert(event(
        "role-later", "我仍然喜欢雨声",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    incoherent = memory(
        "role-mixed", source,
        evidence=(
            MemoryEvidence("role-mixed", source.event_id, "mumo", "喜欢雨声", NOW, "source"),
            MemoryEvidence("role-mixed", later.event_id, "mumo", "仍然喜欢雨声", NOW + timedelta(minutes=1), "confirmation"),
        ),
    )
    with pytest.raises(ValueError, match="evidence roles"):
        with database.transaction() as connection:
            memories.apply_consolidation_in_transaction(
                connection,
                conversation_id="conversation-a",
                session_job_id="role-job",
                candidates=(incoherent,),
                reviews=(),
                assessed_at_utc=NOW + timedelta(minutes=2),
            )
    assert memories.count() == 0
    assert database.connection.execute(
        "SELECT COUNT(*) FROM memory_audit_events WHERE session_job_id='role-job'"
    ).fetchone()[0] == 0


@pytest.mark.parametrize("bad_event", [
    {"event_id": "internal-evidence", "actor": "platform", "direction": "internal", "kind": "initiative", "status": "received", "text": "内部提示"},
    {"event_id": "pending-evidence", "actor": "qichi", "direction": "outbound", "kind": "text", "status": "pending", "text": "我记住了"},
])
def test_unreliable_or_platform_events_cannot_be_memory_evidence(repositories, bad_event):
    _, events, memories = repositories
    source = events.insert(event(**bad_event))
    record = memory(
        f"bad-{bad_event['event_id']}", source,
        type="self_expression" if bad_event["actor"] == "qichi" else "preference",
        fact="一条不应入记忆的事实",
    )
    with pytest.raises(ValueError, match="reliable evidence"):
        memories.create(record)
    assert memories.count() == 0


@pytest.mark.parametrize("action,initial_status", [("reject", "candidate"), ("expire", "active")])
def test_terminal_review_requires_later_counterevidence(repositories, action, initial_status):
    database, events, memories = repositories
    source = events.insert(event("terminal-base", "我喜欢雨声"))
    memories.create(memory("terminal-order", source, status=initial_status))
    same_time = events.insert(event(
        "terminal-same-time", "这条不成立了",
        occurred_at_utc=NOW - timedelta(seconds=1),
        received_at_utc=NOW - timedelta(seconds=1),
    ))
    review_item = MemoryReview(
        "terminal-order", action, "unsupported", 0, "unclassified",
        "contradicted_by_user" if action == "reject" else "expired_or_completed",
        (MemoryEvidence(
            "terminal-order", same_time.event_id, "mumo", "这条不成立了",
            same_time.occurred_at_utc, "counterevidence",
        ),),
    )
    with pytest.raises(ValueError, match="later counterevidence"):
        with database.transaction() as connection:
            memories.apply_consolidation_in_transaction(
                connection, conversation_id="conversation-a", session_job_id="order-bad",
                candidates=(), reviews=(review_item,), assessed_at_utc=NOW + timedelta(minutes=1),
            )
    assert memories.get("terminal-order").status == initial_status

    later = events.insert(event(
        "terminal-later", "确认，这条不成立了",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    valid_review = MemoryReview(
        "terminal-order", action, "unsupported", 0, "unclassified",
        "contradicted_by_user" if action == "reject" else "expired_or_completed",
        (MemoryEvidence(
            "terminal-order", later.event_id, "mumo", "这条不成立了",
            later.occurred_at_utc, "counterevidence",
        ),),
    )
    with database.transaction() as connection:
        memories.apply_consolidation_in_transaction(
            connection, conversation_id="conversation-a", session_job_id="order-good",
            candidates=(), reviews=(valid_review,), assessed_at_utc=NOW + timedelta(minutes=2),
        )
    assert memories.get("terminal-order").status == ("rejected" if action == "reject" else "expired")


def test_active_support_review_is_rejected(repositories):
    database, events, memories = repositories
    source = events.insert(event("active-support-base", "我喜欢雨声"))
    memories.create(memory("active-support", source, status="active"))
    later = events.insert(event(
        "active-support-later", "我还是喜欢雨声",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    support = MemoryReview(
        "active-support", "support", "explicit", 2, "ongoing",
        "explicit_user_statement",
        (MemoryEvidence(
            "active-support", later.event_id, "mumo", "还是喜欢雨声",
            later.occurred_at_utc, "source",
        ),),
    )
    with pytest.raises(ValueError, match="active memory"):
        with database.transaction() as connection:
            memories.apply_consolidation_in_transaction(
                connection, conversation_id="conversation-a", session_job_id="active-support-job",
                candidates=(), reviews=(support,), assessed_at_utc=NOW + timedelta(minutes=2),
            )
    assert memories.get("active-support").status == "active"


def test_confirmation_presentation_requires_current_later_trigger_and_time_order(repositories):
    database, events, memories = repositories
    before = events.insert(event(
        "presentation-before", "更早的消息",
        occurred_at_utc=NOW - timedelta(minutes=1),
        received_at_utc=NOW - timedelta(minutes=1),
    ))
    source = events.insert(event("presentation-source", "我可能喜欢雨声"))
    candidate_record = memory(
        "presentation-order", source, certainty="ambiguous", importance=2,
        temporal_scope="unclassified", assessment_reason_code="ambiguous_scope",
    )
    memories.create(candidate_record)
    current = events.insert(event("presentation-current", "最近又下雨了"))
    database.connection.execute(
        "INSERT INTO conversation_cursors (conversation_id, context_version) VALUES (?, ?)",
        ("conversation-a", 1),
    )

    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="presentation-order",
        fragment_key="before", trigger_event_id=before.event_id, context_version=1,
        presented_at_utc=NOW,
    ) is False
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="presentation-order",
        fragment_key="future", trigger_event_id=current.event_id, context_version=1,
        presented_at_utc=NOW - timedelta(seconds=1),
    ) is False
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="presentation-order",
        fragment_key="current", trigger_event_id=current.event_id, context_version=1,
        presented_at_utc=NOW,
    ) is True

    newer = events.insert(event("presentation-newer", "又补充了一句"))
    database.connection.execute(
        "UPDATE conversation_cursors SET context_version = 2 WHERE conversation_id = ?",
        ("conversation-a",),
    )
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="presentation-order",
        fragment_key="stale-trigger", trigger_event_id=current.event_id, context_version=2,
        presented_at_utc=NOW + timedelta(days=8),
    ) is False

    other = events.insert(event("presentation-other", "另一会话", conversation_id="conversation-b"))
    assert memories.try_record_confirmation_presentation(
        conversation_id="conversation-a", memory_id="presentation-order",
        fragment_key="wrong-conversation", trigger_event_id=other.event_id, context_version=2,
        presented_at_utc=NOW + timedelta(days=8),
    ) is False


def _consolidate(database, memories, candidates, *, job="job-dedupe"):
    with database.transaction() as connection:
        return memories.apply_consolidation_in_transaction(
            connection,
            conversation_id="conversation-a",
            session_job_id=job,
            candidates=candidates,
            reviews=(),
            assessed_at_utc=NOW + timedelta(minutes=1),
        )


def test_a_restated_preference_is_not_stored_twice(repositories):
    """2026-09-12：真机上有 4 对一字不差、6 对只换了说法的偏好，各自成了一张卡。"""

    database, events, memories = repositories
    first = events.insert(event("dedupe-1", "我喜欢强势一点的玩法"))
    second = events.insert(event("dedupe-2", "我就喜欢那种强势的玩法"))
    kept = _consolidate(database, memories, (memory(
        "dedupe-a", first,
        fact="用户表示自己喜欢强势的玩法，并希望角色在亲密互动中更浪更荡、想要什么直接说。",
    ),))
    repeated = _consolidate(database, memories, (memory(
        "dedupe-b", second,
        fact="用户希望角色在亲密互动中更浪更荡、想要什么就直接说，并喜欢强势的玩法。",
    ),), job="job-dedupe-2")

    assert kept.created_memory_ids == ("dedupe-a",)
    assert repeated.created_memory_ids == (), "换句话说的同一条偏好不该再建一张卡"
    assert repeated.restated_memory_ids == ("dedupe-b",), "跳过也要留下计数"
    assert database.connection.execute(
        "SELECT COUNT(*) FROM memory_records WHERE type='preference'").fetchone()[0] == 1


def test_a_different_preference_sharing_words_is_still_stored(repositories):
    database, events, memories = repositories
    first = events.insert(event("dedupe-3", "我喜欢下雨天"))
    second = events.insert(event("dedupe-4", "下雨天我还喜欢出门走走"))
    _consolidate(database, memories, (memory(
        "dedupe-c", first, fact="用户喜欢下雨天在家听雨声。",
    ),))
    other = _consolidate(database, memories, (memory(
        "dedupe-d", second, fact="用户喜欢下雨天出门散步。",
    ),), job="job-dedupe-2")

    assert other.created_memory_ids == ("dedupe-d",), "共用了几个字不等于同一条偏好"


def test_two_different_nights_are_never_deduped(repositories):
    """不同夜晚的片段共用同一个模板开头（实测 0.53），绝不能当成重复。"""

    database, events, memories = repositories
    first = events.insert(event("dedupe-5", "那一晚"))
    second = events.insert(event("dedupe-6", "另一晚"))
    _consolidate(database, memories, (memory(
        "dedupe-e", first, type="episode",
        fact="2026年9月10日凌晨，用户与角色在对话中连续接续完成了一段成人共同想象：用户以侍奉为主。",
    ),))
    other = _consolidate(database, memories, (memory(
        "dedupe-f", second, type="episode",
        fact="2026年9月12日凌晨，用户与角色在对话中连续接续完成了一段成人共同想象：用户以主人身份。",
    ),), job="job-dedupe-2")

    assert other.created_memory_ids == ("dedupe-f",), "两个夜晚是两件事"

