from __future__ import annotations

import sqlite3
import threading
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.outbox_repository import OutboxRepository


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)


def insert_event(database: Database, event_id: str = "event") -> str:
    event = ConversationEvent(
        event_id=event_id,
        platform_event_id=f"platform-event-{event_id}",
        platform_message_id=f"platform-message-{event_id}",
        conversation_id="conversation",
        sequence=0,
        direction="outbound",
        actor="qichi",
        kind="text",
        text="text",
        message_segments=(MessageSegment("text", {"text": "text"}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=NOW,
        received_at_utc=NOW,
        status="received",
        metadata={},
    )
    return EventRepository(database).insert(event).event_id


def payload(action_kind: str = "text", **extra: object) -> dict[str, object]:
    result: dict[str, object] = {"action_kind": action_kind, "data": {"nested": [None, "value"]}}
    result.update(extra)
    return result


def test_create_intent_is_idempotent_only_for_identical_immutable_identity(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        created = repository.create_intent("operation", event_id, payload(), NOW)
        repeated = repository.create_intent("operation", event_id, payload(), NOW + timedelta(seconds=1))
        assert repeated == created
        assert created.status == "pending"
        assert created.attempt_count == 0
        assert created.created_at_utc == NOW
        assert created.payload["data"]["nested"] == (None, "value")
        with pytest.raises(ValueError, match="identity"):
            repository.create_intent("operation", event_id, payload("face"), NOW)
        other_event = insert_event(database, "other")
        with pytest.raises(ValueError, match="identity"):
            repository.create_intent("operation", other_event, payload(), NOW)
    finally:
        database.close()


def test_public_outbox_records_are_immutable_and_do_not_expose_intent_mutation(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        record = repository.create_intent("operation", event_id, payload(), NOW)
        with pytest.raises(FrozenInstanceError):
            record.status = "sent"  # type: ignore[misc]
        with pytest.raises(TypeError):
            record.payload["action_kind"] = "poke"  # type: ignore[index]
        with pytest.raises(TypeError):
            record.payload["data"]["nested"] += ("changed",)  # type: ignore[index,operator]
        assert repository.get("operation") == record
    finally:
        database.close()


@pytest.mark.parametrize("action_kind", ["", "unknown", 1, None])
def test_create_intent_rejects_invalid_action_kind_before_sql(tmp_path, action_kind):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        with pytest.raises((TypeError, ValueError)):
            OutboxRepository(database).create_intent("operation", event_id, payload(action_kind), NOW)
    finally:
        database.close()


@pytest.mark.parametrize("invalid_value", ["", 1, None])
def test_create_intent_rejects_invalid_ids_and_naive_timestamp(tmp_path, invalid_value):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        with pytest.raises((TypeError, ValueError)):
            repository.create_intent(invalid_value, event_id, payload(), NOW)  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            repository.create_intent("operation", invalid_value, payload(), NOW)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            repository.create_intent("operation", event_id, payload(), NOW.replace(tzinfo=None))
    finally:
        database.close()


def test_create_intent_rejects_invalid_json_and_foreign_key(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        repository = OutboxRepository(database)
        with pytest.raises(TypeError):
            repository.create_intent("not-json", "missing", {"action_kind": "text", "bad": {1}}, NOW)
        with pytest.raises(sqlite3.IntegrityError):
            repository.create_intent("missing-event", "missing", payload(), NOW)
    finally:
        database.close()


@pytest.mark.parametrize("reaction_payload", [payload("reaction"), payload("reaction", set=False)])
def test_reaction_intent_requires_explicit_idempotent_set_true(tmp_path, reaction_payload):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        with pytest.raises(ValueError, match="set=true"):
            OutboxRepository(database).create_intent("reaction", event_id, reaction_payload, NOW)
    finally:
        database.close()


@pytest.mark.parametrize("terminal_status", ["sent", "unknown", "failed"])
def test_legal_delivery_transitions_increment_once_and_preserve_response(tmp_path, terminal_status):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(), NOW)
        dispatched = repository.begin_dispatch("operation", NOW + timedelta(seconds=1))
        terminal = repository.complete_dispatch(
            "operation", terminal_status, {"result": {"id": 1}}, NOW + timedelta(seconds=2)
        )
        assert dispatched.status == "dispatched"
        assert dispatched.attempt_count == 1
        assert terminal.status == terminal_status
        assert terminal.attempt_count == 1
        assert terminal.response == {"result": {"id": 1}}
    finally:
        database.close()


@pytest.mark.parametrize("operation", ["begin", "complete-sent", "complete-unknown", "complete-failed", "reaction-recovery"])
def test_illegal_delivery_transitions_are_rejected_without_mutation(tmp_path, operation):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(), NOW)
        if operation == "begin":
            repository.begin_dispatch("operation", NOW)
            action = lambda: repository.begin_dispatch("operation", NOW)
        elif operation == "complete-sent":
            action = lambda: repository.complete_dispatch("operation", "sent", None, NOW)
        elif operation == "complete-unknown":
            action = lambda: repository.complete_dispatch("operation", "unknown", None, NOW)
        elif operation == "complete-failed":
            action = lambda: repository.complete_dispatch("operation", "failed", None, NOW)
        else:
            action = lambda: repository.recover_idempotent_reaction("operation", NOW)
        with pytest.raises(ValueError):
            action()
        expected_status = "dispatched" if operation == "begin" else "pending"
        assert repository.get("operation").status == expected_status
    finally:
        database.close()


@pytest.mark.parametrize("action_kind", ["text", "face", "poke"])
def test_unknown_non_idempotent_actions_cannot_be_retried(tmp_path, action_kind):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(action_kind), NOW)
        repository.begin_dispatch("operation", NOW)
        repository.complete_dispatch("operation", "unknown", None, NOW)
        with pytest.raises(ValueError):
            repository.recover_idempotent_reaction("operation", NOW)
        with pytest.raises(ValueError):
            repository.begin_dispatch("operation", NOW)
        assert repository.get("operation").status == "unknown"
        assert repository.get("operation").attempt_count == 1
    finally:
        database.close()


def test_unknown_idempotent_reaction_set_true_can_recover_across_reopen(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    event_id = insert_event(database)
    repository = OutboxRepository(database)
    repository.create_intent("reaction", event_id, payload("reaction", set=True), NOW)
    repository.begin_dispatch("reaction", NOW)
    repository.complete_dispatch("reaction", "unknown", None, NOW)
    database.close()
    reopened = Database(path)
    try:
        repository = OutboxRepository(reopened)
        recovered = repository.recover_idempotent_reaction("reaction", NOW + timedelta(seconds=1))
        assert recovered.status == "pending"
        assert recovered.attempt_count == 1
        assert repository.get("reaction") == recovered
        dispatched = repository.begin_dispatch("reaction", NOW + timedelta(seconds=2))
        assert dispatched.attempt_count == 2
        assert repository.complete_dispatch("reaction", "sent", {"ok": True}, NOW + timedelta(seconds=3)).status == "sent"
    finally:
        reopened.close()


def test_reopen_outstanding_work_includes_pending_dispatched_and_unknown_only(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    event_id = insert_event(database)
    repository = OutboxRepository(database)
    repository.create_intent("pending", event_id, payload(), NOW)
    repository.create_intent("dispatched", event_id, payload("face"), NOW)
    repository.begin_dispatch("dispatched", NOW + timedelta(seconds=1))
    repository.create_intent("unknown", event_id, payload("text"), NOW)
    repository.begin_dispatch("unknown", NOW + timedelta(seconds=1))
    repository.complete_dispatch("unknown", "unknown", None, NOW + timedelta(seconds=2))
    repository.create_intent("sent", event_id, payload("face"), NOW)
    repository.begin_dispatch("sent", NOW + timedelta(seconds=1))
    repository.complete_dispatch("sent", "sent", {"ok": True}, NOW + timedelta(seconds=2))
    database.close()
    reopened = Database(path)
    try:
        repository = OutboxRepository(reopened)
        outstanding = {record.operation_key: record.status for record in repository.outstanding()}
        assert outstanding == {"pending": "pending", "dispatched": "dispatched", "unknown": "unknown"}
        with pytest.raises(ValueError):
            repository.begin_dispatch("dispatched", NOW + timedelta(seconds=3))
        assert repository.complete_dispatch("dispatched", "unknown", None, NOW + timedelta(seconds=3)).status == "unknown"
        with pytest.raises(ValueError):
            repository.recover_idempotent_reaction("unknown", NOW + timedelta(seconds=3))
    finally:
        reopened.close()


def test_failed_actions_are_terminal_and_pending_recovery_only_lists_pending(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("pending", event_id, payload(), NOW)
        repository.create_intent("failed", event_id, payload("face"), NOW)
        repository.begin_dispatch("failed", NOW)
        repository.complete_dispatch("failed", "failed", {"error": "no retry"}, NOW)
        assert [record.operation_key for record in repository.recoverable_pending()] == ["pending"]
        with pytest.raises(ValueError):
            repository.begin_dispatch("failed", NOW)
    finally:
        database.close()


@pytest.mark.parametrize("action_kind", ["text", "face", "poke"])
def test_pending_non_reaction_with_prior_attempt_fails_closed(tmp_path, action_kind):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(action_kind), NOW)
        database.connection.execute(
            "UPDATE outbox SET attempt_count = ? WHERE operation_key = ?", (1, "operation")
        )
        with pytest.raises(ValueError, match="pending"):
            repository.get("operation")
        with pytest.raises(ValueError, match="pending"):
            repository.recoverable_pending()
    finally:
        database.close()


def test_timestamps_cannot_move_backward_and_rejection_leaves_record_unchanged(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(), NOW)
        with pytest.raises(ValueError, match="backward"):
            repository.begin_dispatch("operation", NOW - timedelta(seconds=1))
        assert repository.get("operation").status == "pending"
        repository.begin_dispatch("operation", NOW + timedelta(seconds=1))
        with pytest.raises(ValueError, match="backward"):
            repository.complete_dispatch("operation", "sent", None, NOW)
        assert repository.get("operation").status == "dispatched"
    finally:
        database.close()


@pytest.mark.parametrize(
    "column, value",
    [
        ("status", "corrupt"),
        ("attempt_count", -1),
        ("created_at_utc", "not-a-timestamp"),
        ("updated_at_utc", "2026-08-27T20:00:00+08:00"),
        ("payload_json", '{"action_kind":"invalid"}'),
    ],
)
def test_persisted_outbox_rows_fail_closed_when_corrupt(tmp_path, column, value):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(), NOW)
        database.connection.execute(f"UPDATE outbox SET {column} = ? WHERE operation_key = ?", (value, "operation"))
        with pytest.raises((TypeError, ValueError)):
            repository.get("operation")
    finally:
        database.close()


@pytest.mark.parametrize("enumerate_records", ["outstanding", "recoverable_pending"])
@pytest.mark.parametrize(
    "column, value",
    [
        ("status", "corrupt"),
        ("payload_json", '{"action_kind":"invalid"}'),
        ("updated_at_utc", "not-a-timestamp"),
    ],
)
def test_recovery_enumeration_fails_closed_for_corrupted_rows(tmp_path, enumerate_records, column, value):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(), NOW)
        database.connection.execute(
            f"UPDATE outbox SET {column} = ? WHERE operation_key = ?", (value, "operation")
        )
        with pytest.raises((TypeError, ValueError)):
            getattr(repository, enumerate_records)()
    finally:
        database.close()


@pytest.mark.parametrize(
    "column, value",
    [
        ("operation_key", ""),
        ("event_id", ""),
        ("status", "dispatched"),
        ("status", "sent"),
        ("status", "unknown"),
        ("status", "failed"),
    ],
)
def test_persisted_outbox_ids_and_status_attempt_combinations_fail_closed(tmp_path, column, value):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event_id = insert_event(database)
        repository = OutboxRepository(database)
        repository.create_intent("operation", event_id, payload(), NOW)
        if column == "status":
            database.connection.execute(
                "UPDATE outbox SET status = ?, attempt_count = ? WHERE operation_key = ?",
                (value, 0, "operation"),
            )
        else:
            if column == "event_id":
                database.connection.execute("PRAGMA foreign_keys = OFF")
            database.connection.execute(
                f"UPDATE outbox SET {column} = ? WHERE operation_key = ?", (value, "operation")
            )
        with pytest.raises(ValueError):
            repository.outstanding()
    finally:
        database.close()


def test_concurrent_dispatch_only_one_connection_increments_attempt(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    setup = Database(path)
    event_id = insert_event(setup)
    OutboxRepository(setup).create_intent("operation", event_id, payload(), NOW)
    setup.close()
    barrier = threading.Barrier(2)
    successes: list[int] = []
    errors: list[BaseException] = []

    def dispatch() -> None:
        database = None
        try:
            database = Database(path)
            barrier.wait(timeout=5)
            successes.append(OutboxRepository(database).begin_dispatch("operation", NOW).attempt_count)
        except BaseException as error:
            errors.append(error)
        finally:
            if database is not None:
                database.close()

    threads = [threading.Thread(target=dispatch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert successes == [1]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
