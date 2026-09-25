from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository


UTC_NOW = datetime(2026, 8, 27, 12, 34, 56, tzinfo=timezone.utc)


def make_event(event_id: str, **overrides: object) -> ConversationEvent:
    values: dict[str, object] = {
        "event_id": event_id, "platform_event_id": f"pe-{event_id}", "platform_message_id": f"pm-{event_id}", "conversation_id": "conversation-a", "sequence": 999, "direction": "inbound", "actor": "mumo", "kind": "text", "text": "original text", "message_segments": (MessageSegment("text", {"text": "first"}), MessageSegment("face", {"id": 14}), MessageSegment("text", {"text": "last"})), "reply_to_event_id": None, "reply_to_platform_message_id": None, "occurred_at_utc": UTC_NOW, "received_at_utc": UTC_NOW + timedelta(seconds=1), "status": "received", "metadata": {"nested": {"values": [1, {"ok": True}]}, "empty": None},
    }
    values.update(overrides)
    return ConversationEvent(**values)  # type: ignore[arg-type]


def test_close_reopen_round_trip_preserves_event_and_raw_payload(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    raw_payload = {"post_type": "message", "nested": {"items": [None, "value"]}}
    database = Database(path)
    repository = EventRepository(database)
    target = repository.insert(make_event("target"))
    inserted = repository.insert(
        make_event(
            "roundtrip",
            reply_to_event_id=target.event_id,
            reply_to_platform_message_id=target.platform_message_id,
        ),
        raw_payload=raw_payload,
    )
    database.close()
    reopened = Database(path)
    try:
        restored = EventRepository(reopened).get("roundtrip")
        assert restored == inserted
        assert restored.sequence == 1
        assert restored.message_segments[1].type == "face"
        assert restored.metadata == inserted.metadata
        assert restored.reply_to_event_id == target.event_id
        assert restored.reply_to_platform_message_id == target.platform_message_id
        assert restored.occurred_at_utc == UTC_NOW
        assert restored.received_at_utc == UTC_NOW + timedelta(seconds=1)
        payload = reopened.connection.execute("SELECT raw_payload_json FROM conversation_events WHERE event_id = ?", ("roundtrip",)).fetchone()[0]
        assert json.loads(payload) == raw_payload
    finally:
        reopened.close()


def test_outbound_platform_message_map_is_hydrated_without_mutating_immutable_event(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        event = EventRepository(database).insert(
            make_event(
                "mapped-outbound",
                direction="outbound",
                actor="qichi",
                platform_message_id=None,
                status="sent",
            )
        )
        database.connection.execute(
            "INSERT INTO platform_message_map "
            "(platform_message_id, event_id, source, created_at_utc) "
            "VALUES (?, ?, ?, ?)",
            ("701", event.event_id, "sender", UTC_NOW.isoformat()),
        )

        restored = EventRepository(database).get(event.event_id)

        assert restored.platform_message_id == "701"
        assert database.connection.execute(
            "SELECT platform_message_id FROM conversation_events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()[0] is None
    finally:
        database.close()


@pytest.mark.parametrize("column, value", [("text", "changed"), ("actor", "qichi"), ("conversation_id", "other"), ("sequence", 22), ("direction", "outbound"), ("kind", "poke"), ("message_segments_json", "[]"), ("reply_to_event_id", "other-event"), ("reply_to_platform_message_id", "other-platform-message"), ("occurred_at_utc", "2026-08-28T00:00:00+00:00"), ("received_at_utc", "2026-08-28T00:00:00+00:00"), ("metadata_json", "{}"), ("raw_payload_json", "{}"), ("platform_event_id", "new-platform-event"), ("platform_message_id", "new-platform-message")])
def test_event_fields_are_immutable_at_database_level(tmp_path, column, value):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        EventRepository(database).insert(make_event("immutable"), raw_payload={"one": 1})
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            database.connection.execute(f"UPDATE conversation_events SET {column} = ? WHERE event_id = ?", (value, "immutable"))
    finally:
        database.close()


def test_status_update_succeeds_but_combined_status_and_text_update_is_rejected(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        repository = EventRepository(database)
        repository.insert(make_event("status"))
        repository.update_status("status", "sent")
        assert repository.get("status").status == "sent"
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            database.connection.execute("UPDATE conversation_events SET status = ?, text = ? WHERE event_id = ?", ("failed", "changed", "status"))
        with pytest.raises(KeyError):
            repository.update_status("missing", "sent")
        with pytest.raises(ValueError):
            repository.update_status("status", "")
        assert repository.get("status").status == "sent"
        with pytest.raises(TypeError):
            repository.update_status("status", 1)  # type: ignore[arg-type]
        assert repository.get("status").status == "sent"
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "UPDATE conversation_events SET status = ? WHERE event_id = ?",
                ("", "status"),
            )
        assert repository.get("status").status == "sent"
    finally:
        database.close()


@pytest.mark.parametrize("duplicate_overrides", [{"platform_event_id": "pe-original", "platform_message_id": None}, {"platform_event_id": None, "platform_message_id": "pm-original"}, {"platform_event_id": "pe-original", "platform_message_id": "pm-original"}])
def test_platform_identity_duplicates_return_existing_without_consuming_sequence(tmp_path, duplicate_overrides):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        repository = EventRepository(database)
        original = repository.insert(make_event("original"))
        duplicate = repository.insert(make_event("duplicate", **duplicate_overrides))
        next_event = repository.insert(make_event("next"))
        assert duplicate.event_id == original.event_id
        assert next_event.sequence == 1
    finally:
        database.close()


def test_platform_identity_cross_conflicts_are_rejected_without_sequence_gap(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        repository = EventRepository(database)
        first = repository.insert(make_event("first"))
        second = repository.insert(make_event("second"))
        with pytest.raises(ValueError, match="platform identity conflict"):
            repository.insert(make_event("cross", platform_event_id=first.platform_event_id, platform_message_id=second.platform_message_id))
        with pytest.raises(ValueError, match="platform identity conflict"):
            repository.insert(make_event("mismatch", platform_event_id=first.platform_event_id, platform_message_id="pm-new"))
        assert repository.insert(make_event("third")).sequence == 2
    finally:
        database.close()


def test_concurrent_connections_allocate_unique_contiguous_sequences(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    barrier = threading.Barrier(2)
    sequences: list[int] = []
    errors: list[BaseException] = []

    def insert_event(event_id: str) -> None:
        database = None
        try:
            database = Database(path)
            barrier.wait(timeout=5)
            sequences.append(EventRepository(database).insert(make_event(event_id)).sequence)
        except BaseException as error:
            errors.append(error)
        finally:
            if database is not None:
                database.close()

    threads = [threading.Thread(target=insert_event, args=(f"concurrent-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert sorted(sequences) == [0, 1]


def test_handle_derivation_and_resolution_validate_conversation_actor_and_visibility(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        repository = EventRepository(database)
        inbound = repository.insert(make_event("inbound"))
        outbound = repository.insert(make_event("outbound", direction="outbound", actor="qichi", platform_event_id="pe-outbound", platform_message_id="pm-outbound"))
        internal = repository.insert(make_event("internal", direction="internal", actor="platform", platform_event_id=None, platform_message_id=None))
        assert repository.derive_handle("conversation-a", inbound.event_id) == "M0"
        assert repository.derive_handle("conversation-a", outbound.event_id) == "Q1"
        assert repository.derive_handle("conversation-a", internal.event_id) is None
        assert repository.resolve_handle("conversation-a", "M0").event_id == inbound.event_id
        assert repository.resolve_handle("conversation-a", "Q1").event_id == outbound.event_id
        with pytest.raises(KeyError): repository.derive_handle("wrong-conversation", inbound.event_id)
        with pytest.raises(KeyError): repository.resolve_handle("wrong-conversation", "M0")
        with pytest.raises(KeyError): repository.resolve_handle("conversation-a", "M1")
        with pytest.raises(KeyError): repository.resolve_handle("conversation-a", "Q2")
        with pytest.raises(ValueError): repository.resolve_handle("conversation-a", "P0")
        with pytest.raises(ValueError): repository.resolve_handle("conversation-a", "M00")
    finally:
        database.close()
