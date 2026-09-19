from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pytest

from qichi.observability import TurnTraceRepository
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.migrations import SCHEMA_VERSION
from qichi.domain.events import ConversationEvent, MessageSegment


NOW = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)


def trigger() -> ConversationEvent:
    return ConversationEvent(
        event_id="trigger-1", platform_event_id="platform-1", platform_message_id="10",
        conversation_id="owner", sequence=0, direction="inbound", actor="mumo", kind="text",
        text="hello", message_segments=(MessageSegment("text", {"text": "hello"}),),
        reply_to_event_id=None, reply_to_platform_message_id=None,
        occurred_at_utc=NOW, received_at_utc=NOW, status="received", metadata={},
    )


def test_trace_is_append_only_idempotent_and_survives_restart(tmp_path):
    path = tmp_path / "trace.sqlite3"
    with Database(path) as database:
        EventRepository(database).insert(trigger())
        repository = TurnTraceRepository(database)
        first = repository.append_once(
            trigger_event_id="trigger-1", conversation_id="owner", source="dialogue",
            phase="context", occurred_at_utc=NOW,
            details={"context_version": 1, "selected_event_ids": ["trigger-1"]},
        )
        assert repository.append_once(
            trigger_event_id="trigger-1", conversation_id="owner", source="dialogue",
            phase="context", occurred_at_utc=NOW,
            details={"context_version": 1, "selected_event_ids": ["trigger-1"]},
        ) == first
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            database.connection.execute("UPDATE turn_trace_events SET phase='failure'")
    with Database(path) as database:
        restored = TurnTraceRepository(database).list_for_trace(first.trace_id)
        assert restored == (first,)


def test_trace_conflict_and_sensitive_fields_fail_closed(tmp_path):
    with Database(tmp_path / "trace.sqlite3") as database:
        EventRepository(database).insert(trigger())
        repository = TurnTraceRepository(database)
        repository.append_once(
            trigger_event_id="trigger-1", conversation_id="owner", source="dialogue",
            phase="received", occurred_at_utc=NOW, details={"context_version": 1},
        )
        with pytest.raises(ValueError, match="identity conflict"):
            repository.append_once(
                trigger_event_id="trigger-1", conversation_id="owner", source="dialogue",
                phase="received", occurred_at_utc=NOW, details={"context_version": 2},
            )
        for key in ("prompt", "messages", "reasoning", "api_key", "raw_payload", "text"):
            with pytest.raises(ValueError, match="sensitive"):
                repository.append_once(
                    trigger_event_id="trigger-1", conversation_id="owner", source="dialogue",
                    phase="failure", occurred_at_utc=NOW, details={key: "secret"},
                )


def test_schema_upgrade_preserves_existing_events(tmp_path):
    path = tmp_path / "upgrade.sqlite3"
    with Database(path) as database:
        EventRepository(database).insert(trigger())
        assert database.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key='schema_version'"
        ).fetchone()[0] == str(SCHEMA_VERSION)
        assert database.connection.execute(
            "SELECT COUNT(*) FROM conversation_events"
        ).fetchone()[0] == 1
