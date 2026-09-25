from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import qichi.storage.migrations as migrations
from qichi.storage.database import Database
from qichi.storage.migrations import SCHEMA_VERSION


EXPECTED_TABLES = {
    "conversation_events", "platform_message_map", "message_links", "outbox",
    "memory_records", "memory_evidence", "conversation_cursors", "runtime_meta",
    "turn_trace_events", "memory_audit_events", "memory_session_jobs",
    "memory_confirmation_presentations", "memory_job_events",
    "memory_fragments", "memory_fragment_events", "memory_detail_records",
    "memory_detail_evidence",
}

I3_MEMORY_IDS = {
    "04e2296c-ba1c-503b-abf4-d61fe590ef44",
    "8bac2494-9f8b-5fa2-a882-ef5c96c77ceb",
    "b7aca90e-9216-59e0-99ed-990dc11efe65",
    "de392c29-a61b-5e25-9131-5a637f7132c4",
    "71cf09df-f555-54b4-93e5-56d147feb789",
    "9bd8e649-0444-5978-9b6e-0403ab70dd35",
    "6fc794ad-5548-5f92-a5b4-5bd1879a7d59",
}
CANDIDATE_MEMORY_ID = "7e77ed72-eb5e-59ff-847d-f290c02c081e"


def create_version_one_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        migration = (
            Path(__file__).resolve().parents[1] / "migrations" / "0001_initial.sql"
        ).read_text(encoding="utf-8")
        connection.executescript(migration)
        occurred_at = "2026-08-27T00:00:00+00:00"
        connection.execute(
            "INSERT INTO conversation_events (event_id, platform_event_id, "
            "platform_message_id, conversation_id, sequence, direction, actor, kind, "
            "text, message_segments_json, reply_to_event_id, reply_to_platform_message_id, "
            "occurred_at_utc, received_at_utc, status, metadata_json, raw_payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-event", "legacy-platform-event", "10001", "owner", 0,
                "inbound", "mumo", "text", "legacy text",
                json.dumps([{"type": "text", "data": {"text": "legacy text"}}]),
                None, None, occurred_at, occurred_at, "received", "{}", None,
            ),
        )
        connection.execute(
            "INSERT INTO platform_message_map "
            "(platform_message_id, event_id, source, created_at_utc) VALUES (?, ?, ?, ?)",
            ("10001", "legacy-event", "test", occurred_at),
        )
        connection.execute(
            "INSERT INTO conversation_cursors "
            "(conversation_id, context_version, last_processed_sequence, "
            "last_user_activity_utc, presence_topic_cursor) VALUES (?, ?, ?, ?, ?)",
            ("owner", 3, 0, occurred_at, 0),
        )
        connection.execute(
            "INSERT INTO runtime_meta (key, value_json, updated_at_utc) VALUES (?, ?, ?)",
            ("schema_version", "1", occurred_at),
        )
        connection.commit()
    finally:
        connection.close()


def create_version_two_database(path: Path, *, with_known_shape: bool = False) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        root = Path(__file__).resolve().parents[1] / "migrations"
        connection.executescript((root / "0001_initial.sql").read_text(encoding="utf-8"))
        connection.executescript((root / "0002_turn_traces.sql").read_text(encoding="utf-8"))
        if with_known_shape:
            agreement_ids = {
                "04e2296c-ba1c-503b-abf4-d61fe590ef44",
                "8bac2494-9f8b-5fa2-a882-ef5c96c77ceb",
                "b7aca90e-9216-59e0-99ed-990dc11efe65",
                "de392c29-a61b-5e25-9131-5a637f7132c4",
            }
            active_rows = [
                (memory_id, "agreement" if memory_id in agreement_ids else "preference", "active")
                for memory_id in sorted(I3_MEMORY_IDS)
            ]
            active_rows.extend(
                [(f"active-{index:02d}", "episode" if index < 2 else "preference", "active") for index in range(9)]
            )
            memory_rows = (
                active_rows
                + [(CANDIDATE_MEMORY_ID, "agreement", "candidate")]
                + [(f"expired-{index:02d}", "preference", "expired") for index in range(15)]
                + [(f"rejected-{index:02d}", "preference", "rejected") for index in range(29)]
            )
            assert len(memory_rows) == 61
            occurred_at = "2026-09-04T00:00:00+00:00"
            for sequence in range(72):
                quote = f"quote-{sequence:02d}"
                connection.execute(
                    "INSERT INTO conversation_events "
                    "(event_id, conversation_id, sequence, direction, actor, kind, text, "
                    "message_segments_json, occurred_at_utc, received_at_utc, status, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"event-{sequence:02d}", "owner", sequence, "inbound", "mumo", "text",
                        quote, "[]", occurred_at, occurred_at, "received", "{}",
                    ),
                )
            for index, (memory_id, memory_type, status) in enumerate(memory_rows):
                connection.execute(
                    "INSERT INTO memory_records "
                    "(memory_id, type, normalized_fact, modality, status, valid_from_utc, "
                    "valid_until_utc, supersedes_id, created_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory_id, memory_type, f"fact-{index:02d}", "explicit_statement", status,
                        occurred_at, None, None, occurred_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_evidence "
                    "(memory_id, event_id, actor, exact_quote, occurred_at_utc) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (memory_id, f"event-{index:02d}", "mumo", f"quote-{index:02d}", occurred_at),
                )
            first_memory_id = memory_rows[0][0]
            for sequence in range(61, 72):
                connection.execute(
                    "INSERT INTO memory_evidence "
                    "(memory_id, event_id, actor, exact_quote, occurred_at_utc) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (first_memory_id, f"event-{sequence:02d}", "mumo", f"quote-{sequence:02d}", occurred_at),
                )
        connection.execute(
            "INSERT INTO runtime_meta (key, value_json, updated_at_utc) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc",
            ("schema_version", "2", "2026-09-04T00:00:00+00:00"),
        )
        connection.commit()
    finally:
        connection.close()


def test_schema_pragmas_version_and_required_columns(tmp_path):
    before_initialization = datetime.now(timezone.utc)
    database = Database(tmp_path / "qichi.sqlite3")
    after_initialization = datetime.now(timezone.utc)
    try:
        tables = {row[0] for row in database.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert EXPECTED_TABLES <= tables
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert database.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert database.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert database.connection.execute("SELECT value_json FROM runtime_meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        applied_at = database.connection.execute(
            "SELECT updated_at_utc FROM runtime_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        applied_at_datetime = datetime.fromisoformat(applied_at)
        assert applied_at_datetime.tzinfo is not None
        assert applied_at_datetime.utcoffset() is not None
        assert before_initialization <= applied_at_datetime <= after_initialization
        event_columns = {row[1] for row in database.connection.execute("PRAGMA table_info(conversation_events)")}
        assert {"event_id", "platform_event_id", "platform_message_id", "conversation_id", "sequence", "direction", "actor", "kind", "text", "message_segments_json", "reply_to_event_id", "reply_to_platform_message_id", "occurred_at_utc", "received_at_utc", "status", "metadata_json", "raw_payload_json"} <= event_columns
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(platform_message_map)")} >= {"platform_message_id", "event_id", "source", "created_at_utc"}
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(message_links)")} >= {"link_id", "source_event_id", "target_event_id", "relation", "status"}
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(runtime_meta)")} >= {"key", "value_json", "updated_at_utc"}
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(memory_records)")} >= {
            "certainty", "importance", "temporal_scope", "assessment_reason_code",
            "assessed_at_utc", "privacy_class", "recall_policy",
        }
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(memory_evidence)")} >= {
            "evidence_role",
        }
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(memory_session_jobs)")} >= {
            "job_id", "conversation_id", "revision", "fragment_key", "start_sequence",
            "end_sequence", "anchor_event_id", "anchor_sequence", "anchor_received_at_utc",
            "deadline_utc", "context_version", "status", "claim_token", "claim_owner",
            "claim_lease_until_utc", "next_retry_at_utc", "attempt_count",
            "failure_category", "created_at_utc", "updated_at_utc",
        }
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(memory_fragments)")} >= {
            "fragment_id", "conversation_id", "fragment_key", "start_sequence", "end_sequence",
            "start_event_id", "end_event_id", "started_at_utc", "ended_at_utc", "fragment_type",
            "reality_scope", "summary", "privacy_class", "recall_policy", "status", "closed_at_utc",
            "created_at_utc",
        }
        assert {row[1] for row in database.connection.execute("PRAGMA table_info(memory_detail_records)")} >= {
            "detail_id", "fragment_id", "ordinal", "detail_kind", "actor", "reality_scope",
            "normalized_detail", "exact_quote", "source_event_id", "occurred_at_utc", "certainty",
            "temporal_scope", "status", "privacy_class", "recall_policy",
        }
    finally:
        database.close()


def test_foreign_key_violation_is_rejected(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute("INSERT INTO platform_message_map (platform_message_id, event_id, source, created_at_utc) VALUES (?, ?, ?, ?)", ("missing-platform-message", "missing-event", "test", "2026-08-27T00:00:00+00:00"))
    finally:
        database.close()


def test_reopen_migration_is_idempotent(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    first = Database(path)
    first.close()
    second = Database(path)
    try:
        assert second.connection.execute("SELECT value_json FROM runtime_meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
    finally:
        second.close()


def test_memory_job_events_is_append_only(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        database.connection.execute(
            "INSERT INTO memory_job_events "
            "(job_event_id,job_id,conversation_id,revision,fragment_key,start_sequence,end_sequence,action,occurred_at_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("evt-1", "job-1", "owner", 1, "frag", 0, 0, "failed", "2026-09-07T00:00:00+00:00"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            database.connection.execute("UPDATE memory_job_events SET action='completed' WHERE job_event_id='evt-1'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            database.connection.execute("DELETE FROM memory_job_events WHERE job_event_id='evt-1'")
    finally:
        database.close()


def test_schema_four_migration_records_legacy_failed_job(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    create_version_two_database(path)
    connection = sqlite3.connect(path)
    try:
        root = Path(__file__).resolve().parents[1] / "migrations"
        connection.executescript((root / "0003_memory_v2.sql").read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,occurred_at_utc,received_at_utc,status,metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("anchor", "owner", 0, "inbound", "mumo", "text", "hello", "[]", "2026-09-07T00:00:00+00:00", "2026-09-07T00:00:00+00:00", "received", "{}"),
        )
        connection.execute(
            "INSERT INTO memory_session_jobs (job_id,conversation_id,revision,fragment_key,start_sequence,end_sequence,anchor_event_id,anchor_sequence,anchor_received_at_utc,deadline_utc,context_version,status,next_retry_at_utc,attempt_count,failure_category,created_at_utc,updated_at_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("job-failed", "owner", 3, "frag", 0, 0, "anchor", 0, "2026-09-07T00:00:00+00:00", "2026-09-07T00:30:00+00:00", 1, "failed", None, 4, "llm_error", "2026-09-07T00:00:00+00:00", "2026-09-07T00:10:00+00:00"),
        )
        connection.execute("UPDATE runtime_meta SET value_json='3' WHERE key='schema_version'")
        connection.commit()
    finally:
        connection.close()
    database = Database(path)
    try:
        row = database.connection.execute(
            "SELECT job_id,conversation_id,revision,action,failure_category,attempt_count FROM memory_job_events"
        ).fetchone()
        assert tuple(row) == ("job-failed", "owner", 3, "failed", "llm_error", 4)
        assert database.connection.execute("SELECT COUNT(*) FROM memory_job_events").fetchone()[0] == 1
    finally:
        database.close()


def test_empty_version_two_upgrade_uses_safe_defaults_and_database_checks(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    create_version_two_database(path)
    database = Database(path)
    try:
        database.connection.execute(
            "INSERT INTO memory_records "
            "(memory_id, type, normalized_fact, modality, status, valid_from_utc, "
            "valid_until_utc, supersedes_id, created_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "new-memory", "preference", "fact", "explicit_statement", "candidate",
                "2026-09-04T00:00:00+00:00", None, None, "2026-09-04T00:00:00+00:00",
            ),
        )
        row = database.connection.execute(
            "SELECT certainty, importance, temporal_scope, assessment_reason_code, "
            "assessed_at_utc FROM memory_records WHERE memory_id='new-memory'"
        ).fetchone()
        assert tuple(row) == ("unassessed", 0, "unclassified", None, None)
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "UPDATE memory_records SET certainty='certain' WHERE memory_id='new-memory'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "UPDATE memory_records SET importance=4 WHERE memory_id='new-memory'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "UPDATE memory_records SET temporal_scope='forever' WHERE memory_id='new-memory'"
            )
    finally:
        database.close()


def test_known_61_72_upgrade_grades_without_changing_legacy_fields(tmp_path, monkeypatch):
    path = tmp_path / "qichi.sqlite3"
    create_version_two_database(path, with_known_shape=True)
    inspection = sqlite3.connect(path)
    try:
        monkeypatch.setattr(
            migrations,
            "_KNOWN_MEMORY_V2_LEGACY_FINGERPRINT",
            migrations._memory_v2_legacy_fingerprint(inspection),
        )
        records_before = inspection.execute(
            "SELECT memory_id,type,normalized_fact,modality,status,valid_from_utc,"
            "valid_until_utc,supersedes_id,created_at_utc FROM memory_records ORDER BY memory_id"
        ).fetchall()
        evidence_before = inspection.execute(
            "SELECT memory_id,event_id,actor,exact_quote,occurred_at_utc "
            "FROM memory_evidence ORDER BY memory_id,event_id,exact_quote"
        ).fetchall()
    finally:
        inspection.close()

    database = Database(path)
    try:
        records_after = database.connection.execute(
            "SELECT memory_id,type,normalized_fact,modality,status,valid_from_utc,"
            "valid_until_utc,supersedes_id,created_at_utc FROM memory_records ORDER BY memory_id"
        ).fetchall()
        evidence_after = database.connection.execute(
            "SELECT memory_id,event_id,actor,exact_quote,occurred_at_utc "
            "FROM memory_evidence ORDER BY memory_id,event_id,exact_quote"
        ).fetchall()
        assert [tuple(row) for row in records_after] == records_before
        assert [tuple(row) for row in evidence_after] == evidence_before
        assert database.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0] == 61
        assert database.connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 72
        assert database.connection.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE evidence_role='source'"
        ).fetchone()[0] == 72
        assert database.connection.execute(
            "SELECT COUNT(*) FROM memory_audit_events WHERE assessment_reason_code='legacy_manual_review'"
        ).fetchone()[0] == 61
        assert database.connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE status='active' AND certainty='explicit' "
            "AND importance IN (2,3) AND temporal_scope IN ('ongoing','historical') "
            "AND assessment_reason_code='legacy_manual_review' AND assessed_at_utc IS NOT NULL"
        ).fetchone()[0] == 16
        assert {
            row[0] for row in database.connection.execute(
                "SELECT memory_id FROM memory_records WHERE status='active' AND importance=3"
            )
        } == I3_MEMORY_IDS
        assert tuple(database.connection.execute(
            "SELECT certainty,importance,temporal_scope,assessment_reason_code "
            "FROM memory_records WHERE memory_id=?",
            (CANDIDATE_MEMORY_ID,),
        ).fetchone()) == ("ambiguous", 2, "unclassified", "legacy_manual_review")
        assert database.connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE status='expired' AND certainty='explicit' "
            "AND importance=0 AND temporal_scope='unclassified'"
        ).fetchone()[0] == 15
        assert database.connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE status='rejected' AND certainty='unsupported' "
            "AND importance=0 AND temporal_scope='unclassified'"
        ).fetchone()[0] == 29
        assert database.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert database.connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        database.close()


def test_changed_known_legacy_data_rolls_back_schema_three(tmp_path, monkeypatch):
    path = tmp_path / "qichi.sqlite3"
    create_version_two_database(path, with_known_shape=True)
    connection = sqlite3.connect(path)
    try:
        monkeypatch.setattr(
            migrations,
            "_KNOWN_MEMORY_V2_LEGACY_FINGERPRINT",
            migrations._memory_v2_legacy_fingerprint(connection),
        )
        connection.execute(
            "UPDATE memory_records SET normalized_fact='changed' WHERE memory_id=?",
            (CANDIDATE_MEMORY_ID,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="unknown legacy memory data"):
        Database(path)

    inspection = sqlite3.connect(path)
    try:
        assert inspection.execute(
            "SELECT value_json FROM runtime_meta WHERE key='schema_version'"
        ).fetchone()[0] == "2"
        assert "certainty" not in {
            row[1] for row in inspection.execute("PRAGMA table_info(memory_records)")
        }
        assert inspection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            "AND name IN ('memory_audit_events','memory_session_jobs',"
            "'memory_confirmation_presentations')"
        ).fetchone()[0] == 0
    finally:
        inspection.close()


def test_version_one_upgrade_preserves_rows_and_survives_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    create_version_one_database(path)

    first = Database(path)
    try:
        assert first.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = 'schema_version'"
        ).fetchone()[0] == str(SCHEMA_VERSION)
        legacy = first.connection.execute(
            "SELECT event_id, platform_event_id, platform_message_id, conversation_id, "
            "sequence, text, status FROM conversation_events WHERE event_id = ?",
            ("legacy-event",),
        ).fetchone()
        assert tuple(legacy) == (
            "legacy-event", "legacy-platform-event", "10001", "owner", 0,
            "legacy text", "received",
        )
        assert tuple(first.connection.execute(
            "SELECT platform_message_id, event_id, source FROM platform_message_map"
        ).fetchone()) == ("10001", "legacy-event", "test")
        assert tuple(first.connection.execute(
            "SELECT conversation_id, context_version, last_processed_sequence, "
            "last_user_activity_utc, presence_topic_cursor FROM conversation_cursors"
        ).fetchone()) == (
            "owner", 3, 0, "2026-08-27T00:00:00+00:00", 0,
        )
        first.connection.execute(
            "INSERT INTO turn_trace_events "
            "(trace_event_id, trace_id, conversation_id, trigger_event_id, source, phase, "
            "occurred_at_utc, details_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-trace-event", "legacy-trace", "owner", "legacy-event",
                "dialogue", "received", "2026-08-27T00:00:01+00:00", "{}",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            first.connection.execute(
                "UPDATE turn_trace_events SET phase = 'failure' "
                "WHERE trace_event_id = 'legacy-trace-event'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            first.connection.execute(
                "DELETE FROM turn_trace_events WHERE trace_event_id = 'legacy-trace-event'"
            )
        assert first.connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        first.close()

    second = Database(path)
    try:
        assert second.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE event_id = 'legacy-event'"
        ).fetchone()[0] == 1
        assert second.connection.execute(
            "SELECT COUNT(*) FROM turn_trace_events WHERE trace_event_id = 'legacy-trace-event'"
        ).fetchone()[0] == 1
        assert second.connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        second.close()


def test_version_one_upgrade_rolls_back_schema_version_on_failure(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    create_version_one_database(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE turn_trace_events ("
            "trace_event_id TEXT, trace_id TEXT, conversation_id TEXT, "
            "trigger_event_id TEXT, source TEXT, phase TEXT, "
            "occurred_at_utc TEXT, details_json TEXT)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        Database(path)

    inspection = sqlite3.connect(path)
    try:
        assert inspection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "1"
        assert inspection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE event_id = 'legacy-event'"
        ).fetchone()[0] == 1
        assert {
            row[1] for row in inspection.execute("PRAGMA table_info(turn_trace_events)")
        } == {
            "trace_event_id", "trace_id", "conversation_id", "trigger_event_id",
            "source", "phase", "occurred_at_utc", "details_json",
        }
        assert inspection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('index', 'trigger') "
            "AND name IN ('ix_turn_trace_conversation_time', "
            "'immutable_turn_trace_update', 'immutable_turn_trace_delete')"
        ).fetchone()[0] == 0
    finally:
        inspection.close()


def test_concurrent_first_initialization_succeeds(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def open_database() -> None:
        database = None
        try:
            barrier.wait(timeout=5)
            database = Database(path)
            assert database.connection.execute("SELECT value_json FROM runtime_meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        except BaseException as error:
            errors.append(error)
        finally:
            if database is not None:
                database.close()

    threads = [threading.Thread(target=open_database) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors


def test_partial_initialization_recovers(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    connection = sqlite3.connect(path)
    try:
        migration = (Path(__file__).resolve().parents[1] / "migrations" / "0001_initial.sql").read_text(encoding="utf-8")
        connection.execute(migration.split(";", 1)[0])
        connection.commit()
    finally:
        connection.close()
    database = Database(path)
    try:
        tables = {row[0] for row in database.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert EXPECTED_TABLES <= tables
        assert database.connection.execute("SELECT value_json FROM runtime_meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
    finally:
        database.close()


def test_future_schema_version_is_rejected(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE runtime_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at_utc TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO runtime_meta (key, value_json, updated_at_utc) VALUES (?, ?, ?)",
            ("schema_version", str(SCHEMA_VERSION + 1), "2026-08-27T00:00:00+00:00"),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RuntimeError, match="newer"):
        Database(path)
