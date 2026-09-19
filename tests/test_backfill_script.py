from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backfill_detail_timeline as backfill  # noqa: E402

from qichi.domain.events import ConversationEvent, MessageSegment  # noqa: E402
from qichi.domain.memory_details import MemoryDetailDraft  # noqa: E402
from qichi.storage.database import Database  # noqa: E402
from qichi.storage.event_repository import EventRepository  # noqa: E402
from qichi.storage.memory_detail_repository import MemoryDetailRepository  # noqa: E402

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
TABLES = ("memory_fragments", "memory_fragment_events", "memory_detail_records", "memory_detail_evidence")


def event(event_id: str, sequence: int, text: str) -> ConversationEvent:
    return ConversationEvent(
        event_id, None, "pm-" + event_id, "10001", sequence, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    )


def stage_fragment(stage: Database, events: tuple[ConversationEvent, ...]) -> None:
    repository = MemoryDetailRepository(stage)
    fragment = repository.build_fragment(events, "key-1", None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="细节", exact_quote=item.text or "", source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class="ordinary", recall_policy="daily_safe",
            evidence=((item.event_id, "source"),),
        )
        for index, item in enumerate(events)
    )
    details = repository.build_details(fragment, drafts, events)
    with stage.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=events, details=details)


def test_replaying_a_session_keeps_the_stored_fragment_metadata(tmp_path):
    """重放同一段会话时，重新推导的 type/summary 不该让整段落库失败。"""

    database = Database(tmp_path / "replay.sqlite3")
    events = (event("e1", 1, "第一条"),)
    for item in events:
        EventRepository(database).insert(item)
    stage_fragment(database, events)
    stored = database.connection.execute(
        "SELECT fragment_type,summary FROM memory_fragments WHERE fragment_key='key-1'"
    ).fetchone()

    repository = MemoryDetailRepository(database)
    replayed = replace(
        repository.build_fragment(events, "key-1", None, created_at_utc=NOW),
        fragment_type="mixed",
        summary="重放时推导出的另一句话",
    )
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=replayed, events=events, details=())

    after = database.connection.execute(
        "SELECT fragment_type,summary FROM memory_fragments WHERE fragment_key='key-1'"
    ).fetchone()
    assert tuple(after) == tuple(stored), "首写优先，重放不改已存的分类与摘要"
    database.close()


def test_a_fragment_whose_span_changed_is_still_a_conflict(tmp_path):
    database = Database(tmp_path / "conflict.sqlite3")
    first, second = event("e1", 1, "第一条"), event("e2", 2, "第二条")
    for item in (first, second):
        EventRepository(database).insert(item)
    stage_fragment(database, (first,))
    repository = MemoryDetailRepository(database)
    # Same conversation and key, wider span: the identifier is derived, so this is
    # the same fragment identity covering different evidence.
    wider = repository.build_fragment((first, second), "key-1", None, created_at_utc=NOW)

    with pytest.raises(ValueError, match="fragment identity conflict"):
        with database.transaction() as connection:
            repository.store_in_transaction(connection, fragment=wider, events=(first, second), details=())
    database.close()


def prepare(tmp_path):
    production = Database(tmp_path / "production.sqlite3")
    stage = Database(tmp_path / "stage.sqlite3")
    first, second = event("e1", 1, "第一条"), event("e2", 2, "第二条")
    for item in (first, second):
        EventRepository(production).insert(item)
        # The stage fragment carries foreign keys into its own event ledger, so
        # the same events have to exist there as well.
        EventRepository(stage).insert(item)
    stage_fragment(stage, (first, second))
    return production, stage


def counts(database: Database) -> dict:
    return {table: database.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}


def test_dry_run_writes_nothing_and_apply_is_idempotent(tmp_path, capsys):
    production, stage = prepare(tmp_path)
    try:
        assert backfill.main(["--stage", str(stage.path), "--database", str(production.path),
                              "--from-sequence", "1", "--to-sequence", "2"]) == 0
        assert "dry run" in capsys.readouterr().out
        assert counts(production) == {table: 0 for table in TABLES}

        arguments = ["--stage", str(stage.path), "--database", str(production.path),
                     "--from-sequence", "1", "--to-sequence", "2", "--apply",
                     "--backup-dir", str(tmp_path / "backup")]
        assert backfill.main(arguments) == 0
        written = counts(production)
        assert written == {"memory_fragments": 1, "memory_fragment_events": 2,
                           "memory_detail_records": 2, "memory_detail_evidence": 2}
        assert (tmp_path / "backup" / "qichi-before-backfill.sqlite3").exists()

        assert backfill.main(arguments) == 0
        assert counts(production) == written, "a repeated run must not duplicate anything"
    finally:
        production.close()
        stage.close()


def test_a_link_to_an_event_production_does_not_have_blocks_the_write(tmp_path, capsys):
    production, stage = prepare(tmp_path)
    try:
        # An event that exists in the stage ledger (so the stage keeps its own
        # foreign keys happy) but never reached production.
        ghost = event("ghost", 2, "只在暂存里")
        EventRepository(stage).insert(ghost)
        columns = [row[1] for row in stage.connection.execute("PRAGMA table_info(memory_fragment_events)")]
        row = list(stage.connection.execute("SELECT * FROM memory_fragment_events LIMIT 1").fetchone())
        row[columns.index("event_id")] = "ghost"
        row[columns.index("ordinal")] = 99
        stage.connection.execute(
            f"INSERT INTO memory_fragment_events VALUES ({','.join('?' * len(row))})", row
        )
        stage.connection.commit()
        assert backfill.main(["--stage", str(stage.path), "--database", str(production.path),
                              "--from-sequence", "1", "--to-sequence", "2", "--apply",
                              "--backup-dir", str(tmp_path / "backup")]) == 3
        output = capsys.readouterr().out
        assert "verification problem" in output and "ghost" in output
        assert counts(production) == {table: 0 for table in TABLES}
    finally:
        production.close()
        stage.close()


def test_a_pre_v6_production_database_is_refused(tmp_path, capsys):
    production, stage = prepare(tmp_path)
    try:
        production.connection.execute("UPDATE runtime_meta SET value_json='5' WHERE key='schema_version'")
        production.connection.commit()
        assert backfill.main(["--stage", str(stage.path), "--database", str(production.path),
                              "--from-sequence", "1", "--to-sequence", "2", "--apply"]) == 2
        assert "not on schema 6" in capsys.readouterr().out
    finally:
        production.close()
        stage.close()
