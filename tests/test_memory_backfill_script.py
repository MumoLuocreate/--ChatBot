from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backfill_memory_records as backfill  # noqa: E402

from qichi.domain.events import ConversationEvent, MessageSegment  # noqa: E402
from qichi.domain.memory import MemoryEvidence, MemoryRecord  # noqa: E402
from qichi.storage.database import Database  # noqa: E402
from qichi.storage.event_repository import EventRepository  # noqa: E402
from qichi.storage.memory_repository import MemoryRepository  # noqa: E402

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
QUOTE = "我喜欢雨声"


def event(event_id: str, sequence: int, text: str) -> ConversationEvent:
    return ConversationEvent(
        event_id, None, "pm-" + event_id, "10001", sequence, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    )


def stage_memory(stage: Database, memory_id: str, source: ConversationEvent, *, privacy="ordinary", policy="daily_safe"):
    repository = MemoryRepository(stage)
    evidence = MemoryEvidence(memory_id, source.event_id, source.actor, source.text, source.occurred_at_utc, "source")
    return repository.create(MemoryRecord(
        memory_id, "preference", "喜欢雨声", "explicit_statement", "active", source.occurred_at_utc,
        None, None, source.received_at_utc, (evidence,), "explicit", 2, "ongoing",
        "explicit_user_statement", source.received_at_utc, privacy_class=privacy, recall_policy=policy,
    ))


def prepare(tmp_path):
    production, stage = Database(tmp_path / "production.sqlite3"), Database(tmp_path / "stage.sqlite3")
    source = event("e1", 1, QUOTE)
    EventRepository(production).insert(source)
    EventRepository(stage).insert(source)
    stage_memory(stage, "memory-new", source)
    return production, stage, source


def counts(database: Database) -> tuple[int, int]:
    return (database.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0],
            database.connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])


def test_dry_run_then_apply_then_repeat(tmp_path, capsys):
    production, stage, _ = prepare(tmp_path)
    try:
        assert backfill.main(["--stage", str(stage.path), "--database", str(production.path)]) == 0
        assert "dry run" in capsys.readouterr().out
        assert counts(production) == (0, 0)

        arguments = ["--stage", str(stage.path), "--database", str(production.path), "--apply",
                     "--backup-dir", str(tmp_path / "backup")]
        assert backfill.main(arguments) == 0
        assert counts(production) == (1, 1)
        assert (tmp_path / "backup" / "qichi-before-memory-backfill.sqlite3").exists()

        assert backfill.main(arguments) == 0
        assert counts(production) == (1, 1), "a repeated run must not duplicate anything"
    finally:
        production.close()
        stage.close()


def test_a_memory_production_already_has_is_left_alone(tmp_path, capsys):
    production, stage, source = prepare(tmp_path)
    try:
        # The same identifier exists in production with different content: the
        # backfill must skip it rather than overwrite what production holds.
        stage_memory(production, "memory-new", source)
        production.connection.execute(
            "UPDATE memory_records SET normalized_fact = '生产里的版本' WHERE memory_id = 'memory-new'"
        )
        production.connection.commit()
        assert backfill.main(["--stage", str(stage.path), "--database", str(production.path), "--apply",
                              "--backup-dir", str(tmp_path / "backup")]) == 0
        assert "nothing to apply" in capsys.readouterr().out
        assert production.connection.execute(
            "SELECT normalized_fact FROM memory_records WHERE memory_id = 'memory-new'"
        ).fetchone()[0] == "生产里的版本"
    finally:
        production.close()
        stage.close()


def test_evidence_production_never_saw_blocks_the_write(tmp_path, capsys):
    production, stage, _ = prepare(tmp_path)
    try:
        ghost = event("ghost", 2, "只在暂存里")
        EventRepository(stage).insert(ghost)
        stage_memory(stage, "memory-ghost", ghost)
        assert backfill.main(["--stage", str(stage.path), "--database", str(production.path), "--apply",
                              "--backup-dir", str(tmp_path / "backup")]) == 3
        output = capsys.readouterr().out
        assert "verification problem" in output and "evidence event missing" in output
        assert counts(production) == (0, 0)
    finally:
        production.close()
        stage.close()
