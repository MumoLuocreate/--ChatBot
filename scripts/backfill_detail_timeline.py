"""Backfill v6 fragment indexes and detail timelines from a staged replay copy.

The frozen contract requires history to be replayed in isolation first and only
then written to production as a separate, auditable task.  This script is that
task's writer: it copies the staged fragment, fragment-event, detail and
detail-evidence rows for a sequence range, but only after re-verifying every one
of them against the production event ledger.

Dry run by default.  With --apply it first takes a consistent backup (the v6
tables are append-only, so restoring that backup is the only way back) and then
inserts with ON CONFLICT DO NOTHING, which keeps a repeated run harmless because
fragment and detail identifiers are derived rather than random.

Usage:
    python scripts/backfill_detail_timeline.py --stage <copy.sqlite3> --from-sequence 3397 --to-sequence 3615
    python scripts/backfill_detail_timeline.py --stage <copy.sqlite3> --from-sequence 3397 --to-sequence 3615 --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "qichi.sqlite3"
TABLES = ("memory_fragments", "memory_fragment_events", "memory_detail_records", "memory_detail_evidence")


def _ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _quote_is_verbatim(text: str | None, quote: str) -> bool:
    return bool(text) and quote in text


def verify(stage: sqlite3.Connection, production: sqlite3.Connection, low: int, high: int) -> dict:
    fragments = stage.execute(
        "SELECT fragment_id, conversation_id, start_sequence, end_sequence FROM memory_fragments "
        "WHERE start_sequence >= ? AND end_sequence <= ? ORDER BY start_sequence",
        (low, high),
    ).fetchall()
    problems: list[str] = []
    events_checked = details_checked = 0
    for fragment in fragments:
        links = stage.execute(
            "SELECT event_id FROM memory_fragment_events WHERE fragment_id = ?", (fragment["fragment_id"],)
        ).fetchall()
        for link in links:
            row = production.execute(
                "SELECT conversation_id, sequence FROM conversation_events WHERE event_id = ?", (link["event_id"],)
            ).fetchone()
            events_checked += 1
            if row is None:
                problems.append(f"event missing in production: {link['event_id']}")
            elif row["conversation_id"] != fragment["conversation_id"]:
                # The fragment's own numbering is not the event ledger's, so the
                # meaningful checks are that the event exists and belongs to the
                # same conversation as the fragment.
                problems.append(f"event belongs to another conversation: {link['event_id']}")
        for detail in stage.execute(
            "SELECT detail_id, source_event_id, exact_quote FROM memory_detail_records WHERE fragment_id = ?",
            (fragment["fragment_id"],),
        ).fetchall():
            details_checked += 1
            row = production.execute(
                "SELECT text FROM conversation_events WHERE event_id = ?", (detail["source_event_id"],)
            ).fetchone()
            if row is None:
                problems.append(f"detail source event missing: {detail['source_event_id']}")
            elif not _quote_is_verbatim(row["text"], detail["exact_quote"]):
                problems.append(f"quote is not verbatim for detail {detail['detail_id']}")
            for evidence in stage.execute(
                "SELECT event_id FROM memory_detail_evidence WHERE detail_id = ?", (detail["detail_id"],)
            ).fetchall():
                if production.execute(
                    "SELECT 1 FROM conversation_events WHERE event_id = ?", (evidence["event_id"],)
                ).fetchone() is None:
                    problems.append(f"detail evidence missing: {evidence['event_id']}")
    return {
        "fragments": len(fragments), "events": events_checked, "details": details_checked,
        "problems": problems,
    }


def apply_backfill(production: sqlite3.Connection, stage_path: Path, low: int, high: int) -> dict:
    production.execute("ATTACH DATABASE ? AS stage", (str(stage_path),))
    try:
        before = {table: production.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0] for table in TABLES}
        production.execute("BEGIN IMMEDIATE")
        try:
            production.execute(
                "INSERT INTO main.memory_fragments SELECT * FROM stage.memory_fragments "
                "WHERE start_sequence >= ? AND end_sequence <= ? AND true "
                "ON CONFLICT(fragment_id) DO NOTHING",
                (low, high),
            )
            production.execute(
                "INSERT INTO main.memory_fragment_events SELECT * FROM stage.memory_fragment_events "
                "WHERE fragment_id IN (SELECT fragment_id FROM main.memory_fragments "
                "WHERE start_sequence >= ? AND end_sequence <= ?) AND true "
                "ON CONFLICT DO NOTHING",
                (low, high),
            )
            production.execute(
                "INSERT INTO main.memory_detail_records SELECT * FROM stage.memory_detail_records "
                "WHERE fragment_id IN (SELECT fragment_id FROM main.memory_fragments "
                "WHERE start_sequence >= ? AND end_sequence <= ?) AND true "
                "ON CONFLICT(detail_id) DO NOTHING",
                (low, high),
            )
            production.execute(
                "INSERT INTO main.memory_detail_evidence SELECT * FROM stage.memory_detail_evidence "
                "WHERE detail_id IN (SELECT detail_id FROM main.memory_detail_records) AND true "
                "ON CONFLICT DO NOTHING",
            )
            production.execute("COMMIT")
        except BaseException:
            production.execute("ROLLBACK")
            raise
        after = {table: production.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0] for table in TABLES}
        return {"before": before, "after": after}
    finally:
        production.execute("DETACH DATABASE stage")


def backup(production_path: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "qichi-before-backfill.sqlite3"
    source = _ro(production_path)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--from-sequence", type=int, required=True)
    parser.add_argument("--to-sequence", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", default=None)
    arguments = parser.parse_args(argv)
    stage_path, database_path = Path(arguments.stage), Path(arguments.database)
    if not stage_path.exists() or not database_path.exists():
        print("BLOCKED: stage or database path does not exist")
        return 2
    stage, production = _ro(stage_path), sqlite3.connect(database_path, timeout=30, isolation_level=None)
    production.row_factory = sqlite3.Row
    try:
        production.execute("PRAGMA foreign_keys = ON")
        version = production.execute(
            "SELECT value_json FROM runtime_meta WHERE key = 'schema_version'"
        ).fetchone()
        if version is None or int(version[0]) < 6:
            print("BLOCKED: production is not on schema 6")
            return 2
        report = verify(stage, production, arguments.from_sequence, arguments.to_sequence)
        print(f"staged range {arguments.from_sequence}-{arguments.to_sequence}: "
              f"{report['fragments']} fragment(s), {report['events']} event link(s), {report['details']} detail(s)")
        if report["problems"]:
            print(f"BLOCKED: {len(report['problems'])} verification problem(s); nothing written")
            for problem in report["problems"][:10]:
                print("   ", problem)
            return 3
        print("verification: every event link, detail source and quote checked against the production ledger")
        if not arguments.apply:
            print("dry run: nothing written (pass --apply to write)")
            return 0
        directory = Path(arguments.backup_dir) if arguments.backup_dir else database_path.parent.parent / "_backups" / (
            "backfill-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        snapshot = backup(database_path, directory)
        print(f"backup written: {snapshot}")
        result = apply_backfill(production, stage_path, arguments.from_sequence, arguments.to_sequence)
        for table in TABLES:
            print(f"   {table}: {result['before'][table]} -> {result['after'][table]}")
        print("backfill applied; reversing it means restoring the backup, because these tables are append-only")
        return 0
    finally:
        production.close()
        stage.close()


if __name__ == "__main__":
    sys.exit(main())
