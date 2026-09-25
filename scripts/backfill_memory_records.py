"""Backfill memory records that a staged replay produced but production lacks.

The frozen contract forbids sneaking history in through the daily worker, so the
staged replay is the source of truth and this script is the writer.  It copies
only records that production does not already have, in the statuses the replay
produced, together with their evidence rows -- and only after re-verifying every
piece of evidence against the production event ledger: the event must exist, the
actor must match and the quoted words must appear verbatim in that event.

It never touches an existing memory.  A record that already exists in production
is skipped, which also means a replay's incidental changes to old records (a
status promotion, for instance) are deliberately not applied.

Dry run by default.  With --apply it backs up first; the reversal is restoring
that backup, because memory records are append-only in practice.

Usage:
    python scripts/backfill_memory_records.py --stage <copy.sqlite3>
    python scripts/backfill_memory_records.py --stage <copy.sqlite3> --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "qichi.sqlite3"
IMPORTABLE_STATUSES = {"active", "candidate"}


def _ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def verify(stage: sqlite3.Connection, production: sqlite3.Connection) -> dict:
    known = {row[0] for row in production.execute("SELECT memory_id FROM memory_records")}
    candidates = stage.execute("SELECT * FROM memory_records ORDER BY memory_id").fetchall()
    fresh, skipped, problems = [], 0, []
    for record in candidates:
        if record["memory_id"] in known:
            skipped += 1
            continue
        if record["status"] not in IMPORTABLE_STATUSES:
            skipped += 1
            continue
        evidence = stage.execute(
            "SELECT * FROM memory_evidence WHERE memory_id = ? ORDER BY rowid", (record["memory_id"],)
        ).fetchall()
        if not evidence:
            problems.append(f"{record['memory_id'][:8]} has no evidence")
            continue
        for item in evidence:
            row = production.execute(
                "SELECT actor, text FROM conversation_events WHERE event_id = ?", (item["event_id"],)
            ).fetchone()
            if row is None:
                problems.append(f"{record['memory_id'][:8]} evidence event missing: {item['event_id']}")
            elif row["actor"] != item["actor"]:
                problems.append(f"{record['memory_id'][:8]} evidence actor mismatch")
            elif not row["text"] or item["exact_quote"] not in row["text"]:
                problems.append(f"{record['memory_id'][:8]} quote is not verbatim")
        if record["privacy_class"] == "adult" and record["recall_policy"] == "daily_safe":
            problems.append(f"{record['memory_id'][:8]} is adult with daily_safe")
        fresh.append(record["memory_id"])
    return {"fresh": fresh, "skipped": skipped, "problems": problems, "total": len(candidates)}


def apply_backfill(production: sqlite3.Connection, stage_path: Path, memory_ids: list[str]) -> dict:
    production.execute("ATTACH DATABASE ? AS stage", (str(stage_path),))
    try:
        before = {table: production.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
                  for table in ("memory_records", "memory_evidence")}
        production.execute("BEGIN IMMEDIATE")
        try:
            for memory_id in memory_ids:
                production.execute(
                    "INSERT INTO main.memory_records SELECT * FROM stage.memory_records "
                    "WHERE memory_id = ? AND true ON CONFLICT(memory_id) DO NOTHING",
                    (memory_id,),
                )
                production.execute(
                    "INSERT INTO main.memory_evidence SELECT * FROM stage.memory_evidence "
                    "WHERE memory_id = ? AND true ON CONFLICT DO NOTHING",
                    (memory_id,),
                )
            production.execute("COMMIT")
        except BaseException:
            production.execute("ROLLBACK")
            raise
        after = {table: production.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
                 for table in ("memory_records", "memory_evidence")}
        return {"before": before, "after": after}
    finally:
        production.execute("DETACH DATABASE stage")


def backup(production_path: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "qichi-before-memory-backfill.sqlite3"
    source, destination = _ro(production_path), sqlite3.connect(target)
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
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", default=None)
    arguments = parser.parse_args(argv)
    stage_path, database_path = Path(arguments.stage), Path(arguments.database)
    if not stage_path.exists() or not database_path.exists():
        print("BLOCKED: stage or database path does not exist")
        return 2
    stage = _ro(stage_path)
    production = sqlite3.connect(database_path, timeout=30, isolation_level=None)
    production.row_factory = sqlite3.Row
    try:
        production.execute("PRAGMA foreign_keys = ON")
        version = production.execute("SELECT value_json FROM runtime_meta WHERE key = 'schema_version'").fetchone()
        if version is None or int(version[0]) < 6:
            print("BLOCKED: production is not on schema 6")
            return 2
        report = verify(stage, production)
        print(f"staged records: {report['total']} | already present or not importable: {report['skipped']} "
              f"| to write: {len(report['fresh'])}")
        for memory_id in report["fresh"]:
            row = stage.execute(
                "SELECT type, status, certainty, temporal_scope, privacy_class, recall_policy, substr(normalized_fact,1,44) fact "
                "FROM memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            print(f"   {memory_id[:8]} {row['status']}/{row['type']}/{row['certainty']}/{row['temporal_scope']} "
                  f"{row['privacy_class']}/{row['recall_policy']} | {row['fact']}")
        if report["problems"]:
            print(f"BLOCKED: {len(report['problems'])} verification problem(s); nothing written")
            for problem in report["problems"][:8]:
                print("   ", problem)
            return 3
        print("verification: every evidence event exists, actor matches and the quote is verbatim")
        if not arguments.apply:
            print("dry run: nothing written (pass --apply to write)")
            return 0
        if not report["fresh"]:
            print("nothing to apply")
            return 0
        directory = Path(arguments.backup_dir) if arguments.backup_dir else database_path.parent.parent / "_backups" / (
            "memory-backfill-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"backup written: {backup(database_path, directory)}")
        result = apply_backfill(production, stage_path, report["fresh"])
        for table in ("memory_records", "memory_evidence"):
            print(f"   {table}: {result['before'][table]} -> {result['after'][table]}")
        print("memory backfill applied; existing records were not touched")
        return 0
    finally:
        production.close()
        stage.close()


if __name__ == "__main__":
    sys.exit(main())
