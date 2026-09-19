"""Apply the approved P4 privacy review to a database.

Dry run by default: nothing is written unless --apply is passed.  The script
refuses to touch a database below schema 6, and it writes a reversal file with
the previous values before it changes a single row, so the edit can be undone
exactly.  It only updates rows whose current values still match what was
reviewed; anything unexpected is reported and skipped rather than forced.

Usage (deployment window only):
    python scripts/apply_privacy_review.py                 # dry run
    python scripts/apply_privacy_review.py --apply         # writes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "qichi.sqlite3"
MINIMUM_SCHEMA = 6

# Reviewed and approved by the user on 2026-09-11.  Keys are memory_id prefixes.
REVIEW_PLAN = {
    "04e2296c": ("intimate", "topic_only", "relationship boundary and companionship agreement"),
    "8bac2494": ("intimate", "topic_only", "relationship confirmation agreement"),
    "8bb97958": ("adult", "topic_only", "intimacy arrangement, bounded and already expired"),
    "b7aca90e": ("intimate", "topic_only", "shared history and the no-fabrication line"),
    "de392c29": ("intimate", "topic_only", "do not rush him to sleep or dismiss him"),
    "13fc9f1b": ("intimate", "topic_only", "the 09-05 argument and who called the stop"),
    "1d7d10af": ("intimate", "topic_only", "pet-play scene and the stop"),
    "a968850a": ("adult", "explicit_request_only", "intimacy episode"),
    "cf86ef3c": ("adult", "explicit_request_only", "adult roleplay history"),
    "e76bc6df": ("intimate", "topic_only", "pre-reset lovers relationship, historical"),
    "31475359": ("intimate", "topic_only", "do not snipe at him, especially when close"),
    "68100e2f": ("adult", "explicit_request_only", "contrast and submission preference"),
    "6a58b59f": ("adult", "explicit_request_only", "dominant preference in shared imagination"),
    "b2827dc5": ("intimate", "topic_only", "treats Qichi as a partner"),
    "7e77ed72": ("adult", "explicit_request_only", "ambiguous promise about desire, stays candidate"),
    "5b1b9501": ("adult", "topic_only", "consented to a second intimacy"),
    "67d2ce02": ("intimate", "topic_only", "will not take advantage while he is sleepy"),
    "b6eb2dc5": ("intimate", "topic_only", "affectionate contact and returned kisses"),
    "e606bb9a": ("intimate", "topic_only", "affectionate contact and returned kisses"),
    "11e554b2": ("intimate", "topic_only", "wants him, of her own accord"),
}


def schema_version(database: Path) -> int:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        row = connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = 'schema_version'"
        ).fetchone()
    finally:
        connection.close()
    return int(row[0]) if row is not None else 0


def read_state(connection: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    rows = connection.execute(
        "SELECT memory_id, privacy_class, recall_policy FROM memory_records"
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def build_changes(state: dict[str, tuple[str, str]]) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for prefix in sorted(REVIEW_PLAN):
        privacy, policy, reason = REVIEW_PLAN[prefix]
        matches = sorted(memory_id for memory_id in state if memory_id.startswith(prefix))
        if not matches:
            changes.append({"prefix": prefix, "status": "absent", "reason": reason})
            continue
        for memory_id in matches:
            current = state[memory_id]
            target = (privacy, policy)
            if current == target:
                status = "already"
            elif current == ("ordinary", "daily_safe"):
                status = "update"
            else:
                status = "unexpected"
            changes.append({
                "prefix": prefix, "memory_id": memory_id, "status": status,
                "before": list(current), "after": list(target), "reason": reason,
            })
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--reversal", default=None, help="where to write the reversal file")
    arguments = parser.parse_args(argv)
    database = Path(arguments.database)
    if not database.exists():
        print(f"BLOCKED: database not found: {database}")
        raise SystemExit(2)
    version = schema_version(database)
    if version < MINIMUM_SCHEMA:
        print(f"BLOCKED: schema {version} is below {MINIMUM_SCHEMA}; migrate first")
        raise SystemExit(2)

    connection = sqlite3.connect(database, timeout=30, isolation_level=None)
    try:
        changes = build_changes(read_state(connection))
        updates = [item for item in changes if item.get("status") == "update"]
        print(f"reviewed records: {len(REVIEW_PLAN)} | matched: {len(changes)} | to update: {len(updates)}")
        for item in changes:
            print("  ", item.get("status"), item.get("prefix"), item.get("before", "-"), "->", item.get("after", "-"))
        odd = [item for item in changes if item.get("status") in {"absent", "unexpected"}]
        if odd:
            print(f"note: {len(odd)} entr(ies) need a human look before applying")
        if not arguments.apply:
            print("dry run: nothing written (pass --apply to write)")
            return 0
        if not updates:
            print("nothing to apply")
            return 0
        reversal = Path(arguments.reversal) if arguments.reversal else database.with_suffix(".privacy-reversal.json")
        reversal.write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "database": str(database),
            "previous": {item["memory_id"]: item["before"] for item in updates},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"reversal written: {reversal}")
        with connection:
            for item in updates:
                connection.execute(
                    "UPDATE memory_records SET privacy_class = ?, recall_policy = ? WHERE memory_id = ?",
                    (item["after"][0], item["after"][1], item["memory_id"]),
                )
        print(f"applied {len(updates)} update(s)")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
