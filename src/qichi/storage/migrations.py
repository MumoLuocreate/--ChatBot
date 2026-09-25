from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 7

_KNOWN_MEMORY_V2_LEGACY_FINGERPRINT = (
    "5e91ace57ccfd29f6757afecc35502c88e89a7cb77d85bff70fbdb6b19da259c"
)
_MEMORY_V2_LEGACY_QUERIES = (
    (
        "memory_records",
        "SELECT memory_id,type,normalized_fact,modality,status,valid_from_utc,"
        "valid_until_utc,supersedes_id,created_at_utc "
        "FROM memory_records ORDER BY memory_id",
    ),
    (
        "memory_evidence",
        "SELECT memory_id,event_id,actor,exact_quote,occurred_at_utc "
        "FROM memory_evidence ORDER BY memory_id,event_id,exact_quote",
    ),
)


def _memory_v2_legacy_fingerprint(connection: sqlite3.Connection) -> str:
    payload = [
        [name, [list(row) for row in connection.execute(query)]]
        for name, query in _MEMORY_V2_LEGACY_QUERIES
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_memory_v2_legacy_data(connection: sqlite3.Connection) -> None:
    memory_count = connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    evidence_count = connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0]
    if memory_count == 0 and evidence_count == 0:
        return
    if (
        memory_count != 61
        or evidence_count != 72
        or _memory_v2_legacy_fingerprint(connection)
        != _KNOWN_MEMORY_V2_LEGACY_FINGERPRINT
    ):
        raise RuntimeError(
            "schema v3 refuses to grade unknown legacy memory data"
        )


def _execute_script(connection: sqlite3.Connection, sql: str) -> None:
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""
    if statement.strip():
        raise ValueError("migration SQL ends with an incomplete statement")


def migrate(connection: sqlite3.Connection) -> None:
    migration_root = Path(__file__).resolve().parents[3] / "migrations"
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS runtime_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at_utc TEXT NOT NULL)")
        row = connection.execute("SELECT value_json FROM runtime_meta WHERE key = ?", ("schema_version",)).fetchone()
        version = int(row[0]) if row is not None else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database schema version {version} is newer than supported")
        for target_version in range(version + 1, SCHEMA_VERSION + 1):
            matches = sorted(migration_root.glob(f"{target_version:04d}_*.sql"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"database migration {target_version} is missing or ambiguous"
                )
            if target_version == 3:
                _verify_memory_v2_legacy_data(connection)
            _execute_script(connection, matches[0].read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO runtime_meta (key, value_json, updated_at_utc) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at_utc = excluded.updated_at_utc",
                ("schema_version", str(target_version), datetime.now(timezone.utc).isoformat()),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
