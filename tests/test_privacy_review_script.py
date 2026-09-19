from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import apply_privacy_review as review  # noqa: E402

POLICIES = {"daily_safe", "topic_only", "explicit_request_only"}
PRIVACY = {"ordinary", "intimate", "adult"}


def _database(path: Path, schema: int, rows: list[tuple[str, str, str]]) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE runtime_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at_utc TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE memory_records (memory_id TEXT PRIMARY KEY, privacy_class TEXT NOT NULL, recall_policy TEXT NOT NULL)"
    )
    connection.execute("INSERT INTO runtime_meta VALUES ('schema_version', ?, 'x')", (str(schema),))
    connection.executemany("INSERT INTO memory_records VALUES (?, ?, ?)", rows)
    connection.commit()
    connection.close()
    return path


def test_the_plan_matches_the_reviewed_records():
    assert len(review.REVIEW_PLAN) == 20
    for prefix, (privacy, policy, reason) in review.REVIEW_PLAN.items():
        assert len(prefix) == 8 and prefix.isalnum()
        assert privacy in PRIVACY and policy in POLICIES
        assert privacy != "adult" or policy != "daily_safe"
        assert reason


def test_a_pre_v6_database_is_refused(tmp_path, capsys):
    database = _database(tmp_path / "v5.sqlite3", 5, [("04e2296c-0000-0000-0000-000000000000", "ordinary", "daily_safe")])
    with pytest.raises(SystemExit):
        review.main(["--database", str(database), "--apply"])
    assert "below 6" in capsys.readouterr().out


def test_dry_run_reports_without_writing(tmp_path, capsys):
    memory_id = "04e2296c-0000-0000-0000-000000000000"
    database = _database(tmp_path / "v6.sqlite3", 6, [(memory_id, "ordinary", "daily_safe")])
    assert review.main(["--database", str(database)]) == 0
    assert "dry run" in capsys.readouterr().out
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT privacy_class, recall_policy FROM memory_records").fetchone() == ("ordinary", "daily_safe")
    finally:
        connection.close()


def test_apply_writes_the_change_and_a_reversal_file(tmp_path):
    memory_id = "04e2296c-0000-0000-0000-000000000000"
    database = _database(tmp_path / "apply.sqlite3", 6, [(memory_id, "ordinary", "daily_safe")])
    reversal = tmp_path / "reversal.json"
    assert review.main(["--database", str(database), "--apply", "--reversal", str(reversal)]) == 0
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT privacy_class, recall_policy FROM memory_records").fetchone() == ("intimate", "topic_only")
    finally:
        connection.close()
    assert memory_id in reversal.read_text(encoding="utf-8")


def test_an_unexpected_current_value_is_reported_not_forced(tmp_path, capsys):
    memory_id = "04e2296c-0000-0000-0000-000000000000"
    database = _database(tmp_path / "odd.sqlite3", 6, [(memory_id, "intimate", "explicit_request_only")])
    assert review.main(["--database", str(database)]) == 0
    output = capsys.readouterr().out
    assert "unexpected" in output and "need a human look" in output
