"""人工恢复一条被移出的记忆（卡B③，2026-09-22 用户裁定 R1：移出必须可逆）。

模型不能撤销自己的判决（allowed_review_actions 里没有 restore），所以这条路径**只由人走**。
恢复会把记录还原到**移除前**的状态与评级（不是只翻状态），证据一个字不动。

用法：
    python scripts/memory_restore.py --list
    python scripts/memory_restore.py --memory-id <id>            # 预演，不写
    python scripts/memory_restore.py --memory-id <id> --apply    # 真恢复（先自动备份）

注意（运维顺序）：本脚本会打开数据库，而打开即会**执行迁移**。如果生产库还是 schema 6
而线上还跑着旧代码，迁移到 7 之后**运行中的旧进程会报 schema 不匹配**（面板健康转红），
直到重启换上新代码为止。干净的做法是先用新代码重启（重启时完成迁移），再需要时跑本脚本。
因此这里在迁移前会明确警告并要求 --apply。
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qichi.storage.database import Database  # noqa: E402
from qichi.storage.memory_repository import MemoryRepository  # noqa: E402
from qichi.storage.migrations import SCHEMA_VERSION  # noqa: E402


def _read_schema(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


def _restorable(database: Database) -> list[tuple[str, str, str]]:
    rows = database.connection.execute(
        "SELECT m.memory_id, m.status, m.normalized_fact FROM memory_records m "
        "WHERE m.status IN ('rejected', 'expired') AND EXISTS ("
        "  SELECT 1 FROM memory_audit_events a WHERE a.memory_id = m.memory_id "
        "  AND a.action IN ('reject', 'expire') AND a.after_status IN ('rejected', 'expired')"
        ") ORDER BY m.memory_id"
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "qichi.sqlite3")
    parser.add_argument("--memory-id")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if not args.list and not args.memory_id:
        parser.error("either --list or --memory-id is required")
    if args.memory_id and not args.apply:
        print("预演模式（不写库）；加 --apply 才真的恢复。")

    path = args.database
    if not path.exists():
        print(f"database not found: {path}")
        return 2
    schema = _read_schema(path)
    if schema > SCHEMA_VERSION:
        print(f"database schema {schema} is newer than this code ({SCHEMA_VERSION})")
        return 2
    if schema < SCHEMA_VERSION and args.apply:
        print(
            f"警告：库是 schema {schema}，打开即会迁移到 {SCHEMA_VERSION}。"
            "若线上仍跑旧代码，迁移后它会报 schema 不匹配，需要重启换新代码。"
        )

    database = Database(path)
    try:
        memories = MemoryRepository(database)
        if args.list:
            items = _restorable(database)
            print(f"可恢复（有移除事件）的记录 {len(items)} 条：")
            for memory_id, status, fact in items:
                print(f"  [{status:8s}] {memory_id}  {fact[:60]}")
            return 0

        record = memories.get(args.memory_id)
        print(f"目标 {record.memory_id}  当前 {record.status} / {record.certainty} / "
              f"importance={record.importance} / {record.temporal_scope}")
        removal = database.connection.execute(
            "SELECT action, before_status, before_certainty, before_importance, "
            "before_temporal_scope, occurred_at_utc FROM memory_audit_events "
            "WHERE memory_id=? AND action IN ('reject','expire') "
            "ORDER BY occurred_at_utc DESC LIMIT 1",
            (record.memory_id,),
        ).fetchone()
        if removal is None:
            print("这条没有移除事件（迁移前的遗留数据），无法确定移除前的评级——fail closed。")
            return 1
        print(f"移除事件 {removal[0]} @ {removal[5]}：将恢复到 {removal[1]} / "
              f"{removal[2]} / importance={removal[3]} / {removal[4]}")
        if not args.apply:
            return 0

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = ROOT / "_backups" / f"memory-restore-{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        database.connection.execute("VACUUM INTO ?", (str(backup / "qichi.sqlite3"),))
        print(f"已备份到 {backup / 'qichi.sqlite3'}")
        restored = memories.restore(record.memory_id)
        print(f"已恢复 {restored.memory_id} -> {restored.status} / {restored.certainty} / "
              f"importance={restored.importance} / {restored.temporal_scope} / "
              f"{restored.assessment_reason_code}")
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
