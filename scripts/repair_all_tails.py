# -*- coding: utf-8 -*-
"""按「洞」批量补录记忆明细——调用 scripts/repair_memory_window.py --start/--end --force。

为什么改成按洞（2026-09-17 第二次批量踩到）：
    原来按「片段」补（--tail-of），用的是**该片段自己**的明细盖到哪；可一个片段的尾巴常常
    已经被别的片段盖住了，真正的洞只是中间一小截。于是子脚本会把整个尾巴（真机 224 条）
    重抽一遍，写出一个重复且截断的新片段——白花钱、还多出一层重复索引。
    现在改成：把全库的片段覆盖合并成一层，找**连续未覆盖的洞**，去重后每个洞补一次。

纪律：
- 默认只列洞，不调用模型（零成本）；加 --apply 才真补。
- 批次开头一次 VACUUM INTO 备份；子脚本 --no-backup（但报告目录仍会建）。
- 逐个跑、失败即停；子进程输出落 _backups/.../logs/<seq>.log。

用法：
    python scripts/repair_all_tails.py                 # 只列洞
    python scripts/repair_all_tails.py --apply --limit 3
    python scripts/repair_all_tails.py --apply
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from memory_gaps import DETAIL_TAIL_MARGIN as TAIL_MARGIN  # noqa: E402

DEFAULT_DATABASE = PROJECT_ROOT / "data" / "qichi.sqlite3"
REPAIR = PROJECT_ROOT / "scripts" / "repair_memory_window.py"


def holes(database_path: pathlib.Path) -> list[dict]:
    """明细覆盖层里的连续空洞（只在已有片段跨度之内，去重、最新在前）。

    跨越多个片段的同一个洞只会出现一次——两个片段常常指向同一个洞（补录会产生重叠片段）。
    """

    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        spans = [
            (int(row["start_sequence"]), int(row["end_sequence"]))
            for row in connection.execute(
                "SELECT start_sequence, end_sequence FROM memory_fragments ORDER BY start_sequence"
            )
        ]
        covered: set = set()
        for row in connection.execute(
            "SELECT e.sequence AS sequence FROM memory_detail_records d "
            "JOIN memory_detail_evidence de ON de.detail_id = d.detail_id "
            "JOIN conversation_events e ON e.event_id = de.event_id"
        ):
            covered.add(int(row["sequence"]))
    finally:
        connection.close()

    # 把片段跨度合并成一层，再找这一层里的连续空洞。
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    found: list[dict] = []
    for start, end in merged:
        sequence = start
        while sequence <= end:
            if sequence in covered:
                sequence += 1
                continue
            run_start = sequence
            while sequence <= end and sequence not in covered:
                sequence += 1
            run_length = sequence - run_start
            if run_length > TAIL_MARGIN:
                found.append({"start": run_start, "end": sequence - 1, "missing": run_length})
    found.sort(key=lambda item: item["start"], reverse=True)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=pathlib.Path, default=DEFAULT_DATABASE)
    parser.add_argument("--apply", action="store_true", help="真的补（默认只列洞）")
    parser.add_argument("--limit", type=int, default=None, help="这次最多补几个洞")
    parser.add_argument("--skip", default="", help="已知补不了的洞，形如 6268-6295,5401-5465")
    parser.add_argument("--backup-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    skip = {item.strip() for item in args.skip.split(",") if item.strip()}
    todo = [item for item in holes(args.database) if "%d-%d" % (item["start"], item["end"]) not in skip]
    if args.limit is not None:
        todo = todo[: args.limit]
    print("待补的洞 %d 个（合计 %d 条事件）：" % (len(todo), sum(item["missing"] for item in todo)))
    for item in todo:
        print("  seq %d..%d  %d 条" % (item["start"], item["end"], item["missing"]))
    if not todo:
        print("没有要补的洞。")
        return 0
    if not args.apply:
        print()
        print("（只列洞：没有调用模型。确认后加 --apply）")
        return 0

    from copy_replay import copy_database

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_dir = args.backup_dir or (PROJECT_ROOT / "_backups" / ("memory-repair-holes-" + stamp))
    backup_dir.mkdir(parents=True, exist_ok=True)
    log_dir = backup_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / "qichi.sqlite3"
    print()
    print("批次备份（一次）：%s" % backup)
    copy_database(args.database, backup)
    print("备份完成：%d 字节" % backup.stat().st_size)

    python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    results = []
    consecutive_failures = 0
    for item in todo:
        label = "%d-%d" % (item["start"], item["end"])
        log_path = log_dir / (label + ".log")
        print()
        print("== 补洞 seq %s ==" % label)
        with log_path.open("w", encoding="utf-8") as log:
            code = subprocess.run(
                [python, str(REPAIR), "--start", str(item["start"]), "--end", str(item["end"]),
                 "--force", "--apply", "--no-backup", "--backup-dir", str(backup_dir / label)],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
            ).returncode
        written = [
            line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith("写入片段")
        ]
        print("  退出码 %d；%s" % (code, "；".join(written) if written else "（见日志）"))
        results.append({**item, "exit_code": code, "written": written, "log": log_path.name})
        if code != 0:
            consecutive_failures += 1
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            print("  （这个洞没补成功，日志 %s）" % log_path.name)
            # 2026-09-17：失败分两类。**内容层**（parse_error：模型给的证据引用了洞外的事件、
            # 或 review 内部不一致 → all_items_invalid）是特定窗口的问题，不该拦住其余二十几个洞；
            # **系统层**（llm_error：账户 402 / 供应商拒绝）必须立刻停，否则只会一路白跑。
            if "llm_error" in log_text:
                print()
                print("系统层失败（llm_error）：立刻停下。本批次备份 %s" % backup)
                break
            if consecutive_failures >= 4:
                print()
                print("内容层失败连续 %d 个：停下来看看。本批次备份 %s" % (consecutive_failures, backup))
                break
        else:
            consecutive_failures = 0

    report = {"at_utc": datetime.now(timezone.utc).isoformat(), "database": str(args.database),
              "backup": str(backup), "holes": len(todo), "results": results}
    report_path = backup_dir / "repair-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("报告：%s" % report_path)
    succeeded = sum(1 for item in results if item["exit_code"] == 0)
    print("成功 %d / 尝试 %d（洞总数 %d）" % (succeeded, len(results), len(todo)))
    return 0 if succeeded == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
