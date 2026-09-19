# -*- coding: utf-8 -*-
"""盘点「哪些 seq 没有记忆片段覆盖、里面有没有他的话」——只读，不调用任何模型。

为什么要有它：2026-09-16 发现 seq 7306-7320（他醒来那 15 条）因为抽取解析失败被
quarantine，而 quarantine 游标只增不减（worker._snapshot_tx 用 max(watermark, quarantine)
当扫描起点），于是这一段**再也不会成为任何一次抽取的候选**——运行时没有任何人会回头看它。
那次是靠人工挖出来的；这个脚本把那次挖掘固化成一次命令。

判据（什么是真损失）：
- 片段覆盖之外的区间里，只要有**用户的原话**（direction=inbound、kind=text），
  这一段就是「本可以进记忆、但没进」→ 需要 repair_memory_window.py 补录；
- 只有角色的主动开场（initiative 及其紧跟着的她自己那几条）的区间**不是损失**：
  片段从他的话开始，是被设计成这样的（历史 24 处这种缺口全是这一类）。
- 记忆层起点（第一个片段的 start_sequence）之前的区间不计：那时还没有片段层。

另外它还报**明细覆盖**（2026-09-17）：片段存下来了，但明细时间线只盖到前半段——摘要不进
上下文，明细是那段记忆的唯一入口，所以尾巴对她就等于不存在（真机：172 条的片段只盖到第 57 条，
她因此翻不到凌晨那句「什么时候会想要我」）。判据：明细条数 > 0 且「片段末尾 - 明细最后一条」
> 3（末尾几条允许没有自己的明细）。这一类默认只报告，加 --strict-details 才影响退出码。

退出码：0 = 没有含他原话的缺口；1 = 有（可直接当巡检用）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "qichi.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


# 末尾几条允许没有自己的明细（收尾、被合并进上一条、本来就没什么可记）。
# 2026-09-17：3 → 6。抽取器每条明细平均覆盖 1-2 条事件，所以一段 28 条事件的片段里，
# 出现 3-5 条没有自己明细的零散事件是正常的；用 3 会把这类噪声误报成缺口（第一次批量
# 补完后，已补好的 6a33b1fb 与新建的 7525c425 都各差 4 条而重新上榜）。
DETAIL_TAIL_MARGIN = 6


def dangling_evidence(connection: sqlite3.Connection) -> list[dict]:
    """证据指向了**不存在的事件**——这类引用会让整轮对话直接失败。

    2026-09-17：副本回放脚本为了「回到那个时刻」删掉了近期事件，却留下指向它们的明细证据；
    于是 app 的 events.get() 抛 KeyError，**整轮装配失败、她一个字都发不出来**——我在多情景
    验证里就撞上了（H 组三次静默），一度误判成 ambiguous 会崩。生产库里悬空引用是 0，
    但这件事说明需要一个常设检查：恢复/裁剪事件之后，证据必须跟着检查。
    """

    return [
        dict(row)
        for row in connection.execute(
            "SELECT 'memory_evidence' AS source, me.event_id AS event_id, COUNT(*) AS refs "
            "FROM memory_evidence me LEFT JOIN conversation_events e ON e.event_id = me.event_id "
            "WHERE e.event_id IS NULL GROUP BY me.event_id "
            "UNION ALL "
            "SELECT 'memory_detail_evidence', de.event_id, COUNT(*) "
            "FROM memory_detail_evidence de LEFT JOIN conversation_events e ON e.event_id = de.event_id "
            "WHERE e.event_id IS NULL GROUP BY de.event_id "
            "UNION ALL "
            "SELECT 'memory_detail_records.source_event_id', d.source_event_id, COUNT(*) "
            "FROM memory_detail_records d LEFT JOIN conversation_events e ON e.event_id = d.source_event_id "
            "WHERE d.source_event_id IS NOT NULL AND e.event_id IS NULL "
            "GROUP BY d.source_event_id"
        )
    ]


def longest_uncovered_run(start: int, end: int, covered: set) -> tuple[int, int]:
    """区间里**最长的一段连续没人覆盖**：返回（起点, 长度）。

    2026-09-17：这是「尾巴要不要补」的判据。抽取器会把两条事件并成一条明细，所以
    「某条事件没有自己的明细」是正常的零散小洞；真正要补的是**整块连续缺口**。
    按「有没有被别的片段盖住」的单片段口径会把已补好的片段重新报成缺口。
    """

    best_start, best_length = start, 0
    run_start, run_length = start, 0
    for sequence in range(start, end + 1):
        if sequence in covered:
            if run_length > best_length:
                best_start, best_length = run_start, run_length
            run_start, run_length = sequence + 1, 0
        else:
            if run_length == 0:
                run_start = sequence
            run_length += 1
    if run_length > best_length:
        best_start, best_length = run_start, run_length
    return best_start, best_length


def _detail_coverage(connection: sqlite3.Connection) -> list[dict]:
    """每个片段的明细盖到了哪里——只统计「尾巴至今没人盖」的片段。

    2026-09-17 起：尾巴可以被**另一个片段**补上（补录会造出这种重叠），
    所以先看它自己的明细盖到哪，再看那之后的事件是不是已经被别的片段盖住了。
    """

    rows = connection.execute(
        "SELECT f.fragment_id, f.start_sequence, f.end_sequence, "
        "COUNT(DISTINCT d.detail_id) AS details, MAX(e.sequence) AS covered_to "
        "FROM memory_fragments f "
        "LEFT JOIN memory_detail_records d ON d.fragment_id = f.fragment_id "
        "LEFT JOIN memory_detail_evidence de ON de.detail_id = d.detail_id "
        "LEFT JOIN conversation_events e ON e.event_id = de.event_id "
        "GROUP BY f.fragment_id, f.start_sequence, f.end_sequence "
        "ORDER BY f.start_sequence"
    ).fetchall()
    covered_all: set = set()
    for row in connection.execute(
        "SELECT e.sequence AS sequence FROM memory_detail_records d "
        "JOIN memory_detail_evidence de ON de.detail_id = d.detail_id "
        "JOIN conversation_events e ON e.event_id = de.event_id"
    ):
        covered_all.add(int(row["sequence"]))

    report = []
    for row in rows:
        details = int(row["details"] or 0)
        covered_to = row["covered_to"]
        if not details or covered_to is None:
            continue  # 没有明细是另一条路径的事（时间线为空由 job 账本说明）
        end = int(row["end_sequence"])
        gap = end - int(covered_to)
        if gap <= DETAIL_TAIL_MARGIN:
            continue
        # 2026-09-17：判据是「尾巴里最长的一段**连续**无人覆盖」——零散小洞是抽取器合并事件的
        # 正常结果，整块缺口才是要补的。
        tail_start = int(covered_to) + 1
        run_start, run_length = longest_uncovered_run(tail_start, end, covered_all)
        if run_length <= DETAIL_TAIL_MARGIN:
            continue  # 尾巴已经被（别的片段或自己的）明细盖住了
        report.append(
            {
                "fragment_id": row["fragment_id"],
                "start": int(row["start_sequence"]),
                "end": end,
                "details": details,
                "covered_to": int(covered_to),
                "gap": gap,
                "still_missing": run_length,
                "uncovered_from": run_start,
            }
        )
    return report


def _scan_boundary(connection: sqlite3.Connection) -> int | None:
    """worker 已经轮到哪一条：水位与隔离游标取大（quarantine 本身就是扫描边界）。

    高过它的缺口只是「还没轮到」（刚聊完、还没到静默判定），不是丢了——不区分就会把
    正在排队的那一段报成真损失。库里没有任何水位记录时返回 None（无从判断，按老规矩报）。
    """

    row = connection.execute(
        "SELECT MAX(CAST(value_json AS INTEGER)) FROM runtime_meta "
        "WHERE key LIKE 'memory_worker:%:processed_sequence' "
        "OR key LIKE 'memory_worker:%:quarantined_sequence'"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _covered(fragments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(fragments):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(item[0], item[1]) for item in merged]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--json", action="store_true", help="输出 JSON（给面板或别的脚本用）")
    parser.add_argument(
        "--strict-details",
        action="store_true",
        help="明细没盖到片段末尾也算不合格（默认只报告）",
    )
    args = parser.parse_args(argv)
    connection = _connect(args.database)

    fragments = [
        (int(row["start_sequence"]), int(row["end_sequence"]))
        for row in connection.execute("SELECT start_sequence, end_sequence FROM memory_fragments")
    ]
    if not fragments:
        print("memory_gaps: 库里没有任何片段")
        return 1
    floor = min(start for start, _ in fragments)
    highest = connection.execute("SELECT MAX(sequence) FROM conversation_events").fetchone()[0]
    highest = int(highest or floor)

    quarantined = {
        (int(row["start_sequence"]), int(row["end_sequence"]))
        for row in connection.execute(
            "SELECT start_sequence, end_sequence FROM memory_job_events WHERE action='quarantined'"
        )
    }

    gaps: list[tuple[int, int]] = []
    cursor = floor
    for start, end in _covered(fragments):
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= highest:
        gaps.append((cursor, highest))

    boundary = _scan_boundary(connection)
    report = []
    real_losses = 0
    for start, end in gaps:
        row = connection.execute(
            "SELECT "
            "SUM(CASE WHEN direction='inbound' AND kind='text' THEN 1 ELSE 0 END) AS theirs, "
            "SUM(CASE WHEN direction='outbound' AND kind='text' THEN 1 ELSE 0 END) AS hers, "
            "SUM(CASE WHEN kind='initiative' THEN 1 ELSE 0 END) AS initiatives "
            "FROM conversation_events WHERE sequence BETWEEN ? AND ?",
            (start, end),
        ).fetchone()
        theirs = int(row["theirs"] or 0)
        pending = boundary is not None and start > boundary
        is_loss = theirs > 0 and not pending
        real_losses += 1 if is_loss else 0
        was_quarantined = any(q_start <= end and q_end >= start for q_start, q_end in quarantined)
        report.append(
            {
                "start": start,
                "end": end,
                "count": end - start + 1,
                "his_messages": theirs,
                "her_messages": int(row["hers"] or 0),
                "initiative_events": int(row["initiatives"] or 0),
                "quarantined": was_quarantined,
                "pending": pending,
                "is_loss": is_loss,
            }
        )

    truncated = _detail_coverage(connection)

    if args.json:
        print(json.dumps(
            {"floor": floor, "highest": highest, "gaps": report, "truncated_details": truncated},
            ensure_ascii=False, indent=2,
        ))
    else:
        print("记忆层起点 seq %d，最后事件 seq %d，片段 %d 个，缺口 %d 处" % (
            floor, highest, len(fragments), len(gaps)))
        for item in report:
            if item["is_loss"]:
                tag = "【真损失：含他 %d 条原话】" % item["his_messages"]
            elif item.get("pending"):
                tag = "（还没轮到：worker 的水位还在后面）"
            else:
                tag = "（只有角色的主动开场）"
            mark = " [来自隔离]" if item["quarantined"] else ""
            print("  seq %5d-%5d  %2d 条  他 %2d / 她 %2d / 主动 %d  %s%s" % (
                item["start"], item["end"], item["count"], item["his_messages"],
                item["her_messages"], item["initiative_events"], tag, mark))
            if item["is_loss"]:
                print("      补录：python scripts/repair_memory_window.py --start %d --end %d"
                      % (item["start"], item["end"]))
        print()
        if real_losses:
            print("结论：**%d 处缺口含他的原话**，需要补录。" % real_losses)
        else:
            print("结论：没有含他原话的缺口（其余缺口都是主动开场，按设计不进片段）。")
        print()
        if truncated:
            print("明细覆盖：**%d 个片段的明细没盖到末尾**（末尾 %d 条以内算正常）：" % (
                len(truncated), DETAIL_TAIL_MARGIN))
            for item in truncated:
                print("  %s seq %d-%d  明细 %d 条，只盖到 %d（还差 %d 条没人盖）" % (
                    item["fragment_id"][:8], item["start"], item["end"],
                    item["details"], item["covered_to"], item.get("still_missing", item["gap"])))
        else:
            print("明细覆盖：每个有明细的片段都盖到了自己的末尾。")
    dangling = dangling_evidence(connection)
    print()
    if dangling:
        print("证据悬空：**%d 个事件 id 被证据引用，但事件已经不存在**——这类引用会让整轮对话失败：" % len(dangling))
        for item in dangling[:10]:
            print("  %s  %s × %d" % (item["source"], str(item["event_id"])[:8], item["refs"]))
        if len(dangling) > 10:
            print("  …（还有 %d 个）" % (len(dangling) - 10))
    else:
        print("证据悬空：没有。每条证据指向的事件都还在。")
    connection.close()
    if real_losses:
        return 1
    if dangling:
        return 1
    if args.strict_details and truncated:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
