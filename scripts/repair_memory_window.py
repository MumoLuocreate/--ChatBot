# -*- coding: utf-8 -*-
"""补录一个被隔离（quarantine）的记忆窗口——走生产抽取路径，默认只抽不写。

背景：2026-09-15 凌晨 DeepSeek 官方崩溃，后台模型连续超时并返回残缺 JSON，
seq 6566–6600（35 条，正是用户纠正她的那一段）在两次尝试后被永久隔离，
那一段从此不在记忆层里（doc/修复计划-20260915-记忆补录与折半重抽.md）。

这个脚本只做一件事：把点名的一段原文交给**与生产完全相同的抽取器**，把结果打印出来；
加 --apply 才写库，写之前一定先做一份 VACUUM INTO 备份，且**不碰**
（水位 / 隔离游标 / memory_session_jobs）——缺口在水位下方，worker 本来就不会再碰它。

2026-09-17 起加了一个模式：**补一个已有片段的「尾巴」**。背景是明细时间线只盖到片段
前半段（47 个片段里 26 个），尾巴那截对她就等于不存在（真机：她因此翻不到凌晨那句
「什么时候会想要我」）。新片段不再截断，但**存量片段的尾巴仍然空着**——这个模式把尾巴
按新规则落成**一个新片段**：不动任何已存在的行（原文账本不可变、片段事件只追加），
只是把「切晚了的那一刀」补上。用法：

    python scripts/repair_memory_window.py --start 6566 --end 6600                    # 只看
    python scripts/repair_memory_window.py --start 6566 --end 6600 --apply            # 写
    python scripts/repair_memory_window.py --tail-of 46bb6fd9                         # 只看尾巴
    python scripts/repair_memory_window.py --tail-of 46bb6fd9 --apply                 # 补尾巴
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def hydrate_environment() -> None:
    """把用户环境里的密钥搬进来（与其它回放脚本同一条约定）。"""

    import os
    import winreg

    names = ("QICHI_OWNER_QQ", "NAPCAT_WS_URL", "NAPCAT_HTTP_URL", "NAPCAT_ACCESS_TOKEN",
             "SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY",
             "TAVILY_API_KEY", "SERPAPI_API_KEY")
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            for name in names:
                if os.environ.get(name):
                    continue
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if value:
                    os.environ[name] = value
    except OSError:
        return


def describe_failure(failure) -> str:
    """把抽取失败说清楚。

    2026-09-17：这里原来直接取 failure.message（ExtractionFailure 没有这个属性），于是
    **失败分支一走就 AttributeError**，把真正的错误（那次是账户 402）吞掉了——写成一个
    纯函数是为了让"失败时到底打印了什么"能被测试盯住。
    """

    parts = ["%s / %s" % (getattr(failure, "kind", "?"), getattr(failure, "reason", "?"))]
    details = getattr(failure, "details", None)
    if details:
        parts.append("诊断：%s" % dict(details))
    return "；".join(parts)


def _coverage(database, start: int, end: int) -> list:
    return database.connection.execute(
        "SELECT fragment_id,start_sequence,end_sequence,status FROM memory_fragments "
        "WHERE NOT (end_sequence < ? OR start_sequence > ?) ORDER BY start_sequence",
        (start, end),
    ).fetchall()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--tail-of",
        default=None,
        help="补这个片段（前缀即可）的尾巴：从明细覆盖到的地方往后，落成一个新片段",
    )
    parser.add_argument("--config", type=pathlib.Path, default=PROJECT_ROOT / "config.example.yaml")
    parser.add_argument("--apply", action="store_true", help="真的写库（默认只抽不写）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许补一个「已被片段覆盖、但内部有洞」的区间（批量驱动按洞补时用）",
    )
    parser.add_argument("--backup-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="跳过 VACUUM 备份（批量驱动 scripts/repair_all_tails.py 已经在批次开头统一备份过）",
    )
    args = parser.parse_args()
    if args.tail_of is None and (args.start is None or args.end is None):
        print("repair: 要么给 --start/--end，要么给 --tail-of")
        return 2
    if args.tail_of is not None and (args.start is not None or args.end is not None):
        print("repair: --tail-of 与 --start/--end 只能给一个")
        return 2
    if args.tail_of is None and args.end < args.start:
        print("repair: --end must be >= --start")
        return 2

    hydrate_environment()
    from copy_replay import copy_database
    from qichi.config import load_config
    from qichi.memory.detail_pass import MemoryDetailPass
    from qichi.memory.extractor import MemoryExtractor
    from qichi.memory.worker import MemoryWorker, _drafts_cover_the_window
    # 这一段的装配与 runtime.build_runtime 第 787–824 行**逐字对应**：抽取器、后台模型、
    # 明细补写都必须是生产那一套，否则补录进来的记忆与她自己抽的不是一个来源。
    from qichi.runtime import (
        MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS,
        MEMORY_EXTRACTION_TIMEOUT_SECONDS,
        _MemoryExtractionLLM,
        _build_llm_client,
    )
    from qichi.storage.database import Database
    from qichi.storage.event_repository import EventRepository
    from qichi.storage.memory_detail_repository import (
        MAX_DETAILS_PER_FRAGMENT,
        MemoryDetailRepository,
    )
    from qichi.storage.memory_repository import MemoryRepository

    config = load_config(args.config)
    database_path = PROJECT_ROOT / config.storage.database_path
    database = Database(database_path)
    conversation = str(config.app.owner_qq)
    tail_of = None
    if args.tail_of is not None:
        # 尾巴 = 这个片段的「明细没盖到」的那一段。范围由代码算出来，不接受手填。
        rows = database.connection.execute(
            "SELECT fragment_id,start_sequence,end_sequence FROM memory_fragments "
            "WHERE fragment_id LIKE ? ORDER BY start_sequence",
            (args.tail_of + "%",),
        ).fetchall()
        if len(rows) != 1:
            print("repair: --tail-of 需要唯一命中一个片段，实际命中 %d 个" % len(rows))
            database.close()
            return 2
        row = rows[0]
        covered = database.connection.execute(
            "SELECT MAX(e.sequence) FROM memory_detail_records d "
            "JOIN memory_detail_evidence de ON de.detail_id = d.detail_id "
            "JOIN conversation_events e ON e.event_id = de.event_id WHERE d.fragment_id = ?",
            (row["fragment_id"],),
        ).fetchone()[0]
        args.start = (int(covered) + 1) if covered is not None else int(row["start_sequence"])
        args.end = int(row["end_sequence"])
        tail_of = row["fragment_id"]
        print("片段 %s 覆盖 seq %s..%s；明细盖到 %s → 本次补尾巴 seq %s..%s" % (
            tail_of[:8], row["start_sequence"], row["end_sequence"],
            covered if covered is not None else "（没有明细）", args.start, args.end))
        if args.start > args.end:
            print("repair: 这个片段的明细已经盖到末尾，没有尾巴要补")
            database.close()
            return 0
    elif not args.force:
        covering = _coverage(database, args.start, args.end)
        if covering:
            print("repair: 这一段已经有片段覆盖，什么都不做：")
            for row in covering:
                print("   %s  seq %s..%s  status=%s" % (
                    row["fragment_id"][:8], row["start_sequence"], row["end_sequence"], row["status"]))
            database.close()
            return 0

    repository = MemoryRepository(database)
    extraction_client = _build_llm_client(
        config.llm.provider, config.llm.base_url, config.llm.api_key,
        model=config.llm.background_model,
        temperature=config.llm.primary.temperature, top_p=config.llm.primary.top_p,
        max_output_tokens=MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS,
        timeout_seconds=MEMORY_EXTRACTION_TIMEOUT_SECONDS,
    )
    detail_client = _build_llm_client(
        config.llm.provider, config.llm.base_url, config.llm.api_key,
        model=config.llm.background_model,
        temperature=config.llm.primary.temperature, top_p=config.llm.primary.top_p,
        max_output_tokens=8192, timeout_seconds=90,
    )
    worker = MemoryWorker(
        MemoryExtractor(
            _MemoryExtractionLLM(
                extraction_client, repository, config.app.owner_qq,
                max_output_tokens=MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS,
                local_timezone=config.app.timezone,
            )
        ),
        repository,
        quiet_minutes=30,
        auto_commit=config.memory.auto_commit,
        database=database,
        conversation_id=config.app.owner_qq,
        detail_pass=MemoryDetailPass(detail_client),
    )
    worker.detail_repository = MemoryDetailRepository(database)

    events = EventRepository(database)
    rows = database.connection.execute(
        "SELECT event_id FROM conversation_events WHERE conversation_id=? AND sequence BETWEEN ? AND ? "
        "ORDER BY sequence,event_id", (conversation, args.start, args.end)).fetchall()
    reliable = [events.get(row["event_id"]) for row in rows]
    reliable = [event for event in reliable if worker._is_reliable(event)]
    if not reliable:
        print("repair: 这一段没有可抽取的可靠事件")
        database.close()
        return 1
    snapshot = worker._snapshot_from_events(
        conversation, tuple(reliable), worker._context_version_tx(database.connection, conversation))
    print("区间 %s..%s：原文 %d 条，可抽取 %d 条" % (args.start, args.end, len(rows), len(reliable)))
    print("抽取器：%s（与生产同一个）；明细补写：%s" % (
        config.llm.background_model, "有" if worker.detail_pass is not None else "无"))
    print("调用抽取器（真的走一次后台模型）…")
    result = await worker.extractor.extract(snapshot.events)
    if not result.ok:
        print("抽取失败：%s" % describe_failure(result.failure))
        database.close()
        return 1

    details = tuple(result.details or ())
    detail_failure = None
    if not details and worker.detail_pass is not None:
        # dry-run 也要把明细抽出来给你看：写到库里的到底是什么，先看再定。
        extra, detail_failure, _split = await worker._detail_timeline(snapshot.events)
        details = tuple(extra)
    outcome = result.outcome
    print()
    print("=== 抽取结果（还没写库）===")
    print("outcome: %s / %s" % (outcome.kind if outcome else "-",
                                outcome.reason_code if outcome else "-"))
    print("片段摘要: %s" % (result.fragment.summary if result.fragment else "（无）"))
    print("candidates: %d 条，reviews: %d 条（本轮只写片段与明细；records 层要不要补由你定）" % (
        len(result.candidates), len(result.reviews)))
    if result.fragment is not None:
        print("片段 privacy=%s type=%s scope=%s" % (
            result.fragment.privacy_class, result.fragment.fragment_type, result.fragment.reality_scope))
    print("明细 %d 条，隐私分布 %s" % (
        len(details), dict(Counter(item.privacy_class for item in details))))
    for item in details[:8]:
        print("   #%s [%s/%s] %s" % (item.ordinal, item.privacy_class, item.reality_scope,
                                     str(item.normalized_detail)[:78]))
    if len(details) > 8:
        print("   …（还有 %d 条）" % (len(details) - 8))
    if detail_failure:
        print("明细补写失败：%s（片段仍可落库，明细为空）" % detail_failure)
    if result.reviews:
        print()
        print("=== 它想改的关系记忆（reviews，本轮不写）===")
        for review in result.reviews:
            row = database.connection.execute(
                "SELECT status,type,normalized_fact FROM memory_records WHERE memory_id=?",
                (review.memory_id,)).fetchone()
            current = ("%s/%s" % (row["status"], row["type"])) if row else "（库里没有这条）"
            fact = str(row["normalized_fact"])[:60] if row else ""
            print("   %s → %s（%s）  当前 %s | %s" % (
                review.memory_id[:8], review.action, review.assessment_reason_code, current, fact))
    if result.candidates:
        print()
        print("=== 它想新增的关系记忆（candidates，本轮不写）===")
        for candidate in result.candidates:
            print("   [%s] %s" % (candidate.type, str(candidate.normalized_fact)[:70]))

    if not args.apply:
        print()
        print("（dry-run：没有写任何东西。确认后加 --apply）")
        database.close()
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_dir = args.backup_dir or (PROJECT_ROOT / "_backups" / ("memory-repair-" + stamp))
    backup = backup_dir / "qichi.sqlite3"
    _ = backup  # 报告目录在下面统一创建（--no-backup 也要建，否则写报告会 FileNotFoundError）
    backup_dir.mkdir(parents=True, exist_ok=True)
    if args.no_backup:
        print()
        print("备份: 跳过（--no-backup；批次驱动已在开头统一备份）")
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)
        copy_database(database_path, backup)
        print()
        print("备份: %s（%d 字节）" % (backup, backup.stat().st_size))

    now = datetime.now(timezone.utc)
    # 与 worker **同一条**切分规则（超上限 **或** 没盖到末尾）：补录出来的片段同样不许截断，
    # 否则补一次还是半截，等于白补。切分交给生产那套 _stores_within_cap（含兜底明细 pass）。
    units = None
    if len(details) > MAX_DETAILS_PER_FRAGMENT or not _drafts_cover_the_window(
        snapshot.events, tuple(details)
    ):
        print()
        print("明细 %d 条没盖满这一段（%d 条事件）：按生产同一套规则切分重抽，一条不裁。"
              % (len(details), len(snapshot.events)))
        units = await worker._stores_within_cap(
            snapshot.events, result.fragment, tuple(details)
        )
        for unit_events, _spec, unit_drafts in units:
            print("   单元 seq %s..%s：明细 %d 条" % (
                unit_events[0].sequence, unit_events[-1].sequence, len(unit_drafts)))
    if units is None:
        units = ((snapshot.events, result.fragment, tuple(details)),)

    written = []
    with database.transaction() as connection:
        for unit_events, unit_spec, unit_drafts in units:
            unit_key = (
                snapshot.fragment_key
                if unit_events == snapshot.events
                else worker._snapshot_key(conversation, unit_events, snapshot.context_version)
            )
            fragment = worker.detail_repository.build_fragment(
                unit_events, unit_key, unit_spec, created_at_utc=now)
            built = worker.detail_repository.build_details(
                fragment, tuple(unit_drafts), unit_events)
            worker.detail_repository.store_in_transaction(
                connection, fragment=fragment, events=unit_events, details=built)
            written.append((fragment.fragment_id, len(built),
                            unit_events[0].sequence, unit_events[-1].sequence))
    for fragment_id, count, first, last in written:
        print("写入片段 %s（seq %s..%s，明细 %d 条）" % (fragment_id[:8], first, last, count))
    report = {"repaired_at_utc": now.isoformat(), "start": args.start, "end": args.end,
              "fragments": [{"fragment_id": item[0], "details": item[1],
                             "start": item[2], "end": item[3]} for item in written],
              "candidates_not_written": len(result.candidates), "backup": str(backup)}
    (backup_dir / "repair-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告: %s" % (backup_dir / "repair-report.json"))
    database.close()
    return 0


if __name__ == "__main__":
    # 2026-09-17：原来这里是裸的 raise SystemExit(asyncio.run(main()))，于是**任何 import
    # 都会跑一遍 main()**（argparse 立刻以 2 退出），测试没法 import 它、也看不出失败分支
    # 打得对不对。批量驱动走的是子进程，所以一直没暴露。
    raise SystemExit(asyncio.run(main()))
