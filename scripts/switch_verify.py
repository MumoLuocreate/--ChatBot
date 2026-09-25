# -*- coding: utf-8 -*-
r"""切换 flash 之后的三条命题复核（只读、不调用模型、不改任何东西）。

用法：
    .venv\Scripts\python.exe scripts\switch_verify.py
    .venv\Scripts\python.exe scripts\switch_verify.py --switch 2026-09-15T18:15:10 --window 200

判据（先写判据再动手，见 doc/方案-20260916-全量切flash.md §2）：

命题 A：flash 时代她「点名重复」的比率不高于 pro 时代。
  判别观测：等量窗口内她已发送消息里的点名条数占比（词表代理：重复|又说一遍|复述|模板|旧话|翻出来）
            + 均字数 + 最高频骨架频次。
  情景覆盖：真机自然对话（不按语域切分——切语域就需要分类，是红线）。亲密段另有三臂副本数据，不重跑。
  判决线：flash 窗口 >= 2x pro 窗口 且 >= 3 条 → 判为存在问题；flash 窗口 < 100 条 → 样本不足，不下结论。
  成本：0。

命题 B：她仍然会自愿用语音。判别观测：meta 里带 voice 的她方消息数 + 「想发但发不出」的降级计数。
  判决线：切换后 24h 内 >= 1 次自愿语音且 0 次降级 → 元决策还在；连续 24h 0 次 → 单独处理。成本：0。

命题 C：成本下降。判别观测：turn_trace_events 按天按模型的 token 汇总。判决线：调用量与上下文量级不变
  即成立（单价差按 09-13 记录：pro 折月 19~22 美元、flash 4~5）。成本：0。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "qichi.sqlite3"

# 词表型代理：只作走势参考，不能当判据本身（AGENTS.md 变更纪律）。
NAMING = re.compile(r"重复|又说一遍|再说一遍|复述|模板|旧话|翻出来")
DEGRADED = "voice_only_turn_degraded"


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _naming_rows(con: sqlite3.Connection, start: str, end: str, limit: int, newest_first: bool):
    order = "DESC" if newest_first else "ASC"
    rows = list(
        con.execute(
            "SELECT text, occurred_at_utc FROM conversation_events "
            "WHERE direction='outbound' AND status='sent' AND text IS NOT NULL "
            "AND occurred_at_utc >= ? AND occurred_at_utc < ? "
            f"ORDER BY sequence {order} LIMIT ?",
            (start, end, limit),
        )
    )
    if newest_first:
        rows.reverse()
    return rows


def _skeletons(texts, sizes=(4, 5, 6, 8), top=3):
    counts: collections.Counter[str] = collections.Counter()
    for text in texts:
        for size in sizes:
            for start in range(len(text) - size + 1):
                piece = text[start : start + size]
                if piece.strip() != piece or any(ch.isdigit() for ch in piece):
                    continue
                if sum(1 for ch in piece if ch.isalpha()) < 3:
                    continue
                counts[piece] += 1
    return [(p, c) for p, c in counts.most_common(40) if c >= 3][:top]


def _window_report(label: str, rows, total_hint: str) -> tuple[int, int]:
    texts = [t for t, _ in rows]
    hits = [t for t in texts if NAMING.search(t)]
    avg = sum(len(t) for t in texts) / max(1, len(texts))
    rate = 100.0 * len(hits) / max(1, len(texts))
    print(f"  {label}: {len(texts)} 条 / 点名 {len(hits)} 条 = {rate:.1f}% / 均 {avg:.0f} 字  {total_hint}")
    for piece, count in _skeletons(texts):
        print(f"      · 高频骨架「{piece}」x{count}")
    for text in hits[:3]:
        print(f"      · 点名样本：{text[:56]}")
    return len(texts), len(hits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch", default="2026-09-15T18:15:10", help="flash 上线的 UTC 时刻")
    parser.add_argument("--window", type=int, default=200, help="每侧取多少条她已发送的消息")
    parser.add_argument("--database", type=Path, default=DATABASE)
    args = parser.parse_args(argv)
    con = _connect(args.database)

    print("== 命题 A：点名重复（词表代理，只作走势参考）==")
    pro_rows = _naming_rows(con, "0000", args.switch, args.window, newest_first=True)
    flash_rows = _naming_rows(con, args.switch, "9999", args.window, newest_first=False)
    pro_n, pro_hit = _window_report("pro 尾窗", pro_rows, "(切换前最近 N 条)")
    flash_n, flash_hit = _window_report("flash 首窗", flash_rows, "(切换后最早 N 条)")
    if flash_n < 100:
        print(f"  判决：flash 窗口只有 {flash_n} 条 < 100 → **样本不足，不下结论**")
    else:
        pro_rate = pro_hit / max(1, pro_n)
        flash_rate = flash_hit / max(1, flash_n)
        if flash_hit >= 3 and flash_rate >= 2 * pro_rate:
            print(f"  判决：flash {flash_rate:.1%} vs pro {pro_rate:.1%} → **判为存在问题**（走补偿候选）")
        else:
            print(f"  判决：flash {flash_rate:.1%} vs pro {pro_rate:.1%} → 未达判决线，不判为问题")

    print()
    print("== 命题 B：她自愿用语音 ==")
    total = 0
    for era, start, end in (("pro", "0000", args.switch), ("flash", args.switch, "9999")):
        rows = list(
            con.execute(
                "SELECT occurred_at_utc, metadata_json FROM conversation_events "
                "WHERE metadata_json LIKE '%\"voice\"%' AND occurred_at_utc >= ? AND occurred_at_utc < ? "
                "ORDER BY sequence",
                (start, end),
            )
        )
        total += len(rows)
        days = collections.Counter(ts[:10] for ts, _ in rows)
        print(f"  {era}: 语音消息 {len(rows)} 条  {dict(days) if days else ''}")
        for ts, _ in rows[-2:]:
            print(f"      · {ts}")
    degraded = list(
        con.execute(
            "SELECT occurred_at_utc, details_json FROM turn_trace_events "
            "WHERE details_json LIKE ? ORDER BY occurred_at_utc",
            (f"%{DEGRADED}%",),
        )
    )
    print(f"  「想发但发不出」降级记录：{len(degraded)} 条（历史全部；切换后 {sum(1 for ts, _ in degraded if ts >= args.switch)} 条）")

    print()
    print("== 命题 C：token 与成本（按天、按模型）==")
    agg: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for ts, dj in con.execute(
        "SELECT occurred_at_utc, details_json FROM turn_trace_events WHERE phase='generation'"
    ):
        d = json.loads(dj)
        key = (ts[:10], str(d.get("model_id")))
        row = agg[key]
        row[0] += 1
        row[1] += int(d.get("input_tokens") or 0)
        row[2] += int(d.get("output_tokens") or 0)
        row[3] += int(d.get("reasoning_tokens") or 0)
        row[4] += int(d.get("cache_hit_tokens") or 0)
    for key in sorted(agg)[-8:]:
        calls, inp, out, reason, cache = agg[key]
        print(f"  {key[0]}  {key[1]:<18} 调用 {calls:>4} / 输入 {inp:>9} / 输出 {out:>7} / 推理 {reason:>6} / 命中 {cache:>8}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
