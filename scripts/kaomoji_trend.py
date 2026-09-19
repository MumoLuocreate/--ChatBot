# -*- coding: utf-8 -*-
"""她的表情使用趋势：按周统计种类数与集中度（只读）。

用途：颜文字的"范围塌缩"这件事，本项目已经用 A/B 栽过两次（项目约定：
单窗口分布不能推断趋势）。所以不再造对照，改成**按周看账本自己的趋势**：
真实对话的样本量远大于任何回放，趋势真假由它自己证明。

  python scripts/kaomoji_trend.py            # 全库按周
  python scripts/kaomoji_trend.py --weeks 6  # 只看最近 6 周
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qichi.dialogue.context_builder import _expression_tokens  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=pathlib.Path, default=PROJECT_ROOT / "data" / "qichi.sqlite3")
    parser.add_argument("--conversation", default=None, help="缺省=所有会话")
    parser.add_argument("--weeks", type=int, default=0, help="只看最近 N 周；0=全部")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()

    zone = ZoneInfo(args.timezone)
    connection = sqlite3.connect("file:%s?mode=ro" % pathlib.Path(args.database).as_posix(), uri=True)
    per_week: dict[str, Counter] = defaultdict(Counter)
    per_day: dict[str, list] = defaultdict(list)
    query = ("SELECT text, occurred_at_utc FROM conversation_events "
             "WHERE direction='outbound' AND status='sent' AND text IS NOT NULL")
    parameters: tuple = ()
    if args.conversation:
        query += " AND conversation_id = ?"
        parameters = (str(args.conversation),)
    for text, occurred_at in connection.execute(query, parameters):
        moment = datetime.fromisoformat(occurred_at).astimezone(zone)
        per_week[moment.strftime("%Y-%W")].update(_expression_tokens(text or ""))
        per_day[moment.strftime("%Y-%m-%d")].extend(tokens)
    connection.close()

    weeks = sorted(per_week)
    if args.weeks:
        weeks = weeks[-args.weeks:]
    print("周次        表情总数  种类  最多的一款（占比）")
    for week in weeks:
        counter = per_week[week]
        if not counter:
            print("%-11s %8d %5d  —" % (week, 0, 0))
            continue
        total = sum(counter.values())
        token, count = counter.most_common(1)[0]
        print("%-11s %8d %5d  %s ×%d（%.0f%%）" % (week, total, len(counter), token, count, 100.0 * count / total))
    print()
    print("（种类数只数她真用过的表情 token；『（无表情）』不计入。）")
    return 0


raise SystemExit(main())
