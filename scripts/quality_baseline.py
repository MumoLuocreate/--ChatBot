"""Validate or summarize the offline dialogue quality baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qichi.quality import load_ratings_jsonl, load_scenarios_jsonl, summarize_ratings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Qichi's offline quality baseline")
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("--ratings", type=Path)
    args = parser.parse_args(argv)
    scenarios = load_scenarios_jsonl(args.scenarios)
    ratings = () if args.ratings is None else load_ratings_jsonl(args.ratings)
    summary = summarize_ratings(scenarios, ratings)
    print(json.dumps({
        "scenario_count": summary.scenario_count,
        "sample_count": summary.sample_count,
        "axis_means": dict(summary.axis_means),
        "total_mean": summary.total_mean,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
