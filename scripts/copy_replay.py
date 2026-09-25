"""Replay recall questions against a throwaway copy of the live database.

Before a change to recall, context assembly or the capability envelope reaches
production, run it here: the script copies the live database, builds the real
runtime against that copy with a real model client, and reports per question what
the recall gate decided, what was injected, and what the character answered.

Two guarantees, both structural rather than promised:

* the live database is always opened read-only, and every run works on a fresh
  copy under the work directory;
* the OneBot client is a recorder, so nothing is ever sent to QQ.

Usage:
    python scripts/copy_replay.py
    python scripts/copy_replay.py --question "细说一下九号那天中午我们讲了啥"
    python scripts/copy_replay.py --context-only        # 不调用模型，只看上下文
    python scripts/copy_replay.py --out runtime/copy-replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qichi.dialogue.llm_client import DeepSeekLLMClient, LLMGeneration  # noqa: E402

DEFAULT_DATABASE = PROJECT_ROOT / "data" / "qichi.sqlite3"
DEFAULT_CONFIG = PROJECT_ROOT / "config.example.yaml"
DEFAULT_WORKDIR = PROJECT_ROOT / "_tmp" / "copy-replay"

# The questions that have actually caught something.  Add one whenever a real
# machine failure is fixed, so the same failure cannot come back unnoticed.
DEFAULT_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("命中-要求细说九号", "细说一下九号那天中午我们讲了啥"),
    ("命中-详细说今天中午", "详细说一下今天中午我们都聊了什么"),
    ("边界-只说日期不带请求", "九号中午我们聊了啥"),
    ("不误判-日常闲聊", "揉揉你，在忙吗"),
    ("不误判-普通提问", "今天天气怎么样，你那边冷不冷"),
)

BANNER = "副本回放：只读源库 + 假 OneBot（不会发出任何 QQ 消息）"


def running_bot_qq() -> str | None:
    """The bot account the live process logged in as, when it is on disk."""

    marker = PROJECT_ROOT / "runtime" / "qichi-ready.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8")).get("bot_qq")
    except (OSError, ValueError):
        return None
    return None if value is None else str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", action="append", default=None,
                        help="要回放的问题，可重复；缺省用内置问题集")
    parser.add_argument("--label", action="append", default=None,
                        help="与 --question 一一对应的标签，可省略")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--bot-qq", default=None,
                        help="机器人账号；缺省读 runtime/qichi-ready.json 里正在运行的那个")
    parser.add_argument("--context-only", action="store_true",
                        help="不调用模型，只报告会注入什么（零成本）")
    parser.add_argument("--keep", action="store_true", help="保留副本文件（默认保留，便于复查）")
    parser.add_argument("--out", type=Path, default=None, help="把结果 JSON 写到这个路径")
    return parser


def load_questions(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    if not args.question:
        return DEFAULT_QUESTIONS
    labels = list(args.label or [])
    questions = []
    for index, question in enumerate(args.question):
        label = labels[index] if index < len(labels) else f"问题 {index + 1}"
        questions.append((label, question))
    return tuple(questions)


def copy_database(source: Path, target: Path) -> None:
    """Take a consistent copy; the live file is only ever opened read-only.

    VACUUM INTO is preferred because it is one pass; the online backup API has to
    restart whenever the source is written, and the bot writes constantly while
    it is running -- which is exactly when this tool is used.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    origin = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        try:
            origin.execute("VACUUM INTO ?", (str(target),))
            return
        except sqlite3.Error:
            pass
        destination = sqlite3.connect(target)
        try:
            with destination:
                origin.backup(destination)
        finally:
            destination.close()
    finally:
        origin.close()


def render_report(results: list[dict]) -> str:
    lines = []
    for item in results:
        lines.append(f"[{item['label']}] {item['question']}")
        lines.append(
            f"  判据={item['reason']} 片段={item['fragments']} 明细={item['details']} "
            f"明细token={item['details_tokens']}"
        )
        # 2026-09-12 T1：钥匙（授权）与定位分开看。运行中的判据还没换，这一行是
        # 冻结规则**本该**怎么答；两者不一致的地方就是 T2 要动的地方。
        lines.append(
            f"  钥匙={item.get('key')} 逐字命中={item.get('match_count')} "
            f"索引一致={item.get('indexed')} 空日子说明={item.get('recall_note')}"
        )
        for line in item["index_lines"]:
            lines.append(f"  索引: {line}")
        for line in item["detail_labels"]:
            lines.append(f"  明细块: {line}")
        if item.get("reply") is not None:
            lines.append(f"  她的回答: {item['reply']}")
        lines.append("")
    return "\n".join(lines)


class RecordingLLM(DeepSeekLLMClient):
    """The real client, remembering what it was asked and what it answered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple] = []
        self.replies: list[str] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        result = await super().generate(messages, **kwargs)
        self.replies.append(result.text or "")
        return result


class NullLLM(DeepSeekLLMClient):
    """context-only mode: reports the context without leaving the machine."""

    def __init__(self):
        super().__init__("https://example.invalid/v1", "unused", model="deepseek-v4-flash")
        self.calls: list[tuple] = []
        self.replies: list[str] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        self.replies.append("")
        return LLMGeneration("（context-only：没有调用模型）", "primary", "none", 0, 0, 0.0)


class RecordingOneBot:
    """Records outbound messages instead of sending them anywhere."""

    def __init__(self):
        self.sent: list = []
        self.get_image_calls: list[str] = []

    async def send_private_msg(self, user_id, message):
        self.sent.append({"user_id": str(user_id), "message": message})
        return {"message_id": 9_500_000_000 + len(self.sent)}

    async def get_image(self, file):
        self.get_image_calls.append(file)
        raise RuntimeError("copy replay never fetches platform images")


def _raw(message_id: int, at: datetime, text: str, owner: str, bot: str) -> dict:
    return {
        "post_type": "message", "message_type": "private", "sub_type": "friend",
        "self_id": int(bot), "user_id": owner, "target_id": owner,
        "sender": {"user_id": owner}, "message_id": message_id, "time": at.timestamp(),
        "message": [{"type": "text", "data": {"text": text}}],
    }


async def replay(args: argparse.Namespace) -> list[dict]:
    from qichi.config import load_config
    from qichi.runtime import MODEL_MANIFESTS, build_runtime, load_provider_capability_evidence
    from qichi.storage.database import Database

    config = load_config(args.config)
    bot_qq = args.bot_qq or running_bot_qq()
    if not bot_qq:
        raise RuntimeError("bot account is unknown; pass --bot-qq")
    manifest = MODEL_MANIFESTS[config.llm.primary.model]
    # 2026-09-14：证据文件按主模型推导 —— 写死成 flash 会让 pro 配置在这里直接报
    # ModelCapabilityError（和 start_production.ps1 是同一类坑）。
    evidence = load_provider_capability_evidence(
        PROJECT_ROOT / "runtime" / f"{config.llm.primary.model}-capability.json",
        provider=config.llm.provider,
        model_id=config.llm.primary.model,
    )
    capability = manifest.capability_for(
        config.llm.primary.model, provider=config.llm.provider, provider_evidence=evidence
    )
    counter = manifest.load_token_counter(PROJECT_ROOT / "runtime" / "model-cache" / "v4-tokenizer.json")

    results: list[dict] = []
    now = datetime.now(timezone.utc)
    questions = load_questions(args)
    for index, (label, question) in enumerate(questions, start=1):
        # Long runs must not look stuck: one line per question, flushed, and the
        # model call itself can take tens of seconds on a full production context.
        print(f"[{index}/{len(questions)}] {label}: 复制副本并装配上下文…", file=sys.stderr, flush=True)
        copy = args.workdir / f"copy-{index}.sqlite3"
        copy_database(args.database, copy)
        local = replace(config, storage=replace(config.storage, database_path=str(copy)))
        database = Database(copy)
        if args.context_only:
            llm = NullLLM()
        else:
            llm = RecordingLLM(
                local.llm.base_url, local.llm.api_key, model=local.llm.primary.model,
                temperature=local.llm.primary.temperature, top_p=local.llm.primary.top_p,
                max_output_tokens=local.llm.primary.max_output_tokens,
                timeout_seconds=local.llm.primary.timeout_seconds,
            )
        components = build_runtime(
            local, project_root=PROJECT_ROOT, database=database, onebot_client=RecordingOneBot(),
            bot_qq=bot_qq, model_capability=capability, token_counter=counter, llm_client=llm,
        )
        try:
            await components.application.handle_onebot(
                _raw(950000 + index, now, question, str(config.app.owner_qq), bot_qq),
                received_at_utc=now,
            )
            row = database.connection.execute(
                "SELECT details_json FROM turn_trace_events WHERE phase='context' ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            trace = json.loads(row["details_json"]) if row is not None else {}
            blob = "\n".join(message.content for message in llm.calls[-1]) if llm.calls else ""
            results.append({
                "label": label,
                "question": question,
                "reason": trace.get("memory_detail_reason"),
                "key": trace.get("memory_detail_key"),
                "match_count": trace.get("memory_detail_match_count"),
                "indexed": trace.get("memory_detail_indexed"),
                "recall_note": trace.get("memory_recall_note"),
                "fragments": len(trace.get("memory_detail_fragments") or []),
                "details": trace.get("memory_detail_count"),
                "details_tokens": (trace.get("category_tokens") or {}).get("memory_details"),
                "index_lines": [line.strip() for line in blob.splitlines()
                                if line.startswith("- ") and ("未展开" in line or "已在本轮展开" in line)],
                "detail_labels": [line.strip() for line in blob.splitlines() if line.startswith("[片段 ")],
                "reply": (llm.replies[-1] if llm.replies else "")[:400].replace("\n", " / "),
            })
        finally:
            await llm.close()
            database.close()
        print(f"[{index}/{len(questions)}] 完成：判据={results[-1]['reason']} "
              f"钥匙={results[-1]['key']} 明细={results[-1]['details']}", file=sys.stderr, flush=True)
        if not args.keep:
            copy.unlink()
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(BANNER, file=sys.stderr)
    try:
        results = asyncio.run(replay(args))
    except Exception as error:  # noqa: BLE001 - the report is the product here
        print(f"copy_replay: replay failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 1
    print(render_report(results))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"copy_replay: wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
