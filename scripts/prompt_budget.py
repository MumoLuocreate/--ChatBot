# -*- coding: utf-8 -*-
"""提示分块占用表：一轮里她到底被交代了多少件事，各占多少。

只读源库 + 副本 + 空模型（零外呼、不调用任何模型）。用途：任何"提示变复杂了"的说法，
先跑它拿数字，再谈改不改。见 doc/设计-20260915-提示分块收敛.md。

用法：
    python scripts/prompt_budget.py
    python scripts/prompt_budget.py --question "兔子，在吗"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
from dataclasses import replace
from datetime import datetime, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _hydrate_user_environment() -> None:
    """把注册表里的持久环境变量补进本进程（与 start_production.ps1 的白名单一致）。"""

    try:
        import winreg
    except ImportError:
        return
    names = ("QICHI_OWNER_QQ", "NAPCAT_WS_URL", "NAPCAT_HTTP_URL", "NAPCAT_ACCESS_TOKEN",
             "SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            for name in names:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if value and not os.environ.get(name):
                    os.environ[name] = value
    except OSError:
        return


def _section_of(line: str) -> str | None:
    """一行是不是块抬头。索引类抬头带括号说明，不以 ] 结尾，所以不能用 endswith。"""

    if not line.startswith("["):
        return None
    head = line.split("（")[0].strip()
    if "]" not in head:
        return None
    return head[: head.index("]") + 1]


async def measure(question: str, bot_qq: str) -> int:
    from copy_replay import NullLLM, RecordingOneBot, copy_database, _raw
    from qichi.config import load_config
    from qichi.runtime import MODEL_MANIFESTS, build_runtime, load_provider_capability_evidence
    from qichi.storage.database import Database

    config = load_config(PROJECT_ROOT / "config.example.yaml")
    manifest = MODEL_MANIFESTS[config.llm.primary.model]
    evidence = load_provider_capability_evidence(
        PROJECT_ROOT / "runtime" / (config.llm.primary.model + "-capability.json"),
        provider=config.llm.provider, model_id=config.llm.primary.model)
    capability = manifest.capability_for(
        config.llm.primary.model, provider=config.llm.provider, provider_evidence=evidence)
    counter = manifest.load_token_counter(PROJECT_ROOT / "runtime" / "model-cache" / "v4-tokenizer.json")
    work = PROJECT_ROOT / "_tmp" / "prompt-budget"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "copy.sqlite3"
    copy_database(PROJECT_ROOT / "data" / "qichi.sqlite3", copy)
    local = replace(config, storage=replace(config.storage, database_path=str(copy)))
    database = Database(copy)
    llm = NullLLM()
    components = build_runtime(
        local, project_root=PROJECT_ROOT, database=database, onebot_client=RecordingOneBot(),
        bot_qq=bot_qq, model_capability=capability, token_counter=counter, llm_client=llm)
    try:
        await components.application.handle_onebot(
            _raw(999_100, datetime.now(timezone.utc), question, str(config.app.owner_qq), bot_qq),
            received_at_utc=datetime.now(timezone.utc))
    finally:
        database.close()
    messages = llm.calls[-1]
    total = sum(len(message.content) for message in messages)
    print("问题: %s" % question)
    print("消息 %d 条，共 %d 字（%d token）" % (len(messages), total, counter(messages and "")) if False else
          "消息 %d 条，共 %d 字" % (len(messages), total))
    rows: list[tuple[str, int]] = []
    for index, message in enumerate(messages, start=1):
        label = None
        size = 0
        for line in message.content.splitlines():
            head = _section_of(line)
            if head is not None:
                if label is not None:
                    rows.append((label, size))
                label, size = head, 0
            size += len(line) + 1
        rows.append((label or ("<无抬头 #%d role=%s>" % (index, message.role)), size))
    for label, size in sorted(rows, key=lambda item: -item[1]):
        print("   %6d  %5.1f%%  %s" % (size, 100.0 * size / total, label))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="兔子，在吗")
    parser.add_argument("--bot-qq", default=None)
    args = parser.parse_args(argv)
    _hydrate_user_environment()
    bot_qq = args.bot_qq
    if not bot_qq:
        try:
            import json
            bot_qq = str(json.loads((PROJECT_ROOT / "runtime" / "qichi-ready.json").read_text(encoding="utf-8"))["bot_qq"])
        except Exception:
            bot_qq = "10001"
    return asyncio.run(measure(args.question, bot_qq))


if __name__ == "__main__":
    raise SystemExit(main())
