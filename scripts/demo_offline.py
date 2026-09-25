"""离线端到端演示：不需要 NapCat、不需要任何 API key、零成本。

跑的是**真正的**引擎链路——真实的事件持久化、真实的上下文装配、真实的发送与
outbox——只把两个外部依赖换成了假的：

* 模型：返回固定回复，并把每一轮**真正装配出来的上下文**打印给你看；
* OneBot/NapCat：只记录它被要求发送什么，不连任何网络。

所以这个脚本能说明的是「引擎把什么喂给模型、又把什么发了出去」，
而不是「模型会说什么」。后者需要你自己的 key 和角色核心。

    python scripts/demo_offline.py
    python scripts/demo_offline.py --db demo.sqlite3   # 保留数据库自己查
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.storage.database import Database


OWNER_QQ = "10001"
BOT_QQ = "20001"
NOW = datetime(2026, 9, 20, 20, 0, tzinfo=timezone.utc)
CONTEXT_WINDOW = 262_144

# 合成对话：措辞是中性示例，不对应任何真实往来。
SESSION = (
    ("在吗", "在呢，刚把桌子收拾完。"),
    ("今天降温了，你记得添件衣服", "记下了。你也别硬扛着。"),
    ("我晚点再来找你", "好，我在这儿。"),
)


class CharCounter:
    """按字符计数的替身：真实运行时用带哈希校验的 tokenizer 产物。"""

    def count_text(self, text: str) -> int:
        return len(text)


class OfflineLLM:
    """固定回复的假模型；它唯一的用处是把装配好的上下文交出来。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[object, ...]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        text = self.replies.pop(0) if self.replies else "（演示到此为止。）"
        return LLMGeneration(text, "primary", "offline-demo", 1, 1, 1.0)

    async def close(self) -> None:
        return None


class OfflineOneBot:
    """不连网的假 OneBot：只记录发送动作。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self.connection_id = "offline-demo"

    async def send_private_msg(self, user_id, message):
        self.sent.append((str(user_id), message))
        return {"message_id": 900 + len(self.sent)}

    async def close(self) -> None:
        return None


def build_application(database: Database, llm: OfflineLLM, onebot: OfflineOneBot) -> G0Application:
    evidence = ProviderCapabilityEvidence(
        "offline", "offline-demo", CONTEXT_WINDOW, "https://example.invalid/offline-demo", NOW
    )
    capability = ModelCapability("offline-demo", CONTEXT_WINDOW, "offline", evidence)
    builder = ContextBuilder(
        CharCounter(),
        capability,
        preferred_window_tokens=CONTEXT_WINDOW,
        max_window_tokens=CONTEXT_WINDOW,
        output_reserve_tokens=1_024,
    )
    engine = DialogueEngine(
        llm,
        OutputGuard(CharCounter(), 2_048),
        ResponseProtocol(face_keys=(), reaction_keys=()),
    )
    return G0Application(
        database,
        builder,
        engine,
        onebot,
        owner_qq=OWNER_QQ,
        bot_qq=BOT_QQ,
        # 角色核心由部署方提供；这里给的是最小占位，见 doc/运行时角色核心.md。
        role_core="你是角色。你不补证据没有提供的细节。",
        clock=lambda: NOW,
    )


def raw_message(message_id: int, text: str, *, timestamp: datetime) -> dict:
    return {
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "self_id": int(BOT_QQ),
        "user_id": OWNER_QQ,
        "target_id": OWNER_QQ,
        "sender": {"user_id": OWNER_QQ},
        "message_id": message_id,
        "time": timestamp.timestamp(),
        "message": [{"type": "text", "data": {"text": text}}],
    }


def describe_context(messages) -> None:
    print("   装配出来的上下文：")
    for index, message in enumerate(messages):
        content = message.content if isinstance(message.content, str) else str(message.content)
        head = content.replace("\n", " / ")[:96]
        print(f"     [{index}] {message.role:<9} {len(content):>6} 字符  {head}")


async def _run(args: argparse.Namespace) -> int:
    temporary = None
    if args.db is None:
        temporary = tempfile.TemporaryDirectory()
        db_path = Path(temporary.name) / "demo.sqlite3"
    else:
        db_path = Path(args.db).resolve()
        if db_path.exists():
            db_path.unlink()

    database = Database(db_path)
    llm = OfflineLLM([reply for _, reply in SESSION])
    onebot = OfflineOneBot()
    application = build_application(database, llm, onebot)

    print("=== 离线端到端演示（无网络、无密钥、零成本）===")
    print(f"数据库: {db_path}")
    print(f"对话对象 QQ: {OWNER_QQ}   机器人 QQ: {BOT_QQ}")
    print("说明: 模型是假替身，只回固定文本；下面每一轮的上下文都是引擎真正装配出来的。")
    print()

    try:
        for index, (user_text, expected_reply) in enumerate(SESSION, start=1):
            at = NOW + timedelta(minutes=2 * index)
            print(f"--- 第 {index} 轮 ---")
            print(f"   收到: {user_text}")
            delivered = await application.handle_onebot(
                raw_message(500 + index, user_text, timestamp=at), received_at_utc=at
            )
            describe_context(llm.calls[-1])
            text = getattr(delivered, "text", None)
            print(f"   发出: {text}")
            assert text == expected_reply, (text, expected_reply)
            print()

        events = database.connection.execute(
            "SELECT direction, actor, text, status FROM conversation_events ORDER BY sequence"
        ).fetchall()
        print("=== 落库的事件（原文持久化）===")
        for row in events:
            print(f"   {row['direction']:<8} {row['actor']:<6} {row['status']:<9} {row['text']}")
        outbox = database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        traces = database.connection.execute(
            "SELECT COUNT(*) FROM turn_trace_events"
        ).fetchone()[0]
        print()
        print(f"outbox 行数: {outbox}   轮次追踪事件: {traces}")
        print("（轮次追踪只记结构事实，不记对话原文——可以自己 SELECT 看一眼。）")
    finally:
        await onebot.close()
        database.close()
        if temporary is not None:
            temporary.cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=None, help="保留演示数据库到指定路径")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
