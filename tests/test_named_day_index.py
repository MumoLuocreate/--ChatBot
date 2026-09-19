from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory_details import MemoryDetailDraft
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MemoryDetailRepository

NOW = datetime(2026, 9, 11, 6, 0, tzinfo=timezone.utc)  # 本地 14:00
OWNER, BOT = "123456", "10001"
SHANGHAI = ZoneInfo("Asia/Shanghai")


class Counter:
    def count_text(self, text):
        return len(text)


class FakeLLM:
    async def generate(self, messages, **kwargs):
        return LLMGeneration("好", "primary", "fake", 1, 1, 1.0)


class FakeNapCat:
    async def send_private_msg(self, user_id, message):
        return {"message_id": 1}


def _event(tree, sequence, at, text, actor="mumo", direction="inbound"):
    return tree.insert(ConversationEvent(
        f"e{sequence}", None, f"pm-{sequence}", OWNER, sequence, direction, actor, "text", text,
        (MessageSegment("text", {"text": text}),), None, None, at, at,
        "received" if direction == "inbound" else "sent", {},
    ))


def _fragment(database, events, key):
    repository = MemoryDetailRepository(database)
    fragment = repository.build_fragment(events, key, None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail=item.text, exact_quote=item.text, source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class="adult", recall_policy="explicit_request_only",
            evidence=((item.event_id, "source"),),
        )
        for index, item in enumerate(events)
    )
    details = repository.build_details(fragment, drafts, events)
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=events, details=details)
    return fragment


def application(database):
    evidence = ProviderCapabilityEvidence("fake", "fake-v4", 262_144, "local", NOW)
    builder = ContextBuilder(
        Counter(), ModelCapability("fake-v4", 262_144, "fake", evidence), output_reserve_tokens=1024
    )
    engine = DialogueEngine(
        FakeLLM(), OutputGuard(Counter(), 2048), ResponseProtocol(face_keys=(), reaction_keys=())
    )
    return G0Application(
        database, builder, engine, FakeNapCat(), owner_qq=OWNER, bot_qq=BOT,
        role_core="你是角色。", clock=lambda: NOW, local_zone=SHANGHAI,
    )


def test_a_named_day_keeps_its_fragment_without_any_recall_request(tmp_path):
    """只说日期、不带回顾请求时，索引仍要保住那天的行（明细照旧封着）。"""

    database = Database(tmp_path / "named-day.sqlite3")
    try:
        events = EventRepository(database)
        first = _event(events, 1, NOW - timedelta(days=2, hours=1), "九号中午的第一句原话")
        second = _event(events, 2, NOW - timedelta(days=2), "九号中午的第二句原话")
        fragment = _fragment(database, (first, second), "ninth")
        client = application(database)

        named = _event(events, 3, NOW, "九号中午我们聊了啥")
        assert client._named_day_fragments(named, (first, second), NOW) == (fragment.fragment_id,)

        other_day = _event(events, 4, NOW, "昨天中午我们聊了啥")
        assert client._named_day_fragments(other_day, (first, second), NOW) == (), "别的日子不许顺手带出九号"

        no_day = _event(events, 5, NOW, "在忙吗")
        assert client._named_day_fragments(no_day, (), NOW) == ()
    finally:
        database.close()


def test_the_naming_window_is_three_turns_for_the_index_too(tmp_path):
    """2026-09-12 T3：钥匙和索引钉住共用同一个窗口，也只认过去的日子。"""

    database = Database(tmp_path / "named-window.sqlite3")
    try:
        events = EventRepository(database)
        first = _event(events, 1, NOW - timedelta(days=2, hours=1), "九号中午的第一句原话")
        second = _event(events, 2, NOW - timedelta(days=2), "九号中午的第二句原话")
        fragment = _fragment(database, (first, second), "ninth")
        client = application(database)

        named = _event(events, 3, NOW - timedelta(minutes=6), "九号中午我们聊了啥")
        near = tuple(
            _event(events, 4 + index, NOW - timedelta(minutes=5 - index), text)
            for index, text in enumerate(("嗯嗯", "我在"))
        )
        far = tuple(
            _event(events, 6 + index, NOW - timedelta(minutes=3 - index), text)
            for index, text in enumerate(("嗯嗯", "我在", "然后呢"))
        )
        current = _event(events, 9, NOW, "那天的事再说说")

        assert client._named_day_fragments(current, (named, *near), NOW) == (fragment.fragment_id,)
        assert client._named_day_fragments(current, (named, *far), NOW) == (), "出了三条窗口就不再继承"

        today = _event(events, 10, NOW - timedelta(minutes=1), "今天中午我们聊了啥")
        assert client._named_day_fragments(current, (today,), NOW) == (), "今天不是回顾目标"
    finally:
        database.close()


def test_a_named_day_without_stored_fragments_pins_nothing(tmp_path):
    database = Database(tmp_path / "empty-day.sqlite3")
    try:
        events = EventRepository(database)
        client = application(database)

        current = _event(events, 1, NOW, "八号那天我们聊了啥")

        assert client._named_day_fragments(current, (), NOW) == ()
    finally:
        database.close()
