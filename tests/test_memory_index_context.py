from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
import sys

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory_details import MemoryDetailDraft
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MemoryDetailRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import NOW, OWNER, FakeLLM, FakeNapCat, app, seed_event  # noqa: E402

SECRET_QUOTE = "SECRET-QUOTE-MUST-NEVER-APPEAR"
SECRET_FACT = "SECRET-FACT-MUST-NEVER-APPEAR"


def _fragment(database, events, *, privacy: str, policy: str, key: str):
    repository = MemoryDetailRepository(database)
    fragment = repository.build_fragment(events, key, None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail=SECRET_FACT, exact_quote=item.text, source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class=privacy, recall_policy=policy, evidence=((item.event_id, "source"),),
        )
        for index, item in enumerate(events)
    )
    details = repository.build_details(fragment, drafts, events)
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=events, details=details)


def _turn(application, events, text: str, event_id: str) -> str:
    event = events.insert(ConversationEvent(
        event_id, "pe-" + event_id, "pm-" + event_id, OWNER, 999, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    ))
    built = asyncio.run(application._build_input(event, 1))
    return " ".join(str(message) for message in built.role_messages)


def test_the_index_is_resident_and_carries_no_content(tmp_path):
    database = Database(tmp_path / "index-context.sqlite3")
    try:
        events = EventRepository(database)
        first = seed_event(events, "adult-1", SECRET_QUOTE, at=NOW - timedelta(days=1))
        second = seed_event(events, "adult-2", SECRET_QUOTE, at=NOW - timedelta(days=1) + timedelta(minutes=5))
        _fragment(database, (first, second), privacy="adult", policy="explicit_request_only", key="idx-1")
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "随便聊聊", "ctx-1")

        assert "[最近片段索引]" in rendered, "the index must be resident in ordinary conversation"
        assert "含成人内容" in rendered, "the level is what tells her the topic exists"
        assert "细节未展开" in rendered
        assert SECRET_QUOTE not in rendered
        assert SECRET_FACT not in rendered
    finally:
        database.close()


def test_without_history_the_index_block_is_absent(tmp_path):
    database = Database(tmp_path / "index-empty.sqlite3")
    try:
        events = EventRepository(database)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "随便聊聊", "ctx-2")

        assert "[最近片段索引]" not in rendered
    finally:
        database.close()
