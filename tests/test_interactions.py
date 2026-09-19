from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.domain.dialogue import DialogueResult
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.storage.database import Database
from qichi.storage.outbox_repository import OutboxRepository
from qichi.transport.onebot_client import OneBotActionError, OneBotTimeoutError
from qichi.transport.normalizer import normalize_event
from qichi.transport.sender import Sender

NOW = datetime(2026, 8, 28, 10, tzinfo=timezone.utc)
OWNER = "10001"
BOT = "20001"


class Counter:
    def count_text(self, text):
        return len(text)


class FakeLLM:
    def __init__(self, outputs, before=None):
        self.outputs = list(outputs)
        self.before = before
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.before:
            await self.before(len(self.calls))
        return LLMGeneration(self.outputs.pop(0), "primary", "fake", 1, 1, 1)


class FakeNapCat:
    def __init__(self, poke_error=None, reaction_error=None, poke_started=None, poke_release=None, poke_response=None):
        self.pokes = []
        self.texts = []
        self.reactions = []
        self.poke_error = poke_error
        self.reaction_error = reaction_error
        self.poke_started = poke_started
        self.poke_release = poke_release
        # The real NapCat answers send_poke with no data payload at all.
        self.poke_response = {"ok": True} if poke_response is None else poke_response

    async def send_poke(self, user_id):
        self.pokes.append(str(user_id))
        if self.poke_started:
            self.poke_started.set()
        if self.poke_release:
            await self.poke_release.wait()
        if self.poke_error:
            raise self.poke_error
        return self.poke_response

    async def send_private_msg(self, user_id, message):
        self.texts.append((str(user_id), message))
        return {"message_id": 700 + len(self.texts)}

    async def set_msg_emoji_like(self, *args, **kwargs):
        self.reactions.append((args, kwargs))
        if self.reaction_error:
            raise self.reaction_error
        return {"ok": True}


def poke(*, sender=OWNER, target=BOT, timestamp=NOW.timestamp()):
    return {
        "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
        "self_id": int(BOT), "user_id": int(OWNER), "sender_id": int(sender),
        "target_id": int(target), "time": timestamp,
    }


def message(message_id, text, timestamp=NOW.timestamp()):
    return {
        "post_type": "message", "message_type": "private", "sub_type": "friend",
        "self_id": int(BOT), "user_id": OWNER, "target_id": OWNER,
        "sender": {"user_id": OWNER}, "message_id": message_id, "time": timestamp,
        "message": [{"type": "text", "data": {"text": text}}],
    }


def make_app(database, llm, napcat, *, clock=lambda: NOW, worker=None, engine=None, face_catalog=None):
    evidence = ProviderCapabilityEvidence("fake", "fake", 262144, "test", NOW)
    builder = ContextBuilder(Counter(), ModelCapability("fake", 262144, "fake", evidence), output_reserve_tokens=10)
    engine = engine or DialogueEngine(llm, OutputGuard(Counter(), 2048), ResponseProtocol(face_keys={"shy"}, reaction_keys={"heart"}))
    return G0Application(database, builder, engine, napcat, owner_qq=OWNER, bot_qq=BOT, role_core="你是角色。", clock=clock, memory_worker=worker, face_catalog=face_catalog or {}, reaction_catalog={"heart": 66})


class RecordingEngine:
    def __init__(self, text="互动回复"):
        self.inputs = []
        self.text = text

    async def generate(self, value):
        self.inputs.append(value)
        return DialogueResult(self.text, None, None, "primary", value.context_version)


def test_normalizer_filters_non_owner_and_preserves_poke_structure():
    non_owner = poke()
    non_owner["user_id"] = 99999
    assert normalize_event(non_owner, owner_qq=OWNER, bot_qq=BOT, received_at_utc=NOW) is None
    event = normalize_event(poke(), owner_qq=OWNER, bot_qq=BOT, received_at_utc=NOW)
    assert event is not None
    assert event.kind == "poke" and event.text is None and event.actor == "mumo"
    assert event.visible_handle == "M0"


@pytest.mark.asyncio
async def test_poke_fingerprint_is_stable_but_distinguishes_time_and_direction(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["一次", "两次"])
    app = make_app(db, llm, napcat)
    try:
        payload = poke()
        assert await app.handle_onebot(payload, received_at_utc=NOW)
        assert await app.handle_onebot(payload, received_at_utc=NOW) is None
        later = poke(timestamp=NOW.timestamp() + 1)
        assert await app.handle_onebot(later, received_at_utc=NOW + timedelta(seconds=1))
        outbound = poke(sender=BOT, target=OWNER)
        assert await app.handle_onebot(outbound, received_at_utc=NOW) is None
        non_owner = poke()
        non_owner["user_id"] = 99999
        assert await app.handle_onebot(non_owner, received_at_utc=NOW) is None
        assert db.connection.execute("SELECT COUNT(*) FROM conversation_events WHERE kind='poke'").fetchone()[0] == 2
        assert len(napcat.pokes) == 2 and len(llm.calls) == 2
        assert db.connection.execute("SELECT context_version FROM conversation_cursors").fetchone()[0] == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_poke_persists_action_before_generation_and_uses_interaction(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["我也戳回来了"])
    app = make_app(db, llm, napcat)
    try:
        result = await app.handle_onebot(poke(), received_at_utc=NOW)
        assert result is not None and result.text == "我也戳回来了"
        assert napcat.pokes == [OWNER]
        assert len(llm.calls) == 1
        assert "poke" in "\n".join(message.content for message in llm.calls[0])
        prompt = "\n".join(message.content for message in llm.calls[0])
        assert "[[qq:react:<key>]]" not in prompt
        assert "qq_face" not in prompt.split("本轮可执行平台动作:", 1)[1].splitlines()[0]
        assert "reaction" not in prompt.split("本轮可执行平台动作:", 1)[1].splitlines()[0]
        row = db.connection.execute("SELECT kind, text FROM conversation_events WHERE direction='inbound'").fetchone()
        assert tuple(row) == ("poke", None)
        assert db.connection.execute("SELECT COUNT(*) FROM outbox WHERE payload_json LIKE '%poke%'").fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_duplicate_poke_does_not_regenerate_or_repoke(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["一次"])
    app = make_app(db, llm, napcat)
    try:
        payload = poke()
        assert await app.handle_onebot(payload, received_at_utc=NOW)
        assert await app.handle_onebot(payload, received_at_utc=NOW) is None
        assert len(llm.calls) == 1 and len(napcat.pokes) == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    (
        (OneBotTimeoutError("timeout"), "unknown"),
        (OneBotActionError("rejected"), "failed"),
    ),
)
@pytest.mark.asyncio
async def test_poke_failure_is_terminal_and_does_not_block_text(
    tmp_path, error, expected_status
):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat(error)
    llm = FakeLLM(["戳回", "文字"])
    app = make_app(db, llm, napcat)
    try:
        payload = poke()
        first = await app.handle_onebot(payload, received_at_utc=NOW)
        assert first is not None and first.text == "戳回"
        event_id = db.connection.execute(
            "SELECT event_id FROM conversation_events WHERE kind='poke'"
        ).fetchone()[0]
        operation = OutboxRepository(db).get(Sender.poke_operation_key(event_id))
        assert operation.payload["action_kind"] == "poke"
        assert operation.status == expected_status
        napcat.poke_error = None
        assert await app.handle_onebot(payload, received_at_utc=NOW) is None
        await app.handle_onebot(message(1, "继续"), received_at_utc=NOW + timedelta(seconds=1))
        assert napcat.pokes == [OWNER]
        assert len(llm.calls) == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_new_text_during_poke_generation_cancels_stale_interaction(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def before(call_number):
        if call_number == 1:
            entered.set()
            await release.wait()

    llm = FakeLLM(["过时", "新文字"], before)
    app = make_app(db, llm, napcat)
    try:
        old = asyncio.create_task(app.handle_onebot(poke(), received_at_utc=NOW))
        await entered.wait()
        new = asyncio.create_task(app.handle_onebot(message(2, "抢先", NOW.timestamp() + 1), received_at_utc=NOW + timedelta(seconds=1)))
        await asyncio.sleep(0)
        release.set()
        old_result, new_result = await asyncio.gather(old, new)
        assert old_result is None and new_result is not None and new_result.text == "新文字"
        assert [item[1][0]["data"]["text"] for item in napcat.texts] == ["新文字"]
        poke_sequence = db.connection.execute("SELECT sequence FROM conversation_events WHERE kind='poke'").fetchone()[0]
        processed = db.connection.execute("SELECT last_processed_sequence FROM conversation_cursors").fetchone()[0]
        assert processed >= poke_sequence
    finally:
        db.close()


@pytest.mark.asyncio
async def test_engine_protocol_tail_reaction_survives_text_success(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["收到\n[[qq:reply:M0]]\n[[qq:react:heart]]"])
    app = make_app(db, llm, napcat)
    try:
        result = await app.handle_onebot(message(1, "你好"), received_at_utc=NOW)
        assert result is not None and result.status == "sent"
        assert napcat.reactions == [(('1', '66'), {'set': True})]
        prompt = "\n".join(message.content for message in llm.calls[0])
        assert "本轮可执行平台动作: text, reply, reaction" in prompt
        assert "消息回应格式 [[qq:react:<key>]]，keys=heart" in prompt
    finally:
        db.close()


@pytest.mark.asyncio
async def test_interaction_source_and_worker_filter_poke_but_keep_sent_text(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    class Worker:
        def notify_reliable_activity(self, conversation_id):
            self.conversation_id = conversation_id
            return True
    worker = Worker()
    recorder = RecordingEngine()
    app = make_app(db, FakeLLM([]), napcat, worker=worker, engine=recorder)
    try:
        result = await app.handle_onebot(poke(), received_at_utc=NOW)
        assert result is not None
        assert recorder.inputs[0].source == "interaction"
        assert worker.conversation_id == OWNER
        assert db.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE kind='poke'"
        ).fetchone()[0] == 1
        sent = db.connection.execute(
            "SELECT metadata_json FROM conversation_events WHERE actor='qichi' AND status='sent'"
        ).fetchall()
        assert any('"source": "interaction"' in row[0] for row in sent)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_reaction_failure_does_not_change_sent_text(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat(reaction_error=OneBotTimeoutError("unknown"))
    llm = FakeLLM(["正文\n[[qq:react:heart]]"])
    app = make_app(db, llm, napcat)
    try:
        result = await app.handle_onebot(message(1, "你好"), received_at_utc=NOW)
        assert result is not None and result.status == "sent" and result.text == "正文"
        reaction = OutboxRepository(db).outstanding()
        assert any(item.payload["action_kind"] == "reaction" and item.status == "unknown" for item in reaction)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_new_text_during_poke_dispatch_cancels_before_llm(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    started = asyncio.Event()
    release = asyncio.Event()
    napcat = FakeNapCat(poke_started=started, poke_release=release)
    llm = FakeLLM(["文字"])
    app = make_app(db, llm, napcat)
    try:
        old = asyncio.create_task(app.handle_onebot(poke(), received_at_utc=NOW))
        await started.wait()
        new = asyncio.create_task(app.handle_onebot(message(3, "抢先", NOW.timestamp() + 1), received_at_utc=NOW + timedelta(seconds=1)))
        await asyncio.sleep(0)
        release.set()
        old_result, new_result = await asyncio.gather(old, new)
        assert old_result is None and new_result is not None
        assert len(llm.calls) == 1 and napcat.texts[0][1][0]["data"]["text"] == "文字"
        poke_sequence = db.connection.execute("SELECT sequence FROM conversation_events WHERE kind='poke'").fetchone()[0]
        assert db.connection.execute("SELECT last_processed_sequence FROM conversation_cursors").fetchone()[0] >= poke_sequence
    finally:
        db.close()


@pytest.mark.asyncio
async def test_concurrent_duplicate_pokes_are_processed_once(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["一次"])
    app = make_app(db, llm, napcat)
    try:
        payload = poke()
        results = await asyncio.gather(
            app.handle_onebot(payload, received_at_utc=NOW),
            app.handle_onebot(payload, received_at_utc=NOW),
        )
        assert sum(result is not None for result in results) == 1
        assert len(napcat.pokes) == 1 and len(llm.calls) == 1
        assert db.connection.execute("SELECT context_version FROM conversation_cursors").fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_app_exposes_face_only_when_catalog_has_executable_key(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    recorder = RecordingEngine()
    app = make_app(db, FakeLLM([]), napcat, engine=recorder, face_catalog={"shy": 6})
    try:
        await app.handle_onebot(message(10, "你好"), received_at_utc=NOW)
        prompt = "\n".join(item.content for item in recorder.inputs[0].role_messages)
        assert "qq_face" in prompt
        assert "QQ face 格式 [[qq:face:<key>]]，keys=shy" in prompt

        invalid_recorder = RecordingEngine("仍是普通文字")
        invalid_app = make_app(
            db,
            FakeLLM([]),
            napcat,
            engine=invalid_recorder,
            face_catalog={"shy": True},
        )
        await invalid_app.handle_onebot(
            message(11, "继续", NOW.timestamp() + 1),
            received_at_utc=NOW + timedelta(seconds=1),
        )
        invalid_prompt = "\n".join(
            item.content for item in invalid_recorder.inputs[0].role_messages
        )
        assert "qq_face" not in invalid_prompt.split(
            "本轮可执行平台动作:", 1
        )[1].splitlines()[0]
        assert "[[qq:face:<key>]]" not in invalid_prompt
    finally:
        db.close()


@pytest.mark.asyncio
async def test_sender_poke_requires_mumo_received_event(tmp_path):
    from qichi.domain.events import ConversationEvent

    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    sender = Sender(db, napcat, face_catalog={}, reaction_catalog={})
    fields = dict(
        event_id="poke-event", platform_event_id="poke:x", platform_message_id=None,
        conversation_id=OWNER, sequence=0, direction="inbound", actor="mumo", kind="poke",
        text=None, message_segments=(), reply_to_event_id=None, reply_to_platform_message_id=None,
        occurred_at_utc=NOW, received_at_utc=NOW, status="received", metadata={},
    )
    try:
        event = ConversationEvent(**fields)
        from qichi.storage.event_repository import EventRepository

        EventRepository(db).insert(event)
        assert await sender.send_poke(event, owner_qq=OWNER, occurred_at_utc=NOW) == "sent"
        assert napcat.pokes == [OWNER]
        for changes in ({"actor": "qichi"}, {"status": "sent"}):
            invalid = ConversationEvent(**{**fields, **changes, "event_id": "invalid-" + changes["status"] if "status" in changes else "invalid-actor"})
            with pytest.raises(ValueError):
                await sender.send_poke(invalid, owner_qq=OWNER, occurred_at_utc=NOW)
        assert len(napcat.pokes) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_sender_poke_accepts_a_reply_without_payload(tmp_path):
    """NapCat answers send_poke with no data payload at all.

    That is a success -- the envelope already said status ok -- and it must not be
    recorded as unknown, which is what left fifteen real pokes stuck.
    """
    from qichi.domain.events import ConversationEvent
    from qichi.storage.event_repository import EventRepository
    from qichi.storage.outbox_repository import OutboxRepository

    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    napcat.poke_response = None
    sender = Sender(db, napcat, face_catalog={}, reaction_catalog={})
    fields = dict(
        event_id="poke-event-bare", platform_event_id="poke:bare", platform_message_id=None,
        conversation_id=OWNER, sequence=0, direction="inbound", actor="mumo", kind="poke",
        text=None, message_segments=(), reply_to_event_id=None, reply_to_platform_message_id=None,
        occurred_at_utc=NOW, received_at_utc=NOW, status="received", metadata={},
    )
    try:
        event = ConversationEvent(**fields)
        EventRepository(db).insert(event)
        assert await sender.send_poke(event, owner_qq=OWNER, occurred_at_utc=NOW) == "sent"
        record = OutboxRepository(db).get(Sender.poke_operation_key(event.event_id))
        assert record.status == "sent"
    finally:
        db.close()
