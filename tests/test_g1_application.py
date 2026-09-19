from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json

import pytest

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository
from qichi.transport.onebot_client import OneBotActionError, OneBotTimeoutError


NOW = datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc)
OWNER = "10001"
BOT = "20001"


class Counter:
    def count_text(self, text):
        return len(text)


class FakeLLM:
    def __init__(self, outputs, before=None):
        self.outputs, self.before, self.calls = list(outputs), before, []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        if self.before:
            await self.before(len(self.calls), messages)
        return LLMGeneration(self.outputs.pop(0), "primary", "fake-v4", 1, 1, 1.0)


class FakeNapCat:
    def __init__(self, errors=()):
        self.errors, self.sent, self.next_id = list(errors), [], 700

    async def send_private_msg(self, user_id, message):
        self.sent.append((str(user_id), message))
        if self.errors:
            error = self.errors.pop(0)
            if error:
                raise error
        result = {"message_id": self.next_id}
        self.next_id += 1
        return result

    async def set_msg_emoji_like(self, *args, **kwargs):
        raise AssertionError("T19A does not send reactions")


class NotAwaitableWorker:
    class Poison:
        def __await__(self):
            raise AssertionError("worker enqueue must not be awaited")
            yield

    def __init__(self):
        self.calls = []

    def notify_reliable_activity(self, conversation_id):
        self.calls.append((conversation_id,))
        return True


def raw(message_id, text, at=NOW, *, reply_to=None):
    message = []
    if reply_to is not None:
        message.append({"type": "reply", "data": {"id": str(reply_to)}})
    message.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message", "message_type": "private", "sub_type": "friend",
        "self_id": int(BOT), "user_id": OWNER, "target_id": OWNER,
        "sender": {"user_id": OWNER}, "message_id": message_id, "time": at.timestamp(),
        "message": message,
    }


def builder():
    evidence = ProviderCapabilityEvidence("fake", "fake-v4", 262_144, "local", NOW)
    capability = ModelCapability("fake-v4", 262_144, "fake", evidence)
    return ContextBuilder(Counter(), capability, output_reserve_tokens=1024)


def app(database, llm, napcat, worker=None, clock=lambda: NOW):
    engine = DialogueEngine(llm, OutputGuard(Counter(), 2048),
                            ResponseProtocol(face_keys=(), reaction_keys=()))
    return G0Application(
        database, builder(), engine, napcat, owner_qq=OWNER, bot_qq=BOT,
        role_core="你是角色。", clock=clock, memory_worker=worker,
    )


def seed_event(
    repository,
    event_id,
    text,
    actor="mumo",
    direction="inbound",
    at=NOW - timedelta(hours=1),
    platform_message_id=None,
):
    return repository.insert(ConversationEvent(
        event_id, f"pe-{event_id}", platform_message_id or f"pm-{event_id}", OWNER, 999, direction, actor, "text", text,
        (MessageSegment("text", {"text": text}),), None, None, at, at,
        "sent" if direction == "outbound" else "received",
        {"generation_metadata": {"source": "dialogue"}} if direction == "outbound" else {},
    ))


def seed_memory(
    memories, memory_id, source, fact, quote, *, type="preference", status="active",
    supersedes=None, certainty="explicit", importance=2, temporal_scope="ongoing",
    reason="explicit_user_statement", evidence_role="source", additional_evidence=(),
):
    evidence = MemoryEvidence(
        memory_id, source.event_id, source.actor, quote, source.occurred_at_utc, evidence_role,
    )
    return memories.create(MemoryRecord(
        memory_id, type, fact, "explicit_statement", status, source.occurred_at_utc,
        None, supersedes, source.received_at_utc, (evidence, *additional_evidence),
        certainty, importance, temporal_scope, reason, source.received_at_utc,
    ))


def seed_relationships(database):
    events, memories = EventRepository(database), MemoryRepository(database)
    preference_source = seed_event(events, "preference-source", "我喜欢雨声")
    agreement_source = seed_event(events, "agreement-source", "我们约好周末聊书")
    agreement_acceptance = seed_event(
        events, "agreement-acceptance", "好，周末聊书", "qichi", "outbound",
    )
    old_source = seed_event(events, "old-source", "我以前喜欢清晨风")
    correction_source = seed_event(events, "correction-source", "其实现在更喜欢傍晚风")
    joke_source = seed_event(events, "joke-source", "开玩笑，我每天数云")
    qichi_source = seed_event(events, "qichi-source", "我说过我很在意用户", "qichi", "outbound")
    seed_memory(memories, "preference", preference_source, "用户喜欢雨声", "喜欢雨声")
    seed_memory(
        memories, "agreement", agreement_source, "周末一起聊书", "约好周末聊书",
        type="agreement", certainty="confirmed", reason="bilateral_agreement",
        evidence_role="proposal", additional_evidence=(MemoryEvidence(
            "agreement", agreement_acceptance.event_id, "qichi", "好，周末聊书",
            agreement_acceptance.occurred_at_utc, "acceptance",
        ),),
    )
    seed_memory(memories, "old", old_source, "用户喜欢清晨风", "以前喜欢清晨风")
    seed_memory(memories, "correction", correction_source, "用户现在更喜欢傍晚风", "现在更喜欢傍晚风",
                type="correction", supersedes="old", reason="user_correction", evidence_role="correction")
    seed_memory(memories, "joke", joke_source, "用户每天数云", "每天数云", status="candidate",
                certainty="unsupported", importance=0, temporal_scope="unclassified", reason="unsupported_or_transient")
    seed_memory(memories, "self", qichi_source, "角色表达过在意用户", "很在意用户", type="self_expression")


def test_recent_history_starts_after_a_real_long_gap(tmp_path):
    database = Database(tmp_path / "session-history.sqlite3")
    repository = EventRepository(database)
    try:
        old_user = seed_event(repository, "old-user", "旧会话内容", at=NOW - timedelta(hours=2))
        old_reply = seed_event(
            repository,
            "old-reply",
            "旧的角色台词",
            actor="qichi",
            direction="outbound",
            at=NOW - timedelta(hours=1, minutes=59),
        )
        new_user = seed_event(repository, "new-user", "新会话内容", at=NOW - timedelta(minutes=10))
        current = ConversationEvent(
            "current",
            "pe-current",
            "pm-current",
            OWNER,
            999,
            "inbound",
            "mumo",
            "text",
            "当前输入",
            (MessageSegment("text", {"text": "当前输入"}),),
            None,
            None,
            NOW,
            NOW,
            "received",
            {},
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        history = application._events_before(current)

        assert [event.event_id for event in history] == [new_user.event_id]
        assert old_user.event_id not in {event.event_id for event in history}
        assert old_reply.event_id not in {event.event_id for event in history}
    finally:
        database.close()


def test_recent_history_keeps_continuous_turns_inside_the_gap_threshold(tmp_path):
    database = Database(tmp_path / "continuous-history.sqlite3")
    repository = EventRepository(database)
    try:
        first = seed_event(repository, "first", "第一句", at=NOW - timedelta(minutes=30))
        second = seed_event(
            repository,
            "second",
            "第二句",
            actor="qichi",
            direction="outbound",
            at=NOW - timedelta(minutes=29),
        )
        current = ConversationEvent(
            "current",
            "pe-current",
            "pm-current",
            OWNER,
            999,
            "inbound",
            "mumo",
            "text",
            "当前输入",
            (MessageSegment("text", {"text": "当前输入"}),),
            None,
            None,
            NOW,
            NOW,
            "received",
            {},
        )
        history = app(database, FakeLLM(["回复"]), FakeNapCat())._events_before(current)

        assert [event.event_id for event in history] == [first.event_id, second.event_id]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_second_round_reads_relationships_retrieval_and_notifies_reliable_activity(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    seed_relationships(database)
    worker, llm, napcat = NotAwaitableWorker(), FakeLLM(["第一轮回复", "第二轮回复"]), FakeNapCat()
    application = app(database, llm, napcat, worker, clock=lambda: NOW + timedelta(hours=2))
    try:
        await application.handle_onebot(raw(101, "先说一句"), received_at_utc=NOW)
        await application.handle_onebot(raw(102, "傍晚风 在意用户", NOW + timedelta(minutes=1)),
                                        received_at_utc=NOW + timedelta(minutes=1))
        prompt = "\n".join(message.content for message in llm.calls[1])
        assert len(llm.calls) == 2
        assert "[关系记忆工作集 |" in prompt
        assert "memory_id=preference" in prompt
        assert "[过去背景 | memory_id=preference" not in prompt
        assert "memory_id=agreement" in prompt and "status=active" in prompt
        assert "[用户已确认 | memory_id=correction; type=correction; status=active]" in prompt
        assert "memory_id=correction" in prompt and "[过去背景 | memory_id=correction" not in prompt
        assert "用户每天数云" not in prompt
        assert "[过去背景 | memory_id=self; type=self_expression; status=active]" in prompt
        assert "actor=qichi" in prompt and "[用户已确认 | memory_id=self" not in prompt
        assert len(worker.calls) == 2
        assert worker.calls == [(OWNER,), (OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_preference_memory_is_globally_discoverable_but_only_topic_retrieved_with_evidence(tmp_path):
    database = Database(tmp_path / "preference-scope.sqlite3")
    seed_relationships(database)
    llm = FakeLLM(["晚饭吃了吗", "雨声确实很适合放空"])
    application = app(database, llm, FakeNapCat(), clock=lambda: NOW + timedelta(hours=2))
    try:
        await application.handle_onebot(raw(201, "今晚吃饭了吗"), received_at_utc=NOW)
        unrelated_prompt = "\n".join(item.content for item in llm.calls[0])
        assert "memory_id=agreement" in unrelated_prompt
        assert "memory_id=correction" in unrelated_prompt
        assert "[关系记忆工作集 |" in unrelated_prompt
        assert "memory_id=preference" in unrelated_prompt
        assert "[过去背景 | memory_id=preference" not in unrelated_prompt

        await application.handle_onebot(
            raw(202, "你还记得我喜欢雨声吗", NOW + timedelta(minutes=1)),
            received_at_utc=NOW + timedelta(minutes=1),
        )
        related_prompt = "\n".join(item.content for item in llm.calls[1])
        assert "[过去背景 | memory_id=preference; type=preference; status=active]" in related_prompt
        assert "memory_id=preference" in related_prompt
    finally:
        database.close()


@pytest.mark.asyncio
async def test_reliable_quoted_text_participates_in_memory_retrieval(tmp_path):
    database = Database(tmp_path / "quoted-memory.sqlite3")
    events, memories = EventRepository(database), MemoryRepository(database)
    source = seed_event(
        events,
        "quoted-preference",
        "我喜欢窗外细碎雨声",
        platform_message_id="600",
    )
    seed_memory(
        memories,
        "rain-preference",
        source,
        "用户喜欢窗外细碎雨声",
        "喜欢窗外细碎雨声",
    )
    llm = FakeLLM(["记得这句。"])
    try:
        await app(database, llm, FakeNapCat(), clock=lambda: NOW + timedelta(hours=2)).handle_onebot(
            raw(202, "那这句呢", NOW, reply_to=600),
            received_at_utc=NOW,
        )
        prompt = "\n".join(message.content for message in llm.calls[0])
        assert "[过去背景 | memory_id=rain-preference;" in prompt
        trace = database.connection.execute(
            "SELECT details_json FROM turn_trace_events WHERE phase='context'"
        ).fetchone()
        details = json.loads(trace[0])
        assert details["quoted_event_id"] == source.event_id
        assert details["selected_memory_ids"] == ["rain-preference"]
        assert details["memory_scores"]["rain-preference"] > 0
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error,status", [
    (OneBotActionError("rejected"), "failed"),
    (OneBotTimeoutError("timeout"), "unknown"),
])
async def test_failed_or_unknown_delivery_still_schedules_memory_activity(tmp_path, error, status):
    database = Database(tmp_path / f"{status}.sqlite3")
    worker = NotAwaitableWorker()
    try:
        delivered = await app(database, FakeLLM(["未送达回复"]), FakeNapCat((error,)), worker).handle_onebot(
            raw(101, "第一轮"), received_at_utc=NOW
        )
        assert delivered.status == status
        # The inbound text was durable before delivery was attempted. Memory
        # must still get a 30-minute job even when the outbound result is not
        # visible to the owner.
        assert worker.calls == [(OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_structural_generation_failure_keeps_failed_input_for_memory_recovery(tmp_path):
    database = Database(tmp_path / "generation-failure.sqlite3")
    worker = NotAwaitableWorker()
    try:
        result = await app(database, FakeLLM(["", ""]), FakeNapCat(), worker).handle_onebot(
            raw(101, "我明确喜欢雨声"), received_at_utc=NOW
        )
        assert result is None
        event = EventRepository(database).resolve_handle(OWNER, "M0")
        assert event.status == "failed"
        assert len(worker.calls) == 1
        assert worker.calls == [(OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_context_assembly_failure_still_schedules_memory_activity(tmp_path):
    database = Database(tmp_path / "context-failure.sqlite3")
    worker = NotAwaitableWorker()
    application = app(database, FakeLLM(["不会被调用"]), FakeNapCat(), worker)

    async def fail_context(*args, **kwargs):
        raise RuntimeError("context unavailable")

    application._build_input = fail_context
    try:
        result = await application.handle_onebot(
            raw(101, "我明确喜欢雨声"), received_at_utc=NOW
        )
        assert result is None
        assert worker.calls == [(OWNER,)]
        assert EventRepository(database).resolve_handle(OWNER, "M0").status == "failed"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_context_version_cancellation_still_schedules_memory_activity(tmp_path):
    database = Database(tmp_path / "context-cancelled.sqlite3")
    worker = NotAwaitableWorker()
    application = app(database, FakeLLM(["不会被调用"]), FakeNapCat(), worker)
    application._context_version = lambda _conversation_id: 999
    try:
        result = await application.handle_onebot(
            raw(101, "草稿失效也要保留这条文字的记忆窗口"), received_at_utc=NOW
        )
        assert result is None
        assert worker.calls == [(OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_received_trace_failure_still_schedules_memory_activity(tmp_path):
    database = Database(tmp_path / "trace-failure.sqlite3")
    worker = NotAwaitableWorker()
    application = app(database, FakeLLM(["不会被调用"]), FakeNapCat(), worker)

    def fail_trace(*args, **kwargs):
        raise RuntimeError("trace unavailable")

    application._append_trace = fail_trace
    try:
        with pytest.raises(RuntimeError, match="trace unavailable"):
            await application.handle_onebot(
                raw(101, "可观测性故障也不能吞掉记忆机会"), received_at_utc=NOW
            )
        assert worker.calls == [(OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_invalid_dialogue_result_still_schedules_memory_activity(tmp_path):
    database = Database(tmp_path / "invalid-result.sqlite3")
    worker = NotAwaitableWorker()
    application = app(database, FakeLLM(["不会被调用"]), FakeNapCat(), worker)

    class InvalidEngine:
        async def generate(self, _input):
            return object()

    application.dialogue_engine = InvalidEngine()
    try:
        result = await application.handle_onebot(
            raw(101, "这条输入仍是记忆证据"), received_at_utc=NOW
        )
        assert result is None
        assert worker.calls == [(OWNER,)]
        assert EventRepository(database).resolve_handle(OWNER, "M0").status == "failed"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_dispatch_guard_rejection_still_schedules_memory_activity(tmp_path):
    database = Database(tmp_path / "guard-rejected.sqlite3")
    worker = NotAwaitableWorker()
    application = app(database, FakeLLM(["草稿"]), FakeNapCat(), worker)

    async def reject_dispatch(*args, **kwargs):
        return None

    application.sender.send = reject_dispatch
    try:
        result = await application.handle_onebot(
            raw(101, "这条消息不能因为取消而丢失记忆机会"), received_at_utc=NOW
        )
        assert result is None
        assert worker.calls == [(OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_unexpected_sender_exception_still_schedules_memory_activity(tmp_path):
    database = Database(tmp_path / "sender-exception.sqlite3")
    worker = NotAwaitableWorker()
    application = app(database, FakeLLM(["草稿"]), FakeNapCat(), worker)

    async def explode_sender(*args, **kwargs):
        raise RuntimeError("sender state update failed")

    application.sender.send = explode_sender
    try:
        with pytest.raises(RuntimeError, match="sender state update failed"):
            await application.handle_onebot(
                raw(101, "发送异常也不能让这条输入失去记忆机会"), received_at_utc=NOW
            )
        assert worker.calls == [(OWNER,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_sqlite_memory_read_path_survives_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    seed_relationships(database)
    database.close()
    reopened = Database(path)
    llm = FakeLLM(["记得。"])
    try:
        await app(reopened, llm, FakeNapCat(), NotAwaitableWorker(), clock=lambda: NOW + timedelta(hours=2)).handle_onebot(
            raw(101, "还记得傍晚风吗"), received_at_utc=NOW
        )
        prompt = "\n".join(message.content for message in llm.calls[0])
        assert "memory_id=correction" in prompt
        assert "用户现在更喜欢傍晚风" in prompt
        assert "用户喜欢清晨风" not in prompt
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_new_message_still_invalidates_old_draft_with_memory_enabled(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    entered, release = asyncio.Event(), asyncio.Event()

    async def block(call_number, _messages):
        if call_number == 1:
            entered.set()
            await release.wait()

    llm, napcat, worker = FakeLLM(["旧草稿", "新回复"], block), FakeNapCat(), NotAwaitableWorker()
    application = app(database, llm, napcat, worker)
    try:
        old = asyncio.create_task(application.handle_onebot(raw(101, "旧话题"), received_at_utc=NOW))
        await entered.wait()
        new = asyncio.create_task(application.handle_onebot(
            raw(102, "新话题", NOW + timedelta(seconds=1)), received_at_utc=NOW + timedelta(seconds=1)))
        for _ in range(100):
            if database.connection.execute("SELECT COUNT(*) FROM conversation_events WHERE direction='inbound'").fetchone()[0] == 2:
                break
            await asyncio.sleep(0)
        release.set()
        old_result, new_result = await asyncio.gather(old, new)
        assert old_result is None and new_result.text == "新回复"
        assert len(llm.calls) == 2 and len(napcat.sent) == 1
        # Both durable user inputs retain a memory window; the real worker
        # coalesces these notifications into one session job.
        assert worker.calls == [(OWNER,), (OWNER,)]
    finally:
        database.close()
