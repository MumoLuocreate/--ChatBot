from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json

import pytest

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMConnectionError, LLMGeneration, LLMTimeoutError
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.transport.onebot_client import OneBotActionError, OneBotTimeoutError


NOW = datetime(2026, 8, 27, 10, tzinfo=timezone.utc)
OWNER_QQ = "10001"
BOT_QQ = "20001"


class CharacterCounter:
    def count_text(self, text):
        return len(text)


class FakeSiliconFlow:
    def __init__(self, outputs, before_generate=None):
        self.outputs = list(outputs)
        self.before_generate = before_generate
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.before_generate is not None:
            await self.before_generate(len(self.calls), messages)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return LLMGeneration(output, "primary", "fake-v4f", 1, 1, 1.0)


class FakeNapCat:
    def __init__(self, first_message_id=700, errors=()):
        self.next_message_id = first_message_id
        self.errors = list(errors)
        self.sent = []

    async def send_private_msg(self, user_id, message):
        self.sent.append((str(user_id), message))
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        message_id = self.next_message_id
        self.next_message_id += 1
        return {"message_id": message_id}

    async def set_msg_emoji_like(self, message_id, emoji_id, set=True):
        raise AssertionError("G0 must not send reactions")


def raw_message(message_id, text, *, reply_to=None, timestamp=NOW):
    message = []
    if reply_to is not None:
        message.append({"type": "reply", "data": {"id": str(reply_to)}})
    message.append({"type": "text", "data": {"text": text}})
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
        "message": message,
    }


def context_builder():
    evidence = ProviderCapabilityEvidence(
        "fake", "fake-v4f", 262_144, "local-test", NOW
    )
    capability = ModelCapability("fake-v4f", 262_144, "fake", evidence)
    return ContextBuilder(
        CharacterCounter(), capability, output_reserve_tokens=1_024
    )


def application(database, napcat, siliconflow, *, clock=lambda: NOW, auto_quote_current_message=False, get_msg_async=None):
    engine = DialogueEngine(
        siliconflow,
        OutputGuard(CharacterCounter(), 2_048),
        ResponseProtocol(face_keys=(), reaction_keys=()),
    )
    return G0Application(
        database,
        context_builder(),
        engine,
        napcat,
        owner_qq=OWNER_QQ,
        bot_qq=BOT_QQ,
        role_core="你是角色。",
        clock=clock,
        auto_quote_current_message=auto_quote_current_message,
        get_msg_async=get_msg_async,
    )


@pytest.mark.asyncio
async def test_inbound_is_persisted_before_one_generation_and_plain_output_is_not_auto_quoted(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()

    async def assert_persisted(call_number, _messages):
        assert call_number == 1
        row = database.connection.execute(
            "SELECT direction, actor, text, status FROM conversation_events"
        ).fetchone()
        assert tuple(row) == ("inbound", "mumo", "你好", "received")
        assert database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0

    siliconflow = FakeSiliconFlow(["在呢。"], assert_persisted)
    try:
        delivered = await application(database, napcat, siliconflow).handle_onebot(
            raw_message(101, "你好"), received_at_utc=NOW
        )

        assert len(siliconflow.calls) == 1
        assert delivered is not None and delivered.text == "在呢。"
        assert napcat.sent == [
            (OWNER_QQ, [{"type": "text", "data": {"text": "在呢。"}}])
        ]
        cursor = database.connection.execute(
            "SELECT context_version, last_processed_sequence FROM conversation_cursors"
        ).fetchone()
        assert tuple(cursor) == (1, 0)
        traces = database.connection.execute(
            "SELECT phase, details_json FROM turn_trace_events ORDER BY occurred_at_utc, phase"
        ).fetchall()
        by_phase = {row["phase"]: json.loads(row["details_json"]) for row in traces}
        assert set(by_phase) == {"received", "context", "generation", "delivery"}
        assert by_phase["received"]["context_version"] == 1
        assert by_phase["context"]["selected_history_event_ids"] == []
        assert by_phase["context"]["selected_memory_ids"] == []
        assert by_phase["context"]["quote_resolution_status"] == "none"
        assert by_phase["context"]["input_tokens"] > 0
        assert by_phase["generation"]["model_id"] == "fake-v4f"
        assert by_phase["generation"]["attempt_count"] == 1
        assert by_phase["delivery"]["status"] == "sent"
        assert by_phase["delivery"]["outbound_event_id"] == delivered.event_id
        serialized = "\n".join(row["details_json"] for row in traces)
        assert "你好" not in serialized
        assert "在呢" not in serialized
    finally:
        database.close()


@pytest.mark.asyncio
async def test_user_quote_is_visible_to_model_and_plain_output_uses_native_quote_when_enabled(tmp_path):
    database = Database(tmp_path / "quoted-native.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["我接着你引用的那句说。"])
    first = application(database, napcat, siliconflow)
    try:
        await first.handle_onebot(raw_message(101, "原始消息"), received_at_utc=NOW)
        siliconflow.outputs.append("我对应这条说。")
        second = application(
            database,
            napcat,
            siliconflow,
            auto_quote_current_message=True,
        )
        delivered = await second.handle_onebot(
            raw_message(102, "引用后继续", reply_to=700, timestamp=NOW + timedelta(seconds=1)),
            received_at_utc=NOW + timedelta(seconds=1),
        )

        prompt = "\n".join(message.content for message in siliconflow.calls[-1])
        assert "[直接引用 | actor=qichi; handle=Q1;" in prompt
        assert "我接着你引用的那句说。" in prompt
        assert delivered is not None
        assert delivered.reply_to_event_id is not None
        assert napcat.sent[-1][1][0] == {"type": "reply", "data": {"id": "700"}}
    finally:
        database.close()


@pytest.mark.asyncio
async def test_auto_quote_mode_does_not_quote_an_ordinary_message(tmp_path):
    database = Database(tmp_path / "ordinary-no-quote.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["普通接话。"])
    try:
        delivered = await application(
            database,
            napcat,
            siliconflow,
            auto_quote_current_message=True,
        ).handle_onebot(raw_message(101, "普通消息"), received_at_utc=NOW)
        assert delivered is not None
        assert delivered.reply_to_event_id is None
        assert napcat.sent[0][1] == [
            {"type": "text", "data": {"text": "普通接话。"}}
        ]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_generation_failure_sends_one_non_semantic_status_notice(tmp_path):
    database = Database(tmp_path / "generation-failure-notice.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["", ""])
    try:
        result = await application(database, napcat, siliconflow).handle_onebot(
            raw_message(101, "这条不能悄悄吞掉"), received_at_utc=NOW
        )
        assert result is None
        assert len(napcat.sent) == 1
        assert napcat.sent[0][1] == [
            {"type": "text", "data": {"text": "消息已收到，但本次处理未完成，请稍后重试。"}}
        ]
        notice = database.connection.execute(
            "SELECT direction, actor, kind, status, text FROM conversation_events "
            "WHERE direction='internal' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        assert tuple(notice) == (
            "internal",
            "platform",
            "processing_failure_notice",
            "sent",
            "消息已收到，但本次处理未完成，请稍后重试。",
        )
        traces = database.connection.execute(
            "SELECT phase, details_json FROM turn_trace_events ORDER BY occurred_at_utc, phase"
        ).fetchall()
        by_phase = {row["phase"]: json.loads(row["details_json"]) for row in traces}
        assert set(by_phase) == {"received", "context", "failure"}
        assert by_phase["failure"]["stage"] == "generation"
        assert by_phase["failure"]["failure_category"].startswith("output_guard:")
        await application(database, napcat, siliconflow).handle_onebot(
            raw_message(101, "这条不能悄悄吞掉"), received_at_utc=NOW
        )
        assert len(napcat.sent) == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_quote_resolution_failure_is_converged_and_not_silent(tmp_path):
    database = Database(tmp_path / "quote-failure-notice.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["不应生成"])

    async def fetch(_message_id):
        raise RuntimeError("temporary quote lookup failure")

    try:
        result = await application(
            database,
            napcat,
            siliconflow,
            get_msg_async=fetch,
        ).handle_onebot(
            raw_message(101, "引用失败也要有反馈", reply_to=999),
            received_at_utc=NOW,
        )
        assert result is None
        assert siliconflow.calls == []
        assert len(napcat.sent) == 1
        status = database.connection.execute(
            "SELECT status FROM conversation_events WHERE direction='inbound'"
        ).fetchone()[0]
        assert status == "failed"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_one_generation_with_blank_line_paragraphs_sends_distinct_qq_messages(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat(first_message_id=700)
    siliconflow = FakeSiliconFlow(["先说这一句。\n\n再补这一句。"])
    try:
        delivered = await application(database, napcat, siliconflow).handle_onebot(
            raw_message(101, "分开说"), received_at_utc=NOW
        )

        assert len(siliconflow.calls) == 1
        assert delivered is not None and delivered.text == "再补这一句。"
        assert [call[1] for call in napcat.sent] == [
            [{"type": "text", "data": {"text": "先说这一句。"}}],
            [{"type": "text", "data": {"text": "再补这一句。"}}],
        ]
        rows = database.connection.execute(
            "SELECT text, status FROM conversation_events "
            "WHERE direction='outbound' ORDER BY sequence"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("先说这一句。", "sent"),
            ("再补这一句。", "sent"),
        ]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_bidirectional_quotes_and_platform_id_mapping_survive_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    first_napcat = FakeNapCat(first_message_id=700)
    first_llm = FakeSiliconFlow(["你自己说的。\n[[qq:reply:M0]]"])
    try:
        first = await application(database, first_napcat, first_llm).handle_onebot(
            raw_message(101, "我会记得"), received_at_utc=NOW
        )
        assert first is not None
        assert first.reply_to_event_id == EventRepository(database).resolve_handle(OWNER_QQ, "M0").event_id
        assert first_napcat.sent[0][1][0] == {"type": "reply", "data": {"id": "101"}}
        assert database.connection.execute(
            "SELECT event_id FROM platform_message_map WHERE platform_message_id = '700'"
        ).fetchone()[0] == first.event_id
        first_event_id = first.event_id
        first_reply_target_id = first.reply_to_event_id
    finally:
        database.close()

    reopened = Database(path)
    second_napcat = FakeNapCat(first_message_id=701)
    second_llm = FakeSiliconFlow(["这次引用得很准。"])
    try:
        restored_first = EventRepository(reopened).get(first_event_id)
        assert restored_first.reply_to_event_id == first_reply_target_id
        assert restored_first.reply_to_platform_message_id == "101"
        second = await application(
            reopened,
            second_napcat,
            second_llm,
            clock=lambda: NOW + timedelta(minutes=1),
        ).handle_onebot(
            raw_message(102, "那这句呢", reply_to=700, timestamp=NOW + timedelta(minutes=1)),
            received_at_utc=NOW + timedelta(minutes=1),
        )
        prompt = "\n".join(message.content for message in second_llm.calls[0])
        assert "[直接引用 | actor=qichi; handle=Q1;" in prompt
        assert "你自己说的。" in prompt
        assert second is not None and second.reply_to_event_id is None
        link = reopened.connection.execute(
            "SELECT source_event_id, target_event_id, relation, status FROM message_links"
        ).fetchone()
        assert tuple(link) == (
            EventRepository(reopened).resolve_handle(OWNER_QQ, "M2").event_id,
            first.event_id,
            "reply",
            "resolved",
        )
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_new_inbound_persists_during_generation_and_stale_draft_is_never_sent(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block_first(call_number, _messages):
        if call_number == 1:
            entered.set()
            await release.wait()

    siliconflow = FakeSiliconFlow(["过时草稿", "新消息回复"], block_first)
    app = application(database, napcat, siliconflow)
    try:
        old = asyncio.create_task(
            app.handle_onebot(raw_message(101, "旧话题"), received_at_utc=NOW)
        )
        await entered.wait()
        new = asyncio.create_task(
            app.handle_onebot(
                raw_message(102, "新话题", timestamp=NOW + timedelta(seconds=1)),
                received_at_utc=NOW + timedelta(seconds=1),
            )
        )

        for _ in range(100):
            count = database.connection.execute(
                "SELECT COUNT(*) FROM conversation_events WHERE direction = 'inbound'"
            ).fetchone()[0]
            if count == 2:
                break
            await asyncio.sleep(0)
        assert count == 2
        assert database.connection.execute(
            "SELECT context_version FROM conversation_cursors"
        ).fetchone()[0] == 2

        release.set()
        old_result, new_result = await asyncio.gather(old, new)
        assert old_result is None
        assert new_result is not None and new_result.text == "新消息回复"
        assert [call[1][0]["data"]["text"] for call in napcat.sent] == ["新消息回复"]
        assert len(siliconflow.calls) == 2
        assert "新话题" in "\n".join(message.content for message in siliconflow.calls[1])
        assert database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_new_inbound_after_generation_check_still_wins_atomic_dispatch(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["过时草稿", "新消息回复"])
    app = application(database, napcat, siliconflow)
    original_send = app.sender.send
    user_task = None
    send_calls = 0

    async def race_send(*args, **kwargs):
        nonlocal send_calls, user_task
        send_calls += 1
        if send_calls == 1:
            user_task = asyncio.create_task(
                app.handle_onebot(
                    raw_message(102, "新话题", timestamp=NOW + timedelta(seconds=1)),
                    received_at_utc=NOW + timedelta(seconds=1),
                )
            )
            for _ in range(100):
                row = database.connection.execute(
                    "SELECT context_version FROM conversation_cursors "
                    "WHERE conversation_id = ?",
                    (OWNER_QQ,),
                ).fetchone()
                if row is not None and row[0] == 2:
                    break
                await asyncio.sleep(0)
            else:
                raise AssertionError("new inbound did not commit before dispatch")
        return await original_send(*args, **kwargs)

    app.sender.send = race_send
    try:
        old_result = await app.handle_onebot(
            raw_message(101, "旧话题"), received_at_utc=NOW
        )
        assert old_result is None
        assert user_task is not None
        new_result = await user_task
        assert new_result is not None and new_result.text == "新消息回复"
        assert [call[1][0]["data"]["text"] for call in napcat.sent] == ["新消息回复"]
        assert database.connection.execute(
            "SELECT COUNT(*) FROM outbox"
        ).fetchone()[0] == 1
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_error, expected_reason",
    [
        (LLMTimeoutError("provider timeout"), "llm:timeout"),
        (LLMConnectionError("provider disconnected"), "llm:connection"),
    ],
)
async def test_llm_service_failure_converges_inbound_and_next_message_still_replies(
    tmp_path, caplog, service_error, expected_reason
):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow([service_error, "第二条正常回复"])
    app = application(database, napcat, siliconflow)
    try:
        assert await app.handle_onebot(
            raw_message(101, "第一条"), received_at_utc=NOW
        ) is None
        failed = database.connection.execute(
            "SELECT status FROM conversation_events "
            "WHERE direction='inbound' AND platform_message_id='101'"
        ).fetchone()
        assert failed[0] == "failed"
        assert database.connection.execute(
            "SELECT last_processed_sequence FROM conversation_cursors"
        ).fetchone()[0] == 0
        assert expected_reason in caplog.text

        result = await app.handle_onebot(
            raw_message(102, "第二条"), received_at_utc=NOW + timedelta(seconds=1)
        )
        assert result is not None and result.text == "第二条正常回复"
        assert [call[1][0]["data"]["text"] for call in napcat.sent] == [
            "消息已收到，但本次处理未完成，请稍后重试。",
            "第二条正常回复",
        ]
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_error, expected_status, visible",
    [
        (None, "sent", True),
        (OneBotActionError("rejected"), "failed", False),
        (OneBotTimeoutError("timeout"), "unknown", False),
    ],
)
async def test_only_delivered_outbound_text_enters_the_next_prompt(
    tmp_path, first_error, expected_status, visible
):
    database = Database(tmp_path / f"{expected_status}.sqlite3")
    napcat = FakeNapCat(errors=(first_error,))
    secret = "ONLY_DELIVERED_TEXT_MAY_BE_HISTORY"
    siliconflow = FakeSiliconFlow([secret, "第二轮回复"])
    app = application(database, napcat, siliconflow, clock=lambda: NOW + timedelta(minutes=2))
    try:
        first = await app.handle_onebot(raw_message(101, "第一轮"), received_at_utc=NOW)
        assert first is not None and first.status == expected_status
        await app.handle_onebot(
            raw_message(102, "第二轮", timestamp=NOW + timedelta(minutes=1)),
            received_at_utc=NOW + timedelta(minutes=1),
        )

        second_prompt = "\n".join(message.content for message in siliconflow.calls[1])
        assert (secret in second_prompt) is visible
        assert "第一轮" in second_prompt
        if not visible:
            assert "120s" in second_prompt
    finally:
        database.close()


@pytest.mark.asyncio
async def test_duplicate_inbound_is_idempotent_without_advancing_context_or_generating(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["只回复一次"])
    app = application(database, napcat, siliconflow)
    payload = raw_message(101, "重复投递")
    try:
        first = await app.handle_onebot(payload, received_at_utc=NOW)
        duplicate = await app.handle_onebot(payload, received_at_utc=NOW)

        assert first is not None and duplicate is None
        assert len(siliconflow.calls) == len(napcat.sent) == 1
        assert database.connection.execute(
            "SELECT context_version FROM conversation_cursors"
        ).fetchone()[0] == 1
        assert database.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE direction = 'inbound'"
        ).fetchone()[0] == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_existing_platform_identity_conflict_is_rejected_by_repository_contract(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    siliconflow = FakeSiliconFlow(["不应生成"])
    try:
        EventRepository(database).insert_inbound(
            ConversationEvent(
                event_id="existing",
                platform_event_id=f"message:{BOT_QQ}:101",
                platform_message_id="999",
                conversation_id=OWNER_QQ,
                sequence=0,
                direction="inbound",
                actor="mumo",
                kind="text",
                text="existing",
                message_segments=(MessageSegment("text", {"text": "existing"}),),
                reply_to_event_id=None,
                reply_to_platform_message_id=None,
                occurred_at_utc=NOW,
                received_at_utc=NOW,
                status="received",
                metadata={},
            )
        )
        with pytest.raises(ValueError, match="platform identity conflict"):
            await application(database, napcat, siliconflow).handle_onebot(
                raw_message(101, "冲突投递"), received_at_utc=NOW
            )
        assert siliconflow.calls == [] and napcat.sent == []
        assert database.connection.execute("SELECT COUNT(*) FROM conversation_cursors").fetchone()[0] == 0
    finally:
        database.close()
