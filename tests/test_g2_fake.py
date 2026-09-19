from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.outbox_repository import OutboxRepository
from qichi.transport.quote_resolver import QuoteResolver
from qichi.transport.onebot_client import OneBotActionError, OneBotTimeoutError

from test_interactions import (
    BOT,
    NOW,
    OWNER,
    FakeLLM,
    FakeNapCat,
    make_app,
    message,
    poke,
)


class TextFailingNapCat(FakeNapCat):
    async def send_private_msg(self, user_id, message):
        raise OneBotActionError("send failed")


class IdNapCat(FakeNapCat):
    async def send_private_msg(self, user_id, message):
        self.texts.append((str(user_id), message))
        return {"message_id": 700 + len(self.texts)}


@pytest.mark.asyncio
async def test_g2_owner_only_text_history_and_context_version(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = IdNapCat()
    llm = FakeLLM(["第一轮", "第二轮"])
    app = make_app(db, llm, napcat)
    try:
        assert await app.handle_onebot(message(1, "你好"), received_at_utc=NOW)
        assert await app.handle_onebot(message(2, "还记得吗", NOW.timestamp() + 1), received_at_utc=NOW + timedelta(seconds=1))
        prompt = "\n".join(item.content for item in llm.calls[1])
        assert "你好" in prompt and "第一轮" in prompt
        assert db.connection.execute("SELECT context_version FROM conversation_cursors").fetchone()[0] == 2
        assert await app.handle_onebot({"post_type": "message", "message_type": "group", "group_id": 1}, received_at_utc=NOW) is None
        assert await app.handle_onebot({"post_type": "message_sent", "message_type": "private", "self_id": int(BOT), "user_id": int(BOT), "target_id": int(OWNER), "sender": {"user_id": int(BOT)}, "message_id": 90, "time": NOW.timestamp(), "message": [{"type": "text", "data": {"text": "out"}}]}, received_at_utc=NOW) is None
        assert len(llm.calls) == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_g2_face_reaction_tail_and_unicode_are_sent_as_one_message(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["好呀 🙂\n[[qq:reply:M0]]\n[[qq:face:shy]]"])
    app = make_app(db, llm, napcat, face_catalog={"shy": 6})
    try:
        result = await app.handle_onebot(message(10, "你好"), received_at_utc=NOW)
        assert result is not None and result.text == "好呀 🙂"
        assert napcat.texts[0][1] == [
            {"type": "reply", "data": {"id": "10"}},
            {"type": "text", "data": {"text": "好呀 🙂"}},
            {"type": "face", "data": {"id": "6"}},
        ]
        assert napcat.reactions == []
        assert "[[qq:" not in napcat.texts[0][1][1]["data"]["text"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_g2_four_quote_directions_preserve_handles_sources_and_platform_targets(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = IdNapCat()
    llm = FakeLLM(
        [
            "owner cites owner\n[[qq:reply:M0]]",
            "owner cites qichi\n[[qq:reply:M2]]",
            "qichi cites owner\n[[qq:reply:M4]]",
            "qichi cites self\n[[qq:reply:Q3]]",
            "invalid stays plain",
        ]
    )
    app = make_app(db, llm, napcat)
    events = EventRepository(db)
    try:
        first = await app.handle_onebot(message(1, "owner source"), received_at_utc=NOW)
        assert first is not None and first.status == "sent"
        assert events.get(db.connection.execute("SELECT event_id FROM platform_message_map WHERE platform_message_id = '701'").fetchone()[0]).event_id == first.event_id
        second = await app.handle_onebot(
            {**message(2, "owner cites owner"), "message": [{"type": "reply", "data": {"id": "1"}}, {"type": "text", "data": {"text": "owner cites owner"}}]},
            received_at_utc=NOW + timedelta(seconds=1),
        )
        assert second is not None and second.status == "sent"
        second_inbound = events.resolve_handle(OWNER, "M2")
        assert second_inbound.reply_to_platform_message_id == "1"
        assert QuoteResolver(db).resolve(second_inbound.event_id, lambda _: pytest.fail("local owner quote must resolve")).event_id == events.resolve_handle(OWNER, "M0").event_id
        second_prompt = "\n".join(item.content for item in llm.calls[1])
        assert "[直接引用 | actor=mumo; handle=M0;" in second_prompt
        assert "owner source" in second_prompt
        third = await app.handle_onebot(
            {**message(3, "owner cites qichi"), "message": [{"type": "reply", "data": {"id": "701"}}, {"type": "text", "data": {"text": "owner cites qichi"}}]},
            received_at_utc=NOW + timedelta(seconds=2),
        )
        assert third is not None and third.status == "sent"
        third_inbound = events.resolve_handle(OWNER, "M4")
        assert third_inbound.reply_to_platform_message_id == "701"
        assert QuoteResolver(db).resolve(third_inbound.event_id, lambda _: pytest.fail("local qichi quote must resolve")).event_id == first.event_id
        third_prompt = "\n".join(item.content for item in llm.calls[2])
        assert "[直接引用 | actor=qichi; handle=Q1;" in third_prompt
        assert "owner cites owner" in third_prompt
        fourth = await app.handle_onebot(message(4, "qichi cites owner"), received_at_utc=NOW + timedelta(seconds=3))
        assert third.reply_to_event_id == events.resolve_handle(OWNER, "M4").event_id
        assert fourth is not None and fourth.reply_to_event_id == events.resolve_handle(OWNER, "Q3").event_id
        invalid = await app.handle_onebot(
            {**message(5, "invalid"), "message": [{"type": "reply", "data": {"id": "999999"}}, {"type": "text", "data": {"text": "invalid"}}]},
            received_at_utc=NOW + timedelta(seconds=4),
        )
        assert invalid is not None and invalid.reply_to_event_id is None
        assert napcat.texts[0][1][0] == {"type": "reply", "data": {"id": "1"}}
        assert napcat.texts[1][1][0] == {"type": "reply", "data": {"id": "2"}}
        assert napcat.texts[2][1][0] == {"type": "reply", "data": {"id": "3"}}
        assert napcat.texts[3][1][0] == {"type": "reply", "data": {"id": "702"}}
        for event in (first, second, third, fourth):
            restored = events.get(event.event_id)
            assert restored.reply_to_event_id is not None
            assert restored.reply_to_platform_message_id is not None
            assert restored.visible_handle.startswith("Q")
        assert events.get(first.reply_to_event_id).text == "owner source"
        assert events.get(second.reply_to_event_id).text == "owner cites owner"
        assert events.get(third.reply_to_event_id).text == "owner cites qichi"
        assert events.get(fourth.reply_to_event_id).text == "owner cites qichi"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_g2_reaction_targets_exact_message_and_requires_sent_text(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["收到\n[[qq:react:heart]]"])
    app = make_app(db, llm, napcat)
    try:
        result = await app.handle_onebot(message(11, "第一条"), received_at_utc=NOW)
        assert result is not None and result.status == "sent"
        assert napcat.reactions == [(('11', '66'), {'set': True})]
        failing_db = Database(tmp_path / "failing.sqlite3")
        failing_napcat = TextFailingNapCat()
        failing_llm = FakeLLM(["不会送达\n[[qq:react:heart]]"])
        failing_app = make_app(failing_db, failing_llm, failing_napcat)
        assert (await failing_app.handle_onebot(message(12, "第二条"), received_at_utc=NOW + timedelta(seconds=1))) is not None
        assert failing_napcat.reactions == []
        failing_db.close()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_g2_reaction_unknown_requires_explicit_recovery(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat(reaction_error=OneBotTimeoutError("unknown"))
    llm = FakeLLM(["正文\n[[qq:react:heart]]"])
    app = make_app(db, llm, napcat)
    try:
        result = await app.handle_onebot(message(20, "消息"), received_at_utc=NOW)
        assert result is not None and result.status == "sent"
        key = next(item.operation_key for item in OutboxRepository(db).outstanding() if item.payload["action_kind"] == "reaction")
        assert OutboxRepository(db).get(key).status == "unknown"
        assert len(napcat.reactions) == 1
        await app.sender._dispatch_stored_reaction(result, NOW + timedelta(milliseconds=1))
        assert len(napcat.reactions) == 1
        OutboxRepository(db).recover_idempotent_reaction(key, NOW + timedelta(seconds=1))
        napcat.reaction_error = None
        await app.sender._dispatch_stored_reaction(result, NOW + timedelta(seconds=2))
        assert len(napcat.reactions) == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_g2_poke_unknown_is_not_replayed_after_sqlite_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    db = Database(path)
    napcat = IdNapCat(poke_error=OneBotTimeoutError("unknown"))
    llm = FakeLLM(["普通引用\n[[qq:reply:M0]]", "互动正文"])
    app = make_app(db, llm, napcat)
    first = await app.handle_onebot(message(40, "普通"), received_at_utc=NOW)
    assert first is not None and first.status == "sent"
    await app.handle_onebot(poke(), received_at_utc=NOW + timedelta(seconds=1))
    first_target = EventRepository(db).get(first.reply_to_event_id)
    assert first.reply_to_event_id == first_target.event_id
    assert first.reply_to_platform_message_id == first_target.platform_message_id
    assert QuoteResolver(db).resolve(first.event_id, lambda _: pytest.fail("local map must resolve")) is not None
    db.close()
    calls_before = (len(llm.calls), len(napcat.pokes))
    reopened = Database(path)
    try:
        events = EventRepository(reopened)
        restored_outbound = events.get(first.event_id)
        assert restored_outbound.text == "普通引用"
        assert restored_outbound.reply_to_event_id == first_target.event_id
        assert restored_outbound.reply_to_platform_message_id == first_target.platform_message_id
        assert reopened.connection.execute(
            "SELECT event_id FROM platform_message_map WHERE platform_message_id = ?",
            ("701",),
        ).fetchone()[0] == first.event_id
        assert reopened.connection.execute(
            "SELECT text FROM conversation_events WHERE direction='outbound' AND kind='text' AND text = ?",
            ("互动正文",),
        ).fetchone()[0] == "互动正文"
        poke_row = reopened.connection.execute(
            "SELECT platform_event_id FROM conversation_events WHERE kind='poke'"
        ).fetchone()
        assert poke_row[0] is not None
        poke_outbox = reopened.connection.execute(
            "SELECT status FROM outbox WHERE payload_json LIKE '%\"action_kind\":\"poke\"%'"
        ).fetchone()
        assert poke_outbox[0] == "unknown"
        reopened_app = make_app(reopened, llm, napcat)
        assert await reopened_app.handle_onebot(poke(), received_at_utc=NOW + timedelta(seconds=2)) is None
        assert (len(llm.calls), len(napcat.pokes)) == calls_before
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_g2_duplicate_and_concurrent_poke_are_idempotent(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM(["只回一次"])
    app = make_app(db, llm, napcat)
    try:
        payload = poke()
        results = await asyncio.gather(
            app.handle_onebot(payload, received_at_utc=NOW),
            app.handle_onebot(payload, received_at_utc=NOW),
        )
        assert sum(result is not None for result in results) == 1
        assert len(llm.calls) == 1
        assert napcat.pokes == [OWNER]
        assert db.connection.execute("SELECT COUNT(*) FROM conversation_events WHERE kind='poke'").fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_g2_poke_after_immediate_text_has_one_new_generation_and_processed_cursor(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    started = asyncio.Event()
    release = asyncio.Event()
    napcat = FakeNapCat(poke_started=started, poke_release=release)
    llm = FakeLLM(["新文字"])
    app = make_app(db, llm, napcat)
    try:
        old = asyncio.create_task(app.handle_onebot(poke(), received_at_utc=NOW))
        await started.wait()
        new = asyncio.create_task(app.handle_onebot(message(30, "抢占", NOW.timestamp() + 1), received_at_utc=NOW + timedelta(seconds=1)))
        for _ in range(100):
            inbound_texts = db.connection.execute(
                "SELECT COUNT(*) FROM conversation_events WHERE direction='inbound' AND kind='text'"
            ).fetchone()[0]
            context_version = db.connection.execute(
                "SELECT context_version FROM conversation_cursors"
            ).fetchone()[0]
            if inbound_texts == 1 and context_version == 2:
                break
            await asyncio.sleep(0)
        assert inbound_texts == 1 and context_version == 2
        release.set()
        old_result, new_result = await asyncio.gather(old, new)
        assert old_result is None and new_result is not None and new_result.text == "新文字"
        assert len(llm.calls) == 1
        poke_sequence = db.connection.execute("SELECT sequence FROM conversation_events WHERE kind='poke'").fetchone()[0]
        processed = db.connection.execute("SELECT last_processed_sequence FROM conversation_cursors").fetchone()[0]
        assert processed >= poke_sequence
        assert [item[1][0]["data"]["text"] for item in napcat.texts] == ["新文字"]
    finally:
        db.close()


def test_g2_protocol_and_plain_text_regression():
    protocol = ResponseProtocol(face_keys={"shy"}, reaction_keys={"heart"})
    result = protocol.parse("普通 [方括号] JSON {\"x\": 1} 🙂", source="dialogue", current_event_handle="M1", context_version=1)
    assert result.text.endswith("🙂")
    assert protocol.parse(
        "x\n[[qq:reply:M1]]\n[[qq:react:heart]]",
        source="dialogue",
        current_event_handle="M1",
        context_version=1,
    ).text == "x"


@pytest.mark.asyncio
async def test_structural_generation_failure_marks_inbound_failed_and_allows_next_message(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    llm = FakeLLM([
        "x\n[[qq:reply:M1]]\n[[qq:reply:Q2]]",
        "x\n[[qq:reply:M1]]\n[[qq:reply:Q2]]",
        "恢复了",
    ])
    app = make_app(db, llm, napcat)
    try:
        assert await app.handle_onebot(message(1, "第一次"), received_at_utc=NOW) is None
        failed = db.connection.execute(
            "SELECT status FROM conversation_events "
            "WHERE direction='inbound' AND platform_message_id='1'"
        ).fetchone()
        assert failed[0] == "failed"
        assert db.connection.execute(
            "SELECT last_processed_sequence FROM conversation_cursors"
        ).fetchone()[0] == 0
        result = await app.handle_onebot(
            message(2, "第二次"), received_at_utc=NOW + timedelta(seconds=1)
        )
        assert result is not None and result.text == "恢复了"
        assert len(napcat.texts) == 2
        assert napcat.texts[0][1] == [
            {"type": "text", "data": {"text": "消息已收到，但本次处理未完成，请稍后重试。"}}
        ]
    finally:
        db.close()
