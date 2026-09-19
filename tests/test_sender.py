from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from qichi.domain.dialogue import DialogueResult, ExpressionIntent
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.outbox_repository import OutboxRepository
from qichi.transport.onebot_client import (
    OneBotActionError,
    OneBotConnectionError,
    OneBotHTTPClientError,
    OneBotHTTPRedirectError,
    OneBotHTTPServerError,
    OneBotProtocolError,
    OneBotTimeoutError,
)
from qichi.transport.sender import Sender


NOW = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)


class FakeOneBotClient:
    def __init__(self, text_result=None, text_error=None, reaction_error=None):
        self.text_result = text_result if text_result is not None else {"message_id": 700}
        self.text_error = text_error
        self.reaction_error = reaction_error
        self.text_calls: list[tuple[int | str, list[dict]]] = []
        self.reaction_calls: list[tuple[int | str, int | str, bool]] = []

    async def send_private_msg(self, user_id, message):
        self.text_calls.append((user_id, message))
        if self.text_error:
            raise self.text_error
        return self.text_result

    async def set_msg_emoji_like(self, message_id, emoji_id, set=True):
        self.reaction_calls.append((message_id, emoji_id, set))
        if self.reaction_error:
            raise self.reaction_error
        return {"ok": True}


def inbound(event_id: str, *, conversation_id="42", platform_message_id="100"):
    return ConversationEvent(
        event_id=event_id, platform_event_id=f"event-{event_id}", platform_message_id=platform_message_id,
        conversation_id=conversation_id, sequence=0, direction="inbound", actor="mumo", kind="text",
        text="source", message_segments=(MessageSegment("text", {"text": "source"}),),
        reply_to_event_id=None, reply_to_platform_message_id=None, occurred_at_utc=NOW,
        received_at_utc=NOW, status="received", metadata={},
    )


def result(*, reply=None, expression=None, text="hello 🙂"):
    return DialogueResult(text=text, reply_target=reply, expression_intent=expression, model_route="primary", context_version=1)


def sender(database, client):
    return Sender(database, client, face_catalog={"annoyed": "14"}, reaction_catalog={"heart": "128512"})


def test_blank_line_parts_send_as_independent_persisted_qq_messages(tmp_path):
    class IncrementingClient(FakeOneBotClient):
        def __init__(self):
            super().__init__()
            self.next_message_id = 700

        async def send_private_msg(self, user_id, message):
            self.text_calls.append((user_id, message))
            response = {"message_id": self.next_message_id}
            self.next_message_id += 1
            return response

    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            source = EventRepository(database).insert(inbound("source"))
            client = IncrementingClient()
            split_result = DialogueResult(
                text="先回这一句。\n\n再补这一句。",
                reply_target=source.handle,
                expression_intent=ExpressionIntent("face", "annoyed", None),
                model_route="primary",
                context_version=1,
                message_parts=("先回这一句。", "再补这一句。"),
            )

            delivered = await sender(database, client).send(
                split_result,
                event_id="delivery-group",
                conversation_id="42",
                owner_qq=42,
                occurred_at_utc=NOW,
            )

            assert delivered is not None
            assert delivered.event_id == "delivery-group"
            assert delivered.text == "再补这一句。"
            assert len(client.text_calls) == 2
            assert client.text_calls[0][1] == [
                {"type": "reply", "data": {"id": "100"}},
                {"type": "text", "data": {"text": "先回这一句。"}},
            ]
            assert client.text_calls[1][1] == [
                {"type": "text", "data": {"text": "再补这一句。"}},
                {"type": "face", "data": {"id": "14"}},
            ]

            rows = database.connection.execute(
                "SELECT event_id, sequence, text, status, reply_to_event_id, metadata_json "
                "FROM conversation_events WHERE direction = 'outbound' ORDER BY sequence"
            ).fetchall()
            assert len(rows) == 2
            assert [row["text"] for row in rows] == ["先回这一句。", "再补这一句。"]
            assert [row["status"] for row in rows] == ["sent", "sent"]
            assert rows[0]["reply_to_event_id"] == source.event_id
            assert rows[1]["reply_to_event_id"] is None
            assert database.connection.execute(
                "SELECT COUNT(*) FROM platform_message_map WHERE platform_message_id IN ('700', '701')"
            ).fetchone()[0] == 2
            assert database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2
        finally:
            database.close()

    asyncio.run(scenario())


def test_multi_message_partial_unknown_never_resends_an_already_sent_part(tmp_path):
    class SecondCallTimesOut(FakeOneBotClient):
        def __init__(self):
            super().__init__()
            self.call_number = 0

        async def send_private_msg(self, user_id, message):
            self.call_number += 1
            self.text_calls.append((user_id, message))
            if self.call_number == 2:
                raise OneBotTimeoutError("timeout")
            return {"message_id": 700 + self.call_number}

    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = SecondCallTimesOut()
            hook_calls = 0

            def commit_after_first_sent(connection):
                nonlocal hook_calls
                hook_calls += 1
                connection.execute(
                    "INSERT INTO runtime_meta(key, value_json, updated_at_utc) "
                    "VALUES ('test:multi-part-commit', 'true', ?)",
                    (NOW.isoformat(),),
                )

            split_result = DialogueResult(
                text="第一条\n\n第二条",
                reply_target=None,
                expression_intent=None,
                model_route="primary",
                context_version=1,
                message_parts=("第一条", "第二条"),
            )
            transport = sender(database, client)

            first = await transport.send(
                split_result,
                event_id="delivery-group",
                conversation_id="42",
                owner_qq=42,
                occurred_at_utc=NOW,
                dispatch_guard=lambda _connection: True,
                sent_commit_hook=commit_after_first_sent,
            )
            repeated = await transport.send(
                split_result,
                event_id="delivery-group",
                conversation_id="42",
                owner_qq=42,
                occurred_at_utc=NOW,
            )

            assert first is not None and first.status == "unknown"
            assert hook_calls == 1
            assert database.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key='test:multi-part-commit'"
            ).fetchone()[0] == "true"
            assert repeated is not None and repeated.event_id == first.event_id
            assert repeated.status == "unknown"
            assert len(client.text_calls) == 2
            rows = database.connection.execute(
                "SELECT text, status FROM conversation_events "
                "WHERE direction='outbound' ORDER BY sequence"
            ).fetchall()
            assert [tuple(row) for row in rows] == [
                ("第一条", "sent"),
                ("第二条", "unknown"),
            ]
        finally:
            database.close()

    asyncio.run(scenario())


def test_text_reply_face_order_unicode_metadata_mapping_and_reopen(tmp_path):
    async def scenario():
        path = tmp_path / "qichi.sqlite3"
        database = Database(path)
        try:
            source = EventRepository(database).insert(inbound("source"))
            client = FakeOneBotClient()
            delivered = await sender(database, client).send(
                result(reply=source.handle, expression=ExpressionIntent("face", "annoyed", None)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert [segment.to_dict() for segment in delivered.message_segments] == [
                {"type": "reply", "data": {"id": "100"}},
                {"type": "text", "data": {"text": "hello 🙂"}},
                {"type": "face", "data": {"id": "14"}},
            ]
            assert delivered.reply_to_event_id == source.event_id
            assert delivered.metadata["expression"] == {
                "requested": {"kind": "face", "key": "annoyed", "target_event_handle": None},
                "face_segment_key": "annoyed",
                "requested_reply_target": "M0",
                "reaction": None,
            }
            assert client.text_calls == [("42", [segment.to_dict() for segment in delivered.message_segments])]
            assert OutboxRepository(database).get(Sender.text_operation_key("outbound")).status == "sent"
            assert delivered.status == "sent"
            assert database.connection.execute("SELECT event_id, source FROM platform_message_map WHERE platform_message_id = ?", ("700",)).fetchone()[:] == ("outbound", "sender")
        finally:
            database.close()
        reopened = Database(path)
        try:
            assert EventRepository(reopened).get("outbound").text == "hello 🙂"
            assert reopened.connection.execute("SELECT event_id FROM platform_message_map WHERE platform_message_id = ?", ("700",)).fetchone()[0] == "outbound"
        finally:
            reopened.close()

    asyncio.run(scenario())


def test_guarded_second_worker_never_redispatches_existing_dispatched_intent(tmp_path):
    class BlockingClient(FakeOneBotClient):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def send_private_msg(self, user_id, message):
            self.text_calls.append((user_id, message))
            self.entered.set()
            await self.release.wait()
            return self.text_result

    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        client = BlockingClient()
        try:
            first = asyncio.create_task(
                sender(database, client).send(
                    result(),
                    event_id="guarded-outbound",
                    conversation_id="42",
                    owner_qq=42,
                    occurred_at_utc=NOW,
                    dispatch_guard=lambda _connection: True,
                )
            )
            await client.entered.wait()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert OutboxRepository(database).get(
                Sender.text_operation_key("guarded-outbound")
            ).status == "dispatched"

            with pytest.raises(RuntimeError, match="awaiting reconciliation"):
                await sender(database, client).send(
                    result(),
                    event_id="guarded-outbound",
                    conversation_id="42",
                    owner_qq=42,
                    occurred_at_utc=NOW,
                    dispatch_guard=lambda _connection: True,
                )
            assert len(client.text_calls) == 1
        finally:
            database.close()

    asyncio.run(scenario())


def test_invalid_reply_and_face_degrade_to_unchanged_plain_text(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient()
            delivered = await sender(database, client).send(
                result(reply="M99", expression=ExpressionIntent("face", "unknown", None), text="正文 🙄"),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert [segment.to_dict() for segment in delivered.message_segments] == [{"type": "text", "data": {"text": "正文 🙄"}}]
            assert delivered.reply_to_event_id is None
            assert client.text_calls[0][1] == [{"type": "text", "data": {"text": "正文 🙄"}}]
        finally:
            database.close()

    asyncio.run(scenario())


def test_reaction_is_independent_and_only_after_text_success(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            source = EventRepository(database).insert(inbound("source"))
            client = FakeOneBotClient()
            await sender(database, client).send(
                result(reply=source.handle, expression=ExpressionIntent("reaction", "heart", source.handle)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert len(client.text_calls) == 1
            assert client.reaction_calls == [("100", "128512", True)]
            assert OutboxRepository(database).get(Sender.reaction_operation_key("outbound")).status == "sent"
        finally:
            database.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error, expected_status",
    [(OneBotTimeoutError("timeout"), "unknown"), (OneBotConnectionError("network"), "unknown"), (OneBotHTTPServerError("server"), "unknown"), (OneBotProtocolError("bad"), "unknown"), (OneBotActionError("rejected"), "failed"), (OneBotHTTPClientError("client"), "failed"), (OneBotHTTPRedirectError("redirect"), "failed")],
)
def test_text_failure_evidence_never_retries(error, expected_status, tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient(text_error=error)
            transport = sender(database, client)
            await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            repeated = await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert OutboxRepository(database).get(Sender.text_operation_key("outbound")).status == expected_status
            assert repeated.status == expected_status
            assert len(client.text_calls) == 1
        finally:
            database.close()

    asyncio.run(scenario())


def test_reverse_map_and_non_decimal_catalog_values_are_strictly_handled(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            source = EventRepository(database).insert(
                ConversationEvent(
                    event_id="source", platform_event_id=None, platform_message_id=None,
                    conversation_id="42", sequence=0, direction="inbound", actor="mumo", kind="text",
                    text="source", message_segments=(MessageSegment("text", {"text": "source"}),),
                    reply_to_event_id=None, reply_to_platform_message_id=None, occurred_at_utc=NOW,
                    received_at_utc=NOW, status="received", metadata={},
                )
            )
            database.connection.execute(
                "INSERT INTO platform_message_map (platform_message_id, event_id, source, created_at_utc) VALUES (?, ?, ?, ?)",
                ("321", source.event_id, "test", NOW.isoformat()),
            )
            client = FakeOneBotClient()
            strict = Sender(database, client, face_catalog={"annoyed": True}, reaction_catalog={"heart": "bad"})
            delivered = await strict.send(
                result(reply=source.handle, expression=ExpressionIntent("face", "annoyed", None)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert [segment.to_dict() for segment in delivered.message_segments] == [
                {"type": "reply", "data": {"id": "321"}},
                {"type": "text", "data": {"text": "hello 🙂"}},
            ]
            client.text_result = {"message_id": 701}
            await strict.send(
                result(expression=ExpressionIntent("reaction", "heart", source.handle)),
                event_id="reaction-outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert client.reaction_calls == []
        finally:
            database.close()

    asyncio.run(scenario())


def test_cross_conversation_or_unmapped_actions_degrade_and_text_failure_skips_reaction(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            other = EventRepository(database).insert(inbound("other", conversation_id="43"))
            client = FakeOneBotClient(text_error=OneBotTimeoutError("timeout"))
            await sender(database, client).send(
                result(reply=other.handle, expression=ExpressionIntent("reaction", "missing", other.handle)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert client.text_calls == [("42", [{"type": "text", "data": {"text": "hello 🙂"}}])]
            assert client.reaction_calls == []
            assert OutboxRepository(database).get(Sender.text_operation_key("outbound")).status == "unknown"

            no_platform = EventRepository(database).insert(
                ConversationEvent(
                    event_id="unmapped", platform_event_id=None, platform_message_id=None,
                    conversation_id="42", sequence=0, direction="inbound", actor="mumo", kind="text",
                    text="unmapped", message_segments=(MessageSegment("text", {"text": "unmapped"}),),
                    reply_to_event_id=None, reply_to_platform_message_id=None, occurred_at_utc=NOW,
                    received_at_utc=NOW, status="received", metadata={},
                )
            )
            plain = FakeOneBotClient()
            delivered = await sender(database, plain).send(
                result(reply=no_platform.handle), event_id="plain", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert [segment.type for segment in delivered.message_segments] == ["text"]
        finally:
            database.close()

    asyncio.run(scenario())


def test_reaction_unknown_is_independent_and_recoverable(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            source = EventRepository(database).insert(inbound("source"))
            client = FakeOneBotClient(reaction_error=OneBotConnectionError("network"))
            transport = sender(database, client)
            await transport.send(
                result(expression=ExpressionIntent("reaction", "heart", source.handle)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            key = Sender.reaction_operation_key("outbound")
            assert OutboxRepository(database).get(key).status == "unknown"
            assert len(client.text_calls) == 1 and len(client.reaction_calls) == 1
            assert OutboxRepository(database).recover_idempotent_reaction(key, NOW).status == "pending"
        finally:
            database.close()

    asyncio.run(scenario())


def test_mapping_collision_marks_non_retriable_and_duplicate_or_concurrent_send_calls_once(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            events = EventRepository(database)
            events.insert(inbound("other", platform_message_id="700"))
            client = FakeOneBotClient()
            transport = sender(database, client)
            with pytest.raises(ValueError, match="mapping collision"):
                await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert OutboxRepository(database).get(Sender.text_operation_key("outbound")).status == "unknown"
            await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert len(client.text_calls) == 1
        finally:
            database.close()

        database = Database(tmp_path / "concurrent.sqlite3")
        try:
            class BlockingOneBotClient(FakeOneBotClient):
                def __init__(self):
                    super().__init__()
                    self.entered = asyncio.Event()
                    self.release = asyncio.Event()

                async def send_private_msg(self, user_id, message):
                    self.text_calls.append((user_id, message))
                    self.entered.set()
                    await self.release.wait()
                    return self.text_result

            client = BlockingOneBotClient()
            first_sender = sender(database, client)
            second_sender = sender(database, client)
            first = asyncio.create_task(
                first_sender.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            )
            await client.entered.wait()
            await second_sender.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert len(client.text_calls) == 1
            client.release.set()
            delivered = await first
            record = OutboxRepository(database).get(Sender.text_operation_key("outbound"))
            assert delivered.status == record.status == "sent"
            assert record.attempt_count == 1
            assert database.connection.execute("SELECT COUNT(*) FROM platform_message_map WHERE event_id = ?", ("outbound",)).fetchone()[0] == 1
        finally:
            database.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("owner_qq", [True, "not-a-number", "", 43])
def test_owner_binding_rejects_before_any_persistence_or_network(owner_qq, tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient()
            with pytest.raises(ValueError, match="owner_qq"):
                await sender(database, client).send(
                    result(), event_id="outbound", conversation_id="42", owner_qq=owner_qq, occurred_at_utc=NOW,
                )
            assert database.connection.execute("SELECT COUNT(*) FROM conversation_events").fetchone()[0] == 0
            assert database.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
            assert client.text_calls == []
        finally:
            database.close()

    asyncio.run(scenario())


def test_reverse_mapping_collision_and_catalog_drift_reuse_persisted_plan(tmp_path):
    class ReverseBoundClient(FakeOneBotClient):
        def __init__(self, database):
            super().__init__()
            self.database = database

        async def send_private_msg(self, user_id, message):
            self.text_calls.append((user_id, message))
            self.database.connection.execute(
                "INSERT INTO platform_message_map (platform_message_id, event_id, source, created_at_utc) VALUES (?, ?, ?, ?)",
                ("999", "outbound", "test", NOW.isoformat()),
            )
            return self.text_result

    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            collision_client = ReverseBoundClient(database)
            transport = sender(database, collision_client)
            with pytest.raises(ValueError, match="mapping collision"):
                await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert OutboxRepository(database).get(Sender.text_operation_key("outbound")).status == "unknown"
            assert EventRepository(database).get("outbound").status == "unknown"
            assert database.connection.execute("SELECT platform_message_id FROM platform_message_map WHERE event_id = ?", ("outbound",)).fetchone()[0] == "999"
            await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert len(collision_client.text_calls) == 1
        finally:
            database.close()

        database = Database(tmp_path / "drift.sqlite3")
        try:
            client = FakeOneBotClient()
            initial = Sender(database, client, face_catalog={}, reaction_catalog={})
            original = result(expression=ExpressionIntent("reaction", "heart", "M0"))
            first = await initial.send(original, event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert [segment.type for segment in first.message_segments] == ["text"]
            changed = Sender(database, client, face_catalog={"annoyed": "14"}, reaction_catalog={"heart": "128512"})
            repeated = await changed.send(original, event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert repeated.status == "sent"
            assert [segment.type for segment in repeated.message_segments] == ["text"]
            assert client.reaction_calls == [] and len(client.text_calls) == 1
            with pytest.raises(ValueError, match="identity conflict"):
                await changed.send(result(text="different"), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            with pytest.raises(ValueError, match="identity conflict"):
                await changed.send(
                    result(reply="M0", expression=ExpressionIntent("reaction", "heart", "M0")),
                    event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
                )
        finally:
            database.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("response", [{}, {"message_id": True}, {"message_id": "not-decimal"}])
def test_invalid_success_message_id_is_unknown_without_map_or_retry(response, tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient(text_result=response)
            transport = sender(database, client)
            first = await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            repeated = await transport.send(result(), event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert first.status == repeated.status == "unknown"
            assert len(client.text_calls) == 1
            assert database.connection.execute("SELECT COUNT(*) FROM platform_message_map WHERE event_id = ?", ("outbound",)).fetchone()[0] == 0
        finally:
            database.close()

    asyncio.run(scenario())


def test_reaction_server_unknown_and_action_failed_do_not_change_sent_text(tmp_path):
    async def scenario():
        database = Database(tmp_path / "server.sqlite3")
        try:
            source = EventRepository(database).insert(inbound("source"))
            client = FakeOneBotClient(reaction_error=OneBotHTTPServerError("server"))
            transport = sender(database, client)
            event = await transport.send(
                result(expression=ExpressionIntent("reaction", "heart", source.handle)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            key = Sender.reaction_operation_key("outbound")
            assert event.status == "sent"
            assert OutboxRepository(database).get(key).status == "unknown"
            assert OutboxRepository(database).recover_idempotent_reaction(key, NOW).status == "pending"
        finally:
            database.close()

        database = Database(tmp_path / "action.sqlite3")
        try:
            source = EventRepository(database).insert(inbound("source"))
            client = FakeOneBotClient(reaction_error=OneBotActionError("rejected"))
            transport = sender(database, client)
            event = await transport.send(
                result(expression=ExpressionIntent("reaction", "heart", source.handle)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert event.status == "sent"
            assert OutboxRepository(database).get(Sender.reaction_operation_key("outbound")).status == "failed"
            await transport.send(
                result(expression=ExpressionIntent("reaction", "heart", source.handle)),
                event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW,
            )
            assert len(client.reaction_calls) == 1
        finally:
            database.close()

    asyncio.run(scenario())


def test_face_catalog_drift_keeps_existing_event_text_only(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient()
            original = result(expression=ExpressionIntent("face", "annoyed", None))
            initial = Sender(database, client, face_catalog={}, reaction_catalog={})
            first = await initial.send(original, event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            changed = Sender(database, client, face_catalog={"annoyed": "14"}, reaction_catalog={})
            repeated = await changed.send(original, event_id="outbound", conversation_id="42", owner_qq=42, occurred_at_utc=NOW)
            assert [segment.type for segment in first.message_segments] == ["text"]
            assert [segment.type for segment in repeated.message_segments] == ["text"]
            assert len(client.text_calls) == 1
        finally:
            database.close()

    asyncio.run(scenario())

# --- 语音段：独立幂等键、未知态绝不重发（TTS P1-4，见 §3.5）---


def voice_event(event_id="voice-part", *, conversation_id="42", text="嗯，在呢。"):
    return ConversationEvent(
        event_id=event_id, platform_event_id=None, platform_message_id=None,
        conversation_id=conversation_id, sequence=0, direction="outbound", actor="qichi", kind="text",
        text=text, message_segments=(MessageSegment("record", {"file": "file:///E:/voice.wav"}),),
        reply_to_event_id=None, reply_to_platform_message_id=None,
        occurred_at_utc=NOW, received_at_utc=NOW, status="pending",
        metadata={"voice": {"spoken": text}},
    )


def record_payload(user_id="42", file="file:///E:/voice.wav"):
    return {"action_kind": "record", "user_id": user_id,
            "message": [{"type": "record", "data": {"file": file}}]}


def test_voice_part_dispatches_once_and_records_the_platform_message_id(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient(text_result={"message_id": 12345})
            event = EventRepository(database).insert(voice_event())

            status = await sender(database, client).dispatch_record(
                event=event, payload=record_payload(), occurred_at_utc=NOW
            )

            assert status == "sent"
            assert client.text_calls == [("42", [{"type": "record", "data": {"file": "file:///E:/voice.wav"}}])]
            assert EventRepository(database).get("voice-part").status == "sent"
            row = database.connection.execute(
                "SELECT platform_message_id FROM platform_message_map WHERE event_id = ?", ("voice-part",)
            ).fetchone()
            assert row["platform_message_id"] == "12345"
        finally:
            database.close()

    asyncio.run(scenario())


def test_voice_timeout_is_unknown_and_never_resent(tmp_path):
    """语音不幂等：超时收成 unknown，再调一次也不许外呼。"""

    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient(text_error=OneBotTimeoutError("boom"))
            outbox = OutboxRepository(database)
            event = EventRepository(database).insert(voice_event())
            part_sender = sender(database, client)

            first = await part_sender.dispatch_record(event=event, payload=record_payload(), occurred_at_utc=NOW)
            second = await part_sender.dispatch_record(event=event, payload=record_payload(), occurred_at_utc=NOW)

            assert first == "unknown" and second == "unknown"
            assert len(client.text_calls) == 1, "未知态之后绝不允许再发一次"
            assert outbox.get(Sender.record_operation_key("voice-part")).status == "unknown"
            assert EventRepository(database).get("voice-part").status == "unknown"
        finally:
            database.close()

    asyncio.run(scenario())


def test_voice_payload_identity_is_immutable_per_event(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient()
            event = EventRepository(database).insert(voice_event())
            part_sender = sender(database, client)
            await part_sender.dispatch_record(event=event, payload=record_payload(), occurred_at_utc=NOW)

            with pytest.raises(ValueError, match="immutable intent identity"):
                await part_sender.dispatch_record(
                    event=event, payload=record_payload(file="file:///E:/other.wav"), occurred_at_utc=NOW
                )
            assert len(client.text_calls) == 1
        finally:
            database.close()

    asyncio.run(scenario())


def test_voice_and_text_fallbacks_keep_two_separate_operation_keys(tmp_path):
    """合成失败要退回文字：两者各有各的幂等键，不能互相顶掉。"""

    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient()
            event = EventRepository(database).insert(voice_event())
            part_sender = sender(database, client)

            voice_status = await part_sender.dispatch_record(
                event=event, payload=record_payload(), occurred_at_utc=NOW
            )
            text_payload = {"action_kind": "text", "user_id": "42",
                            "message": [{"type": "text", "data": {"text": "嗯，在呢。"}}]}
            text_status = await part_sender.dispatch_text(
                event=event, payload=text_payload, occurred_at_utc=NOW
            )

            assert (voice_status, text_status) == ("sent", "sent")
            assert len(client.text_calls) == 2
            assert Sender.record_operation_key("voice-part") != Sender.text_operation_key("voice-part")
            rows = database.connection.execute("SELECT operation_key FROM outbox").fetchall()
            assert len(rows) == 2
        finally:
            database.close()

    asyncio.run(scenario())


def test_dispatch_helpers_refuse_the_wrong_action_kind(tmp_path):
    async def scenario():
        database = Database(tmp_path / "qichi.sqlite3")
        try:
            client = FakeOneBotClient()
            event = EventRepository(database).insert(voice_event())
            part_sender = sender(database, client)
            text_payload = {"action_kind": "text", "user_id": "42", "message": []}

            with pytest.raises(ValueError, match="record"):
                await part_sender.dispatch_record(event=event, payload=text_payload, occurred_at_utc=NOW)
            with pytest.raises(ValueError, match="text"):
                await part_sender.dispatch_text(event=event, payload=record_payload(), occurred_at_utc=NOW)
            assert client.text_calls == []
        finally:
            database.close()

    asyncio.run(scenario())

