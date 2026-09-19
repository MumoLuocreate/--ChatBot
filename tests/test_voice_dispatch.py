"""语音编排：整段送达、颜文字残段、失败退回文字、以及重入不重发。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.transport.onebot_client import OneBotTimeoutError
from qichi.transport.sender import Sender
from qichi.voice.director import VoiceClip, VoiceUnavailable
from qichi.voice.dispatch import VoiceDispatcher, VoiceJob

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
PART_TEXT = "嗯，在呢。这声兔兔叫得挺顺口的嘛(￣▽￣)"


class FakeOneBotClient:
    def __init__(self, *, error=None):
        self.error = error
        self.calls: list[tuple[object, list[dict]]] = []

    async def send_private_msg(self, user_id, message):
        self.calls.append((user_id, message))
        if self.error:
            raise self.error
        return {"message_id": 700 + len(self.calls)}


class FakeDirector:
    def __init__(self, *, path, residue="(￣▽￣)", spoken="嗯，在呢。这声兔兔叫得挺顺口的嘛", error=None):
        self.calls: list[tuple[str, str]] = []
        self._clip = VoiceClip(spoken=spoken, residue=residue, instructions="慵懒松弛", path=path, from_cache=False)
        self.error = error

    async def render(self, text, *, situation=""):
        self.calls.append((text, situation))
        if self.error:
            raise self.error
        return self._clip


def job(group="grp", index=1, count=2):
    from qichi.transport.sender import Sender

    return VoiceJob(
        group_event_id=group,
        event_id=Sender.voice_part_event_id(group, index),
        part_index=index,
        part_count=count,
        part_text=PART_TEXT,
        conversation_id="42",
        owner_qq="42",
        situation="他在叫她兔兔",
    )


def build(tmp_path, *, client_error=None, director_error=None, clip_path=None, residue="(￣▽￣)"):
    database = Database(tmp_path / "qichi.sqlite3")
    client = FakeOneBotClient(error=client_error)
    if clip_path is None:
        clip_path = tmp_path / "voice.wav"
        clip_path.write_bytes(b"RIFF")
    director = FakeDirector(path=clip_path, residue=residue, error=director_error)
    sender = Sender(database, client, face_catalog={}, reaction_catalog={})
    dispatcher = VoiceDispatcher(database=database, sender=sender, director=director, clock=lambda: NOW)
    return database, client, director, dispatcher


def test_voice_and_residue_both_reach_the_ledger_and_qq(tmp_path):
    async def scenario():
        database, client, director, dispatcher = build(tmp_path)
        try:
            outcome = await dispatcher.deliver(job())

            assert outcome.status == "sent" and outcome.reason == "voice"
            assert [call[1][0]["type"] for call in client.calls] == ["record", "text"]
            assert client.calls[1][1] == [{"type": "text", "data": {"text": "(￣▽￣)"}}]
            assert director.calls == [(PART_TEXT, "他在叫她兔兔")]

            events = EventRepository(database)
            voice_event = events.get(outcome.event_id)
            assert voice_event.text == PART_TEXT, "账本里必须是她的原话"
            assert [segment.type for segment in voice_event.message_segments] == ["record"]
            assert voice_event.metadata["voice"]["spoken"] == "嗯，在呢。这声兔兔叫得挺顺口的嘛"
            residue_id = str(__import__("uuid").uuid5(__import__("uuid").NAMESPACE_URL,
                                              "qichi:voice-residue:%s" % outcome.event_id))
            assert events.get(residue_id).text == "(￣▽￣)"
            assert database.connection.execute("SELECT COUNT(*) AS n FROM outbox").fetchone()["n"] == 2
        finally:
            database.close()

    asyncio.run(scenario())


def test_without_residue_only_the_voice_is_sent(tmp_path):
    async def scenario():
        database, client, _director, dispatcher = build(tmp_path, residue="")
        try:
            outcome = await dispatcher.deliver(job())
            assert [call[1][0]["type"] for call in client.calls] == ["record"]
            assert database.connection.execute("SELECT COUNT(*) AS n FROM outbox").fetchone()["n"] == 1
        finally:
            database.close()

    asyncio.run(scenario())


def test_synthesis_failure_sends_her_words_as_text_instead(tmp_path):
    """合成挂了不许吞掉她的话：原样以文字发出，不重写、不兜底。"""

    async def scenario():
        database, client, director, dispatcher = build(tmp_path, director_error=VoiceUnavailable("no audio"))
        try:
            outcome = await dispatcher.deliver(job())

            assert outcome.reason.startswith("degraded:")
            assert [call[1][0]["type"] for call in client.calls] == ["text"]
            assert client.calls[0][1] == [{"type": "text", "data": {"text": PART_TEXT}}], "原话，含颜文字"
            assert EventRepository(database).get(outcome.event_id).text == PART_TEXT
            assert director.calls, "仍然试过合成"
        finally:
            database.close()

    asyncio.run(scenario())


def test_unknown_delivery_never_sends_the_residue(tmp_path):
    """派发未知时连颜文字也不发：不能假装那条语音已经到了。"""

    async def scenario():
        database, client, _director, dispatcher = build(tmp_path, client_error=OneBotTimeoutError("boom"))
        try:
            outcome = await dispatcher.deliver(job())

            assert outcome.status == "unknown"
            assert [call[1][0]["type"] for call in client.calls] == ["record"]
            assert database.connection.execute("SELECT COUNT(*) AS n FROM outbox").fetchone()["n"] == 1
        finally:
            database.close()

    asyncio.run(scenario())


def test_reentry_reuses_the_recorded_audio_and_never_resends(tmp_path):
    async def scenario():
        database, client, director, dispatcher = build(tmp_path)
        try:
            first = await dispatcher.deliver(job())
            calls_after_first = len(client.calls)
            second = await dispatcher.deliver(job())

            assert second.status == "sent"
            assert len(director.calls) == 1, "重入不许重新合成"
            assert len(client.calls) == calls_after_first, "重入不许再发一次"
        finally:
            database.close()

    asyncio.run(scenario())


def test_dispatcher_rejects_a_non_job(tmp_path):
    async def scenario():
        database, _client, _director, dispatcher = build(tmp_path)
        try:
            with pytest.raises(TypeError):
                await dispatcher.deliver("nope")
        finally:
            database.close()

    asyncio.run(scenario())
