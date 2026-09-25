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


def job(group="grp", index=1, count=2, source="dialogue"):
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
        source=source,
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


# --- 2026-09-20：语音段必须按它这一轮的真实来源进记忆 --------------------
# 判据见 doc/问题冻结-20260920-语音进记忆.md：语音只是投递形态，不是另一种内容；
# 但它自己造事件时不写 generation_metadata，而记忆的可靠判据只看那个字段。

def _reliable(event) -> bool:
    from qichi.memory.worker import MemoryWorker

    return MemoryWorker._is_reliable(event)


def _residue_id(part_event_id: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, "qichi:voice-residue:%s" % part_event_id))


def test_a_dialogue_voice_event_reaches_memory(tmp_path):
    """命中：对话里的语音段带上真实来源后，记忆能收，且正文仍是她的原话。"""

    async def scenario():
        database, _client, _director, dispatcher = build(tmp_path)
        try:
            outcome = await dispatcher.deliver(job())
            event = EventRepository(database).get(outcome.event_id)

            assert event.metadata["generation_metadata"]["source"] == "dialogue"
            assert event.text == PART_TEXT
            assert _reliable(event) is True
        finally:
            database.close()

    asyncio.run(scenario())


def test_an_interaction_voice_event_reaches_memory(tmp_path):
    """命中：戳一戳那一轮发的语音，与文字段同一条判据。"""

    async def scenario():
        database, _client, _director, dispatcher = build(tmp_path)
        try:
            outcome = await dispatcher.deliver(job(source="interaction"))
            event = EventRepository(database).get(outcome.event_id)

            assert event.metadata["generation_metadata"]["source"] == "interaction"
            assert _reliable(event) is True
        finally:
            database.close()

    asyncio.run(scenario())


def test_degraded_voice_text_reaches_memory(tmp_path):
    """命中：合成失败退回文字，那同样是她的原话，也要能进记忆。"""

    async def scenario():
        database, _client, _director, dispatcher = build(
            tmp_path, director_error=VoiceUnavailable("no audio")
        )
        try:
            outcome = await dispatcher.deliver(job())
            event = EventRepository(database).get(outcome.event_id)

            assert outcome.reason.startswith("degraded:")
            assert event.metadata["generation_metadata"]["source"] == "dialogue"
            assert _reliable(event) is True
        finally:
            database.close()

    asyncio.run(scenario())


def test_an_initiative_voice_event_reaches_memory(tmp_path):
    """命中（卡⑤ 2026-09-21 用户裁定）：主动开口也进记忆。

    此前主动来源被 _is_reliable 一律挡在整合窗口外，副作用是窗口残缺——她的开场
    不在片段里，提取器读到的是没有前因的对话。现在准入，边界交给取证角色矩阵。
    """

    async def scenario():
        database, _client, _director, dispatcher = build(tmp_path)
        try:
            outcome = await dispatcher.deliver(job(source="initiative"))
            event = EventRepository(database).get(outcome.event_id)

            assert event.metadata["generation_metadata"]["source"] == "initiative"
            assert _reliable(event) is True
        finally:
            database.close()

    asyncio.run(scenario())


def test_the_residue_deliberately_stays_out_of_memory(tmp_path):
    """不误判：残段只是从语音里剥出来的颜文字，而语音正文里已经含它一次。
    进了记忆会在明细里留一条重复的装饰行，所以它有意不带来源。"""

    async def scenario():
        database, _client, _director, dispatcher = build(tmp_path)
        try:
            outcome = await dispatcher.deliver(job())
            events = EventRepository(database)
            residue = events.get(_residue_id(outcome.event_id))

            assert "generation_metadata" not in residue.metadata
            assert _reliable(residue) is False
            # 边角不是丢内容：这段颜文字确实已经在语音正文里。
            assert residue.text in events.get(outcome.event_id).text
        finally:
            database.close()

    asyncio.run(scenario())


def test_an_invalid_source_is_rejected_at_construction():
    """不误判：来源是白名单，非法即拒绝（fail closed），不静默当成对话。"""

    with pytest.raises(ValueError, match="source"):
        job(source="poke")


def test_a_source_less_voice_event_stays_unreliable():
    """不误判 / 向后兼容：库存里没有来源的语音事件不会被误判成可靠。"""

    legacy = ConversationEvent(
        event_id="legacy-voice",
        platform_event_id=None,
        platform_message_id=None,
        conversation_id="42",
        sequence=1,
        direction="outbound",
        actor="qichi",
        kind="text",
        text=PART_TEXT,
        message_segments=(MessageSegment("record", {"file": "file:///x.wav"}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=NOW,
        received_at_utc=NOW,
        status="sent",
        metadata={"voice": {"spoken": PART_TEXT, "residue": "", "instructions": "", "file": "x.wav", "part_index": 0, "part_count": 1}},
    )

    assert _reliable(legacy) is False


def test_dispatcher_rejects_a_non_job(tmp_path):
    async def scenario():
        database, _client, _director, dispatcher = build(tmp_path)
        try:
            with pytest.raises(TypeError):
                await dispatcher.deliver("nope")
        finally:
            database.close()

    asyncio.run(scenario())
