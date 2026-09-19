"""端到端：语音真的在文字之后到达，失败也不吞掉她的话（TTS P2 回放）。"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.dialogue import DialogueResult
from qichi.storage.database import Database
from qichi.transport.sender import Sender
from qichi.voice.director import VoiceClip, VoiceUnavailable
from qichi.voice.dispatch import VoiceDispatcher

NOW = datetime(2026, 9, 14, 10, tzinfo=timezone.utc)
OWNER = "10001"
BOT = "20001"


class Counter:
    def count_text(self, text):
        return len(text)


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate(self, messages, **kwargs):
        return LLMGeneration(self.outputs.pop(0), "primary", "fake", 1, 1, 1)


class FakeNapCat:
    def __init__(self, error=None, record_only=False):
        self.texts: list[tuple[str, list[dict]]] = []
        self.error = error
        self.record_only = record_only

    async def send_private_msg(self, user_id, message):
        is_record = any(segment.get("type") == "record" for segment in message)
        self.texts.append((str(user_id), message))   # 先记下这次尝试，再决定是否报错
        if self.error and (is_record or not self.record_only):
            raise self.error
        return {"message_id": 700 + len(self.texts)}

    async def set_msg_emoji_like(self, *args, **kwargs):
        return {"ok": True}


class RecordingDirector:
    """记录「开始合成时，QQ 上已经发出去几条」——这就是卡-3 的顺序判据。"""

    def __init__(self, napcat, *, path, error=None):
        self.napcat = napcat
        self.path = path
        self.error = error
        self.render_calls: list[tuple[str, str]] = []
        self.sent_when_rendering: list[int] = []

    async def render(self, text, *, situation=""):
        self.render_calls.append((text, situation))
        self.sent_when_rendering.append(len(self.napcat.texts))
        if self.error:
            raise self.error
        return VoiceClip(spoken="去掉颜文字的那句", residue="(￣▽￣)",
                         instructions="在「谁求了」之后停半拍", path=self.path, from_cache=False)


class BrokenDispatcher:
    """投递器整个没跑起来 —— 这一支必须退回文字，不能让她静默。"""

    async def deliver(self, job):
        raise RuntimeError("boom")


class ScriptedEngine:
    def __init__(self, outcome):
        self.outcome = outcome
        self.inputs = []

    async def generate(self, value):
        self.inputs.append(value)
        return self.outcome


def message(message_id=1, text="在吗", timestamp=NOW.timestamp()):
    return {
        "post_type": "message", "message_type": "private", "sub_type": "friend",
        "self_id": int(BOT), "user_id": OWNER, "target_id": OWNER,
        "sender": {"user_id": OWNER}, "message_id": message_id, "time": timestamp,
        "message": [{"type": "text", "data": {"text": text}}],
    }


def make_app(database, napcat, outcome, *, factory=None):
    evidence = ProviderCapabilityEvidence("fake", "fake", 262144, "test", NOW)
    builder = ContextBuilder(Counter(), ModelCapability("fake", 262144, "fake", evidence), output_reserve_tokens=10)
    engine = ScriptedEngine(outcome)
    return G0Application(
        database, builder, engine, napcat, owner_qq=OWNER, bot_qq=BOT, role_core="你是角色。",
        clock=lambda: NOW, face_catalog={}, reaction_catalog={}, voice_dispatcher_factory=factory,
        # 声明语音能力就必须同时给出长度边界（2026-09-15 §2.42）；不给会被能力层直接拒。
        voice_max_chars=120 if factory is not None else None,
    )


def voiced_outcome(version=0):
    return DialogueResult(
        "先回这一句。\n\n再说这一句(￣▽￣)", None, None, "primary", version,
        message_parts=("先回这一句。", "再说这一句(￣▽￣)"), voice_part_index=2,
    )


def voiced_single_outcome(version=0):
    """整轮只有一段，而那一段被标成语音（她那种"一句话"的回复）。"""

    return DialogueResult(
        "在呢(￣▽￣)", None, None, "primary", version,
        message_parts=("在呢(￣▽￣)",), voice_part_index=1,
    )


async def drain(app):
    for _ in range(50):
        tasks = [task for task in list(app._voice_tasks) if not task.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_voice_is_dispatched_after_the_text_and_carries_the_residue(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF")
    try:
        director = RecordingDirector(napcat, path=clip)
        sender = None

        def factory(real_sender):
            nonlocal sender
            sender = real_sender
            return VoiceDispatcher(database=database, sender=real_sender, director=director, clock=lambda: NOW)

        app = make_app(database, napcat, voiced_outcome(), factory=factory)
        await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text", "record", "text"], "文字先到，语音随后，颜文字残段最后"
        assert napcat.texts[0][1] == [{"type": "text", "data": {"text": "先回这一句。"}}], "语音那一段不重复发文字"
        assert napcat.texts[2][1] == [{"type": "text", "data": {"text": "(￣▽￣)"}}]
        assert director.sent_when_rendering == [1], "开始合成时，文字已经发出去了（不阻塞热路径）"
        assert director.render_calls[0][0] == "再说这一句(￣▽￣)"
        # 2026-09-15 方向 1：写手要看到「他的原话 + 时间差 + 她这一轮的完整分句」。
        situation = director.render_calls[0][1]
        assert "在吗" in situation, "他刚说的那一句必须在情境里"
        assert "用户（" in situation, "往来要带钟点"
        assert ("分钟前" in situation or "刚刚" in situation), "还要带距现在多久"
        assert "1. 先回这一句。" in situation, "她这一轮的文字段也要给写手看"
        assert "2. 再说这一句(￣▽￣)" in situation
        assert "←这一句用语音说" in situation
    finally:
        database.close()


@pytest.mark.asyncio
async def test_synthesis_failure_falls_back_to_her_own_words(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        director = RecordingDirector(napcat, path=tmp_path / "missing.wav",
                                     error=VoiceUnavailable("no audio"))

        def factory(real_sender):
            return VoiceDispatcher(database=database, sender=real_sender, director=director, clock=lambda: NOW)

        app = make_app(database, napcat, voiced_outcome(), factory=factory)
        await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text", "text"], "合成失败时那一段原样以文字发出，不吞掉、不兜底"
        assert napcat.texts[1][1] == [{"type": "text", "data": {"text": "再说这一句(￣▽￣)"}}]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_unknown_delivery_never_resends_and_never_sends_the_residue(tmp_path):
    """派发未知时：不补发、不重发，也不假装成功地去发颜文字残段。"""

    from qichi.transport.onebot_client import OneBotTimeoutError

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat(error=OneBotTimeoutError("boom"), record_only=True)
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF")
    try:
        director = RecordingDirector(napcat, path=clip)

        def factory(real_sender):
            return VoiceDispatcher(database=database, sender=real_sender, director=director, clock=lambda: NOW)

        app = make_app(database, napcat, voiced_outcome(), factory=factory)
        await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text", "record"], "未知态之后不再补发，也不发颜文字残段"
        rows = {
            row["status"]: row["n"]
            for row in database.connection.execute(
                "SELECT status, COUNT(*) AS n FROM outbox GROUP BY status"
            ).fetchall()
        }
        assert rows == {"sent": 1, "unknown": 1}, "文字一条 sent；语音一条 unknown，绝无重发"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_without_a_voice_channel_the_marker_never_swallows_a_part(tmp_path):
    """没就绪时不摘段：她的每个字都照常以文字发出。"""

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        app = make_app(database, napcat, voiced_outcome(), factory=None)
        await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text", "text"], "语音没就绪 -> 两段都走文字"
        assert napcat.texts[1][1] == [{"type": "text", "data": {"text": "再说这一句(￣▽￣)"}}]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_a_single_part_voice_turn_speaks_without_any_text_first(tmp_path):
    """用户 2026-09-14 裁定：整轮只有一段且被标成语音时，就只发语音。"""

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF")
    try:
        director = RecordingDirector(napcat, path=clip)

        def factory(real_sender):
            return VoiceDispatcher(database=database, sender=real_sender, director=director, clock=lambda: NOW)

        app = make_app(database, napcat, voiced_single_outcome(), factory=factory)
        delivered = await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["record", "text"], "语音先到、颜文字残段跟在后面；前面不再有文字组"
        assert napcat.texts[1][1] == [{"type": "text", "data": {"text": "(￣▽￣)"}}]
        assert director.sent_when_rendering == [0], "开始合成时一条都还没发 —— 这一轮本来就没有文字组"
        assert director.render_calls[0][0] == "在呢(￣▽￣)"
        assert delivered is not None, "这一轮要返回真的那条语音事件"
        assert delivered.message_segments[0].type == "record"
        assert delivered.status == "sent"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_a_single_part_voice_turn_falls_back_to_her_words_without_duplicating(tmp_path):
    """合成失败：那一段原样以文字发出，应用层不再重复发第二遍。"""

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        director = RecordingDirector(napcat, path=tmp_path / "missing.wav",
                                     error=VoiceUnavailable("no audio"))

        def factory(real_sender):
            return VoiceDispatcher(database=database, sender=real_sender, director=director, clock=lambda: NOW)

        app = make_app(database, napcat, voiced_single_outcome(), factory=factory)
        delivered = await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text"], "只发一条文字，且不是固定兜底"
        assert napcat.texts[0][1] == [{"type": "text", "data": {"text": "在呢(￣▽￣)"}}]
        assert delivered is not None and delivered.text == "在呢(￣▽￣)"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_a_single_part_voice_turn_degrades_when_the_dispatcher_itself_fails(tmp_path, caplog):
    """投递器整个没跑起来：退回文字路径，并留下 §2.40 写明却一直没实现的记录。"""

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        def factory(real_sender):
            return BrokenDispatcher()

        app = make_app(database, napcat, voiced_single_outcome(), factory=factory)
        with caplog.at_level(logging.WARNING):
            delivered = await app.handle_onebot(message(), received_at_utc=NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text"], "退回文字，绝不静默"
        assert napcat.texts[0][1] == [{"type": "text", "data": {"text": "在呢(￣▽￣)"}}]
        assert "voice_only_turn_degraded" in caplog.text, "这一轮为什么没出声，账本上必须看得见"
        assert delivered is not None
    finally:
        database.close()


class QueueEngine:
    """按顺序吐出预先排好的结果（第一轮对话、第二轮主动开口）。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def generate(self, value):
        return self.outcomes.pop(0)


@pytest.mark.asyncio
async def test_an_initiative_turn_can_speak_without_a_text_group(tmp_path):
    """2026-09-14：主动开口这一路以前完全没接语音，她标了语音也只会当文字发出去。"""

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF")
    try:
        director = RecordingDirector(napcat, path=clip)

        def factory(real_sender):
            return VoiceDispatcher(database=database, sender=real_sender, director=director, clock=lambda: NOW)

        app = make_app(database, napcat, DialogueResult(
            "在呢。", None, None, "primary", 0, message_parts=("在呢。",),
        ), factory=factory)
        await app.handle_onebot(message(), received_at_utc=NOW)

        cursor = database.connection.execute(
            "SELECT context_version, last_user_activity_utc FROM conversation_cursors "
            "WHERE conversation_id = ?", (OWNER,),
        ).fetchone()
        assert cursor is not None, "第一轮之后应当有会话游标"
        version = int(cursor["context_version"])
        activity = datetime.fromisoformat(str(cursor["last_user_activity_utc"]))
        app.dialogue_engine = QueueEngine([voiced_single_outcome(version=version)])

        delivered = await app.generate_initiative(OWNER, version, activity, NOW)
        await drain(app)

        kinds = [call[1][0]["type"] for call in napcat.texts]
        assert kinds == ["text", "record", "text"], "他那一轮的文字、她主动开口的语音、颜文字残段"
        assert delivered is not None and delivered.message_segments[0].type == "record"
        assert director.render_calls[-1][0] == "在呢(￣▽￣)"
    finally:
        database.close()
