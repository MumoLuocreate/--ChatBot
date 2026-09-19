from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from qichi.app import G0Application
from qichi.config import VisionSettings
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration, LLMImageRejectedError
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.storage.database import Database
from qichi.transport.onebot_client import OneBotActionError

NOW = datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc)
OWNER, BOT = "10001", "20001"
PNG = b"\x89PNG\r\n\x1a\n" + b"landed-png"


class Counter:
    def count_text(self, text):
        return len(text)


def vision(**overrides) -> VisionSettings:
    values = dict(
        enabled=True,
        max_images_per_turn=2,
        max_image_bytes=1024 * 1024,
        download_timeout_seconds=10,
        keep_original_images=True,
        max_total_media_bytes=200 * 1024 * 1024,
        max_media_age_days=90,
    )
    values.update(overrides)
    return VisionSettings(**values)


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        return LLMGeneration(self.outputs.pop(0), "primary", "fake-v4", 1, 1, 1.0)


class RejectingLLM(FakeLLM):
    """Refuses the first request that carries pictures, like a cautious endpoint."""

    def __init__(self, outputs):
        super().__init__(outputs)
        self.rejected = False

    async def generate(self, messages, **kwargs):
        if any(message.images for message in messages) and not self.rejected:
            self.calls.append(tuple(messages))
            self.rejected = True
            raise LLMImageRejectedError("LLM rejected an image request (400)")
        return await super().generate(messages, **kwargs)


class FakeNapCat:
    def __init__(self, payloads=None, error=None):
        self.payloads, self.error = payloads or {}, error
        self.calls, self.sent = [], []

    async def get_image(self, file):
        self.calls.append(file)
        if self.error is not None:
            raise self.error
        return self.payloads.get(file, {})

    async def send_private_msg(self, user_id, message):
        self.sent.append((str(user_id), message))
        return {"message_id": 700}


class CountingNapCat(FakeNapCat):
    """每一轮换一个平台消息 id：多轮测试会在第二次发送上撞同一个 id。"""

    def __init__(self, payloads=None, error=None):
        super().__init__(payloads, error)
        self._next = 700

    async def send_private_msg(self, user_id, message):
        self.sent.append((str(user_id), message))
        self._next += 1
        return {"message_id": self._next}


def raw(message_id, *, text=None, images=1, at=NOW, with_url=True):
    message = []
    if text is not None:
        message.append({"type": "text", "data": {"text": text}})
    for index in range(images):
        data = {"file": f"token-{index}"}
        if with_url:
            data["url"] = f"https://cdn.test/{index}.png"
        message.append({"type": "image", "data": data})
    return {
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "self_id": int(BOT),
        "user_id": OWNER,
        "target_id": OWNER,
        "sender": {"user_id": OWNER},
        "message_id": message_id,
        "time": at.timestamp(),
        "message": message,
    }


def builder():
    evidence = ProviderCapabilityEvidence("fake", "fake-v4", 262_144, "local", NOW)
    return ContextBuilder(
        Counter(),
        ModelCapability("fake-v4", 262_144, "fake", evidence),
        output_reserve_tokens=1024,
    )


def application(database, llm, napcat, *, settings, media_root, carry_turns=1):
    engine = DialogueEngine(
        llm, OutputGuard(Counter(), 2048), ResponseProtocol(face_keys=(), reaction_keys=())
    )
    return G0Application(
        database,
        builder(),
        engine,
        napcat,
        owner_qq=OWNER,
        bot_qq=BOT,
        role_core="你是角色。",
        clock=lambda: NOW,
        vision=settings,
        get_image_async=napcat.get_image,
        media_root=media_root,
        image_carry_turns=carry_turns,
    )


def prompt_of(call) -> str:
    return "\n".join(message.content for message in call)


def carriers(call):
    return [message for message in call if message.images]


def trace_details(database, phase: str) -> list[dict]:
    rows = database.connection.execute(
        "SELECT details_json FROM turn_trace_events WHERE phase = ? ORDER BY rowid", (phase,)
    ).fetchall()
    return [json.loads(row["details_json"]) for row in rows]


@pytest.mark.asyncio
async def test_the_switch_being_off_never_fetches_and_never_attaches(tmp_path):
    database = Database(tmp_path / "off.sqlite3")
    media_root = tmp_path / "data"
    napcat, llm = FakeNapCat(), FakeLLM(["嗯，我在这儿"])
    client = application(
        database, llm, napcat, settings=vision(enabled=False), media_root=media_root
    )
    try:
        await client.handle_onebot(raw(101, text="看看这个", images=2))
    finally:
        database.close()

    assert napcat.calls == [], "开关关闭时不得发起任何取图"
    assert len(llm.calls) == 1
    assert carriers(llm.calls[0]) == []
    assert "不含视觉结果" in prompt_of(llm.calls[0])
    assert not (media_root / "media").exists(), "关闭时不得留下任何文件"


@pytest.mark.asyncio
async def test_a_switch_that_is_on_attaches_this_turn_picture(tmp_path):
    cached = tmp_path / "cached.png"
    cached.write_bytes(PNG)
    database = Database(tmp_path / "on.sqlite3")
    media_root = tmp_path / "data"
    napcat = FakeNapCat({"token-0": {"file": str(cached)}})
    llm = FakeLLM(["我看到了"])
    client = application(database, llm, napcat, settings=vision(), media_root=media_root)
    try:
        await client.handle_onebot(raw(101, text="看看这个"))
        event_id = database.connection.execute(
            "SELECT event_id FROM conversation_events WHERE platform_message_id = '101'"
        ).fetchone()["event_id"]
        context = trace_details(database, "context")[0]
    finally:
        database.close()

    assert napcat.calls == ["token-0"]
    assert len(carriers(llm.calls[0])) == 1
    carried = carriers(llm.calls[0])[0]
    assert carried.role == "user" and carried.content == "看看这个"
    assert carried.images[0].content_type == "image/png"
    assert "已随本轮提供 1 张" in prompt_of(llm.calls[0])
    landed = carried.images[0].path
    assert landed.endswith(f"{event_id}-0.png")
    from pathlib import Path

    assert Path(landed).read_bytes() == PNG
    assert context["images_attached"] == 1
    assert context["image_statuses"] == ["stored"]
    assert context["vision_skipped"] is None


@pytest.mark.asyncio
async def test_pictures_beyond_the_cap_are_reported_as_unexpanded(tmp_path):
    cached = tmp_path / "cached.png"
    cached.write_bytes(PNG)
    database = Database(tmp_path / "cap.sqlite3")
    napcat = FakeNapCat({"token-0": {"file": str(cached)}, "token-1": {"file": str(cached)}})
    llm = FakeLLM(["看到了"])
    client = application(database, llm, napcat, settings=vision(), media_root=tmp_path / "data")
    try:
        await client.handle_onebot(raw(101, text="三张", images=3))
        context = trace_details(database, "context")[0]
    finally:
        database.close()

    assert napcat.calls == ["token-0", "token-1"], "超出上限的图片不得被取回"
    assert "另有 1 张超出本轮上限未展开，不得描述其内容" in prompt_of(llm.calls[0])
    assert context["images_attached"] == 2
    assert context["images_unexpanded"] == 1


@pytest.mark.asyncio
async def test_a_picture_that_cannot_be_fetched_is_reported_and_never_guessed(tmp_path):
    database = Database(tmp_path / "failed.sqlite3")
    napcat = FakeNapCat(error=OneBotActionError("OneBot get_image returned status='failed' retcode=200"))
    llm = FakeLLM(["那是张什么图？我没看到"])
    client = application(database, llm, napcat, settings=vision(), media_root=tmp_path / "data")
    try:
        # 段里没有 url：取不到就是取不到，没有任何别的来源可试
        await client.handle_onebot(raw(101, text="看看这个", with_url=False))
        context = trace_details(database, "context")[0]
    finally:
        database.close()

    assert carriers(llm.calls[0]) == []
    rendered = prompt_of(llm.calls[0])
    assert "不含视觉结果" in rendered
    assert "另有 1 张本轮未取到，不得猜测内容" in rendered
    assert context["images_attached"] == 0
    assert context["images_unavailable"] == 1
    assert context["image_statuses"] == ["unavailable"]


@pytest.mark.asyncio
async def test_an_endpoint_that_refuses_pictures_falls_back_to_text(tmp_path):
    cached = tmp_path / "cached.png"
    cached.write_bytes(PNG)
    database = Database(tmp_path / "rejected.sqlite3")
    napcat = FakeNapCat({"token-0": {"file": str(cached)}})
    llm = RejectingLLM(["那我就先当没看到"])
    client = application(database, llm, napcat, settings=vision(), media_root=tmp_path / "data")
    try:
        await client.handle_onebot(raw(101, text="看看这个"))
        generation = trace_details(database, "generation")
    finally:
        database.close()

    assert len(llm.calls) == 2, "被拒绝后应当只重试一次，且不带图"
    assert len(carriers(llm.calls[0])) == 1
    assert carriers(llm.calls[1]) == []
    assert "不含视觉结果" in prompt_of(llm.calls[1]), "重试那轮的措辞必须改成没看到"
    assert "已随本轮提供" not in prompt_of(llm.calls[1])
    assert any(item.get("vision_degraded") == "endpoint_rejected" for item in generation)


@pytest.mark.asyncio
async def test_a_carried_picture_says_it_is_the_earlier_one_not_a_new_send(tmp_path):
    """2026-09-16 真机：宽限窗口把旧图挂到纯文字轮上，她只能读成「他又发了一张」。"""

    database = Database(tmp_path / "carry.sqlite3")
    media_root = tmp_path / "data"
    try:
        cached = tmp_path / "cached.png"
        cached.write_bytes(PNG)
        llm = FakeLLM(["嗯，看到了", "好饱呀，那歇会儿"])
        napcat = CountingNapCat({"token-0": {"file": str(cached)}})
        app = application(
            database, llm, napcat, settings=vision(), media_root=media_root, carry_turns=3
        )
        await app.handle_onebot(raw(1, images=1), received_at_utc=NOW)
        await app.handle_onebot(
            raw(2, text="好饱呀", images=0, at=NOW + timedelta(seconds=40)),
            received_at_utc=NOW + timedelta(seconds=40),
        )

        second = prompt_of(llm.calls[-1])
        assert len(carriers(llm.calls[-1])) == 1, "宽限窗口要把那张图挂上来"
        assert "存档重挂、不是他新发的图" in second
        assert "handle=M0" in second, "来源要说清是哪条消息（第一条入站的句柄）"
        # 「上一条是图片消息、本轮你看不到那张图」在重挂轮是假的，两句事实不许打架。
        assert "上一条输入是图片消息" not in second
        assert trace_details(database, "context")[-1]["images_carried"] == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_beyond_the_window_no_carry_line_and_no_picture(tmp_path):
    database = Database(tmp_path / "carry-off.sqlite3")
    media_root = tmp_path / "data"
    try:
        cached = tmp_path / "cached.png"
        cached.write_bytes(PNG)
        llm = FakeLLM(["嗯，看到了", "好", "还在忙"])
        napcat = CountingNapCat({"token-0": {"file": str(cached)}})
        app = application(
            database, llm, napcat, settings=vision(), media_root=media_root, carry_turns=1
        )
        await app.handle_onebot(raw(1, images=1), received_at_utc=NOW)
        await app.handle_onebot(
            raw(2, text="好饱呀", images=0, at=NOW + timedelta(seconds=40)),
            received_at_utc=NOW + timedelta(seconds=40),
        )
        await app.handle_onebot(
            raw(3, text="在吗", images=0, at=NOW + timedelta(seconds=80)),
            received_at_utc=NOW + timedelta(seconds=80),
        )

        third = prompt_of(llm.calls[-1])
        assert carriers(llm.calls[-1]) == []
        assert "本轮附带的图来自更早那条消息" not in third
        assert trace_details(database, "context")[-1]["images_carried"] == 0
    finally:
        database.close()


@pytest.mark.asyncio
async def test_originals_are_dropped_when_the_configuration_says_so(tmp_path):
    cached = tmp_path / "cached.png"
    cached.write_bytes(PNG)
    database = Database(tmp_path / "nokeep.sqlite3")
    napcat = FakeNapCat({"token-0": {"file": str(cached)}})
    llm = FakeLLM(["看到了"])
    client = application(
        database, llm, napcat, settings=vision(keep_original_images=False), media_root=tmp_path / "data"
    )
    try:
        await client.handle_onebot(raw(101, text="看看这个"))
    finally:
        database.close()

    assert len(carriers(llm.calls[0])) == 1, "本轮仍然要看得到图"
    from pathlib import Path

    assert not Path(carriers(llm.calls[0])[0].images[0].path).exists(), "回合结束后不保留原图"
