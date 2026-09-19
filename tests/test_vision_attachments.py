from __future__ import annotations

import base64

import pytest
import pytest_asyncio
from aiohttp import web

from qichi.dialogue.llm_client import (
    DeepSeekLLMClient,
    LLMImageRejectedError,
    LLMImageUnavailableError,
    LLMRequestError,
    wire_messages,
)
from qichi.domain.dialogue import ModelImage, ModelMessage

PNG = b"\x89PNG\r\n\x1a\n" + b"picture-bytes"


def picture(path: str = "C:/data/media/2026-09/evt-0.png") -> ModelImage:
    return ModelImage(path=path, content_type="image/png")


def test_a_text_only_message_serializes_exactly_as_before():
    assert ModelMessage(role="user", content="hi").to_dict() == {"role": "user", "content": "hi"}


def test_pictures_round_trip_through_the_dictionary_form():
    message = ModelMessage(role="user", content="看这个", images=(picture(),))

    restored = ModelMessage.from_dict(message.to_dict())

    assert restored == message
    assert message.to_dict()["images"] == [
        {"path": "C:/data/media/2026-09/evt-0.png", "content_type": "image/png"}
    ]


def test_only_a_user_message_may_carry_pictures():
    for role in ("system", "assistant"):
        with pytest.raises(ValueError, match="user message"):
            ModelMessage(role=role, content="x", images=(picture(),))


def test_a_model_image_must_name_a_path_and_an_image_type():
    with pytest.raises(ValueError):
        ModelImage(path="", content_type="image/png")
    with pytest.raises(ValueError):
        ModelImage(path="a.png", content_type="application/pdf")
    with pytest.raises(TypeError):
        ModelMessage(role="user", content="x", images=[picture()])


def test_text_only_messages_keep_plain_string_content():
    payload = wire_messages(
        (ModelMessage("system", "prompt"), ModelMessage("user", "hi")),
        read_image=lambda path: PNG,
    )

    assert payload == [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "hi"},
    ]


def test_a_picture_becomes_a_text_part_plus_a_data_url():
    payload = wire_messages(
        (ModelMessage("user", "看这个", images=(picture(),)),),
        read_image=lambda path: PNG,
    )

    assert payload == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看这个"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")},
                },
            ],
        }
    ]


def test_an_image_only_message_carries_no_invented_text_part():
    payload = wire_messages(
        (ModelMessage("user", "", images=(picture(),)),),
        read_image=lambda path: PNG,
    )

    assert [part["type"] for part in payload[0]["content"]] == ["image_url"]


def test_two_pictures_become_two_parts_in_order():
    payload = wire_messages(
        (
            ModelMessage(
                "user",
                "对比一下",
                images=(picture("a.png"), ModelImage(path="b.jpg", content_type="image/jpeg")),
            ),
        ),
        read_image=lambda path: PNG if path.endswith(".png") else b"\xff\xd8\xffjpeg",
    )

    urls = [part["image_url"]["url"] for part in payload[0]["content"] if part["type"] == "image_url"]
    assert urls == [
        "data:image/png;base64," + base64.b64encode(PNG).decode("ascii"),
        "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xffjpeg").decode("ascii"),
    ]


def test_the_default_reader_reads_the_real_file(tmp_path):
    landed = tmp_path / "evt-0.png"
    landed.write_bytes(PNG)

    payload = wire_messages((ModelMessage("user", "", images=(picture(str(landed)),)),))

    url = payload[0]["content"][0]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == PNG


def test_an_unreadable_or_empty_picture_is_a_typed_error(tmp_path):
    with pytest.raises(LLMImageUnavailableError):
        wire_messages((ModelMessage("user", "", images=(picture(str(tmp_path / "gone.png")),)),))

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(LLMImageUnavailableError):
        wire_messages((ModelMessage("user", "", images=(picture(str(empty)),)),))


@pytest_asyncio.fixture
async def image_server(unused_tcp_port):
    state = {"reject": False}
    calls: list[dict] = []
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        calls.append(await request.json())
        if state["reject"]:
            return web.json_response({"error": "bad request"}, status=400)
        return web.json_response(
            {
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        )

    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", calls, state
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_the_client_sends_parts_for_a_picture_and_a_plain_string_without_one(image_server, tmp_path):
    base, calls, _state = image_server
    landed = tmp_path / "evt-0.png"
    landed.write_bytes(PNG)
    client = DeepSeekLLMClient(base, "KEY")
    try:
        await client.generate((ModelMessage("user", "看这个", images=(picture(str(landed)),)),))
        await client.generate((ModelMessage("user", "just text"),))
    finally:
        await client.close()

    with_image = calls[0]["messages"][0]
    assert isinstance(with_image["content"], list)
    assert with_image["content"][0] == {"type": "text", "text": "看这个"}
    assert with_image["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert calls[1]["messages"][0] == {"role": "user", "content": "just text"}


@pytest.mark.asyncio
async def test_a_rejected_image_request_is_reported_as_such(image_server, tmp_path):
    base, _calls, state = image_server
    landed = tmp_path / "evt-0.png"
    landed.write_bytes(PNG)
    state["reject"] = True
    client = DeepSeekLLMClient(base, "KEY")
    try:
        with pytest.raises(LLMImageRejectedError):
            await client.generate((ModelMessage("user", "看图", images=(picture(str(landed)),)),))
        with pytest.raises(LLMRequestError) as error:
            await client.generate((ModelMessage("user", "没有图"),))
        assert not isinstance(error.value, LLMImageRejectedError)
    finally:
        await client.close()
