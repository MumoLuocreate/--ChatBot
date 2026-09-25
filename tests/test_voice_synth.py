"""语音合成客户端：请求形状、错误分类、绝不泄露密钥。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web

from qichi.voice.synth import (
    SpeechClient,
    SpeechConnectionError,
    SpeechHTTPError,
    SpeechProtocolError,
    SpeechTimeoutError,
)

GENERATION = "/services/aigc/multimodal-generation/generation"


@asynccontextmanager
async def fake_speech(generation, audio=None):
    app = web.Application()
    app.router.add_post(GENERATION, generation)
    if audio is not None:
        app.router.add_get("/audio.wav", audio)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def make_client(base_url: str, **overrides: Any) -> SpeechClient:
    settings = dict(base_url=base_url, api_key="secret-key", model="qwen3-tts-vd-2026-01-26",
                    voice="qwen-tts-vd-qichi_cast2-voice-x", timeout_seconds=2)
    settings.update(overrides)
    return SpeechClient(**settings)  # type: ignore[arg-type]


def test_sends_the_measured_payload_and_returns_audio_bytes():
    async def scenario():
        seen: list[tuple[str | None, dict[str, Any]]] = []

        async def generation(request: web.Request):
            seen.append((request.headers.get("Authorization"), await request.json()))
            return web.json_response({"output": {"audio": {"url": f"http://{request.host}/audio.wav"}}})

        async def audio(_: web.Request):
            return web.Response(body=b"RIFF-audio", content_type="audio/wav")

        async with fake_speech(generation, audio) as base_url:
            client = make_client(base_url)
            try:
                data = await client.synthesize("嗯，在呢。", instructions="慵懒松弛")
            finally:
                await client.close()

        assert data == b"RIFF-audio"
        assert seen == [(
            "Bearer secret-key",
            {"model": "qwen3-tts-vd-2026-01-26",
             "input": {"text": "嗯，在呢。", "voice": "qwen-tts-vd-qichi_cast2-voice-x"},
             "parameters": {"language_type": "Chinese", "instructions": "慵懒松弛"}},
        )]

    asyncio.run(scenario())


def test_instructions_are_optional():
    async def scenario():
        bodies: list[dict[str, Any]] = []

        async def generation(request: web.Request):
            bodies.append(await request.json())
            return web.json_response({"output": {"audio": {"url": f"http://{request.host}/audio.wav"}}})

        async def audio(_: web.Request):
            return web.Response(body=b"a")

        async with fake_speech(generation, audio) as base_url:
            client = make_client(base_url)
            try:
                await client.synthesize("嗯。")
            finally:
                await client.close()

        assert bodies[0]["parameters"] == {"language_type": "Chinese"}

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_http_failures_are_classified(status):
    async def scenario():
        async def generation(_: web.Request):
            return web.Response(status=status, text="nope")

        async with fake_speech(generation) as base_url:
            client = make_client(base_url)
            try:
                with pytest.raises(SpeechHTTPError):
                    await client.synthesize("嗯。")
            finally:
                await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"output": {}},
        {"output": {"audio": {}}},
        {"output": {"audio": {"url": ""}}},
        {"output": {"audio": {"url": "not-absolute"}}},
        {"output": {"audio": {"url": "ftp://example.invalid/x.wav"}}},
    ],
)
def test_missing_or_relative_audio_url_is_a_protocol_error(envelope):
    async def scenario():
        async def generation(_: web.Request):
            return web.json_response(envelope)

        async with fake_speech(generation) as base_url:
            client = make_client(base_url)
            try:
                with pytest.raises(SpeechProtocolError):
                    await client.synthesize("嗯。")
            finally:
                await client.close()

    asyncio.run(scenario())


def test_non_json_response_is_a_protocol_error():
    async def scenario():
        async def generation(_: web.Request):
            return web.Response(body=b"<html>nope</html>", content_type="text/html")

        async with fake_speech(generation) as base_url:
            client = make_client(base_url)
            try:
                with pytest.raises(SpeechProtocolError):
                    await client.synthesize("嗯。")
            finally:
                await client.close()

    asyncio.run(scenario())


def test_empty_audio_is_a_protocol_error():
    async def scenario():
        async def generation(request: web.Request):
            return web.json_response({"output": {"audio": {"url": f"http://{request.host}/audio.wav"}}})

        async def audio(_: web.Request):
            return web.Response(body=b"")

        async with fake_speech(generation, audio) as base_url:
            client = make_client(base_url)
            try:
                with pytest.raises(SpeechProtocolError):
                    await client.synthesize("嗯。")
            finally:
                await client.close()

    asyncio.run(scenario())


def test_timeout_is_classified():
    async def scenario():
        async def generation(_: web.Request):
            await asyncio.sleep(0.5)
            return web.json_response({})

        async with fake_speech(generation) as base_url:
            client = make_client(base_url, timeout_seconds=0.05)
            try:
                with pytest.raises(SpeechTimeoutError):
                    await client.synthesize("嗯。")
            finally:
                await client.close()

    asyncio.run(scenario())


def test_repr_never_contains_the_api_key():
    client = SpeechClient(base_url="https://example.invalid/api/v1", api_key="sk-super-secret",
                          model="m", voice="v")

    assert "sk-super-secret" not in repr(client)
    assert "secret" not in repr(client)


def test_constructor_validates_inputs():
    with pytest.raises(ValueError):
        SpeechClient(base_url="", api_key="k", model="m", voice="v")
    with pytest.raises(ValueError):
        SpeechClient(base_url="https://x/api/v1", api_key="", model="m", voice="v")
    with pytest.raises(ValueError):
        SpeechClient(base_url="https://x/api/v1", api_key="k", model="m", voice="v", timeout_seconds=0)
    with pytest.raises(ValueError):
        SpeechClient(base_url="https://x/api/v1", api_key="k", model="m", voice="v", sample_rate=0)


def test_synthesize_rejects_blank_text():
    async def scenario():
        client = SpeechClient(base_url="https://example.invalid/api/v1", api_key="k", model="m", voice="v")
        with pytest.raises(ValueError):
            await client.synthesize("   ")
        with pytest.raises(ValueError):
            await client.synthesize("好", instructions="")

    asyncio.run(scenario())
