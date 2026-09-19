from __future__ import annotations

import asyncio
import json
import math
import traceback
from dataclasses import FrozenInstanceError

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from qichi.dialogue.llm_client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
    DeepSeekLLMClient,
    SiliconFlowLLMClient,
)
from qichi.domain.dialogue import ModelMessage


@pytest_asyncio.fixture
async def fake_server(unused_tcp_port):
    calls: list[dict] = []
    response_body = {"model": "deepseek-ai/DeepSeek-V4-Flash", "choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        calls.append({"headers": dict(request.headers), "body": await request.json()})
        return web.json_response(response_body)

    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", calls, response_body
    finally:
        await runner.cleanup()


def messages() -> tuple[ModelMessage, ...]:
    return (ModelMessage("system", "PRIVATE_PROMPT"), ModelMessage("user", "hi"))


@pytest.mark.asyncio
async def test_deepseek_client_uses_official_model_id(fake_server):
    base, calls, response = fake_server
    response["model"] = "deepseek-v4-flash"
    client = DeepSeekLLMClient(base, "PRIVATE_KEY")
    try:
        result = await client.generate((ModelMessage("user", "hi"),))
        assert result.model_id == "deepseek-v4-flash"
        assert calls[0]["body"]["model"] == "deepseek-v4-flash"
    finally:
        await client.close()


def exception_graph_text(error: BaseException) -> str:
    pending = [error]
    seen: set[int] = set()
    parts: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        parts.extend(
            (str(current), repr(current), "".join(traceback.format_exception(current)))
        )
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_success_shape_session_reuse_and_one_request(fake_server):
    base, calls, _ = fake_server
    client = SiliconFlowLLMClient(base, "PRIVATE_KEY")
    client_with_sensitive_url = SiliconFlowLLMClient(
        f"{base}/PRIVATE_URL", "PRIVATE_KEY"
    )
    try:
        result = await client.generate(messages())
        first_session = client._session
        await client.generate((ModelMessage("user", "again"),))
        assert client._session is first_session
        assert result.text == "hello"
        assert result.model_route == "primary"
        assert result.model_id == "deepseek-ai/DeepSeek-V4-Flash"
        assert result.input_tokens == 3 and result.output_tokens == 2
        assert result.finish_reason is None and result.reasoning_tokens is None
        assert result.latency_ms >= 0
        assert len(calls) == 2
        assert calls[0]["headers"]["Authorization"] == "Bearer PRIVATE_KEY"
        assert calls[0]["body"] == {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [m.to_dict() for m in messages()],
            "temperature": 0.85,
            "top_p": 0.92,
            "max_tokens": 1200,
        }
        assert client.timeout_seconds == 25.0
        assert "PRIVATE_KEY" not in repr(client)
        assert "PRIVATE_URL" not in repr(client_with_sensitive_url)
        assert repr(result).find("hello") < 0
        with pytest.raises(FrozenInstanceError):
            result.input_tokens = 4
    finally:
        await client.close()
        await client_with_sensitive_url.close()


@pytest.mark.asyncio
async def test_client_session_honors_process_proxy_environment(fake_server):
    base, _, _ = fake_server
    client = SiliconFlowLLMClient(base, "PRIVATE_KEY")
    try:
        await client.generate((ModelMessage("user", "proxy route"),))
        assert client._session is not None
        assert client._session._trust_env is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generation_preserves_finish_reason_and_reasoning_usage(fake_server):
    base, _, response_body = fake_server
    response_body["model"] = "deepseek-v4-flash"
    response_body["choices"][0]["finish_reason"] = "stop"
    response_body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 7}
    client = DeepSeekLLMClient(base, "k")
    try:
        result = await client.generate((ModelMessage("user", "x"),))
        assert result.finish_reason == "stop"
        assert result.reasoning_tokens == 7
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generation_reports_the_prefix_cache_hit_from_either_field_name(fake_server):
    base, _, response_body = fake_server
    response_body["model"] = "deepseek-v4-flash"
    response_body["usage"] = {"prompt_tokens": 64, "completion_tokens": 2, "prompt_cache_hit_tokens": 48}
    client = DeepSeekLLMClient(base, "k")
    try:
        assert (await client.generate((ModelMessage("user", "x"),))).cache_hit_tokens == 48
    finally:
        await client.close()

    response_body["usage"] = {"prompt_tokens": 64, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 32}}
    client = DeepSeekLLMClient(base, "k")
    try:
        assert (await client.generate((ModelMessage("user", "x"),))).cache_hit_tokens == 32
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_absent_cache_field_is_none_and_a_broken_one_fails_closed(fake_server):
    base, _, response_body = fake_server
    response_body["model"] = "deepseek-v4-flash"
    response_body["usage"] = {"prompt_tokens": 64, "completion_tokens": 2}
    client = DeepSeekLLMClient(base, "k")
    try:
        assert (await client.generate((ModelMessage("user", "x"),))).cache_hit_tokens is None
    finally:
        await client.close()

    for broken in ("64", -1, 65):
        response_body["usage"] = {
            "prompt_tokens": 64, "completion_tokens": 2, "prompt_cache_hit_tokens": broken,
        }
        client = DeepSeekLLMClient(base, "k")
        try:
            with pytest.raises(LLMProtocolError):
                await client.generate((ModelMessage("user", "x"),))
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_generation_can_raise_only_the_sidecar_output_budget(fake_server):
    base, calls, response_body = fake_server
    response_body["model"] = "deepseek-ai/DeepSeek-V4-Flash"
    client = SiliconFlowLLMClient(base, "k")
    try:
        await client.generate(
            (ModelMessage("user", "x"),),
            max_output_tokens=2400,
            response_format={"type": "json_object"},
        )
        assert calls[-1]["body"]["max_tokens"] == 2400
        assert calls[-1]["body"]["response_format"] == {"type": "json_object"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generation_can_disable_thinking_without_changing_normal_default(fake_server):
    base, calls, response_body = fake_server
    response_body["model"] = "deepseek-ai/DeepSeek-V4-Flash"
    client = SiliconFlowLLMClient(base, "k")
    try:
        await client.generate(
            (ModelMessage("user", "x"),),
            thinking={"type": "disabled"},
        )
        assert calls[-1]["body"]["thinking"] == {"type": "disabled"}
        assert "thinking" not in (await _request_body_without_thinking(client, calls))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generation_can_override_sampling_without_changing_defaults(fake_server):
    base, calls, response_body = fake_server
    response_body["model"] = "deepseek-ai/DeepSeek-V4-Flash"
    client = SiliconFlowLLMClient(base, "k", temperature=0.5, top_p=0.92)
    try:
        await client.generate(
            (ModelMessage("user", "structured"),), temperature=0.0, top_p=1.0
        )
        assert calls[-1]["body"]["temperature"] == 0.0
        assert calls[-1]["body"]["top_p"] == 1.0

        await client.generate((ModelMessage("user", "dialogue"),))
        assert calls[-1]["body"]["temperature"] == 0.5
        assert calls[-1]["body"]["top_p"] == 0.92
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"temperature": True}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
async def test_sampling_overrides_are_strictly_validated(fake_server, kwargs, field):
    base, calls, _ = fake_server
    client = SiliconFlowLLMClient(base, "k")
    try:
        with pytest.raises(ValueError, match=field):
            await client.generate((ModelMessage("user", "x"),), **kwargs)
        assert calls == []
    finally:
        await client.close()


async def _request_body_without_thinking(client, calls):
    await client.generate((ModelMessage("user", "default"),))
    return calls[-1]["body"]


@pytest.mark.asyncio
async def test_thinking_parameter_is_strictly_validated(fake_server):
    base, _, _ = fake_server
    client = SiliconFlowLLMClient(base, "k")
    try:
        with pytest.raises(ValueError):
            await client.generate((ModelMessage("user", "x"),), thinking={"type": "low"})
        with pytest.raises(ValueError):
            await client.generate((ModelMessage("user", "x"),), thinking={"type": "disabled", "extra": "x"})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_exception_graph_never_contains_sensitive_values(fake_server):
    base, _, response_body = fake_server
    sensitive_prompt = "PRIVATE_PROMPT"
    response_body.clear(); response_body.update({"choices": [{"message": {"content": "PRIVATE_COMPLETION"}}], "usage": {"prompt_tokens": True, "completion_tokens": 1}})
    client = SiliconFlowLLMClient(base, "PRIVATE_KEY")
    try:
        with pytest.raises(LLMProtocolError) as caught:
            await client.generate((ModelMessage("user", sensitive_prompt),))
        rendered = exception_graph_text(caught.value)
        for secret in ("PRIVATE_KEY", "PRIVATE_PROMPT", "PRIVATE_PROVIDER_BODY", "PRIVATE_COMPLETION"):
            assert secret not in rendered
    finally: await client.close()


@pytest.mark.asyncio
async def test_empty_content_is_success(fake_server):
    base, _, response_body = fake_server
    response_body.clear()
    response_body.update({"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    client = SiliconFlowLLMClient(base, "k")
    try:
        assert (await client.generate((ModelMessage("user", "x"),))).text == ""
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_null_content_is_an_empty_success(fake_server):
    # The production shape: a well-formed body whose message carries no visible
    # content at all.  It must reach the engine as an empty completion so the
    # ordinary structural retry applies, instead of dying instantly as a
    # protocol error the retry handlers never see.
    base, _, response_body = fake_server
    response_body.clear()
    response_body.update({"choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 12, "completion_tokens": 0}})
    client = SiliconFlowLLMClient(base, "k")
    try:
        generation = await client.generate((ModelMessage("user", "x"),))
        assert generation.text == ""
        assert generation.finish_reason == "stop"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_non_null_non_string_content_is_still_a_protocol_error(fake_server):
    base, _, response_body = fake_server
    client = SiliconFlowLLMClient(base, "k")
    try:
        for broken in (123, ["x"], {"text": "x"}):
            response_body.clear()
            response_body.update({"choices": [{"message": {"content": broken}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            with pytest.raises(LLMProtocolError):
                await client.generate((ModelMessage("user", "x"),))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_validation_and_closed(fake_server):
    base, _, _ = fake_server
    with pytest.raises(ValueError):
        SiliconFlowLLMClient(base, "k", model="other")
    with pytest.raises(ValueError):
        SiliconFlowLLMClient(base, "k", temperature=math.nan)
    with pytest.raises(ValueError):
        SiliconFlowLLMClient(base, "k", top_p=0)
    with pytest.raises(ValueError):
        SiliconFlowLLMClient(base, "k", timeout_seconds=True)
    client = SiliconFlowLLMClient(base, "k")
    await client.close()
    with pytest.raises(LLMConnectionError):
        await client.generate((ModelMessage("user", "x"),))


@pytest.mark.asyncio
async def test_http_error_mapping(unused_tcp_port):
    sensitive_prompt = "PRIVATE_PROMPT"
    async def handler(request):
        return web.Response(status=int(request.match_info["status"]), text="PRIVATE_PROVIDER_BODY")
    app = web.Application(); app.router.add_post("/{status}/chat/completions", handler)
    runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port); await site.start()
    try:
        for status, error in [(429, LLMRateLimitError), (401, LLMAuthenticationError), (403, LLMAuthenticationError), (404, LLMModelNotFoundError), (400, LLMRequestError), (500, LLMServerError)]:
            client = SiliconFlowLLMClient(f"http://127.0.0.1:{unused_tcp_port}/{status}", "PRIVATE_KEY")
            try:
                with pytest.raises(error) as caught:
                    await client.generate((ModelMessage("user", sensitive_prompt),))
                assert "PRIVATE" not in exception_graph_text(caught.value)
            finally:
                await client.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_protocol_and_redirect(unused_tcp_port):
    async def bad(request):
        return web.Response(text="not json", content_type="text/plain")
    app = web.Application(); app.router.add_post("/chat/completions", bad)
    runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port); await site.start()
    try:
        client = SiliconFlowLLMClient(f"http://127.0.0.1:{unused_tcp_port}", "k")
        try:
            with pytest.raises(LLMProtocolError):
                await client.generate((ModelMessage("user", "x"),))
        finally: await client.close()
    finally: await runner.cleanup()


@pytest.mark.asyncio
async def test_cancellation_propagates(unused_tcp_port):
    async def slow(request):
        await asyncio.sleep(10)
        return web.json_response({})
    app = web.Application(); app.router.add_post("/chat/completions", slow)
    runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port); await site.start()
    client = SiliconFlowLLMClient(
        f"http://127.0.0.1:{unused_tcp_port}", "k", timeout_seconds=5
    )
    task = asyncio.create_task(client.generate((ModelMessage("user", "x"),)))
    await asyncio.sleep(0.05); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    await client.close(); await runner.cleanup()


@pytest.mark.asyncio
async def test_timeout_connection_and_redirect_are_single_attempts(unused_tcp_port):
    calls = {"slow": 0, "redirect": 0}
    async def slow(request):
        calls["slow"] += 1; await asyncio.sleep(1); return web.json_response({})
    async def redirect(request):
        calls["redirect"] += 1; raise web.HTTPFound("/redirect-target")
    async def target(request):
        calls["redirect"] += 100; return web.json_response({})
    app = web.Application(); app.router.add_post("/slow/chat/completions", slow); app.router.add_post("/redir/chat/completions", redirect); app.router.add_post("/redirect-target", target)
    runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port); await site.start()
    try:
        client = SiliconFlowLLMClient(
            f"http://127.0.0.1:{unused_tcp_port}/slow",
            "k",
            timeout_seconds=0.02,
        )
        with pytest.raises(LLMTimeoutError): await client.generate((ModelMessage("user", "x"),))
        assert calls["slow"] == 1; await client.close()
        client = SiliconFlowLLMClient(f"http://127.0.0.1:{unused_tcp_port}/redir", "k")
        with pytest.raises(LLMRequestError): await client.generate((ModelMessage("user", "x"),))
        assert calls["redirect"] == 1; await client.close()
    finally: await runner.cleanup()


@pytest.mark.asyncio
async def test_connection_failure_has_no_retry(monkeypatch):
    client = SiliconFlowLLMClient(
        "http://localhost", "PRIVATE_KEY", timeout_seconds=0.5
    )
    attempts = 0
    class FailedSession:
        def post(self, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise aiohttp.ClientConnectionError()
    async def get_session(): return FailedSession()
    monkeypatch.setattr(client, "_get_session", get_session)
    try:
        with pytest.raises(LLMConnectionError): await client.generate((ModelMessage("user", "x"),))
        assert attempts == 1
    finally: await client.close()


@pytest.mark.asyncio
async def test_protocol_matrix(unused_tcp_port):
    variants = [
        [], {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 1}}]}, {"choices": [{"message": {"content": "x"}}]},
        {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": True, "completion_tokens": 1}},
        {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": -1, "completion_tokens": 1}},
        # A renamed or unexpected model echo is deliberately NOT in this list:
        # the provider renames identifiers on its own schedule (deepseek-v4-flash
        # became deepseek-flash on 2026-09-11) and rejecting the echo took the
        # whole production path down.  The request id is pinned by the config and
        # its capability evidence; the echo is only type-checked.  The dedicated
        # tests below cover the accepted echo, the missing echo and the malformed
        # one.
    ]
    async def handler(request):
        value = variants[int(request.match_info["n"])]
        return web.json_response(value) if value else web.Response(text="[]", content_type="application/json")
    app = web.Application(); app.router.add_post("/{n}/chat/completions", handler)
    runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port); await site.start()
    try:
        for n in range(len(variants)):
            client = SiliconFlowLLMClient(f"http://127.0.0.1:{unused_tcp_port}/{n}", "k")
            try:
                with pytest.raises(LLMProtocolError): await client.generate((ModelMessage("user", "x"),))
            finally: await client.close()
    finally: await runner.cleanup()


def test_strict_input_and_numeric_validation():
    common = {"base_url": "http://localhost", "api_key": "k"}
    for value in [True, math.nan, math.inf, -1]:
        with pytest.raises(ValueError):
            SiliconFlowLLMClient(**common, timeout_seconds=value)
    with pytest.raises(ValueError): SiliconFlowLLMClient(**common, max_output_tokens=True)
    client = SiliconFlowLLMClient(**common)
    async def check():
        for value in [(), [], "x", ("x",), (ModelMessage("user", "x"), "bad")]:
            with pytest.raises((TypeError, ValueError)): await client.generate(value)
        await client.close()
        with pytest.raises(LLMConnectionError): await client.generate((ModelMessage("user", "x"),))
    asyncio.run(check())

@pytest.mark.asyncio
async def test_a_renamed_model_echo_is_accepted(fake_server):
    # The provider renamed deepseek-v4-flash to deepseek-flash on 2026-09-11 and
    # echoed the new name; a strict equality check turned every single
    # generation into llm:protocol.  The request id is what identity means here.
    base, _, response_body = fake_server
    response_body.clear()
    response_body.update({"choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}],
                          "model": "deepseek-flash",
                          "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    client = SiliconFlowLLMClient(base, "k")
    try:
        assert (await client.generate((ModelMessage("user", "x"),))).text == "你好"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_missing_model_echo_is_accepted(fake_server):
    base, _, response_body = fake_server
    response_body.clear()
    response_body.update({"choices": [{"message": {"content": "嗯"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    client = SiliconFlowLLMClient(base, "k")
    try:
        assert (await client.generate((ModelMessage("user", "x"),))).text == "嗯"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_non_string_model_echo_still_fails_closed(fake_server):
    base, _, response_body = fake_server
    client = SiliconFlowLLMClient(base, "k")
    try:
        for broken in (123, ["deepseek-flash"], {"id": "deepseek-flash"}):
            response_body.clear()
            response_body.update({"choices": [{"message": {"content": "x"}}], "model": broken,
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            with pytest.raises(LLMProtocolError):
                await client.generate((ModelMessage("user", "x"),))
    finally:
        await client.close()
