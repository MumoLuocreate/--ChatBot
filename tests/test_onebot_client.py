from __future__ import annotations

import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import WSMsgType, web

from qichi.transport.onebot_client import (
    OneBotActionError,
    OneBotClient,
    OneBotConnectionError,
    OneBotHTTPClientError,
    OneBotHTTPRedirectError,
    OneBotHTTPServerError,
    OneBotProtocolError,
    OneBotTimeoutError,
    OneBotWebSocketConsumerError,
    OneBotWebSocketDisconnectedError,
    OneBotWebSocketHandshakeError,
    OneBotWebSocketProtocolError,
)


@asynccontextmanager
async def fake_onebot(action_handler, ws_handler):
    app = web.Application()
    app.router.add_post("/{action}", action_handler)
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}/ws"
    finally:
        await runner.cleanup()


async def close_ws(_: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse()
    await websocket.prepare(_)
    await websocket.close()
    return websocket


def test_actions_send_onebot_payloads_with_bearer_token_and_ordered_segments():
    async def scenario():
        requests: list[tuple[str, str | None, dict[str, Any]]] = []

        async def actions(request: web.Request):
            requests.append((request.match_info["action"], request.headers.get("Authorization"), await request.json()))
            action = request.match_info["action"]
            data = {"message_id": 99, "user_id": 42, "message": []} if action == "get_msg" else {}
            if action == "send_private_msg":
                data = {"message_id": 101}
            return web.json_response({"status": "ok", "retcode": 0, "data": data})

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                assert await client.send_private_msg(42, [{"type": "text", "data": {"text": "one"}}, {"type": "face", "data": {"id": "14"}}]) == {"message_id": 101}
                assert await client.get_msg(99) == {"message_id": 99, "user_id": 42, "message": []}
                assert await client.set_msg_emoji_like(99, "128512", set=True) == {}
                assert await client.send_poke(42) == {}
                assert await client.get_login_info() == {}
        assert requests == [
            ("send_private_msg", "Bearer secret-token", {"user_id": 42, "message": [{"type": "text", "data": {"text": "one"}}, {"type": "face", "data": {"id": "14"}}]}),
            ("get_msg", "Bearer secret-token", {"message_id": 99}),
            ("set_msg_emoji_like", "Bearer secret-token", {"message_id": 99, "emoji_id": "128512", "set": True}),
            ("send_poke", "Bearer secret-token", {"user_id": 42}),
            ("get_login_info", "Bearer secret-token", {}),
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response, error_type",
    [
        (lambda: web.Response(status=400), OneBotHTTPClientError),
        (lambda: web.Response(status=503), OneBotHTTPServerError),
        (lambda: web.json_response({"status": "failed", "retcode": 1404, "data": None}), OneBotActionError),
        (lambda: web.Response(text="not-json"), OneBotProtocolError),
        (lambda: web.Response(body=b"\xff", content_type="application/json"), OneBotProtocolError),
    ],
)
def test_action_failure_categories(response, error_type):
    async def scenario():
        async def actions(_: web.Request):
            return response()

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(error_type) as error:
                    await client.get_msg(99)
                assert "secret-token" not in str(error.value)

    asyncio.run(scenario())


def test_action_timeout_and_connection_failures_do_not_retry():
    async def scenario():
        calls = 0
        arrived = asyncio.Event()
        release = asyncio.Event()

        async def actions(_: web.Request):
            nonlocal calls
            calls += 1
            arrived.set()
            await release.wait()
            return web.json_response({"status": "ok", "retcode": 0, "data": {}})

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=0.5) as client:
                # 先等请求真的到达服务端，再等客户端超时。原来 timeout=0.01 会与本地
                # 投递赛跑：3.11 的 CI runner 上出现过一次都没到（calls == 0）。
                pending = asyncio.create_task(client.send_poke(42))
                await asyncio.wait_for(arrived.wait(), timeout=5)
                with pytest.raises(OneBotTimeoutError):
                    await pending
            release.set()
        assert calls == 1

        async def disconnect(request: web.Request):
            assert request.transport is not None
            request.transport.close()
            return web.Response()

        async with fake_onebot(disconnect, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(OneBotConnectionError):
                    await client.get_msg(99)

    asyncio.run(scenario())


def test_action_redirect_is_not_followed_or_retried():
    async def scenario():
        calls: list[str] = []

        async def actions(request: web.Request):
            calls.append(request.match_info["action"])
            if request.match_info["action"] == "get_msg":
                raise web.HTTPTemporaryRedirect("/redirect-target")
            return web.json_response({"status": "ok", "retcode": 0, "data": {}})

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(OneBotHTTPRedirectError):
                    await client.get_msg(99)
        assert calls == ["get_msg"]

    asyncio.run(scenario())


def test_token_is_not_exposed_by_client_or_errors():
    async def scenario():
        async def actions(_: web.Request):
            return web.Response(status=500)

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            client = OneBotClient(http_url, ws_url, "secret-token", timeout=1)
            assert "secret-token" not in repr(client)
            async with client:
                with pytest.raises(OneBotHTTPServerError) as error:
                    await client.get_msg(99)
                assert "secret-token" not in str(error.value)

    asyncio.run(scenario())


def test_token_is_not_exposed_by_complete_exception_graphs():
    def exception_graph_text(error: BaseException) -> str:
        pending = [error]
        seen: set[int] = set()
        parts: list[str] = []
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            parts.extend((str(current), repr(current), "".join(traceback.format_exception(current))))
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return "\n".join(parts)

    async def scenario():
        async def http_failure(_: web.Request):
            return web.Response(status=500)

        async def rejected(_: web.Request):
            return web.Response(status=401)

        async with fake_onebot(http_failure, rejected) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(OneBotHTTPServerError) as http_error:
                    await client.get_msg(99)
                assert "secret-token" not in exception_graph_text(http_error.value)
                with pytest.raises(OneBotWebSocketHandshakeError) as ws_error:
                    await anext(client.event_stream())
                assert "secret-token" not in exception_graph_text(ws_error.value)

        async def disconnect(request: web.Request):
            assert request.transport is not None
            request.transport.close()
            return web.Response()

        async with fake_onebot(disconnect, rejected) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(OneBotConnectionError) as http_network_error:
                    await client.get_msg(99)
                assert "secret-token" not in exception_graph_text(http_network_error.value)

        async with fake_onebot(http_failure, disconnect) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(OneBotConnectionError) as ws_network_error:
                    await anext(client.event_stream())
                assert "secret-token" not in exception_graph_text(ws_network_error.value)

    asyncio.run(scenario())


def test_event_stream_yields_raw_mappings_and_normal_close():
    async def scenario():
        async def actions(_: web.Request):
            return web.json_response({"status": "ok", "retcode": 0, "data": {}})

        async def websocket(request: web.Request):
            assert request.headers["Authorization"] == "Bearer secret-token"
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json({"post_type": "message", "message_id": 7})
            await ws.close()
            return ws

        async with fake_onebot(actions, websocket) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                assert [frame async for frame in client.event_stream()] == [{"post_type": "message", "message_id": 7}]

    asyncio.run(scenario())


def test_only_standard_websocket_close_codes_are_clean():
    assert OneBotClient._is_clean_close(1000)
    assert OneBotClient._is_clean_close(1001)
    assert not OneBotClient._is_clean_close(None)
    assert not OneBotClient._is_clean_close(1011)


@pytest.mark.parametrize(
    "mode, error_type",
    [("binary", OneBotWebSocketProtocolError), ("invalid-json", OneBotWebSocketProtocolError), ("error", OneBotWebSocketDisconnectedError)],
)
def test_event_stream_protocol_and_disconnect_errors(mode, error_type):
    async def scenario():
        async def actions(_: web.Request):
            return web.json_response({"status": "ok", "retcode": 0, "data": {}})

        async def websocket(request: web.Request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            if mode == "binary":
                await ws.send_bytes(b"bad")
            elif mode == "invalid-json":
                await ws.send_str("[")
            else:
                await ws.close(code=1011, message=b"failure")
            return ws

        async with fake_onebot(actions, websocket) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(error_type):
                    async for _ in client.event_stream():
                        pass

    asyncio.run(scenario())


def test_event_stream_handshake_single_consumer_cancellation_and_close():
    async def scenario():
        gate = asyncio.Event()

        async def actions(_: web.Request):
            return web.json_response({"status": "ok", "retcode": 0, "data": {}})

        async def rejected(_: web.Request):
            return web.Response(status=401)

        async with fake_onebot(actions, rejected) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(OneBotWebSocketHandshakeError):
                    await anext(client.event_stream())

        async def websocket(request: web.Request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await gate.wait()
            return ws

        async with fake_onebot(actions, websocket) as (http_url, ws_url):
            client = OneBotClient(http_url, ws_url, "secret-token", timeout=1)
            first = client.event_stream()
            waiting = asyncio.create_task(anext(first))
            await asyncio.sleep(0)
            with pytest.raises(OneBotWebSocketConsumerError):
                await anext(client.event_stream())
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            await client.close()
            assert client.closed

    asyncio.run(scenario())


def test_get_image_returns_the_cached_path_and_the_url():
    async def scenario():
        requests: list[tuple[str, dict[str, Any]]] = []

        async def actions(request: web.Request):
            requests.append((request.match_info["action"], await request.json()))
            return web.json_response(
                {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"file": "C:/napcat/cache/a.image", "url": "https://cdn.test/a.png"},
                }
            )

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                data = await client.get_image("abc.image")

        assert data == {"file": "C:/napcat/cache/a.image", "url": "https://cdn.test/a.png"}
        assert requests == [("get_image", {"file": "abc.image"})]

    asyncio.run(scenario())


def test_get_image_refuses_an_empty_token_and_a_payload_that_is_not_a_mapping():
    async def scenario():
        async def actions(request: web.Request):
            return web.json_response({"status": "ok", "retcode": 0, "data": ["nope"]})

        async with fake_onebot(actions, close_ws) as (http_url, ws_url):
            async with OneBotClient(http_url, ws_url, "secret-token", timeout=1) as client:
                with pytest.raises(ValueError):
                    await client.get_image("")
                with pytest.raises(OneBotProtocolError):
                    await client.get_image("abc.image")

    asyncio.run(scenario())

