"""外部检索：客户端只降级不抛、结果裁剪到上限、关掉时什么都不建。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from qichi.config import ConfigError, load_config
from qichi.net import SearchOutcome, TavilySearchClient, WebSearchError, render_external_block
from qichi.runtime import RuntimeAssemblyError, build_search_client


NOW = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def tavily_stub(unused_tcp_port):
    """按请求返回不同结果的小服务器；calls 记录收到的请求体与授权头。"""

    calls: list[dict] = []
    responses: dict[str, tuple[int, object]] = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        calls.append({"body": body, "auth": request.headers.get("Authorization")})
        status, payload = responses.get(body["query"], (200, {"results": []}))
        if isinstance(payload, str):
            return web.Response(status=status, text=payload, content_type="application/json")
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}", calls, responses
    finally:
        await runner.cleanup()


def _client(base: str, **kwargs) -> TavilySearchClient:
    settings = {"timeout_seconds": 5.0, "max_results": 3, "max_chars": 1200}
    settings.update(kwargs)
    return TavilySearchClient("tvly-test", base_url=base, **settings)


@pytest.mark.asyncio
async def test_a_successful_search_keeps_sources_and_order(tavily_stub):
    base, calls, responses = tavily_stub
    responses["示例学院"] = (
        200,
        {
            "query": "示例学院",
            "results": [
                {"title": "示例学院", "url": "https://example.com/a", "content": "  农业  工程  "},
                {"title": "百科", "url": "https://example.com/b", "content": "历史沿革"},
                {"title": "没有网址", "content": "应被跳过"},
            ],
        },
    )
    client = _client(base)
    try:
        outcome = await client.search("  示例学院  ")
    finally:
        await client.close()

    assert outcome.ok is True
    assert outcome.query == "示例学院"
    assert [item.url for item in outcome.results] == ["https://example.com/a", "https://example.com/b"]
    assert outcome.results[0].content == "农业 工程"
    assert calls[0]["auth"] == "Bearer tvly-test"
    assert calls[0]["body"]["max_results"] == 3
    assert calls[0]["body"]["search_depth"] == "basic"


@pytest.mark.asyncio
async def test_content_is_cut_to_the_total_budget(tavily_stub):
    base, _calls, responses = tavily_stub
    long_text = "字" * 500
    responses["长文"] = (
        200,
        {
            "results": [
                {"title": "一", "url": "https://example.com/1", "content": long_text},
                {"title": "二", "url": "https://example.com/2", "content": long_text},
            ]
        },
    )
    client = _client(base, max_chars=600, max_results=3)
    try:
        outcome = await client.search("长文")
    finally:
        await client.close()

    total = sum(len(item.content) for item in outcome.results)
    assert outcome.ok is True
    assert total <= 600
    assert len(outcome.results) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason",
    [(429, "rate_limited"), (401, "unauthorized"), (500, "upstream_error"), (400, "request_rejected")],
)
async def test_http_failures_degrade_instead_of_raising(tavily_stub, status: int, reason: str):
    base, _calls, responses = tavily_stub
    responses["坏"] = (status, {"detail": "nope"})
    client = _client(base)
    try:
        outcome = await client.search("坏")
    finally:
        await client.close()

    assert outcome.ok is False
    assert outcome.degraded_reason == reason
    assert outcome.results == ()


@pytest.mark.asyncio
async def test_empty_and_broken_bodies_degrade(tavily_stub):
    base, _calls, responses = tavily_stub
    responses["空"] = (200, {"results": []})
    responses["坏体"] = (200, "not-json-at-all")
    client = _client(base)
    try:
        empty = await client.search("空")
        broken = await client.search("坏体")
    finally:
        await client.close()

    assert empty.degraded_reason == "no_results"
    assert broken.degraded_reason == "protocol_error"


@pytest.mark.asyncio
async def test_a_slow_server_times_out_without_raising(unused_tcp_port):
    async def handler(_request: web.Request) -> web.Response:
        await asyncio.sleep(3)
        return web.json_response({"results": []})

    app = web.Application()
    app.router.add_post("/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        client = _client(f"http://127.0.0.1:{unused_tcp_port}", timeout_seconds=0.3)
        try:
            outcome = await client.search("慢")
        finally:
            await client.close()
    finally:
        await runner.cleanup()

    assert outcome.degraded_reason == "timeout"


def test_a_query_that_is_empty_or_too_long_is_a_programming_error():
    client = TavilySearchClient("tvly-test")
    with pytest.raises(WebSearchError):
        asyncio.run(client.search("   "))
    with pytest.raises(WebSearchError):
        asyncio.run(client.search("字" * 201))
    asyncio.run(client.close())


def test_the_external_block_states_provenance_and_failure_is_honest():
    ok = SearchOutcome(
        query="示例学院",
        results=(),
        degraded_reason=None,
        elapsed_ms=12,
    )
    block = render_external_block(ok, now=NOW)
    assert "不可信" in block or "不是本机事实" in block

    failed = SearchOutcome(query="示例学院", results=(), degraded_reason="timeout", elapsed_ms=900)
    blocked = render_external_block(failed, now=NOW)
    assert "timeout" in blocked
    assert "没查到" in blocked
    assert "2026-09-13T20:00Z" in blocked


def test_the_switch_is_the_whole_gate(config_path: Path, complete_environment):
    """开关是唯一的门：关着什么都不建，开着才建（2026-09-14 起示例配置是开着的）。"""

    from dataclasses import replace

    config = load_config(config_path, environ=complete_environment)
    assert isinstance(config.net.enabled, bool)
    assert config.net.api_key_env == "TAVILY_API_KEY"
    assert config.net.max_results == 3

    assert build_search_client(replace(config.net, enabled=False), complete_environment) is None
    built = build_search_client(
        replace(config.net, enabled=True), {**complete_environment, "TAVILY_API_KEY": "tvly-x"}
    )
    assert built is not None
    asyncio.run(built.close())


def test_enabled_without_a_key_refuses_to_start(config_path: Path, complete_environment):
    from dataclasses import replace

    config = load_config(config_path, environ=complete_environment)
    enabled = replace(config.net, enabled=True)

    with pytest.raises(RuntimeAssemblyError, match="TAVILY_API_KEY"):
        build_search_client(enabled, complete_environment)
    built = build_search_client(enabled, {**complete_environment, "TAVILY_API_KEY": "tvly-x"})
    assert built is not None
    asyncio.run(built.close())


def test_unknown_net_fields_fail_closed(config_path: Path, complete_environment, tmp_path: Path):
    source = config_path.read_text(encoding="utf-8").replace(
        "  search_depth: basic", "  search_depth: basic\n  nonsense: 1", 1
    )
    path = tmp_path / "net.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_config(path, environ=complete_environment)


def test_an_unverified_search_depth_is_rejected(config_path: Path, complete_environment, tmp_path: Path):
    source = config_path.read_text(encoding="utf-8").replace(
        "  search_depth: basic", "  search_depth: turbo", 1
    )
    path = tmp_path / "net-depth.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="search_depth must be basic, advanced or fast"):
        load_config(path, environ=complete_environment)
