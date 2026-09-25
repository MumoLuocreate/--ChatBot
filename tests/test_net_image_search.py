"""以图搜图：上传换 image_id、再搜；失败一律降级；关掉时什么都不建。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from qichi.config import load_config
from qichi.net import (
    ImageMatch,
    ImageSearchOutcome,
    SerpApiLensClient,
    WebImageSearchError,
    render_image_block,
)
from qichi.runtime import RuntimeAssemblyError, build_image_search_client


NOW = datetime(2026, 9, 13, 22, 30, tzinfo=timezone.utc)


def picture(tmp_path: Path, name: str = "shot.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return path


@pytest_asyncio.fixture
async def lens_stub(unused_tcp_port):
    """假 SerpApi：/image 收 multipart 返回 image_id，/search 按 image_id 返回结果。"""

    uploads: list[dict] = []
    searches: list[dict] = []
    plan: dict[str, tuple[int, object]] = {}

    async def upload(request: web.Request) -> web.Response:
        reader = await request.multipart()
        fields: dict[str, object] = {}
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "image":
                fields["image_bytes"] = len(await part.read())
                fields["filename"] = part.filename
            else:
                fields[part.name] = await part.text()
        uploads.append(fields)
        return web.json_response({"message": "Image uploaded successfully.", "image_id": "img-1"})

    async def search(request: web.Request) -> web.Response:
        params = dict(request.query)
        searches.append(params)
        status, payload = plan.get(params.get("image_id", ""), (200, {"visual_matches": []}))
        if isinstance(payload, str):
            return web.Response(status=status, text=payload, content_type="application/json")
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_post("/image", upload)
    app.router.add_get("/search", search)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}", uploads, searches, plan
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_upload_then_search_keeps_sources_in_order(lens_stub, tmp_path: Path):
    base, uploads, searches, plan = lens_stub
    plan["img-1"] = (
        200,
        {
            "visual_matches": [
                {"title": "Python 官网", "link": "https://python.org", "source": "python.org"},
                {"title": "无链接的条目", "source": "x"},
                {"link": "https://example.com/b"},
            ]
        },
    )
    client = SerpApiLensClient("serpapi-test", base_url=base)
    try:
        outcome = await client.search_image(picture(tmp_path))
    finally:
        await client.close()

    assert outcome.ok is True
    assert outcome.image_id == "img-1"
    assert [item.link for item in outcome.matches] == ["https://python.org", "https://example.com/b"]
    assert outcome.matches[0] == ImageMatch("Python 官网", "https://python.org", "python.org")
    assert uploads[0]["filename"] == "shot.png" and uploads[0]["image_bytes"] == 72
    assert searches[0]["engine"] == "google_lens"
    assert searches[0]["type"] == "all"
    assert searches[0]["api_key"] == "serpapi-test"


@pytest.mark.asyncio
async def test_a_query_can_be_attached_to_the_picture(lens_stub, tmp_path: Path):
    base, _uploads, searches, plan = lens_stub
    plan["img-1"] = (200, {"visual_matches": [{"link": "https://example.com/a"}]})
    client = SerpApiLensClient("serpapi-test", base_url=base)
    try:
        await client.search_image(picture(tmp_path), search_type="products")
    finally:
        await client.close()

    assert searches[0]["type"] == "products"
    # 实测：给 Lens 传 q 会让它直接返回空，所以这个参数再也不发了。
    assert "q" not in searches[0]


@pytest.mark.asyncio
async def test_the_empty_answer_serpapi_gives_is_a_clean_no_results(lens_stub, tmp_path: Path):
    """实测：没匹配时 SerpApi 用 200 + error 字段回答。"""

    base, _uploads, _searches, plan = lens_stub
    plan["img-1"] = (200, {"error": "Google Lens hasn't returned any results for this query."})
    client = SerpApiLensClient("serpapi-test", base_url=base)
    try:
        outcome = await client.search_image(picture(tmp_path))
    finally:
        await client.close()

    assert outcome.ok is False
    assert outcome.degraded_reason == "no_results"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason",
    [(429, "rate_limited"), (401, "unauthorized"), (500, "upstream_error")],
)
async def test_search_failures_degrade(lens_stub, tmp_path: Path, status: int, reason: str):
    base, _uploads, _searches, plan = lens_stub
    plan["img-1"] = (status, {"error": "nope"})
    client = SerpApiLensClient("serpapi-test", base_url=base)
    try:
        outcome = await client.search_image(picture(tmp_path))
    finally:
        await client.close()

    assert outcome.ok is False
    assert outcome.degraded_reason == reason


@pytest.mark.asyncio
async def test_a_too_large_picture_is_refused_before_uploading(lens_stub, tmp_path: Path):
    base, uploads, _searches, _plan = lens_stub
    client = SerpApiLensClient("serpapi-test", base_url=base, max_upload_bytes=8)
    try:
        outcome = await client.search_image(picture(tmp_path))
    finally:
        await client.close()

    assert outcome.degraded_reason == "too_large"
    assert uploads == []


def test_bad_input_is_a_caller_error(tmp_path: Path):
    client = SerpApiLensClient("serpapi-test")
    with pytest.raises(WebImageSearchError):
        asyncio.run(client.search_image(tmp_path / "missing.png"))
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(WebImageSearchError):
        asyncio.run(client.search_image(unsupported))
    with pytest.raises(WebImageSearchError):
        asyncio.run(client.search_image(picture(tmp_path), search_type="guess"))
    asyncio.run(client.close())


def test_the_image_block_is_labelled_external_and_honest_on_failure():
    ok = ImageSearchOutcome(
        image_id="img-1",
        matches=(ImageMatch("Python 官网", "https://python.org", "python.org"),),
        degraded_reason=None,
        elapsed_ms=4000,
    )
    block = render_image_block(ok, now=NOW)
    assert "不可信" in block and "不能当记忆" in block
    assert "https://python.org" in block and "2026-09-13T22:30Z" in block

    failed = ImageSearchOutcome(image_id=None, matches=(), degraded_reason="too_large", elapsed_ms=5)
    blocked = render_image_block(failed, now=NOW)
    assert "too_large" in blocked and "没查到" in blocked


def test_the_switch_gates_the_image_client_too(config_path: Path, complete_environment):
    config = load_config(config_path, environ=complete_environment)

    assert config.net.image_api_key_env == "SERPAPI_API_KEY"
    assert config.net.image_max_matches == 5
    assert build_image_search_client(replace(config.net, enabled=False), complete_environment) is None


def test_enabled_without_the_image_key_refuses_to_start(config_path: Path, complete_environment):
    config = load_config(config_path, environ=complete_environment)
    enabled = replace(config.net, enabled=True, api_key_env="TAVILY_API_KEY")

    with pytest.raises(RuntimeAssemblyError, match="SERPAPI_API_KEY"):
        build_image_search_client(enabled, complete_environment)
    built = build_image_search_client(enabled, {**complete_environment, "SERPAPI_API_KEY": "k"})
    assert built is not None
    asyncio.run(built.close())
