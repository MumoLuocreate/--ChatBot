from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from aiohttp import web

from qichi.domain.events import MessageSegment
from qichi.media.fetch import (
    NETWORK,
    OVERFLOW,
    PROTOCOL,
    STORED,
    TIMEOUT,
    TOO_LARGE,
    UNAVAILABLE,
    UNSUPPORTED_TYPE,
    ImageFetchError,
    ImageFetchOutcome,
    download_image,
    fetch_inbound_images,
    image_segments,
    local_source,
    sniff_content_type,
)
from qichi.transport.onebot_client import (
    OneBotActionError,
    OneBotConnectionError,
    OneBotError,
    OneBotHTTPClientError,
    OneBotHTTPServerError,
    OneBotProtocolError,
    OneBotTimeoutError,
)

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
PNG = b"\x89PNG\r\n\x1a\n" + b"png-payload"
JPEG = b"\xff\xd8\xff" + b"jpeg-payload"
GIF = b"GIF89a" + b"gif-payload"
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"webp-payload"
TEXT = b"absolutely not a picture"


def image(**data) -> MessageSegment:
    return MessageSegment(type="image", data=data)


def text(value: str = "hi") -> MessageSegment:
    return MessageSegment(type="text", data={"text": value})


async def never(token):
    raise AssertionError("get_image must not be called for this case")


async def run_fetch(
    segments,
    data_root: Path,
    *,
    get_image=never,
    download=None,
    max_images: int = 2,
    max_bytes: int = 1024,
    timeout_seconds: float = 5.0,
    event_id: str = "evt-1",
):
    return await fetch_inbound_images(
        segments,
        event_id=event_id,
        data_root=data_root,
        get_image=get_image,
        max_images=max_images,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        download=download,
        now=NOW,
    )


def stored_files(data_root: Path) -> list[Path]:
    root = data_root / "media"
    return sorted(root.rglob("*")) if root.is_dir() else []


def test_two_images_land_under_the_event_name_and_the_third_only_overflows(tmp_path):
    seen: list[str] = []

    async def download(url: str) -> bytes:
        seen.append(url)
        return PNG

    segments = [image(url=f"https://cdn.test/{index}.png") for index in range(3)]

    result = asyncio.run(run_fetch(segments, tmp_path, download=download))

    assert [item.status for item in result.outcomes] == [STORED, STORED, OVERFLOW]
    assert result.stored == (
        tmp_path / "media" / "2026-09" / "evt-1-0.png",
        tmp_path / "media" / "2026-09" / "evt-1-1.png",
    )
    assert result.overflow == 1
    assert result.failures == ()
    assert seen == ["https://cdn.test/0.png", "https://cdn.test/1.png"]
    assert (tmp_path / "media" / "2026-09" / "evt-1-0.png").read_bytes() == PNG
    assert result.outcomes[0].content_type == "image/png"
    assert result.outcomes[0].size_bytes == len(PNG)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (OneBotTimeoutError("get_image timed out"), TIMEOUT),
        (OneBotActionError("retcode=1200"), UNAVAILABLE),
        (OneBotHTTPClientError("OneBot HTTP 404 for get_image"), UNAVAILABLE),
        (OneBotConnectionError("connection failed"), NETWORK),
        (OneBotHTTPServerError("OneBot HTTP 500 for get_image"), NETWORK),
        (OneBotProtocolError("bad envelope"), PROTOCOL),
        (OneBotError("something else"), PROTOCOL),
    ],
)
def test_every_platform_failure_gets_a_name_and_never_a_picture(tmp_path, error, status):
    async def get_image(token):
        raise error

    result = asyncio.run(run_fetch([image(file="token-1")], tmp_path, get_image=get_image))

    assert [item.status for item in result.outcomes] == [status]
    assert result.stored == ()
    assert result.outcomes[0].path is None
    assert stored_files(tmp_path) == [], "失败分支不得留下任何文件"


def test_no_failure_outcome_carries_a_path_or_a_type(tmp_path):
    async def get_image(token):
        raise OneBotActionError("gone")

    result = asyncio.run(run_fetch([image(file="t")], tmp_path, get_image=get_image))

    outcome = result.outcomes[0]
    assert outcome.path is None and outcome.content_type is None and outcome.size_bytes == 0
    assert result.failures == (outcome,)
    assert result.attempted == 1


def test_the_cached_file_wins_over_the_url(tmp_path):
    cached = tmp_path / "napcat-cache" / "shot.image"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PNG)

    async def get_image(token):
        assert token == "file-token"
        return {"file": str(cached), "url": "https://cdn.test/never-read.png"}

    async def download(url):
        raise AssertionError("the cached file must be used when it exists")

    result = asyncio.run(
        run_fetch([image(file="file-token", url="https://cdn.test/from-segment.png")], tmp_path,
                  get_image=get_image, download=download)
    )

    assert result.outcomes[0].status == STORED


def test_the_platform_url_is_used_when_the_cache_is_gone(tmp_path):
    async def get_image(token):
        return {"file": str(tmp_path / "missing.jpg"), "url": "https://cdn.test/real.jpg"}

    async def download(url):
        assert url == "https://cdn.test/real.jpg"
        return JPEG

    result = asyncio.run(run_fetch([image(file="t")], tmp_path, get_image=get_image, download=download))

    assert result.outcomes[0].status == STORED
    assert result.outcomes[0].path.name == "evt-1-0.jpg"
    assert result.outcomes[0].content_type == "image/jpeg"


def test_a_file_url_is_read_from_disk(tmp_path):
    cached = tmp_path / "cache.gif"
    cached.write_bytes(GIF)

    result = asyncio.run(run_fetch([image(url=cached.as_uri())], tmp_path))

    assert result.outcomes[0].status == STORED
    assert result.outcomes[0].content_type == "image/gif"


def test_a_plain_windows_path_is_read_from_disk(tmp_path):
    cached = tmp_path / "cache.webp"
    cached.write_bytes(WEBP)

    result = asyncio.run(run_fetch([image(file=str(cached))], tmp_path,
                                   get_image=lambda token: _payload({"file": str(cached)})))

    assert result.outcomes[0].status == STORED
    assert result.outcomes[0].content_type == "image/webp"


async def _payload(value):
    return value


def test_bytes_decide_the_type_not_the_platform(tmp_path):
    async def download(url):
        return TEXT

    result = asyncio.run(run_fetch([image(url="https://cdn.test/lies.png")], tmp_path, download=download))

    assert result.outcomes[0].status == UNSUPPORTED_TYPE
    assert stored_files(tmp_path) == []


def test_oversized_downloads_are_refused_without_writing(tmp_path):
    async def download(url):
        return PNG + b"x" * 100

    result = asyncio.run(run_fetch([image(url="https://cdn.test/big.png")], tmp_path,
                                   download=download, max_bytes=16))

    assert result.outcomes[0].status == TOO_LARGE
    assert stored_files(tmp_path) == []


def test_oversized_cached_files_are_refused_before_they_are_read(tmp_path):
    cached = tmp_path / "big.png"
    cached.write_bytes(PNG + b"x" * 100)

    result = asyncio.run(run_fetch([image(file=str(cached))], tmp_path,
                                   get_image=lambda token: _payload({"file": str(cached)}), max_bytes=16))

    assert result.outcomes[0].status == TOO_LARGE
    assert stored_files(tmp_path) == []


def test_an_unsupported_scheme_is_a_protocol_error(tmp_path):
    result = asyncio.run(run_fetch([image(url="ftp://cdn.test/a.png")], tmp_path))

    assert result.outcomes[0].status == PROTOCOL


def test_a_segment_without_a_source_is_unavailable(tmp_path):
    cases = [
        image(file="", url=""),
        image(),
    ]
    result = asyncio.run(run_fetch(cases, tmp_path))

    assert [item.status for item in result.outcomes] == [UNAVAILABLE, UNAVAILABLE]


def test_a_payload_without_a_source_is_unavailable(tmp_path):
    async def get_image(token):
        return {}

    result = asyncio.run(run_fetch([image(file="t")], tmp_path, get_image=get_image))

    assert result.outcomes[0].status == UNAVAILABLE


def test_a_non_mapping_payload_is_a_protocol_error(tmp_path):
    async def get_image(token):
        return ["not", "a", "mapping"]

    result = asyncio.run(run_fetch([image(file="t")], tmp_path, get_image=get_image))

    assert result.outcomes[0].status == PROTOCOL


def test_a_missing_cache_file_without_a_url_is_unavailable(tmp_path):
    async def get_image(token):
        return {"file": str(tmp_path / "gone.png")}

    result = asyncio.run(run_fetch([image(file="t")], tmp_path, get_image=get_image))

    assert result.outcomes[0].status == UNAVAILABLE


def test_non_image_segments_are_ignored_entirely(tmp_path):
    segments = [text(), image(url="https://cdn.test/a.png"), text("again")]
    assert len(image_segments(segments)) == 1

    async def download(url):
        return PNG

    result = asyncio.run(run_fetch(segments, tmp_path, download=download))

    assert [item.index for item in result.outcomes] == [0]
    assert result.stored[0].name == "evt-1-0.png"


def test_a_message_without_images_never_touches_the_platform(tmp_path):
    result = asyncio.run(run_fetch([text()], tmp_path))

    assert result.outcomes == ()
    assert result.stored == () and result.overflow == 0 and result.failures == ()


def test_the_segment_url_is_used_when_get_image_cannot_resolve_the_file(tmp_path):
    """真机证据（2026-09-11）：get_image 对解析不到的文件回 failed/retcode=200。"""

    async def get_image(token):
        raise OneBotActionError("OneBot get_image returned status='failed' retcode=200")

    async def download(url):
        assert url == "https://cdn.test/from-segment.png"
        return PNG

    result = asyncio.run(
        run_fetch(
            [image(file="segment-token", url="https://cdn.test/from-segment.png")],
            tmp_path,
            get_image=get_image,
            download=download,
        )
    )

    assert result.outcomes[0].status == STORED
    assert result.stored[0].read_bytes() == PNG


def test_a_platform_failure_is_still_reported_when_the_segment_has_no_url(tmp_path):
    async def get_image(token):
        raise OneBotActionError("failed")

    result = asyncio.run(run_fetch([image(file="segment-token")], tmp_path, get_image=get_image))

    assert result.outcomes[0].status == UNAVAILABLE


def test_the_outcome_shape_has_no_field_for_what_the_picture_shows():
    assert set(ImageFetchOutcome.__dataclass_fields__) == {
        "index",
        "status",
        "path",
        "content_type",
        "size_bytes",
    }


def test_outcome_invariants_are_enforced():
    with pytest.raises(ValueError):
        ImageFetchOutcome(index=0, status=STORED)
    with pytest.raises(ValueError):
        ImageFetchOutcome(index=0, status=TIMEOUT, path=Path("x.png"))
    with pytest.raises(ValueError):
        ImageFetchOutcome(index=0, status="invented")
    with pytest.raises(ValueError):
        ImageFetchError("invented", "detail")


@pytest.mark.parametrize(
    ("content", "expected"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp"),
     (TEXT, None), (b"", None)],
)
def test_content_type_sniffing(content, expected):
    assert sniff_content_type(content) == expected


def test_local_source_understands_windows_paths_and_urls():
    assert local_source(r"C:\napcat\cache\x.png") == Path(r"C:\napcat\cache\x.png")
    assert local_source("file:///C:/napcat/cache/x.png") == Path("C:/napcat/cache/x.png")
    assert local_source("x.png") == Path("x.png")
    assert local_source("https://cdn.test/x.png") is None
    assert local_source("ftp://cdn.test/x.png") is None
    assert local_source("") is None


@asynccontextmanager
async def fake_image_server(handler):
    app = web.Application()
    app.router.add_get("/{name}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def test_the_real_downloader_lands_the_bytes_and_sends_no_authorization_header(tmp_path):
    async def scenario():
        seen: list[dict[str, str]] = []

        async def handler(request: web.Request):
            seen.append(dict(request.headers))
            return web.Response(body=PNG, content_type="image/png")

        async with fake_image_server(handler) as base:
            result = await run_fetch([image(url=f"{base}/a.png")], tmp_path, timeout_seconds=5)

        assert result.outcomes[0].status == STORED
        assert result.stored[0].read_bytes() == PNG
        assert len(seen) == 1
        assert not [key for key in seen[0] if key.lower() == "authorization"], "下载不得携带 NapCat 令牌"

    asyncio.run(scenario())


def test_the_real_downloader_classifies_http_failures(tmp_path):
    async def scenario():
        async def handler(request: web.Request):
            return web.Response(status=404, text="nope")

        async with fake_image_server(handler) as base:
            result = await run_fetch([image(url=f"{base}/missing.png")], tmp_path, timeout_seconds=5)

        assert result.outcomes[0].status == UNAVAILABLE
        assert stored_files(tmp_path) == []

    asyncio.run(scenario())


def test_the_real_downloader_stops_at_the_size_ceiling(tmp_path):
    async def scenario():
        async def handler(request: web.Request):
            return web.Response(body=b"\x89PNG\r\n\x1a\n" + b"x" * 4096, content_type="image/png")

        async with fake_image_server(handler) as base:
            result = await run_fetch([image(url=f"{base}/big.png")], tmp_path,
                                     max_bytes=1024, timeout_seconds=5)

        assert result.outcomes[0].status == TOO_LARGE
        assert stored_files(tmp_path) == []

    asyncio.run(scenario())


def test_the_real_downloader_times_out(tmp_path):
    async def scenario():
        async def handler(request: web.Request):
            await asyncio.sleep(1.0)
            return web.Response(body=PNG, content_type="image/png")

        async with fake_image_server(handler) as base:
            result = await run_fetch([image(url=f"{base}/slow.png")], tmp_path, timeout_seconds=0.05)

        assert result.outcomes[0].status == TIMEOUT
        assert result.stored == ()

    asyncio.run(scenario())


def test_the_real_downloader_follows_a_cdn_redirect(tmp_path):
    async def scenario():
        async def handler(request: web.Request):
            if request.match_info["name"] == "moved.png":
                raise web.HTTPFound("/final.png")
            return web.Response(body=JPEG, content_type="image/jpeg")

        async with fake_image_server(handler) as base:
            result = await run_fetch([image(url=f"{base}/moved.png")], tmp_path, timeout_seconds=5)

        assert result.outcomes[0].status == STORED
        assert result.outcomes[0].content_type == "image/jpeg"

    asyncio.run(scenario())


def test_download_image_reports_a_redirect_without_a_location_as_protocol():
    async def scenario():
        async def handler(request: web.Request):
            return web.Response(status=302)

        async with fake_image_server(handler) as base:
            with pytest.raises(ImageFetchError) as error:
                await download_image(f"{base}/x.png", max_bytes=1024, timeout_seconds=5)

        assert error.value.status == PROTOCOL

    asyncio.run(scenario())
