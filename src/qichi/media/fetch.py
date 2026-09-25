"""Fetch inbound images from the platform and land them under the media root.

This module owns exactly one question: *did the bytes arrive, and where are
they?*  It never looks at what a picture shows, never guesses and never writes
to the ledger.  Every failure becomes a named outcome so the caller can say
"图片没取到" without inventing a single detail about the picture.

Policy lives elsewhere on purpose: size, type and retention rules belong to
qichi.media.store; this module only decides what is worth handing over.
Nothing here is reachable from the conversation path until the vision switch is
on -- the caller owns that gate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import aiohttp

from qichi.domain.events import MessageSegment
from qichi.media.store import MediaStoreError, save_inbound_image
from qichi.transport.onebot_client import (
    OneBotActionError,
    OneBotConnectionError,
    OneBotError,
    OneBotHTTPClientError,
    OneBotHTTPServerError,
    OneBotProtocolError,
    OneBotTimeoutError,
)

IMAGE_SEGMENT_TYPE = "image"

# Every way one image can end.  "stored" is the only one that yields a picture;
# the rest are reasons to tell the model that nothing was seen.
STORED = "stored"
OVERFLOW = "overflow"
UNAVAILABLE = "unavailable"
TIMEOUT = "timeout"
TOO_LARGE = "too_large"
UNSUPPORTED_TYPE = "unsupported_type"
NETWORK = "network"
PROTOCOL = "protocol"

FETCH_STATUSES = frozenset(
    {STORED, OVERFLOW, UNAVAILABLE, TIMEOUT, TOO_LARGE, UNSUPPORTED_TYPE, NETWORK, PROTOCOL}
)

_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_MAGIC_BYTES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class ImageFetchError(RuntimeError):
    """One image could not be fetched; status is the honest reason why."""

    def __init__(self, status: str, detail: str) -> None:
        if status not in FETCH_STATUSES:
            raise ValueError(f"unknown image fetch status {status!r}")
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class ImageFetchOutcome:
    """What happened to the n-th image of one inbound message."""

    index: int
    status: str
    path: Path | None = None
    content_type: str | None = None
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        if self.status not in FETCH_STATUSES:
            raise ValueError(f"unknown image fetch status {self.status!r}")
        if self.status == STORED and self.path is None:
            raise ValueError("a stored image must carry its path")
        if self.path is not None and self.status != STORED:
            raise ValueError("only a stored image may carry a path")


@dataclass(frozen=True)
class ImageFetchResult:
    """The outcome of every image segment in one inbound message, in order."""

    outcomes: tuple[ImageFetchOutcome, ...] = ()

    @property
    def stored(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.outcomes if item.status == STORED and item.path is not None)

    @property
    def overflow(self) -> int:
        """How many images were left unexpanded, to be reported as such."""

        return sum(1 for item in self.outcomes if item.status == OVERFLOW)

    @property
    def failures(self) -> tuple[ImageFetchOutcome, ...]:
        return tuple(item for item in self.outcomes if item.status not in {STORED, OVERFLOW})

    @property
    def attempted(self) -> int:
        return sum(1 for item in self.outcomes if item.status != OVERFLOW)


def image_segments(segments: Sequence[MessageSegment]) -> tuple[MessageSegment, ...]:
    return tuple(segment for segment in segments if segment.type == IMAGE_SEGMENT_TYPE)


def sniff_content_type(content: bytes) -> str | None:
    """Identify an image by its own bytes.

    A server's Content-Type header is a claim, not evidence; only the bytes
    decide whether something is a picture we support.
    """

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    for magic, content_type in _MAGIC_BYTES:
        if content.startswith(magic):
            return content_type
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def local_source(value: str) -> Path | None:
    """Turn a cached-file answer into a local path, or None if it is not one.

    NapCat answers with an absolute Windows path, a file:// url or a plain
    relative name depending on version; a Windows drive letter parses as a
    one-character scheme, so it must not be mistaken for a network url.
    """

    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        path = unquote(parsed.path)
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path) if path else None
    if len(scheme) > 1:
        return None
    return Path(value)


async def download_image(url: str, *, max_bytes: int, timeout_seconds: float) -> bytes:
    """Read one image over HTTP(S) with a hard ceiling on what is kept.

    No authorization header travels with this request: the url regularly points
    at a public CDN, and the NapCat token has no business leaving the machine.
    """

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    chunks: list[bytes] = []
    total = 0
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if 300 <= response.status < 400:
                    raise ImageFetchError(PROTOCOL, "image download redirected")
                if response.status >= 400:
                    raise ImageFetchError(UNAVAILABLE, f"image download answered {response.status}")
                async for chunk in response.content.iter_chunked(_DOWNLOAD_CHUNK_BYTES):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ImageFetchError(TOO_LARGE, "image exceeds the configured size limit")
                    chunks.append(chunk)
    except asyncio.TimeoutError:
        raise ImageFetchError(TIMEOUT, "image download timed out") from None
    except aiohttp.ClientConnectionError:
        raise ImageFetchError(NETWORK, "image download connection failed") from None
    except aiohttp.ClientError:
        raise ImageFetchError(NETWORK, "image download network failed") from None
    return b"".join(chunks)


async def fetch_inbound_images(
    segments: Sequence[MessageSegment],
    *,
    event_id: str,
    data_root: str | Path,
    get_image: Callable[[str], Awaitable[Mapping[str, Any]]],
    max_images: int,
    max_bytes: int,
    timeout_seconds: float,
    download: Callable[[str], Awaitable[bytes]] | None = None,
    now: datetime | None = None,
) -> ImageFetchResult:
    """Land up to max_images pictures of one message and report the rest.

    Expected failures are named per image and never abort the turn; a bug in
    this module is deliberately allowed to surface instead of being swallowed
    into a silent "she saw nothing".
    """

    if type(max_images) is not int or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    fetch = download or (lambda url: download_image(url, max_bytes=max_bytes, timeout_seconds=timeout_seconds))
    images = image_segments(segments)
    outcomes: list[ImageFetchOutcome] = []
    for index, segment in enumerate(images):
        if index >= max_images:
            outcomes.append(ImageFetchOutcome(index=index, status=OVERFLOW))
            continue
        try:
            content = await _read_one(segment, get_image=get_image, fetch=fetch, max_bytes=max_bytes)
        except ImageFetchError as error:
            outcomes.append(ImageFetchOutcome(index=index, status=error.status))
            continue
        content_type = sniff_content_type(content)
        if content_type is None:
            outcomes.append(ImageFetchOutcome(index=index, status=UNSUPPORTED_TYPE))
            continue
        if len(content) > max_bytes:
            outcomes.append(ImageFetchOutcome(index=index, status=TOO_LARGE))
            continue
        try:
            path = save_inbound_image(
                data_root,
                event_id=f"{event_id}-{index}",
                content=content,
                content_type=content_type,
                max_bytes=max_bytes,
                now=now,
            )
        except MediaStoreError:
            outcomes.append(ImageFetchOutcome(index=index, status=PROTOCOL))
            continue
        outcomes.append(
            ImageFetchOutcome(
                index=index, status=STORED, path=path, content_type=content_type, size_bytes=len(content)
            )
        )
    return ImageFetchResult(outcomes=tuple(outcomes))


async def _read_one(
    segment: MessageSegment,
    *,
    get_image: Callable[[str], Awaitable[Mapping[str, Any]]],
    fetch: Callable[[str], Awaitable[bytes]],
    max_bytes: int,
) -> bytes:
    """Read the bytes of one image segment, preferring the local cache."""

    token = _text(segment.data.get("file"))
    url = _text(segment.data.get("url"))
    cached: str | None = None
    if token is not None:
        try:
            payload = await _ask_platform(token, get_image)
        except ImageFetchError:
            # Measured against the live server on 2026-09-11: get_image answers
            # failed/retcode=200 for a file it cannot resolve.  The message
            # segment usually carries a url of its own, and that url is the same
            # picture -- so only give up when there is no other source.
            if url is None:
                raise
        else:
            cached = _text(payload.get("file"))
            answered = _text(payload.get("url"))
            if answered is not None:
                url = answered
    if cached is not None:
        path = local_source(cached)
        if path is not None and path.is_file():
            return _read_file(path, max_bytes=max_bytes)
    if url is not None:
        path = local_source(url)
        if path is not None:
            if not path.is_file():
                raise ImageFetchError(UNAVAILABLE, "image file is no longer on disk")
            return _read_file(path, max_bytes=max_bytes)
        if urlparse(url).scheme.lower() in {"http", "https"}:
            content = await fetch(url)
            if not isinstance(content, bytes):
                raise ImageFetchError(PROTOCOL, "image download did not return bytes")
            return content
        raise ImageFetchError(PROTOCOL, "image url scheme is not supported")
    raise ImageFetchError(UNAVAILABLE, "no image source was provided")


def _read_file(path: Path, *, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        raise ImageFetchError(NETWORK, "image file cannot be read") from None
    if size > max_bytes:
        raise ImageFetchError(TOO_LARGE, "image exceeds the configured size limit")
    try:
        return path.read_bytes()
    except OSError:
        raise ImageFetchError(NETWORK, "image file cannot be read") from None


async def _ask_platform(token: str, get_image: Callable[[str], Awaitable[Mapping[str, Any]]]) -> Mapping[str, Any]:
    try:
        payload = await get_image(token)
    except OneBotTimeoutError:
        raise ImageFetchError(TIMEOUT, "get_image timed out") from None
    except (OneBotActionError, OneBotHTTPClientError):
        raise ImageFetchError(UNAVAILABLE, "get_image could not resolve the image") from None
    except (OneBotConnectionError, OneBotHTTPServerError):
        raise ImageFetchError(NETWORK, "get_image is unreachable") from None
    except OneBotProtocolError:
        raise ImageFetchError(PROTOCOL, "get_image answered with an invalid envelope") from None
    except OneBotError:
        raise ImageFetchError(PROTOCOL, "get_image failed") from None
    if not isinstance(payload, Mapping):
        raise ImageFetchError(PROTOCOL, "get_image payload is not a mapping")
    return payload


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
