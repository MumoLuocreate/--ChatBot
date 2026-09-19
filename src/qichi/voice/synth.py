"""语音合成客户端：一次 HTTP 调用，拿回一段音频字节。

实测形状（2026-09-14，见历史TTS计划 §2.10）：
    POST {base_url}/services/aigc/multimodal-generation/generation
    {"model": ..., "input": {"text": ..., "voice": ...},
     "parameters": {"language_type": "Chinese", "instructions": ...}}
    -> 200, output.audio.url（24 小时有效的音频 URL）-> GET 该 url

本模块只负责"把字念出来"。语气指令由调用方给出（生产里由主模型决定），
这里不判断情绪、不选择音色、不决定要不要用语音。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp


class SpeechError(RuntimeError):
    """语音合成链路失败的基类；实例里绝不包含 API Key。"""


class SpeechTimeoutError(SpeechError):
    pass


class SpeechConnectionError(SpeechError):
    pass


class SpeechHTTPError(SpeechError):
    pass


class SpeechProtocolError(SpeechError):
    pass


_GENERATION_PATH = "/services/aigc/multimodal-generation/generation"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SpeechClient:
    """DashScope 非实时语音合成；同一个实例可重复使用。"""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    voice: str
    timeout_seconds: float = 12.0
    sample_rate: int = 24000
    _session: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _require_text(self.base_url, "base_url").rstrip("/"))
        _require_text(self.api_key, "api_key")
        _require_text(self.model, "model")
        _require_text(self.voice, "voice")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

    def __repr__(self) -> str:
        return (
            f"SpeechClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"voice={self.voice!r}, timeout_seconds={self.timeout_seconds!r})"
        )

    async def synthesize(self, text: str, *, instructions: str | None = None) -> bytes:
        """合成一段音频；失败一律抛 SpeechError 的子类，绝返回半截数据。"""

        spoken = _require_text(text, "text")
        parameters: dict[str, Any] = {"language_type": "Chinese"}
        if instructions is not None:
            parameters["instructions"] = _require_text(instructions, "instructions")
        payload = {
            "model": self.model,
            "input": {"text": spoken, "voice": self.voice},
            "parameters": parameters,
        }
        session = await self._get_session()
        envelope = await self._post(session, payload)
        url = self._audio_url(envelope)
        if urlparse(url).scheme not in {"http", "https"}:
            raise SpeechProtocolError("audio url must be absolute http(s)")
        return await self._download(session, url)

    async def close(self) -> None:
        session = self._session
        if session is not None and not session.closed:
            await session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            object.__setattr__(
                self,
                "_session",
                aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)),
            )
        return self._session

    async def _post(self, session: aiohttp.ClientSession, payload: Mapping[str, Any]) -> Any:
        error: SpeechError | None = None
        envelope: Any = None
        try:
            async with session.post(
                self.base_url + _GENERATION_PATH,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=dict(payload),
                allow_redirects=False,
            ) as response:
                if 300 <= response.status < 400:
                    raise SpeechHTTPError(f"speech synthesis redirected (HTTP {response.status})")
                if response.status >= 400:
                    raise SpeechHTTPError(f"speech synthesis failed (HTTP {response.status})")
                try:
                    envelope = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, ValueError):
                    error = SpeechProtocolError("speech synthesis response is not JSON")
        except SpeechError as raised:
            error = raised
        except TimeoutError:
            error = SpeechTimeoutError("speech synthesis timed out")
        except aiohttp.ClientConnectionError:
            error = SpeechConnectionError("speech synthesis connection failed")
        except aiohttp.ClientError:
            error = SpeechConnectionError("speech synthesis network failed")
        if error is not None:
            raise error
        if not isinstance(envelope, Mapping):
            raise SpeechProtocolError("speech synthesis response must be a mapping")
        return envelope

    @staticmethod
    def _audio_url(envelope: Mapping[str, Any]) -> str:
        output = envelope.get("output")
        if not isinstance(output, Mapping):
            raise SpeechProtocolError("speech synthesis response has no output")
        audio = output.get("audio")
        if not isinstance(audio, Mapping):
            raise SpeechProtocolError("speech synthesis response has no audio")
        url = audio.get("url")
        if not isinstance(url, str) or not url:
            raise SpeechProtocolError("speech synthesis response has no audio url")
        return url

    async def _download(self, session: aiohttp.ClientSession, url: str) -> bytes:
        error: SpeechError | None = None
        content = b""
        try:
            async with session.get(url, allow_redirects=False) as response:
                if response.status >= 400:
                    raise SpeechHTTPError(f"audio download failed (HTTP {response.status})")
                content = await response.read()
        except SpeechError as raised:
            error = raised
        except TimeoutError:
            error = SpeechTimeoutError("audio download timed out")
        except aiohttp.ClientConnectionError:
            error = SpeechConnectionError("audio download connection failed")
        except aiohttp.ClientError:
            error = SpeechConnectionError("audio download network failed")
        if error is not None:
            raise error
        if not content:
            raise SpeechProtocolError("downloaded audio is empty")
        return content
