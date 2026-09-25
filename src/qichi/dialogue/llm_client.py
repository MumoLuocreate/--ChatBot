from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import aiohttp

from qichi.domain.dialogue import ModelImage, ModelMessage, ModelRoute


SILICONFLOW_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_MODEL_ID = "deepseek-v4-flash"
# 2026-09-12 分流：文字走 pro、带图那一轮走 flash（pro 不支持视觉）。
# flash 的旧标识仍然被官方接受，两者都能作为主模型或视觉档。
DEEPSEEK_MODEL_IDS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
# Kept for callers that imported the historical constant.
MODEL_ID = SILICONFLOW_MODEL_ID


class LLMError(RuntimeError):
    """Base class for deliberately non-sensitive model client failures."""


class LLMTimeoutError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMModelNotFoundError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMServerError(LLMError):
    pass


class LLMProtocolError(LLMError):
    pass


class LLMImageUnavailableError(LLMRequestError):
    """A picture attached to this turn could not be read from local storage."""


class LLMImageRejectedError(LLMRequestError):
    """The endpoint refused a request that carried pictures.

    Measured 2026-09-11: the endpoint accepts image input, so this is a
    defensive path -- the caller degrades to text and records why instead of
    pretending she saw something.
    """


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One function call the model asked for.  The engine executes it, not the client."""

    call_id: str
    name: str
    # 原始 JSON 字符串：解析是执行方的事，客户端只负责如实转达（坏参数不该在这里被吞掉）。
    arguments: str


@dataclass(frozen=True, slots=True)
class LLMGeneration:
    text: str = field(repr=False)
    model_route: ModelRoute
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    # Provider-side prefix cache: how many input tokens were served from it.
    # None means the provider did not report the number at all.
    cache_hit_tokens: int | None = None
    # Which tier answered this turn: "text" (primary) or "vision" (the picture
    # model, chosen only when the request carried images).
    model_tier: str | None = None
    # 模型请求调用的工具（2026-09-14 联网）。空元组 = 这一轮没有请求工具。
    tool_calls: tuple[ToolCall, ...] = ()


def _number(value: Any, name: str, low: float, high: float, *, low_exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if (number <= low if low_exclusive else number < low) or number > high:
        bound = f"> {low}" if low_exclusive else f">= {low}"
        raise ValueError(f"{name} must be {bound} and <= {high}")
    return number


def _read_image_file(path: str) -> bytes:
    return Path(path).read_bytes()


def _image_data_url(image: ModelImage, read_image: Callable[[str], bytes]) -> str:
    try:
        content = read_image(image.path)
    except OSError:
        raise LLMImageUnavailableError("attached image could not be read") from None
    if not isinstance(content, bytes) or not content:
        raise LLMImageUnavailableError("attached image is empty")
    return f"data:{image.content_type};base64,{base64.b64encode(content).decode("ascii")}"


def wire_messages(
    messages: Sequence[ModelMessage],
    *,
    read_image: Callable[[str], bytes] | None = None,
) -> list[dict[str, Any]]:
    """Turn domain messages into the provider's message format.

    A message without pictures keeps the plain string content it always had, so
    nothing about text-only turns changes.  A message with pictures becomes the
    multimodal part list: an empty text part is dropped rather than invented,
    because an image-only message has no text to send.
    """

    reader = _read_image_file if read_image is None else read_image
    payload: list[dict[str, Any]] = []
    for message in messages:
        if not message.images:
            payload.append({"role": message.role, "content": message.content})
            continue
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"type": "text", "text": message.content})
        for image in message.images:
            parts.append({"type": "image_url", "image_url": {"url": _image_data_url(image, reader)}})
        payload.append({"role": message.role, "content": parts})
    return payload


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        model: str,
        temperature: float = 0.85,
        top_p: float = 0.92,
        max_output_tokens: int = 1_200,
        timeout_seconds: float = 25.0,
        default_thinking: str = "default",
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if default_thinking not in {"default", "enabled", "disabled"}:
            raise ValueError("default_thinking must be default, enabled or disabled")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = _number(temperature, "temperature", 0.0, 2.0)
        self._top_p = _number(top_p, "top_p", 0.0, 1.0, low_exclusive=True)
        self._max_output_tokens = max_output_tokens
        self._default_thinking = default_thinking
        self._timeout_seconds = _number(
            timeout_seconds,
            "timeout_seconds",
            0.0,
            float("inf"),
            low_exclusive=True,
        )
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleLLMClient("
            f"timeout_seconds={self._timeout_seconds!r}, closed={self._closed!r})"
        )

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> OpenAICompatibleLLMClient:
        await self._get_session()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        max_output_tokens: int | None = None,
        response_format: Mapping[str, str] | None = None,
        thinking: Mapping[str, str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMGeneration:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise TypeError("messages must be a non-empty sequence of ModelMessage")
        if not all(isinstance(message, ModelMessage) for message in messages):
            raise TypeError("messages must contain only ModelMessage")
        budget = self._max_output_tokens if max_output_tokens is None else max_output_tokens
        if type(budget) is not int or budget < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if response_format is not None:
            if not isinstance(response_format, Mapping) or response_format.get("type") != "json_object":
                raise ValueError("response_format must be the json_object format")
        if thinking is not None:
            if (
                not isinstance(thinking, Mapping)
                or set(thinking) != {"type"}
                or thinking.get("type") not in {"enabled", "disabled"}
            ):
                raise ValueError("thinking must be {'type': 'enabled'} or {'type': 'disabled'}")
        sampling_temperature = (
            self._temperature
            if temperature is None
            else _number(temperature, "temperature", 0.0, 2.0)
        )
        sampling_top_p = (
            self._top_p
            if top_p is None
            else _number(top_p, "top_p", 0.0, 1.0, low_exclusive=True)
        )
        # 2026-09-13：配置可以让对话默认关掉思考（pro 开着思考时实测 27 秒一轮）。
        # 显式传入的 thinking 永远优先——引擎的截断重试与后台两条路径都靠这一点。
        if tools is not None:
            if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence) or not tools:
                raise ValueError("tools must be a non-empty sequence of function declarations")
            for declaration in tools:
                if (
                    not isinstance(declaration, Mapping)
                    or declaration.get("type") != "function"
                    or not isinstance(declaration.get("function"), Mapping)
                    or not isinstance(declaration["function"].get("name"), str)
                    or not declaration["function"]["name"].strip()
                ):
                    raise ValueError("every tool must declare one named function")
        effective_thinking = thinking
        if effective_thinking is None and self._default_thinking != "default":
            effective_thinking = {"type": self._default_thinking}
        carries_images = any(message.images for message in messages)
        # 分流只在这一层发生：带图且配了视觉档时改用视觉档，其余一切照旧。
        vision_model = getattr(self, "_vision_model", None)
        model_tier = "vision" if (carries_images and vision_model) else "text"
        model_id = vision_model if model_tier == "vision" else self._model
        session = await self._get_session()
        payload = {
            "model": model_id,
            "messages": wire_messages(messages),
            "temperature": sampling_temperature,
            "top_p": sampling_top_p,
            "max_tokens": budget,
        }
        if response_format is not None:
            payload["response_format"] = {"type": "json_object"}
        if tools is not None:
            payload["tools"] = [dict(declaration) for declaration in tools]
        if effective_thinking is not None:
            payload["thinking"] = {"type": effective_thinking["type"]}
        started = time.monotonic()
        try:
            async with session.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                allow_redirects=False,
            ) as response:
                status = response.status
                if 300 <= status < 400:
                    raise LLMRequestError(f"LLM HTTP redirect ({status})")
                if status == 429:
                    raise LLMRateLimitError("LLM rate limit exceeded")
                if status in (401, 403):
                    raise LLMAuthenticationError(f"LLM authentication failed ({status})")
                if status == 404:
                    raise LLMModelNotFoundError("LLM model endpoint not found")
                if 400 <= status < 500:
                    if carries_images:
                        raise LLMImageRejectedError(f"LLM rejected an image request ({status})")
                    raise LLMRequestError(f"LLM request rejected ({status})")
                if status >= 500:
                    raise LLMServerError(f"LLM server error ({status})")
                try:
                    body = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    raise LLMProtocolError("LLM response is not valid JSON") from None
        except asyncio.TimeoutError:
            raise LLMTimeoutError("LLM request timed out") from None
        except aiohttp.ClientConnectionError:
            raise LLMConnectionError("LLM connection failed") from None
        except aiohttp.ClientError:
            raise LLMConnectionError("LLM network request failed") from None

        (
            text, input_tokens, output_tokens, finish_reason, reasoning_tokens, cache_hit_tokens,
            tool_calls,
        ) = self._parse_body(body)
        return LLMGeneration(
            text=text,
            model_route="primary",
            model_id=model_id,
            model_tier=model_tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=max(0.0, (time.monotonic() - started) * 1000),
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
            cache_hit_tokens=cache_hit_tokens,
            tool_calls=tool_calls,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise LLMConnectionError("LLM client is closed")
        async with self._session_lock:
            if self._closed:
                raise LLMConnectionError("LLM client is closed")
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout_seconds),
                    # The production host reaches external providers through
                    # its configured HTTPS/ALL_PROXY.  Without this, aiohttp
                    # attempts a direct socket connection even when the
                    # Windows session has a working proxy route.
                    trust_env=True,
                )
            return self._session

    def _parse_body(
        self, body: Any
    ) -> tuple[str, int, int, str | None, int | None, int | None, tuple[ToolCall, ...]]:
        if not isinstance(body, dict):
            raise LLMProtocolError("LLM response structure is invalid")
        # The provider echoes its own identifier, which it may rename without
        # asking us: on 2026-09-11 a request for deepseek-v4-flash came back with
        # model="deepseek-flash" and every generation failed as a protocol error.
        # The identity that matters is the one we send, already pinned by the
        # configuration and its capability evidence, so a renamed echo is passed
        # through while a malformed one still fails closed.
        observed_model = body.get("model")
        if observed_model is not None and not isinstance(observed_model, str):
            raise LLMProtocolError("LLM response model is invalid")
        choices = body.get("choices")
        usage = body.get("usage")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMProtocolError("LLM response choices are invalid")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMProtocolError("LLM response content is invalid")
        content = message.get("content")
        if content is None:
            # A provider can answer with a well-formed body that carries no
            # visible content at all: an empty completion, or output withheld by
            # a content policy.  That is an outcome rather than a broken
            # response, so it is passed on as an empty string and the engine
            # decides what an empty completion means for this source.  Anything
            # that is not a string and not null is still a broken response.
            content = ""
        if not isinstance(content, str):
            raise LLMProtocolError("LLM response content is invalid")
        if not isinstance(usage, dict):
            raise LLMProtocolError("LLM response usage is invalid")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if type(input_tokens) is not int or input_tokens < 0 or type(output_tokens) is not int or output_tokens < 0:
            raise LLMProtocolError("LLM response token usage is invalid")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise LLMProtocolError("LLM response finish reason is invalid")
        reasoning_tokens = None
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            value = details.get("reasoning_tokens")
            if type(value) is not int or value < 0:
                raise LLMProtocolError("LLM response reasoning usage is invalid")
            reasoning_tokens = value
        # DeepSeek reports the prefix-cache hit as `prompt_cache_hit_tokens`; the
        # OpenAI-style nested name carries the same number.  The value is only an
        # observation, but a self-contradicting provider answer (a hit larger than
        # the whole prompt, or a non-integer) is still a broken response.
        cache_hit_tokens = None
        raw_hit = usage.get("prompt_cache_hit_tokens")
        if raw_hit is None:
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                raw_hit = prompt_details.get("cached_tokens")
        if raw_hit is not None:
            if type(raw_hit) is not int or raw_hit < 0 or raw_hit > input_tokens:
                raise LLMProtocolError("LLM response cache usage is invalid")
            cache_hit_tokens = raw_hit
        # 工具调用（2026-09-14 联网）：形状不对就整条响应作废——参数是执行方要用的，
        # 吞掉一个坏调用比报错更危险。
        tool_calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls")
        if raw_calls is not None:
            if not isinstance(raw_calls, list):
                raise LLMProtocolError("LLM response tool calls are invalid")
            for item in raw_calls:
                if not isinstance(item, Mapping):
                    raise LLMProtocolError("LLM response tool calls are invalid")
                function = item.get("function")
                name = function.get("name") if isinstance(function, Mapping) else None
                arguments = function.get("arguments") if isinstance(function, Mapping) else None
                call_id = item.get("id")
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(arguments, str)
                ):
                    raise LLMProtocolError("LLM response tool calls are invalid")
                tool_calls.append(
                    ToolCall(
                        call_id=call_id if isinstance(call_id, str) else "",
                        name=name.strip(),
                        arguments=arguments,
                    )
                )
        return (
            content, input_tokens, output_tokens, finish_reason, reasoning_tokens, cache_hit_tokens,
            tuple(tool_calls),
        )


class SiliconFlowLLMClient(OpenAICompatibleLLMClient):
    """Compatibility client for the historical SiliconFlow route."""

    def __init__(self, base_url: str, api_key: str, **kwargs: Any) -> None:
        model = kwargs.pop("model", SILICONFLOW_MODEL_ID)
        if model != SILICONFLOW_MODEL_ID:
            raise ValueError(f"model must be {SILICONFLOW_MODEL_ID}")
        super().__init__(base_url, api_key, model=model, **kwargs)


class DeepSeekLLMClient(OpenAICompatibleLLMClient):
    """Official DeepSeek API client using its OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, **kwargs: Any) -> None:
        model = kwargs.pop("model", DEEPSEEK_MODEL_ID)
        vision_model = kwargs.pop("vision_model", None)
        if model not in DEEPSEEK_MODEL_IDS:
            raise ValueError(f"model must be one of {sorted(DEEPSEEK_MODEL_IDS)}")
        if vision_model is not None and vision_model not in DEEPSEEK_MODEL_IDS:
            raise ValueError(f"vision_model must be one of {sorted(DEEPSEEK_MODEL_IDS)}")
        super().__init__(base_url, api_key, model=model, **kwargs)
        # None 表示不分流：带图也走主模型（与 2026-09-12 之前的行为一致）。
        self._vision_model = vision_model
