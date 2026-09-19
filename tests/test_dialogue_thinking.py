"""对话思考开关：配置决定默认值，显式传入的 thinking 永远优先。"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from qichi.config import ConfigError, load_config
from qichi.dialogue.llm_client import DeepSeekLLMClient
from qichi.domain.dialogue import ModelMessage
from qichi.runtime import _build_llm_client


@pytest_asyncio.fixture
async def capture_server(unused_tcp_port):
    calls: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        calls.append(body)
        return web.json_response(
            {
                "model": body["model"],
                "choices": [{"message": {"content": "好"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", calls
    finally:
        await runner.cleanup()


def _with_thinking(config_path: Path, tmp_path: Path, value: str | None) -> Path:
    source = config_path.read_text(encoding="utf-8")
    # 2026-09-14：只动 llm.primary 下面那一行（4 空格缩进）。initiative 段现在也有
    # 自己的 thinking（2 空格缩进），按 strip() 匹配会把它一起改坏成非法 YAML。
    if value is None:
        source = "\n".join(
            line for line in source.splitlines() if not line.startswith("    thinking:")
        ) + "\n"
    else:
        source = "\n".join(
            f"    thinking: {value}" if line.startswith("    thinking:") else line
            for line in source.splitlines()
        ) + "\n"
    path = tmp_path / "thinking.yaml"
    path.write_text(source, encoding="utf-8")
    return path


def test_shipped_config_keeps_thinking_at_provider_default(
    config_path: Path, complete_environment
) -> None:
    """2026-09-16 起生产是 flash + default：flash 默认含思考，且思考很快。

    09-13 的对照里关掉思考会把「自己复读」推高到 41%，那是给 pro 用的取舍，不再适用。
    """

    config = load_config(config_path, environ=complete_environment)

    assert config.llm.primary.thinking == "default"


def test_omitting_the_key_is_the_same_as_default(
    config_path: Path, complete_environment, tmp_path: Path
) -> None:
    config = load_config(_with_thinking(config_path, tmp_path, None), environ=complete_environment)

    assert config.llm.primary.thinking == "default"


def test_disabled_is_accepted(config_path: Path, complete_environment, tmp_path: Path) -> None:
    config = load_config(
        _with_thinking(config_path, tmp_path, "disabled"), environ=complete_environment
    )

    assert config.llm.primary.thinking == "disabled"


def test_unknown_thinking_values_fail_closed(
    config_path: Path, complete_environment, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="thinking must be default, enabled or disabled"):
        load_config(
            _with_thinking(config_path, tmp_path, "sometimes"), environ=complete_environment
        )


@pytest.mark.asyncio
async def test_a_disabled_default_is_sent_when_the_caller_passes_nothing(capture_server):
    base, calls = capture_server
    client = DeepSeekLLMClient(base, "k", model="deepseek-v4-pro", default_thinking="disabled")
    try:
        await client.generate((ModelMessage("user", "在吗"),))
    finally:
        await client.close()

    assert calls[0]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_an_explicit_thinking_argument_beats_the_default(capture_server):
    base, calls = capture_server
    client = DeepSeekLLMClient(base, "k", model="deepseek-v4-pro", default_thinking="disabled")
    try:
        await client.generate((ModelMessage("user", "在吗"),), thinking={"type": "enabled"})
    finally:
        await client.close()

    assert calls[0]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_the_default_default_leaves_the_parameter_out(capture_server):
    base, calls = capture_server
    client = DeepSeekLLMClient(base, "k", model="deepseek-v4-pro")
    try:
        await client.generate((ModelMessage("user", "在吗"),))
    finally:
        await client.close()

    assert "thinking" not in calls[0]


@pytest.mark.asyncio
async def test_the_runtime_builder_hands_the_setting_to_the_dialogue_client(capture_server):
    """部署时走的就是这条路：配置 → runtime → 说话用的客户端。"""

    base, calls = capture_server
    client = _build_llm_client(
        "deepseek", base, "k", model="deepseek-v4-pro", default_thinking="disabled"
    )
    try:
        await client.generate((ModelMessage("user", "在吗"),))
    finally:
        await client.close()

    assert calls[0]["thinking"] == {"type": "disabled"}


def test_unverified_defaults_fail_closed() -> None:
    with pytest.raises(ValueError):
        DeepSeekLLMClient(
            "https://api.deepseek.com/v1", "k", model="deepseek-v4-pro", default_thinking="maybe"
        )
