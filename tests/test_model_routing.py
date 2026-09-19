"""模型分流：文字走主模型、带图那一轮走视觉档、后台永远走 background_model。"""

from __future__ import annotations

from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from qichi.config import ConfigError, load_config
from qichi.dialogue.llm_client import DeepSeekLLMClient
from qichi.domain.dialogue import ModelImage, ModelMessage


@pytest_asyncio.fixture
async def capture_server(unused_tcp_port):
    """Records the request bodies and echoes the model name back."""

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


def picture(tmp_path: Path) -> ModelImage:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return ModelImage(path=str(path), content_type="image/png")


@pytest.mark.asyncio
async def test_text_turns_use_the_primary_and_picture_turns_use_the_vision_model(
    capture_server, tmp_path: Path
):
    base, calls = capture_server
    client = DeepSeekLLMClient(
        base, "k", model="deepseek-v4-pro", vision_model="deepseek-v4-flash"
    )
    try:
        plain = await client.generate((ModelMessage("user", "在吗"),))
        with_image = await client.generate(
            (ModelMessage("user", "看看这个", images=(picture(tmp_path),)),)
        )
    finally:
        await client.close()

    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[1]["model"] == "deepseek-v4-flash"
    assert plain.model_id == "deepseek-v4-pro" and plain.model_tier == "text"
    assert with_image.model_id == "deepseek-v4-flash" and with_image.model_tier == "vision"
    # 分流是档位不同，不是第二条链路：路由仍然只有 primary 一种。
    assert plain.model_route == with_image.model_route == "primary"


@pytest.mark.asyncio
async def test_without_a_vision_model_pictures_stay_on_the_primary(capture_server, tmp_path: Path):
    base, calls = capture_server
    client = DeepSeekLLMClient(base, "k", model="deepseek-v4-pro")
    try:
        generation = await client.generate(
            (ModelMessage("user", "看看这个", images=(picture(tmp_path),)),)
        )
    finally:
        await client.close()

    assert calls[-1]["model"] == "deepseek-v4-pro"
    assert generation.model_tier == "text"


def test_unverified_model_names_fail_closed():
    with pytest.raises(ValueError):
        DeepSeekLLMClient("https://api.deepseek.com/v1", "k", model="deepseek-v4-pro-0813")
    with pytest.raises(ValueError):
        DeepSeekLLMClient(
            "https://api.deepseek.com/v1", "k", model="deepseek-v4-pro", vision_model="gpt-4o"
        )


def _without_llm_lines(source: str, keys) -> str:
    """去掉 llm 段里这几个键（配置现在显式写了它们，测试要先清干净再重写）。"""

    kept = [
        line
        for line in source.splitlines()
        if not any(line.strip().startswith(f"{key}:") for key in keys)
    ]
    return "\n".join(kept) + "\n"


def _config_with(config_path: Path, tmp_path: Path, **llm_lines: str) -> Path:
    source = _without_llm_lines(config_path.read_text(encoding="utf-8"), llm_lines)
    extra = "".join(f"\n  {key}: {value}" for key, value in llm_lines.items())
    source = source.replace("  api_key_env: DEEPSEEK_API_KEY", "  api_key_env: DEEPSEEK_API_KEY" + extra)
    path = tmp_path / "routing.yaml"
    path.write_text(source, encoding="utf-8")
    return path


def test_defaults_keep_the_single_model_behaviour(
    config_path: Path, complete_environment, tmp_path: Path
):
    """省略这两个键时行为与分流之前一致：不分流、后台跟主模型。"""

    path = tmp_path / "routing-defaults.yaml"
    path.write_text(
        _without_llm_lines(
            config_path.read_text(encoding="utf-8"), ("vision_model", "background_model")
        ),
        encoding="utf-8",
    )
    shipped = load_config(config_path, environ=complete_environment)
    config = load_config(path, environ=complete_environment)

    assert config.llm.primary.model == shipped.llm.primary.model
    assert config.llm.vision_model is None
    assert config.llm.background_model == config.llm.primary.model


def test_config_accepts_pro_with_flash_for_pictures_and_background(
    config_path: Path, complete_environment, tmp_path: Path
):
    source = _config_with(
        config_path,
        tmp_path,
        vision_model="deepseek-v4-flash",
        background_model="deepseek-v4-flash",
    ).read_text(encoding="utf-8")
    path = tmp_path / "routing-pro.yaml"
    path.write_text(
        source.replace("    model: deepseek-v4-flash", "    model: deepseek-v4-pro", 1),
        encoding="utf-8",
    )

    config = load_config(path, environ=complete_environment)

    assert config.llm.primary.model == "deepseek-v4-pro"
    assert config.llm.vision_model == "deepseek-v4-flash"
    assert config.llm.background_model == "deepseek-v4-flash"


@pytest.mark.parametrize(
    "lines",
    [
        {"vision_model": "gpt-4o"},
        {"background_model": "deepseek-v4-pro-0813"},
    ],
)
def test_unverified_routing_models_are_rejected(
    config_path: Path, complete_environment, tmp_path: Path, lines: dict[str, str]
):
    path = _config_with(config_path, tmp_path, **lines)

    with pytest.raises(ConfigError, match="not a verified identifier"):
        load_config(path, environ=complete_environment)


def test_siliconflow_cannot_route_pictures_to_another_provider(
    config_path: Path, complete_environment, tmp_path: Path
):
    shipped_model = load_config(config_path, environ=complete_environment).llm.primary.model
    source = _config_with(config_path, tmp_path, vision_model="deepseek-v4-flash").read_text(
        encoding="utf-8"
    ).replace("provider: deepseek", "provider: siliconflow").replace(
        "api_key_env: DEEPSEEK_API_KEY", "api_key_env: SILICONFLOW_API_KEY"
    ).replace("base_url: https://api.deepseek.com/v1", "base_url: https://api.siliconflow.cn/v1").replace(
        f"\n    model: {shipped_model}", "\n    model: deepseek-ai/DeepSeek-V4-Flash", 1
    )
    path = tmp_path / "routing-siliconflow.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="vision_model is only supported"):
        load_config(path, environ=complete_environment)
