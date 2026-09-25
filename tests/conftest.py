from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def config_path() -> Path:
    return ROOT / "config.example.yaml"


@pytest.fixture
def complete_environment() -> dict[str, str]:
    return {
        "QICHI_OWNER_QQ": "123456",
        "NAPCAT_WS_URL": "ws://127.0.0.1:6700",
        "NAPCAT_HTTP_URL": "http://127.0.0.1:5700",
        "NAPCAT_ACCESS_TOKEN": "test-napcat-token",
        "DEEPSEEK_API_KEY": "test-deepseek-key",
        # 2026-09-14：语音上线后出厂配置 voice.enabled=true，密钥缺失会让配置校验拒绝加载。
        # 这个夹具代表「完整的生产环境」，所以这里也要有它。
        "DASHSCOPE_API_KEY": "test-dashscope-key",
    }
