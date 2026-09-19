from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _runtime_artifacts_ready() -> bool:
    """生产就绪检查需要两件仓库不携带的产物（见 README「仓库不包含什么」）。

    缺件时这条测试**跳过**而不是假装通过：它检验的正是「生产装配是否就绪」，
    没有 tokenizer 与供应商能力证据时，就绪与否根本无法判定。
    """

    if not (ROOT / "runtime" / "model-cache" / "v4-tokenizer.json").is_file():
        return False
    return any((ROOT / "runtime").glob("*-capability.json"))


requires_runtime_artifacts = pytest.mark.skipif(
    not _runtime_artifacts_ready(),
    reason="需要 runtime 产物（tokenizer 与供应商能力证据）；见 README「仓库不包含什么」",
)


@requires_runtime_artifacts
def test_preflight_is_read_only_and_reports_production_assembly(config_path, complete_environment):
    env = os.environ.copy()
    env.update(complete_environment)
    result = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--config", str(config_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK: context profile: 256K (262144), adaptive expansion disabled" in result.stdout
    assert "BLOCKED: QQ face catalog:" not in result.stdout
    assert "OK: provider capability evidence: 1048576 tokens" in result.stdout
    assert "OK: production coordinator and workers:" in result.stdout
    assert complete_environment["DEEPSEEK_API_KEY"] not in result.stdout
    assert complete_environment["NAPCAT_ACCESS_TOKEN"] not in result.stdout
    assert not (ROOT / "ready.json").exists()


def test_preflight_rejects_missing_environment_without_revealing_values(config_path):
    env = os.environ.copy()
    for key in (
        "QICHI_OWNER_QQ",
        "NAPCAT_WS_URL",
        "NAPCAT_HTTP_URL",
        "NAPCAT_ACCESS_TOKEN",
        "DEEPSEEK_API_KEY",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--config", str(config_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "BLOCKED: config:" in result.stdout
