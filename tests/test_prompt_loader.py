from __future__ import annotations

from pathlib import Path

import pytest

from qichi.dialogue.prompt_loader import PromptLoadError, load_runtime_prompt


def test_loads_only_the_runtime_role_core(tmp_path: Path):
    doc = tmp_path / "doc"
    doc.mkdir()
    runtime_core = "# runtime\n\n你是角色。\n"
    (doc / "运行时角色核心.md").write_bytes(runtime_core.encode("utf-8"))
    (doc / "用户画像.md").write_text("UNCONFIRMED_PROFILE_SENTINEL", encoding="utf-8")
    (doc / "角色外貌设定.md").write_text("APPEARANCE_SENTINEL", encoding="utf-8")

    loaded = load_runtime_prompt(tmp_path)

    assert loaded == runtime_core
    assert "UNCONFIRMED_PROFILE_SENTINEL" not in loaded
    assert "APPEARANCE_SENTINEL" not in loaded


def test_missing_empty_and_non_utf8_prompt_fail_closed(tmp_path: Path):
    with pytest.raises(PromptLoadError, match="missing"):
        load_runtime_prompt(tmp_path)

    doc = tmp_path / "doc"
    doc.mkdir()
    prompt = doc / "运行时角色核心.md"
    prompt.write_text("  \n", encoding="utf-8")
    with pytest.raises(PromptLoadError, match="empty"):
        load_runtime_prompt(tmp_path)

    prompt.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(PromptLoadError, match="UTF-8"):
        load_runtime_prompt(tmp_path)


def test_prompt_size_is_bounded_before_decoding(tmp_path: Path):
    doc = tmp_path / "doc"
    doc.mkdir()
    (doc / "运行时角色核心.md").write_bytes(b"x" * 32_769)

    with pytest.raises(PromptLoadError, match="32768"):
        load_runtime_prompt(tmp_path)
@pytest.mark.parametrize("root", [None, "", 1])
def test_project_root_type_is_strict(root):
    with pytest.raises((TypeError, ValueError)):
        load_runtime_prompt(root)


# 开源导出：本文件原有两条 test_runtime_core_keeps_*_boundaries 用例，断言的是私有
# 角色核心里的具体句子（含角色外形与成人互动授权）。公开仓只带通用占位角色
# （doc/运行时角色核心.md），无法满足这些断言，故随私有角色一并移出。
