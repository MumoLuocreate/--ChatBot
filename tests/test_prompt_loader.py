from __future__ import annotations

from pathlib import Path

import pytest

from qichi.dialogue.prompt_loader import PromptLoadError, load_runtime_prompt


ROOT = Path(__file__).resolve().parents[1]


def test_loads_only_the_configured_role_core(tmp_path: Path):
    """引擎只加载配置指定的那一个文件，同目录的其它文档永远不会被带进上下文。"""
    doc = tmp_path / "doc"
    doc.mkdir()
    runtime_core = "# runtime\n\n你是测试角色。\n"
    (doc / "运行时角色核心.md").write_bytes(runtime_core.encode("utf-8"))
    (doc / "其它文档.md").write_text("REFERENCE_SENTINEL", encoding="utf-8")
    (doc / "外貌.md").write_text("APPEARANCE_SENTINEL", encoding="utf-8")

    loaded = load_runtime_prompt(tmp_path)

    assert loaded == runtime_core
    assert "REFERENCE_SENTINEL" not in loaded
    assert "APPEARANCE_SENTINEL" not in loaded


def test_configured_relative_path_selects_the_role_core(tmp_path: Path):
    """persona.system_prompt_file 是唯一的角色来源：换路径就换角色，代码不动。"""
    doc = tmp_path / "doc"
    doc.mkdir()
    (doc / "运行时角色核心.md").write_text("DEFAULT_SENTINEL", encoding="utf-8")
    custom_core = doc / "我的角色.md"
    custom_core.write_text("CUSTOM_SENTINEL", encoding="utf-8")

    assert load_runtime_prompt(tmp_path, Path("doc/我的角色.md")) == "CUSTOM_SENTINEL"
    assert load_runtime_prompt(tmp_path, "doc/运行时角色核心.md") == "DEFAULT_SENTINEL"
    # 绝对路径同样生效，只要它落在项目根目录内。
    assert load_runtime_prompt(tmp_path, custom_core) == "CUSTOM_SENTINEL"


def test_configured_path_cannot_escape_the_project_root(tmp_path: Path):
    """配置不能读项目根目录外的文件；这是边界，不是方便性开关。"""
    outside = tmp_path.parent / "outside-prompt.md"
    outside.write_text("OUTSIDE_SENTINEL", encoding="utf-8")

    with pytest.raises(PromptLoadError, match="inside the project root"):
        load_runtime_prompt(tmp_path, outside)
    with pytest.raises(PromptLoadError, match="inside the project root"):
        load_runtime_prompt(tmp_path, Path("../outside-prompt.md"))


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


@pytest.mark.parametrize("relative", [None, "", 1])
def test_configured_path_type_is_strict(tmp_path: Path, relative):
    with pytest.raises((TypeError, ValueError)):
        load_runtime_prompt(tmp_path, relative)


def test_shipped_example_role_core_loads_without_private_files():
    """克隆下来就能跑：仓库自带的示例角色核心必须真的能加载。"""
    config = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    assert "system_prompt_file: doc/运行时角色核心.example.md" in config

    loaded = load_runtime_prompt(ROOT, Path("doc/运行时角色核心.example.md"))

    assert loaded.strip()
