"""Load the single runtime role prompt without pulling in reference documents."""

from __future__ import annotations

from pathlib import Path


RUNTIME_PROMPT_PATH = Path("doc") / "运行时角色核心.md"
MAX_PROMPT_BYTES = 32_768


class PromptLoadError(ValueError):
    """The runtime prompt cannot be loaded exactly and safely."""


def load_runtime_prompt(
    project_root: str | Path, prompt_path: str | Path | None = None
) -> str:
    """Load the thin role core from one project root.

    默认路径是 RUNTIME_PROMPT_PATH；调用方可以传配置里的相对路径，
    让使用者自备角色核心（路径仍必须落在 project_root 之内）。
    """
    if not isinstance(project_root, (str, Path)):
        raise TypeError("project_root must be a path")
    if isinstance(project_root, str) and not project_root.strip():
        raise ValueError("project_root must not be empty")

    root = Path(project_root).resolve()
    relative = RUNTIME_PROMPT_PATH if prompt_path is None else Path(prompt_path)
    prompt_path = root / relative
    if not prompt_path.is_file():
        raise PromptLoadError("runtime prompt is missing")
    try:
        resolved_prompt = prompt_path.resolve(strict=True)
        resolved_prompt.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PromptLoadError("runtime prompt must stay inside the project root") from exc

    try:
        size = resolved_prompt.stat().st_size
    except OSError as exc:
        raise PromptLoadError("runtime prompt cannot be inspected") from exc
    if size > MAX_PROMPT_BYTES:
        raise PromptLoadError(f"runtime prompt exceeds {MAX_PROMPT_BYTES} bytes")

    try:
        raw = resolved_prompt.read_bytes()
    except OSError as exc:
        raise PromptLoadError("runtime prompt cannot be read") from exc
    if len(raw) > MAX_PROMPT_BYTES:
        raise PromptLoadError(f"runtime prompt exceeds {MAX_PROMPT_BYTES} bytes")
    try:
        prompt = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptLoadError("runtime prompt must be UTF-8") from exc
    if not prompt.strip():
        raise PromptLoadError("runtime prompt is empty")
    return prompt
