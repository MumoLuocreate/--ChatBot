"""Load the single runtime role prompt without pulling in reference documents.

The engine owns no persona. The only role text ever injected is the one file a
deployment points at through `persona.system_prompt_file`; this module loads
exactly that file and nothing else, so reference documents can never leak into
the hot path by accident.
"""

from __future__ import annotations

from pathlib import Path


# Default used when a caller has no configuration at hand. Production paths come
# from `persona.system_prompt_file`, not from this constant.
RUNTIME_PROMPT_PATH = Path("doc") / "运行时角色核心.md"
MAX_PROMPT_BYTES = 32_768


class PromptLoadError(ValueError):
    """The runtime prompt cannot be loaded exactly and safely."""


def load_runtime_prompt(
    project_root: str | Path,
    relative_path: str | Path = RUNTIME_PROMPT_PATH,
) -> str:
    """Load the fixed thin role core from one project root.

    `relative_path` is the configured role-core file. It is resolved against
    `project_root` and must stay inside it, so a mistyped or hostile config
    cannot read arbitrary files. Nothing is substituted on failure: a missing or
    invalid prompt fails closed instead of falling back to a default persona.
    """
    if not isinstance(project_root, (str, Path)):
        raise TypeError("project_root must be a path")
    if isinstance(project_root, str) and not project_root.strip():
        raise ValueError("project_root must not be empty")
    if not isinstance(relative_path, (str, Path)):
        raise TypeError("relative_path must be a path")
    if isinstance(relative_path, str) and not relative_path.strip():
        raise ValueError("relative_path must not be empty")

    root = Path(project_root).resolve()
    prompt_path = root / Path(relative_path)
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
