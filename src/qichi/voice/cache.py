"""语音缓存：按内容寻址落盘，重入不会重复合成。

与 `qichi.media.store` 同形：一个文件一件事、先写 `.part` 再原子替换、
清理是**独立的显式调用**（绝不走对话热路径）。

键 = 模型 + 音色 + 要念的文本 + 语气指令 + 采样率的 sha256。同样的输入必然得到同样的
键，所以 outbox 重入时 payload 里的路径是确定的（`历史TTS计划` §3.5）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import re
from pathlib import Path
from typing import Final


VOICE_ROOT_NAME: Final = "voice"
_KEY_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_EXTENSIONS: Final = frozenset({".wav", ".mp3", ".opus"})


class VoiceCacheError(ValueError):
    """音频无法按缓存策略落盘。"""


def cache_key(
    *,
    model: str,
    voice: str,
    text: str,
    instructions: str | None = None,
    sample_rate: int = 24000,
    container: str = "wav",
) -> str:
    """把一次合成的全部输入折成一个确定性的键。"""

    parts = (model, voice, text, instructions or "", str(sample_rate), container)
    if any(not isinstance(item, str) for item in parts):
        raise TypeError("cache key parts must be strings")
    if not model or not voice or not text:
        raise VoiceCacheError("model, voice and text must all be non-empty")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise VoiceCacheError("sample_rate must be a positive integer")
    if container not in {"wav", "mp3", "opus"}:
        raise VoiceCacheError("container must be wav, mp3 or opus")
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _extension(container: str) -> str:
    extension = "." + container
    if extension not in _EXTENSIONS:
        raise VoiceCacheError("unsupported audio container")
    return extension


def cache_path(data_root: str | Path, *, key: str, container: str = "wav") -> Path:
    """`data/voice/<键前两位>/<键>.<后缀>`——两级散列，避免单目录堆几万个文件。"""

    if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:
        raise VoiceCacheError("cache key must be 64 lowercase hex characters")
    return Path(data_root) / VOICE_ROOT_NAME / key[:2] / (key + _extension(container))


def store_voice(
    data_root: str | Path,
    *,
    key: str,
    content: bytes,
    container: str = "wav",
) -> Path:
    """写入一段音频；已存在就原样返回（内容寻址 ⇒ 同键同内容）。"""

    if not isinstance(content, bytes) or not content:
        raise VoiceCacheError("audio content is empty")
    target = cache_path(data_root, key=key, container=container)
    if target.is_file() and target.stat().st_size == len(content):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(target)
    return target


def voice_files(data_root: str | Path) -> tuple[Path, ...]:
    """所有缓存音频，最旧优先；半截文件不算。"""

    root = Path(data_root) / VOICE_ROOT_NAME
    if not root.is_dir():
        return ()
    files = [item for item in root.rglob("*") if item.is_file() and item.suffix != ".part"]
    return tuple(sorted(files, key=lambda item: (item.stat().st_mtime, item.as_posix())))


def cleanup_voice(
    data_root: str | Path,
    *,
    max_total_bytes: int,
    max_age_days: int,
    now: datetime | None = None,
) -> tuple[Path, ...]:
    """删除超期或超量的音频；只删文件，不动账本里的任何记录。"""

    if type(max_total_bytes) is not int or max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be a positive integer")
    if type(max_age_days) is not int or max_age_days <= 0:
        raise ValueError("max_age_days must be a positive integer")
    moment = now or datetime.now(timezone.utc)
    cutoff = (moment - timedelta(days=max_age_days)).timestamp()
    files = list(voice_files(data_root))
    sizes: dict[Path, int] = {}
    deleted: list[Path] = []
    for item in files:
        if item.stat().st_mtime < cutoff:
            item.unlink()
            deleted.append(item)
        else:
            sizes[item] = item.stat().st_size
    total = sum(sizes.values())
    for item in list(sizes):
        if total <= max_total_bytes:
            break
        total -= sizes.pop(item)
        item.unlink()
        deleted.append(item)
    return tuple(deleted)
