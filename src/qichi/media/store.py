"""Local storage for inbound media.

Images leave this machine exactly like text does -- they are sent to the model
provider -- so what is kept locally is deliberately narrow: one file per event,
size and type checked, written under the project data root, and never re-sent
unless a later turn asks for it.  Retention is an explicit, separate call so the
conversation hot path never walks the media directory.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

MEDIA_ROOT_NAME = "media"
CONTENT_TYPE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class MediaStoreError(ValueError):
    """An inbound image cannot be stored under the configured policy."""


def extension_for(content_type: str | None) -> str | None:
    if not isinstance(content_type, str):
        return None
    return CONTENT_TYPE_EXTENSIONS.get(content_type.split(";")[0].strip().lower())


def save_inbound_image(
    data_root: str | Path,
    *,
    event_id: str,
    content: bytes,
    content_type: str | None,
    max_bytes: int,
    now: datetime | None = None,
) -> Path:
    """Write one inbound image and return its path.

    The ledger event id becomes the file name, so it is restricted to a safe
    character set: a stored id must never be able to steer a path.
    """

    if not isinstance(event_id, str) or _SAFE_EVENT_ID.match(event_id) is None:
        raise MediaStoreError("event_id is not a safe file name")
    if not isinstance(content, bytes) or not content:
        raise MediaStoreError("image content is empty")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if len(content) > max_bytes:
        raise MediaStoreError("image exceeds the configured size limit")
    extension = extension_for(content_type)
    if extension is None:
        raise MediaStoreError("unsupported image content type")
    moment = now or datetime.now(timezone.utc)
    directory = Path(data_root) / MEDIA_ROOT_NAME / f"{moment:%Y-%m}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{event_id}{extension}"
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(target)
    return target


def media_files(data_root: str | Path) -> tuple[Path, ...]:
    """Every stored image, oldest first; partial writes are ignored."""

    root = Path(data_root) / MEDIA_ROOT_NAME
    if not root.is_dir():
        return ()
    files = [item for item in root.rglob("*") if item.is_file() and item.suffix != ".part"]
    return tuple(sorted(files, key=lambda item: (item.stat().st_mtime, item.as_posix())))


def cleanup_media(
    data_root: str | Path,
    *,
    max_total_bytes: int,
    max_age_days: int,
    now: datetime | None = None,
) -> tuple[Path, ...]:
    """Delete expired or surplus image files; never touches the ledger.

    The ledger keeps the path and the recognition result, so removing a file
    degrades an old turn to \"the image is no longer stored\" instead of losing
    the fact that it existed.
    """

    if type(max_total_bytes) is not int or max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be a positive integer")
    if type(max_age_days) is not int or max_age_days <= 0:
        raise ValueError("max_age_days must be a positive integer")
    moment = now or datetime.now(timezone.utc)
    cutoff = (moment - timedelta(days=max_age_days)).timestamp()
    files = list(media_files(data_root))
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
