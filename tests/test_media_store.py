from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qichi.media.store import (
    MediaStoreError,
    cleanup_media,
    extension_for,
    media_files,
    save_inbound_image,
)

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def test_saves_an_inbound_image_under_a_month_directory(tmp_path):
    path = save_inbound_image(
        tmp_path, event_id="evt-1", content=b"png-bytes", content_type="image/png",
        max_bytes=1024, now=NOW,
    )

    assert path == tmp_path / "media" / "2026-09" / "evt-1.png"
    assert path.read_bytes() == b"png-bytes"
    assert not list(path.parent.glob("*.part")), "不留半截文件"


def test_content_type_parameters_and_case_are_understood():
    assert extension_for("IMAGE/JPEG; charset=binary") == ".jpg"
    assert extension_for("application/pdf") is None
    assert extension_for(None) is None


@pytest.mark.parametrize("event_id", ["../evil", "a/b", "", "x" * 200])
def test_unsafe_event_ids_are_refused(tmp_path, event_id):
    with pytest.raises(MediaStoreError):
        save_inbound_image(tmp_path, event_id=event_id, content=b"x", content_type="image/png",
                           max_bytes=10, now=NOW)


def test_size_type_and_empty_content_are_refused(tmp_path):
    with pytest.raises(MediaStoreError):
        save_inbound_image(tmp_path, event_id="e", content=b"x" * 50, content_type="image/png",
                           max_bytes=10, now=NOW)
    with pytest.raises(MediaStoreError):
        save_inbound_image(tmp_path, event_id="e", content=b"x", content_type="application/pdf",
                           max_bytes=10, now=NOW)
    with pytest.raises(MediaStoreError):
        save_inbound_image(tmp_path, event_id="e", content=b"", content_type="image/png",
                           max_bytes=10, now=NOW)


def _write(tmp_path: Path, name: str, age_days: int, size: int) -> Path:
    path = tmp_path / "media" / "2026-09" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = (NOW - timedelta(days=age_days)).timestamp()
    import os
    os.utime(path, (stamp, stamp))
    return path


def test_cleanup_drops_expired_files_first(tmp_path):
    old = _write(tmp_path, "old.png", 120, 10)
    fresh = _write(tmp_path, "fresh.png", 1, 10)

    deleted = cleanup_media(tmp_path, max_total_bytes=10_000, max_age_days=90, now=NOW)

    assert deleted == (old,)
    assert fresh.exists() and not old.exists()
    assert media_files(tmp_path) == (fresh,)


def test_cleanup_drops_the_oldest_files_until_under_the_cap(tmp_path):
    oldest = _write(tmp_path, "a.png", 3, 40)
    middle = _write(tmp_path, "b.png", 2, 40)
    newest = _write(tmp_path, "c.png", 1, 40)

    deleted = cleanup_media(tmp_path, max_total_bytes=80, max_age_days=90, now=NOW)

    assert deleted == (oldest,)
    assert [item.name for item in media_files(tmp_path)] == ["b.png", "c.png"]
    assert middle.exists() and newest.exists()
