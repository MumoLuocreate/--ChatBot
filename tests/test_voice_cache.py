"""语音缓存：确定性、原子写、独立清理。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qichi.voice.cache import (
    VoiceCacheError,
    cache_key,
    cache_path,
    cleanup_voice,
    store_voice,
    voice_files,
)

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
BASE = dict(model="qwen3-tts-vd-2026-01-26", voice="qwen-tts-vd-qichi_cast2-voice-x", text="嗯，在呢。")


def test_key_is_deterministic_and_input_sensitive():
    first = cache_key(**BASE, instructions="慵懒一点", sample_rate=24000, container="wav")
    same = cache_key(**BASE, instructions="慵懒一点", sample_rate=24000, container="wav")
    other_text = cache_key(**{**BASE, "text": "嗯，在呢！"}, instructions="慵懒一点", sample_rate=24000, container="wav")
    other_ins = cache_key(**BASE, instructions="撒娇一点", sample_rate=24000, container="wav")
    no_ins = cache_key(**BASE, instructions=None, sample_rate=24000, container="wav")

    assert first == same
    assert len({first, other_text, other_ins, no_ins}) == 4, "任何一个输入变化都必须换键"
    assert len(first) == 64 and first == first.lower()


def test_key_requires_the_essentials():
    with pytest.raises(VoiceCacheError):
        cache_key(model="", voice="v", text="t")
    with pytest.raises(VoiceCacheError):
        cache_key(model="m", voice="v", text="t", sample_rate=0)
    with pytest.raises(VoiceCacheError):
        cache_key(model="m", voice="v", text="t", container="flac")


def test_path_is_two_level_and_validated(tmp_path):
    key = cache_key(**BASE)

    assert cache_path(tmp_path, key=key) == tmp_path / "voice" / key[:2] / (key + ".wav")
    with pytest.raises(VoiceCacheError):
        cache_path(tmp_path, key="not-a-key")
    with pytest.raises(VoiceCacheError):
        cache_path(tmp_path, key=key.upper())


def test_store_is_atomic_and_idempotent(tmp_path):
    key = cache_key(**BASE)

    path = store_voice(tmp_path, key=key, content=b"audio-bytes")

    assert path.read_bytes() == b"audio-bytes"
    assert not list(path.parent.glob("*.part")), "不留半截文件"
    assert store_voice(tmp_path, key=key, content=b"audio-bytes") == path
    assert voice_files(tmp_path) == (path,)


def test_store_refuses_empty_content(tmp_path):
    with pytest.raises(VoiceCacheError):
        store_voice(tmp_path, key=cache_key(**BASE), content=b"")


def test_cleanup_deletes_by_age_but_never_touches_other_dirs(tmp_path):
    key = cache_key(**BASE)
    fresh = store_voice(tmp_path, key=key, content=b"x" * 10)
    stray = tmp_path / "media" / "2026-09" / "keep.png"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"png")
    old = datetime.now(timezone.utc) - timedelta(days=90)
    import os

    os.utime(fresh, (old.timestamp(), old.timestamp()))

    deleted = cleanup_voice(tmp_path, max_total_bytes=10_000, max_age_days=30, now=NOW)

    assert deleted == (fresh,)
    assert stray.read_bytes() == b"png", "只清语音缓存，不碰别人的目录"


def test_cleanup_enforces_total_size_oldest_first(tmp_path):
    keys = [cache_key(**{**BASE, "text": "句%d。" % i}) for i in range(3)]
    paths = [store_voice(tmp_path, key=k, content=b"y" * 100) for k in keys]
    import os

    for index, path in enumerate(paths):
        stamp = NOW.timestamp() - (len(paths) - index) * 60
        os.utime(path, (stamp, stamp))

    # 三份各 100 字节；上限 250 ⇒ 只须删最旧那一份
    deleted = cleanup_voice(tmp_path, max_total_bytes=250, max_age_days=3650, now=NOW)

    assert deleted == (paths[0],)
    assert voice_files(tmp_path) == (paths[1], paths[2])

    # 上限压到 150 ⇒ 还得再删一份，仍然是最旧的先走
    deleted_again = cleanup_voice(tmp_path, max_total_bytes=150, max_age_days=3650, now=NOW)

    assert deleted_again == (paths[1],)
    assert voice_files(tmp_path) == (paths[2],)


def test_cleanup_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        cleanup_voice(tmp_path, max_total_bytes=0, max_age_days=1)
    with pytest.raises(ValueError):
        cleanup_voice(tmp_path, max_total_bytes=1, max_age_days=0)
