"""语音导演：剥离、指令、缓存命中、以及各种失败不拦路。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from qichi.voice.director import VoiceDirector, VoiceUnavailable
from qichi.voice.synth import SpeechConnectionError


class FakeSpeechClient:
    model = "qwen3-tts-vd-2026-01-26"
    voice = "qwen-tts-vd-qichi_cast2-voice-x"
    sample_rate = 24000

    def __init__(self, *, error=None):
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    async def synthesize(self, text, *, instructions=None):
        self.calls.append((text, instructions))
        if self.error:
            raise self.error
        return b"RIFF-audio-bytes"


def director(tmp_path, client, writer=None, **overrides):
    return VoiceDirector(client=client, data_root=tmp_path, writer=writer, **overrides)


def test_render_strips_kaomoji_and_caches_by_content(tmp_path):
    async def scenario():
        client = FakeSpeechClient()
        writer_calls: list[str] = []

        async def writer(situation):
            writer_calls.append(situation)
            return "慵懒松弛，句尾往下坠"

        d = director(tmp_path, client, writer)
        first = await d.render("嗯，在呢。这声兔兔叫得挺顺口的嘛(￣▽￣)", situation="他在叫她兔兔")
        second = await d.render("嗯，在呢。这声兔兔叫得挺顺口的嘛(￣▽￣)", situation="他在叫她兔兔")

        assert first.spoken == "嗯，在呢。这声兔兔叫得挺顺口的嘛"
        assert first.residue == "(￣▽￣)"
        assert first.instructions == "慵懒松弛，句尾往下坠"
        assert first.from_cache is False and second.from_cache is True
        assert len(client.calls) == 1, "同样的输入第二次必须零调用"
        assert client.calls[0] == ("嗯，在呢。这声兔兔叫得挺顺口的嘛", "慵懒松弛，句尾往下坠")
        assert first.path.is_file() and second.path == first.path
        assert writer_calls == ["他在叫她兔兔", "他在叫她兔兔"]

    asyncio.run(scenario())


def test_a_failing_writer_does_not_block_the_voice(tmp_path):
    async def scenario():
        client = FakeSpeechClient()

        async def writer(_situation):
            raise RuntimeError("flash is down")

        clip = await director(tmp_path, client, writer).render("好。", situation="任何情境")

        assert clip.instructions is None
        assert client.calls == [("好。", None)]

    asyncio.run(scenario())


def test_blank_instructions_are_treated_as_absent(tmp_path):
    async def scenario():
        client = FakeSpeechClient()

        async def writer(_situation):
            return "   "

        clip = await director(tmp_path, client, writer).render("好。", situation="x")

        assert clip.instructions is None

    asyncio.run(scenario())


@pytest.mark.parametrize("text", ["(￣▽￣)", "😀", ""])
def test_text_with_nothing_left_to_say_is_refused(tmp_path, text):
    async def scenario():
        client = FakeSpeechClient()
        with pytest.raises(VoiceUnavailable):
            await director(tmp_path, client).render(text)
        assert client.calls == [], "剥完没内容就不该外呼"

    asyncio.run(scenario())


def test_overlong_text_is_refused(tmp_path):
    async def scenario():
        client = FakeSpeechClient()
        with pytest.raises(VoiceUnavailable):
            await director(tmp_path, client, max_chars=5).render("这一句明显超过五个字")
        assert client.calls == []

    asyncio.run(scenario())


def test_synthesis_failure_becomes_unavailable(tmp_path):
    async def scenario():
        client = FakeSpeechClient(error=SpeechConnectionError("down"))
        with pytest.raises(VoiceUnavailable):
            await director(tmp_path, client).render("好。")

    asyncio.run(scenario())


def test_director_validates_its_settings(tmp_path):
    client = FakeSpeechClient()
    with pytest.raises(ValueError):
        director(tmp_path, client, max_chars=0)
    with pytest.raises(ValueError):
        director(tmp_path, client, container="flac")
