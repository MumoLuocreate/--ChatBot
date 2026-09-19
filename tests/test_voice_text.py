"""颜文字/emoji 剥离：命中与不误判。"""

from __future__ import annotations

import pytest

from qichi.voice.text import SpeechText, is_speakable, prepare_speech


@pytest.mark.parametrize(
    "raw, spoken",
    [
        ("嗯，在呢。这声兔兔叫得挺顺口的嘛(￣▽￣)", "嗯，在呢。这声兔兔叫得挺顺口的嘛"),
        ("行吧，让你亲，反正我也没躲(〃ω〃)", "行吧，让你亲，反正我也没躲"),
        ("谁害羞了，我这是被你亲得有点措手不及(｡•́︿•̀｡)", "谁害羞了，我这是被你亲得有点措手不及"),
        ("好，去吧(￣▽￣)晚上见", "好，去吧晚上见"),
        ("直接写 emoji 也剥掉 😀✨", "直接写 emoji 也剥掉"),
        ("(￣▽￣)", ""),
    ],
)
def test_kaomoji_and_emoji_are_stripped(raw, spoken):
    result = prepare_speech(raw)

    assert result.spoken == spoken
    assert result.changed is True


@pytest.mark.parametrize(
    "raw",
    [
        "（笑）这个我认",
        "你说的 (TTS) 那条链路",
        "那是（2026）年的事",
        "她说了「不借」两个字",
    ],
)
def test_meaningful_parentheses_are_kept(raw):
    result = prepare_speech(raw)

    assert result.spoken == raw
    assert result.changed is False


def test_residue_keeps_what_was_removed():
    result = prepare_speech("好(￣▽￣)😀")

    assert result.spoken == "好"
    assert "(￣▽￣)" in result.residue
    assert "😀" in result.residue


def test_plain_text_is_untouched():
    raw = "行，那我就不追着问了，等你弄出点像样的东西再来显摆。"

    assert prepare_speech(raw).spoken == raw


def test_speakable_requires_leftover_content():
    assert is_speakable("嗯，在呢", max_chars=100) is True
    assert is_speakable("(￣▽￣)", max_chars=100) is False
    assert is_speakable("……", max_chars=100) is False


def test_speakable_rejects_overlong_text():
    assert is_speakable("好" * 20, max_chars=10) is False
    assert is_speakable("好" * 10, max_chars=10) is True


def test_prepare_speech_rejects_non_text():
    with pytest.raises(TypeError):
        prepare_speech(None)  # type: ignore[arg-type]


def test_is_speakable_validates_arguments():
    with pytest.raises(ValueError):
        is_speakable("好", max_chars=0)
    with pytest.raises(TypeError):
        is_speakable(3, max_chars=10)  # type: ignore[arg-type]


def test_speech_text_is_frozen_and_typed():
    result = SpeechText(spoken="好", residue="")
    assert result.changed is False
    with pytest.raises(TypeError):
        SpeechText(spoken=1, residue="")  # type: ignore[arg-type]
