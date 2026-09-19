"""语气指令写手：把情境喂进去、把一条指令拿出来。"""

from __future__ import annotations

import asyncio

import pytest

from qichi.domain.dialogue import ModelMessage
from qichi.voice.instructions import SYSTEM_PROMPT, VoiceInstructionWriter


class FakeGeneration:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    def __init__(self, text="慵懒松弛，句尾往下坠"):
        self.text = text
        self.calls: list[tuple[tuple[ModelMessage, ...], dict]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append((tuple(messages), kwargs))
        return FakeGeneration(self.text)


def test_writer_returns_a_cleaned_instruction_and_disables_thinking():
    async def scenario():
        client = FakeLLM('"在「谁求了」之后停半拍，句尾收住"')
        instruction = await VoiceInstructionWriter(client=client)("[最近往来]\n用户：在吗")

        assert instruction == "在「谁求了」之后停半拍，句尾收住"
        messages, kwargs = client.calls[0]
        assert messages[0] == ModelMessage("system", SYSTEM_PROMPT)
        assert messages[1].content.startswith("[最近往来]")
        assert messages[1].content.endswith("请只输出那条语气指令。")
        assert kwargs["thinking"] == {"type": "disabled"}, "开着思考会把预算吃光、返回空内容"
        assert kwargs["max_output_tokens"] == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("text", ["", "   ", None])
def test_writer_treats_a_blank_answer_as_no_instruction(text):
    async def scenario():
        client = FakeLLM(text)
        assert await VoiceInstructionWriter(client=client)("情境") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("situation", ["", "   ", None])
def test_writer_skips_the_call_without_a_situation(situation):
    async def scenario():
        client = FakeLLM()
        assert await VoiceInstructionWriter(client=client)(situation) is None
        assert client.calls == [], "没有情境就不该外呼"

    asyncio.run(scenario())


def test_the_prompt_gives_information_without_coaching_the_writing():
    """2026-09-15 用户裁定：只提供信息，不指导它怎么写。

    §3.1b 曾经把「写动作与时间点、不要只写情绪名词」写进提示词，结果把盲测赢的那类
    指令（情绪＋气息＋句尾走向）挤掉了（§2.40）。这条测试是那条裁定的回归闸门。
    """

    for coaching in ("停半拍", "换气", "加重", "情绪名词", "语速", "气息", "句尾", "轻重"):
        assert coaching not in SYSTEM_PROMPT, "提示词不许指导它怎么写：%s" % coaching
    for structural in ("语气指令", "语音合成", "不要复述", "不超过 60 字"):
        assert structural in SYSTEM_PROMPT, "产出与边界还要说清楚：%s" % structural


def test_writer_validates_its_client():
    with pytest.raises(TypeError):
        VoiceInstructionWriter(client=object())
    with pytest.raises(ValueError):
        VoiceInstructionWriter(client=FakeLLM(), max_output_tokens=0)
