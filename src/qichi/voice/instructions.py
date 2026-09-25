"""语气指令写手：让 flash 读情境、写一条给合成用的语气描述。

2026-09-14 用户裁定：决定语气的模型**不能是 pro**（分类盲测 5/5、配对试听 3/3，
见 doc/TTS-实施计划-20260914.md §2.30 / §2.34）。

**2026-09-15 用户裁定（现行）：只提供信息，不指导它怎么写。**
起因：§3.1b 曾把「写动作与时间点，不要只写情绪名词」写进提示词，而盲测赢的那批指令
恰恰是「带笑打趣，语速稍快，气息轻巧，句尾微微上扬带俏皮」——**规范把赢的那类词当成了反例**；
同一情境下生产提示词写出的指令机械词 5.00/条、情绪气息词 2.00/条，而盲测那句是 1.50/4.17
（§2.40）。所以提示词只说清**它看到什么、要产出什么**，语气怎么描述归它自己判断。

本模块不判断情绪、不挑通道；它只把情境喂进去、把一条指令拿出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qichi.domain.dialogue import ModelMessage


# 只描述「看到什么、产出什么」。**不写任何写法规范**：写什么语气、要不要写停顿与气息，
# 由 flash 自己从情境里判断（2026-09-15 用户裁定）。
SYSTEM_PROMPT = (
    "你是角色的语音导演。你会看到一段 QQ 私聊的最近往来（括号里是那条消息的钟点与距现在多久），"
    "以及角色这一轮要发出的全部内容，其中标了「←这一句用语音说」的那一段要念出来。"
    "请只为那一段写一条中文的「语气指令」，供语音合成使用。"
    "只输出这条指令本身：不要复述或改写内容，不要加引号，不要解释，不超过 60 字。"
)


@dataclass(frozen=True)
class VoiceInstructionWriter:
    """把 LLM 客户端包成 director 需要的那个异步可调用。"""

    client: object = field(repr=False)
    max_output_tokens: int = 200
    temperature: float = 0.5

    def __post_init__(self) -> None:
        if not callable(getattr(self.client, "generate", None)):
            raise TypeError("client must provide generate")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")

    async def __call__(self, situation: str) -> str | None:
        if not isinstance(situation, str) or not situation.strip():
            return None
        messages = (
            ModelMessage("system", SYSTEM_PROMPT),
            ModelMessage("user", situation.strip() + "\n\n请只输出那条语气指令。"),
        )
        generation = await self.client.generate(
            messages,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            # 显式关思考：这条任务极短，开着只会把预算吃光、返回空内容
            # （2026-09-14 实测过一次，见 §2.29）。
            thinking={"type": "disabled"},
        )
        text = getattr(generation, "text", None)
        if not isinstance(text, str):
            return None
        cleaned = text.strip().strip('"').strip("「」").strip()
        return cleaned or None
