"""语音导演：把她的那句话变成一段落在磁盘上的音频，并把语气指令补上。

分工（2026-09-14 定）：
- **语气指令由 flash 写**，本模块只负责"调用写手、把结果交给合成"；
- 写手失败**不拦住建语音**：没有指令就按默认语气念，而不是整段发不出去；
- 音频按内容寻址缓存：同样的（文本 + 指令）第二次是零调用。

本模块**不判断情绪、不决定这段要不要发语音**——那些归主模型。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .cache import cache_key, cache_path, store_voice
from .synth import SpeechClient, SpeechError
from .text import is_speakable, prepare_speech


class VoiceUnavailable(RuntimeError):
    """这一段现在没法用语音说：剥完没内容、太长，或合成失败。"""


# 输入一段情境文字，输出一条语气指令（或 None 表示不给指令）。
InstructionWriter = Callable[[str], Awaitable[str | None]]


@dataclass(frozen=True)
class VoiceClip:
    spoken: str
    residue: str
    instructions: str | None
    path: Path
    from_cache: bool


@dataclass(frozen=True)
class VoiceDirector:
    client: SpeechClient = field(repr=False)
    data_root: Path
    writer: InstructionWriter | None = field(default=None, repr=False, compare=False)
    max_chars: int = 120
    min_units: int = 1
    container: str = "wav"

    def __post_init__(self) -> None:
        if not isinstance(self.data_root, Path):
            object.__setattr__(self, "data_root", Path(self.data_root))
        if type(self.max_chars) is not int or self.max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        if type(self.min_units) is not int or self.min_units < 1:
            raise ValueError("min_units must be a positive integer")
        if self.container not in {"wav", "mp3"}:
            raise ValueError("container must be wav or mp3")

    async def render(self, text: str, *, situation: str = "") -> VoiceClip:
        """把一段文字变成可发送的音频文件；失败一律抛 VoiceUnavailable。"""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        speech = prepare_speech(text)
        if not is_speakable(speech.spoken, max_chars=self.max_chars, min_units=self.min_units):
            raise VoiceUnavailable("spoken text is empty or too long")

        instructions = None
        if self.writer is not None and situation:
            try:
                candidate = await self.writer(situation)
            except Exception:
                # 写手是辅助调用：它挂了不该让她整段说不出话（2026-09-13 工具层的同一条教训）。
                candidate = None
            if isinstance(candidate, str) and candidate.strip():
                instructions = candidate.strip()

        key = cache_key(
            model=self.client.model,
            voice=self.client.voice,
            text=speech.spoken,
            instructions=instructions,
            sample_rate=self.client.sample_rate,
            container=self.container,
        )
        path = cache_path(self.data_root, key=key, container=self.container)
        if path.is_file():
            return VoiceClip(speech.spoken, speech.residue, instructions, path, True)
        try:
            audio = await self.client.synthesize(speech.spoken, instructions=instructions)
        except SpeechError as error:
            raise VoiceUnavailable(f"speech synthesis failed: {type(error).__name__}") from error
        stored = store_voice(self.data_root, key=key, content=audio, container=self.container)
        return VoiceClip(speech.spoken, speech.residue, instructions, stored, False)
