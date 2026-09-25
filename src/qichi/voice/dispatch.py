"""把"她要说的那一段"变成一条真实的 QQ 语音，并收好每一个失败方向。

时序（2026-09-14 定，见 doc/TTS-实施计划-20260914.md §3.1）：
    文字组发完 -> 【本模块，后台任务】剥离 -> 写语气指令 -> 合成 -> 建事件 -> 派发 -> 颜文字残段

失败方向一律**不吞掉她的话**：
- 剥离后没内容、太长、合成失败 -> 那一段**原样以文字发出**（不是固定兜底）；
- 派发超时/未知 -> 收成 unknown，**永不重发**（Sender 那层保证）；
- 颜文字残段（用户裁定 B）另发一小段文字，不新增任何语义。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.transport.sender import Sender

from .director import VoiceDirector


@dataclass(frozen=True)
class VoiceJob:
    """一次语音投递所需的全部事实；由发送方在文字组发完之后交过来。"""

    group_event_id: str
    # 语音段自己的事件 id（由 Sender.voice_part_event_id 确定性派生）—— 绝不复用组事件 id。
    event_id: str
    part_index: int          # 0 起，只进 metadata，供追溯
    part_count: int
    part_text: str
    conversation_id: str
    owner_qq: str
    situation: str = ""
    reply_target_event_id: str | None = None
    reply_target_platform_message_id: str | None = None
    # 2026-09-20：这一轮由什么触发（dialogue / interaction / initiative）。
    # 语音只是**投递形态**，不是另一种内容——记忆的可靠判据
    # （MemoryWorker._is_reliable）认的就是 generation_metadata.source，
    # 所以这里必须把**真实来源**带下去，不许硬编码。真机 88 条语音因为
    # 缺这个字段而永远进不了记忆，见 doc/问题冻结-20260920-语音进记忆.md。
    source: str = "dialogue"

    def __post_init__(self) -> None:
        if self.source not in {"dialogue", "interaction", "initiative"}:
            raise ValueError("source must be dialogue, interaction, or initiative")


@dataclass(frozen=True)
class VoiceOutcome:
    status: str              # sent / unknown / failed
    reason: str
    event_id: str


class VoiceDispatcher:
    def __init__(
        self,
        *,
        database: Database,
        sender: Sender,
        director: VoiceDirector,
        clock: Callable[[], datetime] | None = None,
    ):
        self.database = database
        self.sender = sender
        self.director = director
        self.events = EventRepository(database)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def deliver(self, job: VoiceJob) -> VoiceOutcome:
        if not isinstance(job, VoiceJob):
            raise TypeError("job must be a VoiceJob")
        timestamp = self._clock()
        part_event_id = job.event_id

        existing = self._existing(part_event_id)
        if existing is not None:
            # 重入：账本里已经有这条，用登记在案的那份音频，绝不重新合成、绝不重发。
            recorded = self._recorded(existing)
            if recorded is None or not Path(str(recorded["file"])).is_file():
                return VoiceOutcome("failed", "recorded_audio_missing", part_event_id)
            event, spoken, residue = existing, existing.text or job.part_text, recorded["residue"]
        else:
            try:
                clip = await self.director.render(job.part_text, situation=job.situation)
            except Exception as error:
                return await self._degrade(job, part_event_id, type(error).__name__)
            event = self._insert_voice_event(part_event_id, job, clip, timestamp)
            spoken, residue = clip.spoken, clip.residue

        status = await self.sender.dispatch_record(
            event=event,
            payload={
                "action_kind": "record",
                "user_id": job.owner_qq,
                "message": [
                    {
                        "type": "record",
                        "data": {"file": Path(self._recorded(event)["file"]).as_uri()},
                    }
                ],
            },
            occurred_at_utc=timestamp,
        )
        if status == "sent" and residue:
            await self._send_residue(job, part_event_id, residue, timestamp)
        return VoiceOutcome(status, "voice", part_event_id)

    # --- 内部 ---

    def _existing(self, event_id: str) -> ConversationEvent | None:
        try:
            return self.events.get(event_id)
        except KeyError:
            return None

    @staticmethod
    def _recorded(event: ConversationEvent) -> Mapping[str, object] | None:
        voice = event.metadata.get("voice") if isinstance(event.metadata, Mapping) else None
        if not isinstance(voice, Mapping) or not isinstance(voice.get("file"), str):
            return None
        return voice

    def _insert_voice_event(self, event_id: str, job: VoiceJob, clip, timestamp: datetime) -> ConversationEvent:
        candidate = ConversationEvent(
            event_id=event_id,
            platform_event_id=None,
            platform_message_id=None,
            conversation_id=job.conversation_id,
            sequence=0,
            direction="outbound",
            actor="qichi",
            kind="text",
            text=job.part_text,
            message_segments=(MessageSegment("record", {"file": Path(clip.path).as_uri()}),),
            reply_to_event_id=job.reply_target_event_id,
            reply_to_platform_message_id=job.reply_target_platform_message_id,
            occurred_at_utc=timestamp,
            received_at_utc=timestamp,
            status="pending",
            metadata={
                "voice": {
                    "spoken": clip.spoken,
                    "residue": clip.residue,
                    "instructions": clip.instructions,
                    "file": str(clip.path),
                    "part_index": job.part_index,
                    "part_count": job.part_count,
                },
                # 只带 source：记忆那一侧只需要它，别的字段由文字组自己承担。
                "generation_metadata": {"source": job.source},
            },
        )
        try:
            return self.events.insert(candidate)
        except Exception:
            return self.events.get(event_id)

    async def _degrade(self, job: VoiceJob, event_id: str, reason: str) -> VoiceOutcome:
        """该段以文字发出——用她自己的原话，不是固定兜底。"""

        candidate = ConversationEvent(
            event_id=event_id,
            platform_event_id=None,
            platform_message_id=None,
            conversation_id=job.conversation_id,
            sequence=0,
            direction="outbound",
            actor="qichi",
            kind="text",
            text=job.part_text,
            message_segments=(MessageSegment("text", {"text": job.part_text}),),
            reply_to_event_id=job.reply_target_event_id,
            reply_to_platform_message_id=job.reply_target_platform_message_id,
            occurred_at_utc=self._clock(),
            received_at_utc=self._clock(),
            status="pending",
            metadata={
                "voice": {"degraded": reason},
                # 降级发出去的同样是她的原话，来源一样要带上，否则这句话进不了记忆。
                "generation_metadata": {"source": job.source},
            },
        )
        try:
            event = self.events.insert(candidate)
        except Exception:
            event = self.events.get(event_id)
        status = await self.sender.dispatch_text(
            event=event,
            payload={
                "action_kind": "text",
                "user_id": job.owner_qq,
                "message": [{"type": "text", "data": {"text": job.part_text}}],
            },
            occurred_at_utc=self._clock(),
        )
        return VoiceOutcome(status, f"degraded:{reason}", event_id)

    async def _send_residue(self, job: VoiceJob, part_event_id: str, residue: str, timestamp: datetime) -> str:
        """裁定 B：被剥下来的颜文字另作一小段文字跟在她那句语音后面。"""

        event_id = str(uuid5(NAMESPACE_URL, f"qichi:voice-residue:{part_event_id}"))
        if self._existing(event_id) is None:
            self.events.insert(
                ConversationEvent(
                    event_id=event_id,
                    platform_event_id=None,
                    platform_message_id=None,
                    conversation_id=job.conversation_id,
                    sequence=0,
                    direction="outbound",
                    actor="qichi",
                    kind="text",
                    text=residue,
                    message_segments=(MessageSegment("text", {"text": residue}),),
                    reply_to_event_id=None,
                    reply_to_platform_message_id=None,
                    occurred_at_utc=timestamp,
                    received_at_utc=timestamp,
                    status="pending",
                    # 有意**不带** generation_metadata：残段只是从语音里剥出来的颜文字，
                    # 而语音事件的 text 里已经含它一次（9393 以「(￣▽￣)」结尾、9394 又是
                    # 「(￣▽￣)」），进记忆会在明细里留一条重复的装饰行。
                    metadata={"voice_residue": {"part_event_id": part_event_id}},
                )
            )
        return await self.sender.dispatch_text(
            event=self.events.get(event_id),
            payload={
                "action_kind": "text",
                "user_id": job.owner_qq,
                "message": [{"type": "text", "data": {"text": residue}}],
            },
            occurred_at_utc=timestamp,
        )
