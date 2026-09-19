"""Minimal G0 assembly for one owner-private dialogue conversation."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from datetime import tzinfo
import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from qichi.dialogue.capability_manifest import (
    MessageSourceFact,
    RuntimeFacts,
    build_capability_manifest,
)
from qichi.dialogue.context_builder import ContextBuildRequest, ContextBuilder
from qichi.dialogue.engine import (
    DialogueEngine,
    DialogueGenerationError,
    DialogueGenerationObservation,
)
from qichi.config import VisionSettings
from qichi.dialogue.llm_client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMImageRejectedError,
    LLMModelNotFoundError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
)
from qichi.domain.dialogue import (
    DialogueInput,
    DialogueOutcome,
    DialogueResult,
    DialogueSkip,
    ModelImage,
    split_voice_part,
)
from qichi.domain.events import ConversationEvent
from qichi.domain.memory_details import MemoryDetailRecord
from qichi.memory.dates import (
    names_a_time_of_day,
    referenced_dates,
    time_of_day_anchors,
    time_of_day_hours,
)
from qichi.media.fetch import STORED, fetch_inbound_images, image_segments
from qichi.memory.index_digest import (
    build_index_line_objects,
    describe_fragment,
    describe_missing_day,
    describe_window,
)
from qichi.memory.retriever import MemoryRetriever
from qichi.observability import TurnTraceRepository
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MAX_DETAILS_PER_FRAGMENT, MemoryDetailRepository
from qichi.storage.memory_repository import MemoryRepository
from qichi.transport.normalizer import NormalizationError, normalize_event
from qichi.transport.quote_resolver import QuoteResolutionError, QuoteResolver
from qichi.transport.sender import Sender


_LOGGER = logging.getLogger(__name__)
_SESSION_GAP = timedelta(minutes=45)
_MEMORY_QUERY_SOURCE_LIMIT = 4_096


_LLM_FAILURE_CATEGORIES = (
    (LLMTimeoutError, "timeout"),
    (LLMConnectionError, "connection"),
    (LLMRateLimitError, "rate_limit"),
    (LLMAuthenticationError, "authentication"),
    (LLMModelNotFoundError, "model_not_found"),
    (LLMRequestError, "request"),
    (LLMServerError, "server"),
    (LLMProtocolError, "protocol"),
)


def _llm_failure_reason(error: LLMError) -> str:
    """Map provider failures to a stable, non-sensitive event category."""
    for error_type, category in _LLM_FAILURE_CATEGORIES:
        if isinstance(error, error_type):
            return f"llm:{category}"
    return "llm:error"


def _decimal_id(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be a decimal identifier")
    normalized = str(value)
    if not normalized or not normalized.isdecimal():
        raise ValueError(f"{field} must be a decimal identifier")
    return normalized


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _executable_catalog_keys(catalog: Mapping[str, int | str]) -> tuple[str, ...]:
    keys: list[str] = []
    for key, platform_id in catalog.items():
        if not isinstance(key, str):
            raise TypeError("catalog keys must be strings")
        try:
            _decimal_id(platform_id, f"catalog.{key}")
        except ValueError:
            continue
        keys.append(key)
    return tuple(sorted(keys))


# The only keyword-driven decision in this project, and its scope is frozen by
# the user-approved exemption recorded in project-pitfalls "P3-2".  It answers
# exactly one question -- did the user explicitly ask to revisit a past episode?
# -- and may only gate the expansion of sensitive detail and verbatim evidence.
# It must never route emotion, tone, topic, intimacy or farewell, and adding a
# word requires explicit user confirmation.
# 2026-09-11 用户裁定：**删掉固定词表**（"细说"/"仔细想想"/"那次"…）。它把"用户
# 怎么说话"变成了"能不能回忆"的前提，限制了对话的自由度。现在的唯一判据是
# 「这条消息本身是否指向某一段」：点了日期、引用了那条消息，或话里的词面真的
# 命中了某段。指向什么就展开什么；什么都指不到，就什么都不展开。

# How far back a single spoken day keeps deciding.  Long enough to cover the
# turns between "就是九号那天呐" and "你再回忆一下那天中午", short enough that a
# day mentioned in a previous topic does not silently govern a later request.
# 冻结计划 §2.1 的 K4：只有**过去**的日期可以从回看窗口继承，而且只看最近 3 条。
# 2026-09-12 在 223 轮真机消息上复算：窗口放到 8 条会把「九号那天」之后的图片问题、
# 闲聊也一起继承（35 轮开门，含「我到床上啦兔子」这种），近 2 条则几乎不触发（23 轮）。
# 「这个对话在说哪一天」只有一个窗口，钥匙（_episode_pointer）和索引钉住
# （_named_day_fragments）共用它。2026-09-12 在 223 轮真机消息上复算：窗口放到 8 条
# 会把「九号那天」之后的图片问题、闲聊也一起继承（35 轮开门，含「我到床上啦兔子」
# 这种），近 2 条则几乎不触发（23 轮）；裁定取 3。
_DAY_LOOKBACK = 3
# 计划 §2.1：只有这三种钥匙能打开成人／explicit_request_only 的原文。
_SENSITIVE_RECALL_KEYS = frozenset({"date_now", "date_inherited", "quote"})
# 计划 §2.1：这几种钥匙只能打开普通明细——点到今天、原样复述一句话，或者上一轮
# 摊错了段、这一轮换一段（T6 的改口）。
_ORDINARY_RECALL_KEYS = frozenset({"today", "verbatim", "correction"})
# 一轮里明细块的 token 上限（2026-09-12 裁定 A：8000）。复算过：点名「前天」时 09-10 的
# 三段合计 94 条 = 13646 token，是日常中位（4771）的三倍；超过这个数就只保留最近的
# 那几段，并把「更早的没展开」写成事实。
#
# 2026-09-17 改成 18000，理由是**单位变了**：用户点名的单位是「一天/一个时段」，
# 而 2026-09-17 的覆盖修复把长夜切成多段（一夜 4 段、每段约 4,400 token），于是
# 8000 只装得下这一段里的**最新一块**——真机上他要核对的那句在第三块，永远进不来，
# 她只能说「我这儿翻不着」（历史诊断 §7）。18000 够装下这样的
# 一整天；仍然装不下时，照旧把「更早的 N 条没有展开」写成事实，不假装完整。
# 排序（fragments_ranked_by_text）在同一天片段多于预算时才起作用。回退：改回 8000。
_DETAIL_TOKEN_BUDGET = 18000

# 2026-09-18 呈现层（历史方案 §5 方案 A，用户裁定 N=3）：
# 常驻三条「她自己记着的事」。**只放 episode**——实测按重要性排出来的全是 preference
# （对她的行为约束），常驻它们等于每轮提醒她该怎样，正是用户最怕的记账化。
# 硬边界：ordinary + daily_safe（成人/亲密内容一律不进常驻层，延续 09-14 裁定）。
# 排序按时间倒序（近的在前），不用 importance。回退：把常量改成 0 即可。
_PINNED_EPISODE_LIMIT = 3
# 点名一天时最多取那天的几段（最新的优先）。真正的裁剪交给 token 预算，这里只
# 防止一天里片段太多时把另一天挤光。
_DAY_FRAGMENT_LIMIT = 8
_DETAIL_BUDGET_NOTE = "（这一轮只展开了最近的 {kept} 条；更早的 {dropped} 条没有展开，需要时请用户点明更早的时间。）"
# 点了日期但那天什么都没存：不展开，并且要说「没有」（T4 才把话说出口）。
_EMPTY_RECALL_KEYS = frozenset({"date_missing"})
# 最近原文足迹（2026-09-12）：常驻「最近几天说过的逐字原话」，起因是用户反馈
# 「角色没法找到原文字段」——转述式追问（不点日期、不引用、非逐字）不给钥匙，
# 明细块不开门，她的常驻层只剩改写。取最近天数 × 每片段头几条，总量封顶。
_FOOTPRINT_DAYS = 2
_FOOTPRINT_PER_FRAGMENT = 4
_FOOTPRINT_LIMIT = 12
# 主动开口单独一套足迹参数（2026-09-15）：这一路的素材只有「他没回」，于是素材不变就复读
# （全库 27 对逐字重复，**27/27 都发生在他一条都没回的时候**）、素材里没有具体内容就只会泛泛问候。
# 给她更多**确实说过的逐字原话**（隐私门不变：仍只取 ordinary + daily_safe），写什么仍归她。
# 普通回复轮沿用上面的常量，逐字不变。
_INITIATIVE_FOOTPRINT_DAYS = 5
_INITIATIVE_FOOTPRINT_PER_FRAGMENT = 6
_INITIATIVE_FOOTPRINT_LIMIT = 24
# 主动开口不占用户等待，所以允许对**超时**多试一次（真机 19:25 那次 90 秒超时，她一个字都没发出来，
# 而他完全不知情；主动开口失败率 14%）。热路径的重试策略完全不动：用户在那儿等着。
_INITIATIVE_TIMEOUT_RETRIES = 1


@dataclass(frozen=True, slots=True)
class _EpisodePointer:
    """One turn's answer to "which episode did the user point at, and how".

    key is the authorization half (plan §2.1): date_now / date_missing / today /
    quote / verbatim / date_inherited / none.  days and fragments are the
    localization half, resolved once here so that the gate and the locator can
    never disagree about which day or which episode they are talking about.
    """

    key: str
    match_count: int = 0
    days: tuple[date, ...] = ()
    fragments: tuple[str, ...] = ()
    # 这些片段只放行普通明细。同一条消息里既点了过去的日子、又带时段词点了今天时，
    # 今天只拿到最窄的权限（2026-09-12 T8，裁定 #5 不变）。
    ordinary_only: tuple[str, ...] = ()


def _past_days(days: tuple[date, ...], today: date) -> tuple[date, ...]:
    """The days in that list that are actually behind us.

    One definition, used by every look-back decision in this module: "今天" is how
    anyone talks about the live conversation, and a day that has not happened yet
    is not a memory.
    """

    return tuple(day for day in days if day < today)


def _spoken_text(history: tuple[ConversationEvent, ...]) -> tuple[str, ...]:
    """The user's own words, newest last.  Her replies are never a clue."""

    return tuple(
        event.text
        for event in history
        if event.direction == "inbound" and event.actor == "mumo" and event.text
    )
_DETAIL_REMAINDER_NOTE = "（本片段另有 {count} 条原文未展开；需要时请用户指出具体内容再回顾。）"

_POLICIES = ("daily_safe", "topic_only", "explicit_request_only")
_PRIVACY = ("ordinary", "intimate", "adult")


def sensitive_detail_allowed(*, privacy_class: str, recall_policy: str, explicit_recall: bool) -> bool:
    """Execute the frozen recall matrix one record at a time.

    The policy decides how far a record may be used: daily_safe anywhere,
    topic_only once the topic actually matched, explicit_request_only only on an
    explicit request or a quote that lands on the record's own evidence.  Adult
    preferences and agreements default to topic_only, so they stay usable when
    the topic really comes up; adult episodes are explicit_request_only and stay
    closed otherwise.  Unknown privacy values, unknown policies and the
    forbidden adult/daily_safe combination all fail closed.
    """
    if privacy_class not in _PRIVACY or recall_policy not in _POLICIES:
        return False
    if privacy_class == "adult" and recall_policy == "daily_safe":
        return False
    if recall_policy == "explicit_request_only":
        return explicit_recall
    return True


def transitive_quote_chain(first, lookup, *, max_depth: int = 3):
    """Follow the reply chain upward, bounded, cycle-safe and single-conversation.

    Quoting is a platform-verified fact: every hop is an exact message link, so
    following it needs no semantic guess.  A chain never crosses conversations
    and never revisits an event.
    """
    if first is None:
        return ()
    chain = []
    seen = set()
    event = first
    while event is not None and len(chain) < max_depth and event.event_id not in seen:
        if event.conversation_id != first.conversation_id:
            break
        seen.add(event.event_id)
        chain.append(event)
        target = event.reply_to_event_id
        event = lookup(target) if target else None
    return tuple(chain)


def _advance_image_carry(
    state: dict[str, tuple],
    *,
    conversation_id: str,
    event_id: str,
    fresh: tuple,
    carry_turns: int,
) -> tuple[tuple, str | None]:
    """宽限窗口状态机（纯函数，好测）：这一轮该挂哪张图、窗口怎么走。

    2026-09-14 真机教训：一轮太短。「先问她能不能查 → 用户过几分钟才回『你查一下』」
    之间隔了六轮，图早就不在窗口里了，她只能拿文字去搜描述，等于白查。

    返回（这一轮要挂的图，图的来源事件 id）。**新图轮的来源就是本轮**（调用方手上已经有它），
    所以只有宽限重挂才回一个 id——那一轮必须把「这是更早那条里的同一张」说出来，否则一张
    没有来由的图会被读成「他又发了一张」（2026-09-16 真机，见历史诊断）。
    """

    if fresh:
        state[conversation_id] = (event_id, tuple(fresh), carry_turns)
        return tuple(fresh), None
    entry = state.get(conversation_id)
    if entry is None:
        return (), None
    stored_event_id, images, remaining = entry
    if stored_event_id == event_id:
        # 同一条事件的二次装配（视觉档被拒后的文字回退）：不挂图，也不消耗窗口。
        return (), None
    if remaining <= 0:
        state.pop(conversation_id, None)
        return (), None
    if remaining == 1:
        state.pop(conversation_id, None)
    else:
        state[conversation_id] = (stored_event_id, images, remaining - 1)
    return tuple(images), stored_event_id


class G0Application:
    """Connect durable owner input to the one dialogue engine and sender."""

    def __init__(
        self,
        database: Database,
        context_builder: ContextBuilder,
        dialogue_engine: DialogueEngine,
        onebot_client: Any,
        *,
        owner_qq: int | str,
        bot_qq: int | str,
        role_core: str,
        clock: Callable[[], datetime] | None = None,
        event_id_factory: Callable[[], str] | None = None,
        outbound_event_id_factory: Callable[[], str] | None = None,
        face_catalog: Mapping[str, int | str] | None = None,
        reaction_catalog: Mapping[str, int | str] | None = None,
        memory_worker: Any | None = None,
        get_msg: Callable[[str], ConversationEvent | None] | None = None,
        get_msg_async: Callable[[str], Awaitable[ConversationEvent | None]] | None = None,
        vision: VisionSettings | None = None,
        get_image_async: Callable[[str], Awaitable[Mapping[str, Any]]] | None = None,
        # 联网工具（2026-09-14）：给了才可能有多一轮；不给时行为与以前逐字一致。
        tool_runner: Any | None = None,
        # 语音（2026-09-14）：给了工厂才建投递器（它需要 Sender）；不给时没有语音能力。
        voice_dispatcher_factory: Callable[[Any], Any] | None = None,
        # 一段最多多少字（2026-09-15 §2.42）：作为**边界事实**进能力行。声明可用就必须
        # 同时给出边界——她两次写了 127 字、超过 120 就静默退回文字，而这边只看到"又没发语音"。
        voice_max_chars: int | None = None,
        image_carry_turns: int = 1,
        media_root: str | Path | None = None,
        # 2026-09-14 用户裁定：主动开口这一路的思考档单独给（见 config.initiative.thinking）。
        # 默认 disabled ⇒ 不给这一项时行为与以前逐字一致。
        initiative_thinking: str = "disabled",
        auto_quote_current_message: bool = False,
        always_include_memory_types: tuple[str, ...] = ("agreement", "correction"),
        memory_candidate_limit: int = 24,
        memory_context_limit: int = 12,
        local_zone: tzinfo | None = None,
    ) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        if not callable(getattr(context_builder, "build", None)):
            raise TypeError("context_builder must provide build")
        if not callable(getattr(dialogue_engine, "generate", None)):
            raise TypeError("dialogue_engine must provide generate")
        if not isinstance(role_core, str) or not role_core.strip():
            raise ValueError("role_core must be non-empty text")
        if memory_worker is not None and not callable(
            getattr(memory_worker, "notify_reliable_activity", None)
        ):
            raise TypeError("memory_worker must provide notify_reliable_activity")
        for factory, field in (
            (clock, "clock"),
            (event_id_factory, "event_id_factory"),
            (outbound_event_id_factory, "outbound_event_id_factory"),
            (get_msg, "get_msg"),
            (get_msg_async, "get_msg_async"),
            (get_image_async, "get_image_async"),
        ):
            if factory is not None and not callable(factory):
                raise TypeError(f"{field} must be callable")
        if vision is not None and not isinstance(vision, VisionSettings):
            raise TypeError("vision must be VisionSettings or None")
        if media_root is not None and not isinstance(media_root, (str, Path)):
            raise TypeError("media_root must be a path or None")
        if tool_runner is not None and (
            not callable(getattr(tool_runner, "declares", None))
            or not callable(getattr(tool_runner, "run", None))
        ):
            raise TypeError("tool_runner must provide declares and run")
        if voice_dispatcher_factory is not None and not callable(voice_dispatcher_factory):
            raise TypeError("voice_dispatcher_factory must be callable")
        if initiative_thinking not in {"default", "enabled", "disabled"}:
            raise ValueError("initiative_thinking must be default, enabled or disabled")
        if type(auto_quote_current_message) is not bool:
            raise TypeError("auto_quote_current_message must be a bool")
        if (
            not isinstance(always_include_memory_types, tuple)
            or not always_include_memory_types
            or not all(isinstance(item, str) and item for item in always_include_memory_types)
            or len(set(always_include_memory_types)) != len(always_include_memory_types)
        ):
            raise TypeError("always_include_memory_types must be a tuple of unique non-empty strings")

        self.database = database
        self.owner_qq = _decimal_id(owner_qq, "owner_qq")
        self.bot_qq = _decimal_id(bot_qq, "bot_qq")
        if self.owner_qq == self.bot_qq:
            raise ValueError("owner_qq and bot_qq must differ")
        self.role_core = role_core
        self.context_builder = context_builder
        self.dialogue_engine = dialogue_engine
        self.events = EventRepository(database)
        self.memories = MemoryRepository(database)
        self.memory_details = MemoryDetailRepository(database)
        self.memory_retriever = MemoryRetriever(
            database,
            candidate_limit=memory_candidate_limit,
            context_limit=memory_context_limit,
        )
        self.turn_traces = TurnTraceRepository(database)
        self.memory_worker = memory_worker
        self._get_msg = get_msg or (lambda _message_id: None)
        self._get_msg_async = get_msg_async
        self.vision = vision
        self._get_image_async = get_image_async
        self._media_root = None if media_root is None else Path(media_root)
        # Pictures landed for a turn that must not be kept are removed when the
        # turn ends, not while the model may still be reading them.
        self._turn_media: dict[str, tuple[Path, ...]] = {}
        self._tool_runner = tool_runner
        # 宽限窗口：上一轮带图，这一轮再把那几张从本地存档挂一次（图可以多挂一轮）。
        self._carried_images: dict[str, tuple] = {}
        self._image_carry_turns = image_carry_turns
        # 每一轮的工具决定与执行结果，供生成阶段与 trace 使用。
        self._turn_tools: dict[str, tuple[tuple[Mapping[str, Any], ...], Path | None]] = {}
        self._turn_tool_usage: dict[str, dict[str, Any]] = {}
        self.auto_quote_current_message = auto_quote_current_message
        self.always_include_memory_types = always_include_memory_types
        self.quotes = QuoteResolver(database)
        self.sender = Sender(
            database,
            onebot_client,
            face_catalog=face_catalog or {},
            reaction_catalog=reaction_catalog or {},
        )
        # 语音投递器要用到 Sender，所以由调用方以工厂形式交进来；不给就是没有语音。
        self._voice_dispatcher = (
            voice_dispatcher_factory(self.sender) if voice_dispatcher_factory is not None else None
        )
        if self._voice_dispatcher is not None and not callable(
            getattr(self._voice_dispatcher, "deliver", None)
        ):
            raise TypeError("voice dispatcher must provide deliver")
        self._voice_tasks: set[Any] = set()
        self.face_catalog = dict(face_catalog or {})
        self.reaction_catalog = dict(reaction_catalog or {})
        self._available_face_keys = _executable_catalog_keys(self.face_catalog)
        self._available_reaction_keys = _executable_catalog_keys(self.reaction_catalog)
        # 只有投递器真的建起来了才声明语音能力：宁可不声明，也不能让她说出做不到的事。
        self._voice_available = self._voice_dispatcher is not None
        if voice_max_chars is not None and (type(voice_max_chars) is not int or voice_max_chars < 1):
            raise TypeError("voice_max_chars must be a positive int or None")
        # 没有语音能力时不带边界事实：宁可少一句，也不能给一个用不上的数字。
        self._voice_max_chars = voice_max_chars if self._voice_available else None
        self._initiative_thinking = initiative_thinking
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Local wall-clock zone for the neutral memory index; the process zone is
        # the honest default when no configured zone is supplied.
        self.local_zone = local_zone or datetime.now().astimezone().tzinfo
        self._event_id_factory = event_id_factory or (lambda: str(uuid4()))
        self._outbound_event_id_factory = outbound_event_id_factory or (lambda: str(uuid4()))
        self._semantic_locks: dict[str, asyncio.Lock] = {}
        self._response_observations: dict[str, dict[str, object]] = {}
        # 上一轮真的摊开过哪些片段（T6 的改口路径要它）：只活在进程内存里，
        # 重启后为空，那也只是回到「没有可改口的上文」。
        self._last_detail_fragments: dict[str, tuple[str, ...]] = {}

    async def handle_onebot(
        self,
        raw_payload: object,
        *,
        received_at_utc: datetime | None = None,
    ) -> ConversationEvent | None:
        received = _aware_utc(
            self._clock() if received_at_utc is None else received_at_utc,
            "received_at_utc",
        )
        try:
            normalized = normalize_event(
                raw_payload,
                bot_qq=self.bot_qq,
                owner_qq=self.owner_qq,
                received_at_utc=received,
                event_id_factory=self._event_id_factory,
            )
        except NormalizationError as error:
            # The payload is not a trustworthy owner event, so there is no
            # safe conversation row to reply to. Keep the consumer alive and
            # expose only the stable failure class in logs.
            _LOGGER.warning("inbound payload rejected during normalization (%s)", type(error).__name__)
            return None
        if normalized is None or normalized.direction != "inbound":
            return None
        if normalized.kind == "poke":
            normalized = replace(normalized, platform_event_id=self._poke_identity(raw_payload))

        persisted, context_version = self._persist_and_advance(normalized, raw_payload)
        if context_version is None or persisted.kind not in {"text", "poke"}:
            return None
        source = "interaction" if persisted.kind == "poke" else "dialogue"
        try:
            self._append_trace(
                persisted,
                source=source,
                phase="received",
                occurred_at_utc=persisted.received_at_utc,
                details={
                    "context_version": context_version,
                    "sequence": persisted.sequence,
                    "event_kind": persisted.kind,
                    "has_quote_target": persisted.reply_to_platform_message_id is not None,
                },
            )
        except Exception:
            # The event is already durable even if observability is degraded;
            # retain its memory window before surfacing the trace failure.
            if persisted.kind in {"text", "poke"}:
                self._notify_memory_activity(persisted.conversation_id)
            raise
        lock = self._semantic_locks.setdefault(persisted.conversation_id, asyncio.Lock())
        try:
            async with lock:
                if self._context_version(persisted.conversation_id) != context_version:
                    self._record_cancelled_trace(persisted, source, "context_version_changed")
                    if persisted.kind == "poke":
                        self._mark_processed(persisted.conversation_id, persisted.sequence)
                    return None
                if persisted.kind == "poke":
                    await self.sender.send_poke(
                        persisted,
                        owner_qq=self.owner_qq,
                        occurred_at_utc=_aware_utc(self._clock(), "clock result"),
                    )
                    if self._context_version(persisted.conversation_id) != context_version:
                        self._record_cancelled_trace(persisted, source, "newer_input_after_poke")
                        self._mark_processed(persisted.conversation_id, persisted.sequence)
                        return None
                try:
                    dialogue_input = await self._build_input(persisted, context_version, source=source)
                except Exception as error:
                    self._record_processing_failure(
                        persisted,
                        "context:assembly",
                        stage="context",
                        source=source,
                    )
                    await self._emit_failure_notice(persisted, "context:assembly")
                    _LOGGER.error("inbound context assembly failed (%s)", type(error).__name__)
                    return None
                generation_started = time.monotonic()
                try:
                    dialogue_input, outcome, generation_observation = await self._generate_turn(
                        persisted,
                        context_version,
                        source=source,
                        dialogue_input=dialogue_input,
                    )
                except DialogueGenerationError as error:
                    self._record_generation_failure(
                        persisted,
                        error.reason,
                        generation_started,
                        source=source,
                        observation=error.observation,
                    )
                    await self._emit_failure_notice(persisted, error.reason)
                    return None
                except LLMError as error:
                    # Provider failures must converge exactly like structural
                    # failures.  Otherwise the WS consumer survives while this
                    # durable inbound event remains an unprocessed blocker.
                    self._record_generation_failure(
                        persisted,
                        _llm_failure_reason(error),
                        generation_started,
                        source=source,
                    )
                    await self._emit_failure_notice(persisted, _llm_failure_reason(error))
                    return None
                if not isinstance(outcome, DialogueResult):
                    self._record_processing_failure(
                        persisted,
                        "dialogue:invalid_result",
                        stage="generation",
                        source=source,
                    )
                    await self._emit_failure_notice(persisted, "dialogue:invalid_result")
                    _LOGGER.error("dialogue engine returned an invalid outcome")
                    return None
                self._record_generation_trace(
                    persisted,
                    source,
                    outcome,
                    generation_started,
                    generation_observation,
                )
                if (
                    self.auto_quote_current_message
                    and persisted.kind == "text"
                    and dialogue_input.quoted_target is not None
                    and self._usable_reply_target(persisted.conversation_id, outcome.reply_target) is None
                ):
                    # Preserve the user's explicit QQ quote as the native reply
                    # target unless the model selected another valid target.
                    outcome = replace(outcome, reply_target=dialogue_input.quoted_target.handle)
                if self._context_version(persisted.conversation_id) != context_version:
                    self._record_cancelled_trace(persisted, source, "newer_input_after_generation")
                    if persisted.kind == "poke":
                        self._mark_processed(persisted.conversation_id, persisted.sequence)
                    return None

                def dialogue_dispatch_guard(connection: Any) -> bool:
                    cursor = connection.execute(
                        "SELECT context_version FROM conversation_cursors "
                        "WHERE conversation_id = ?",
                        (persisted.conversation_id,),
                    ).fetchone()
                    if cursor is None or cursor["context_version"] != context_version:
                        return False
                    newer_inbound = connection.execute(
                        "SELECT 1 FROM conversation_events WHERE conversation_id = ? "
                        "AND direction = 'inbound' AND status = 'received' "
                        "AND sequence > ? LIMIT 1",
                        (persisted.conversation_id, persisted.sequence),
                    ).fetchone()
                    return newer_inbound is None

                # 语音段从这一轮里摘出去之后，剩下的文字段原样走今天那条链路；
                # 语音交给后台，排在文字之后（§3.1）。
                voice_part: tuple[int, int, str] | None = None
                # 2026-09-14 用户裁定：整轮只有一段、而那一段被标成语音时，**就只发语音**。
                # 旧行为（§2.40 的"退回文字"）让她唯一那种"一句话"的回复永远发不出声音，
                # 而且账本上完全看不出她试过。合成失败的方向不在这里兜：
                # VoiceDispatcher._degrade 会把那一段原样以文字发出，绝不吞话。
                voice_only = False
                # 语气写手要看的「这一轮完整分句」（2026-09-15）：语音段常常接在她自己
                # 刚打出去的文字后面，只给它一句就写不出承接。
                reply_parts = tuple(
                    part for part in (outcome.message_parts or (outcome.text or "",))
                    if isinstance(part, str)
                )
                if self._voice_available:
                    text_outcome, spoken = split_voice_part(outcome)
                    if text_outcome is None:
                        voice_only = True
                        text_outcome = outcome
                    if spoken:
                        voice_part = (
                            outcome.voice_part_index - 1,
                            len(outcome.message_parts or (outcome.text,)),
                            spoken,
                        )
                else:
                    # 语音没就绪：**整条当文字发**。绝不因为摘出去而让她的话消失
                    #（能力没声明过，她本来也不该吐这个标记；这一支是防御性的）。
                    text_outcome = outcome
                group_event_id = self._new_id(self._outbound_event_id_factory, "outbound_event_id_factory")
                if voice_only:
                    # 只有语音的一轮没有文字组，派发守卫没有地方交给 Sender，所以在动手前自己查一次。
                    if not dialogue_dispatch_guard(self.database.connection):
                        self._record_cancelled_trace(persisted, source, "dispatch_guard_rejected")
                        if persisted.kind == "poke":
                            self._mark_processed(persisted.conversation_id, persisted.sequence)
                        return None
                    spoken_event = await self._deliver_voice_only(
                        persisted=persisted,
                        group_event_id=group_event_id,
                        voice_part=voice_part,
                        parts=reply_parts,
                        voice_index=outcome.voice_part_index,
                    )
                    if spoken_event is not None:
                        self._response_observations.pop(persisted.event_id, None)
                        self._record_delivery_trace(persisted, source, spoken_event, outcome)
                        self._mark_processed(persisted.conversation_id, persisted.sequence)
                        return spoken_event
                    # 投递器整个没跑起来：退回文字，她的原话一个字都不少。
                    _LOGGER.warning("voice_only_turn_degraded (dispatcher_failed)")
                delivered = await self.sender.send(
                    text_outcome,
                    event_id=group_event_id,
                    conversation_id=persisted.conversation_id,
                    owner_qq=self.owner_qq,
                    occurred_at_utc=_aware_utc(self._clock(), "clock result"),
                    dispatch_guard=dialogue_dispatch_guard,
                    generation_metadata=self._response_observations.pop(persisted.event_id, None),
                )
                if delivered is None:
                    self._record_cancelled_trace(persisted, source, "dispatch_guard_rejected")
                    if persisted.kind == "poke":
                        self._mark_processed(persisted.conversation_id, persisted.sequence)
                    return None
                self._record_delivery_trace(persisted, source, delivered, outcome)
                self._mark_processed(persisted.conversation_id, persisted.sequence)
                if voice_part is not None and self._voice_available:
                    self._schedule_voice_delivery(
                        persisted=persisted,
                        group_event_id=group_event_id,
                        part_index=voice_part[0],
                        part_count=voice_part[1],
                        part_text=voice_part[2],
                        parts=reply_parts,
                        voice_index=outcome.voice_part_index,
                    )
                return delivered
        finally:
            # Every durable text is eligible for the post-send memory window,
            # including failed, cancelled, and unexpectedly interrupted turns.
            # Poke notifications retain the worker's existing filter contract:
            # they may wake a scan for earlier text, but _is_reliable excludes
            # the poke event itself from memory evidence.
            self._discard_turn_media(persisted.event_id)
            if persisted.kind in {"text", "poke"}:
                self._notify_memory_activity(persisted.conversation_id)

    def _voice_situation(
        self,
        persisted: ConversationEvent,
        spoken: str,
        *,
        parts: tuple[str, ...] | None = None,
        voice_index: int | None = None,
        initiative: bool = False,
    ) -> str:
        """给语气写手的一段情境。

        2026-09-15 用户裁定（方向 1）：以前只给「要念的那一句」，写手看不到她自己刚打出去的
        文字段、也看不到时间差，只能孤立地念。拼装归 qichi.voice.situation（纯函数，可单测）；
        这里只负责取出最近往来、交出时钟。导入仍是局部的：voice.enabled 关着时这条链路不存在。
        """

        from qichi.voice.situation import build_situation

        # 必须带上他刚说的那一句：语气写手看不到当前输入就写不出对的语气。
        recent = tuple(self._events_before(persisted)) + (persisted,)
        return build_situation(
            events=recent,
            now=_aware_utc(self._clock(), "clock result"),
            local_zone=self.local_zone,
            spoken=spoken,
            parts=parts,
            voice_index=voice_index,
            initiative=initiative,
        )

    def _build_voice_job(
        self,
        *,
        persisted: ConversationEvent,
        group_event_id: str,
        part_index: int,
        part_count: int,
        part_text: str,
        parts: tuple[str, ...] | None = None,
        voice_index: int | None = None,
        initiative: bool = False,
    ) -> Any:
        from qichi.voice.dispatch import VoiceJob

        return VoiceJob(
            group_event_id=group_event_id,
            event_id=Sender.voice_part_event_id(group_event_id, part_index),
            part_index=part_index,
            part_count=part_count,
            part_text=part_text,
            conversation_id=persisted.conversation_id,
            owner_qq=self.owner_qq,
            situation=self._voice_situation(
                persisted, part_text,
                parts=parts, voice_index=voice_index, initiative=initiative,
            ),
        )

    def _schedule_voice_delivery(
        self,
        *,
        persisted: ConversationEvent,
        group_event_id: str,
        part_index: int,
        part_count: int,
        part_text: str,
        parts: tuple[str, ...] | None = None,
        voice_index: int | None = None,
        initiative: bool = False,
    ) -> None:
        """文字组发完之后才把语音交给后台 —— 绝不阻塞文字回复（用户硬要求）。"""

        job = self._build_voice_job(
            persisted=persisted,
            group_event_id=group_event_id,
            part_index=part_index,
            part_count=part_count,
            part_text=part_text,
            parts=parts,
            voice_index=voice_index,
            initiative=initiative,
        )
        task = asyncio.get_running_loop().create_task(self._deliver_voice(job))
        self._voice_tasks.add(task)
        task.add_done_callback(self._voice_tasks.discard)

    async def _deliver_voice_only(
        self,
        *,
        persisted: ConversationEvent,
        group_event_id: str,
        voice_part: tuple[int, int, str] | None,
        parts: tuple[str, ...] | None = None,
        voice_index: int | None = None,
        initiative: bool = False,
    ) -> ConversationEvent | None:
        """整轮只有语音：不发文字组，就地等这一句说完（用户 2026-09-14 裁定）。

        返回真正发出去的那条事件（语音；合成失败时是投递器降级发出的那条文字）。
        投递器整个没跑起来、或平台没能把这条语音送出去时返回 None，
        由调用方退回文字路径 —— 她这一轮绝不静默。
        """

        if voice_part is None or not self._voice_available:
            return None
        job = self._build_voice_job(
            persisted=persisted,
            group_event_id=group_event_id,
            part_index=voice_part[0],
            part_count=voice_part[1],
            part_text=voice_part[2],
            parts=parts,
            voice_index=voice_index,
            initiative=initiative,
        )
        result = await self._deliver_voice(job)
        if result is None:
            return None
        if result.status != "sent" and not str(result.reason).startswith("degraded:"):
            # 平台没把它送出去、也没有降级文字：这一轮就真的一个字都没有了。
            return None
        try:
            return self.events.get(result.event_id)
        except KeyError:
            _LOGGER.error("voice-only turn was sent but its event is missing (%s)", result.event_id)
            return None

    async def _deliver_voice(self, job: Any) -> Any:
        try:
            outcome = await self._voice_dispatcher.deliver(job)
        except Exception:
            # 语音是旁路：失败只记日志，绝不改已经发出去的文字。
            _LOGGER.exception("voice delivery failed")
            return None
        _LOGGER.info("voice delivery %s (%s)", outcome.status, outcome.reason)
        return outcome

    def _initiative_attempts_since_last_reply(
        self, conversation_id: str
    ) -> tuple[int, datetime | None]:
        """已发出的主动开口次数，以及最近一次的时间。

        2026-09-14 用户裁定：他上一次没有回应时，下一次主动开口要把这件事带进上下文。
        只统计「上一条用户消息之后、由主动尝试触发的已发送消息」，正常一问一答不算。
        """
        row = self.database.connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) AS last_sequence FROM conversation_events "
            "WHERE conversation_id = ? AND direction = 'inbound' AND status = 'received'",
            (conversation_id,),
        ).fetchone()
        start = -1 if row is None else int(row["last_sequence"])
        rows = self.database.connection.execute(
            "SELECT event_id FROM conversation_events WHERE conversation_id = ? "
            "AND sequence > ? ORDER BY sequence",
            (conversation_id, start),
        ).fetchall()
        pending = False
        count = 0
        last_at: datetime | None = None
        for item in rows:
            event = self.events.get(item["event_id"])
            if event.direction == "internal" and event.kind == "initiative":
                pending = True
            elif pending and event.direction == "outbound" and event.status == "sent":
                count += 1
                last_at = event.occurred_at_utc
                pending = False
        return count, last_at

    async def generate_initiative(
        self,
        conversation_id: str,
        expected_context_version: int,
        activity_at_utc: datetime,
        attempt_at_utc: datetime | None = None,
        claim_id: str | None = None,
        outbound_event_id: str | None = None,
        expected_claim_state_json: str | None = None,
        dispatch_allowed_at: Callable[[datetime], bool] | None = None,
    ) -> DialogueOutcome | None:
        """Run the optional initiative attempt through the normal context/engine/sender path."""
        if conversation_id != self.owner_qq:
            raise ValueError("initiative conversation must be the owner conversation")
        if claim_id is not None and (
            outbound_event_id is None or expected_claim_state_json is None
        ):
            raise ValueError("persisted initiative claim requires its complete lease identity")
        if dispatch_allowed_at is not None and not callable(dispatch_allowed_at):
            raise TypeError("dispatch_allowed_at must be callable")
        lock = self._semantic_locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            trigger_event_id = claim_id or self._new_id(self._event_id_factory, "event_id_factory")
            outbound_id = outbound_event_id or self._new_id(
                self._outbound_event_id_factory, "outbound_event_id_factory"
            )
            if claim_id is not None:
                try:
                    existing = self.events.get(trigger_event_id)
                except KeyError:
                    existing = None
                if existing is not None:
                    if (
                        existing.conversation_id != conversation_id
                        or existing.direction != "internal"
                        or existing.actor != "platform"
                        or existing.kind != "initiative"
                    ):
                        raise ValueError("initiative trigger identity conflict")
                    return None
            if self._context_version(conversation_id) != expected_context_version:
                return None
            row = self.database.connection.execute(
                "SELECT event_id FROM conversation_events WHERE conversation_id = ? "
                "AND direction = 'inbound' AND status = 'received' ORDER BY sequence DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            anchor = _aware_utc(activity_at_utc, "activity_at_utc")
            cursor = self.database.connection.execute(
                "SELECT context_version, last_user_activity_utc, presence_topic_cursor "
                "FROM conversation_cursors "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if (
                cursor is None
                or cursor["context_version"] != expected_context_version
                or cursor["last_user_activity_utc"] != anchor.isoformat()
            ):
                return None
            topic_cursor = cursor["presence_topic_cursor"]
            if topic_cursor is not None and (
                type(topic_cursor) is not int or topic_cursor < 0
            ):
                raise ValueError("presence_topic_cursor is invalid")
            if topic_cursor is not None:
                topic_cursor_event = self.database.connection.execute(
                    "SELECT 1 FROM conversation_events WHERE conversation_id = ? "
                    "AND sequence = ? AND direction = 'inbound' AND actor = 'mumo' "
                    "AND status = 'received'",
                    (conversation_id, topic_cursor),
                ).fetchone()
                if topic_cursor_event is None:
                    raise ValueError("presence_topic_cursor has no reliable user event")
            attempt = _aware_utc(
                self._clock() if attempt_at_utc is None else attempt_at_utc,
                "attempt_at_utc",
            )
            history_rows = self.database.connection.execute(
                "SELECT event_id FROM conversation_events WHERE conversation_id = ? "
                "AND direction != 'internal' AND ((direction = 'inbound' AND status = 'received') "
                "OR (direction = 'outbound' AND status = 'sent')) AND sequence > ? "
                "ORDER BY sequence",
                (conversation_id, -1 if topic_cursor is None else topic_cursor),
            ).fetchall()
            topic_history = tuple(
                self.events.get(item["event_id"]) for item in history_rows
            )
            history = self._session_history(
                topic_history,
                boundary_at_utc=attempt,
            )
            topic_events = tuple(
                event
                for event in topic_history
                if event.direction == "inbound" and event.actor == "mumo"
            )
            topic_watermark = max(
                (event.sequence for event in topic_events), default=None
            )
            retrieval_query = "\n".join(
                event.text for event in topic_events if event.text is not None
            )
            # 2026-09-14 用户裁定：他上一次没有回应时，下一次主动开口要把这件事带进上下文。
            # 计数只在「上一条用户消息之后、且由主动尝试触发的已发送消息」上进行，
            # 所以正常一问一答不会被算成主动开口。
            previous_attempts, initiative_previous_at = self._initiative_attempts_since_last_reply(
                conversation_id
            )
            initiative_attempt_index = previous_attempts + 1
            initiative_event = ConversationEvent(
                event_id=trigger_event_id,
                platform_event_id=None,
                platform_message_id=None,
                conversation_id=conversation_id,
                sequence=0,
                direction="internal",
                actor="platform",
                kind="initiative",
                text=None,
                message_segments=(),
                reply_to_event_id=None,
                reply_to_platform_message_id=None,
                occurred_at_utc=attempt,
                received_at_utc=attempt,
                status="received",
                metadata={
                    "activity_at_utc": anchor.isoformat(),
                    "context_version": expected_context_version,
                    "outbound_event_id": outbound_id,
                    "presence_topic_cursor_before": topic_cursor,
                    "topic_watermark": topic_watermark,
                },
            )
            initiative_event = self.events.insert(initiative_event)
            self._append_trace(
                initiative_event,
                source="initiative",
                phase="received",
                occurred_at_utc=initiative_event.received_at_utc,
                details={
                    "context_version": expected_context_version,
                    "sequence": initiative_event.sequence,
                    "event_kind": "initiative",
                    "has_quote_target": False,
                },
            )
            try:
                dialogue_input = await self._build_input(
                    initiative_event,
                    expected_context_version,
                    source="initiative",
                    now=attempt,
                    recent_events=history,
                    current_event_handle=f"I{initiative_event.sequence}",
                    retrieval_query=retrieval_query,
                    initiative_attempt_index=initiative_attempt_index,
                    initiative_previous_at=initiative_previous_at,
                    footprint_days=_INITIATIVE_FOOTPRINT_DAYS,
                    footprint_per_fragment=_INITIATIVE_FOOTPRINT_PER_FRAGMENT,
                    footprint_limit=_INITIATIVE_FOOTPRINT_LIMIT,
                )
            except Exception:
                self._append_trace(
                    initiative_event,
                    source="initiative",
                    phase="failure",
                    details={"stage": "context", "failure_category": "context:assembly"},
                )
                raise
            generation_started = time.monotonic()
            # 2026-09-15：这一路不占用户等待，超时允许再试一次（见 _INITIATIVE_TIMEOUT_RETRIES）。
            timeout_retries = 0
            while True:
                try:
                    # 2026-09-14 用户裁定：只有主动开口这一路显式要思考。关思考的 pro 不做
                    # 「打出来还是说出来」这种元决策（实测 0/11 次选用语音），开思考 4/4 次会
                    # 主动发声；而这一路不占用户等回复的时间。
                    outcome, generation_observation = await self._generate(
                        dialogue_input,
                        thinking=(
                            {"type": self._initiative_thinking}
                            if self._initiative_thinking != "default"
                            else None
                        ),
                    )
                    break
                except LLMTimeoutError as error:
                    if timeout_retries >= _INITIATIVE_TIMEOUT_RETRIES:
                        details = dict(self._generation_failure_details(
                            _llm_failure_reason(error), generation_started, None))
                        details["timeout_retries"] = timeout_retries
                        self._append_trace(
                            initiative_event, source="initiative", phase="failure", details=details,
                        )
                        raise
                    timeout_retries += 1
                    _LOGGER.warning(
                        "initiative generation timed out; retrying once (retry=%d)", timeout_retries
                    )
                    continue
                except DialogueGenerationError as error:
                    self._append_trace(
                        initiative_event,
                        source="initiative",
                        phase="failure",
                        details=self._generation_failure_details(
                            error.reason,
                            generation_started,
                            error.observation,
                        ),
                    )
                    raise
                except LLMError as error:
                    self._append_trace(
                        initiative_event,
                        source="initiative",
                        phase="failure",
                        details=self._generation_failure_details(
                            _llm_failure_reason(error),
                            generation_started,
                            None,
                        ),
                    )
                    raise
            if not isinstance(outcome, (DialogueResult, DialogueSkip)):
                self._append_trace(
                    initiative_event,
                    source="initiative",
                    phase="failure",
                    details={
                        "stage": "generation",
                        "failure_category": "dialogue:invalid_result",
                    },
                )
                raise TypeError("initiative engine must return a dialogue outcome")
            self._record_generation_trace(
                initiative_event,
                "initiative",
                outcome,
                generation_started,
                generation_observation,
            )
            cursor = self.database.connection.execute(
                "SELECT context_version, last_user_activity_utc FROM conversation_cursors "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if (
                cursor is None
                or cursor["context_version"] != expected_context_version
                or cursor["last_user_activity_utc"] != anchor.isoformat()
            ):
                self.events.update_status(initiative_event.event_id, "cancelled")
                self._record_cancelled_trace(
                    initiative_event, "initiative", "conversation_changed_after_generation"
                )
                return None
            if isinstance(outcome, DialogueSkip):
                self.events.update_status(initiative_event.event_id, "skipped")
                self._append_trace(
                    initiative_event,
                    source="initiative",
                    phase="delivery",
                    details={"status": "skipped", "outbound_event_id": None},
                )
                return outcome
            send_at = max(_aware_utc(self._clock(), "clock result"), attempt)

            def initiative_dispatch_guard(connection: Any) -> bool:
                if dispatch_allowed_at is not None:
                    allowed_now = dispatch_allowed_at(
                        _aware_utc(self._clock(), "clock result")
                    )
                    if type(allowed_now) is not bool:
                        raise TypeError("dispatch_allowed_at must return a bool")
                    if not allowed_now:
                        return False
                cursor = connection.execute(
                    "SELECT context_version, last_user_activity_utc FROM conversation_cursors "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if (
                    cursor is None
                    or cursor["context_version"] != expected_context_version
                    or cursor["last_user_activity_utc"] != anchor.isoformat()
                ):
                    return False
                newer_inbound = connection.execute(
                    "SELECT 1 FROM conversation_events WHERE conversation_id = ? "
                    "AND direction = 'inbound' AND status = 'received' AND sequence > ? LIMIT 1",
                    (conversation_id, initiative_event.sequence),
                ).fetchone()
                if newer_inbound is not None:
                    return False
                if claim_id is None:
                    return True
                state_row = connection.execute(
                    "SELECT value_json FROM runtime_meta WHERE key = ?",
                    (f"initiative:{conversation_id}",),
                ).fetchone()
                if state_row is None:
                    return False
                return bool(
                    expected_claim_state_json is not None
                    and state_row["value_json"] == expected_claim_state_json
                )

            def consume_topic_evidence(connection: Any) -> None:
                if topic_watermark is None:
                    return
                updated = connection.execute(
                    "UPDATE conversation_cursors SET presence_topic_cursor = "
                    "CASE WHEN presence_topic_cursor IS NULL OR presence_topic_cursor < ? "
                    "THEN ? ELSE presence_topic_cursor END WHERE conversation_id = ?",
                    (topic_watermark, topic_watermark, conversation_id),
                )
                if updated.rowcount != 1:
                    raise ValueError("initiative topic cursor is unavailable")

            # 2026-09-14：语音段以前只接在对话热路径上，主动开口这一路完全没接 ——
            # 她就算吐了 [[qq:voice:N]] 也只会被当文字发出去（回放实测 [text, text]）。
            voice_part: tuple[int, int, str] | None = None
            voice_only = False
            reply_parts = tuple(
                part for part in (getattr(outcome, "message_parts", None) or (getattr(outcome, "text", None) or "",))
                if isinstance(part, str)
            )
            if self._voice_available and isinstance(outcome, DialogueResult):
                text_outcome, spoken = split_voice_part(outcome)
                if text_outcome is None:
                    voice_only = True
                    text_outcome = outcome
                if spoken:
                    voice_part = (
                        outcome.voice_part_index - 1,
                        len(outcome.message_parts or (outcome.text,)),
                        spoken,
                    )
            else:
                text_outcome = outcome
            if voice_only:
                # 只有语音的一轮没有文字组，派发守卫与话题游标都得自己收。
                if not initiative_dispatch_guard(self.database.connection):
                    self.events.update_status(initiative_event.event_id, "cancelled")
                    self._record_cancelled_trace(
                        initiative_event, "initiative", "dispatch_guard_rejected"
                    )
                    return None
                spoken_event = await self._deliver_voice_only(
                    persisted=initiative_event,
                    group_event_id=outbound_id,
                    voice_part=voice_part,
                    parts=reply_parts,
                    voice_index=getattr(outcome, "voice_part_index", None),
                    initiative=True,
                )
                if spoken_event is not None:
                    with self.database.connection:
                        consume_topic_evidence(self.database.connection)
                    self._response_observations.pop(initiative_event.event_id, None)
                    self._record_delivery_trace(
                        initiative_event, "initiative", spoken_event, outcome
                    )
                    self.events.update_status(
                        initiative_event.event_id,
                        "processed" if spoken_event.status == "sent" else spoken_event.status,
                    )
                    return spoken_event
                # 投递器整个没跑起来：退回文字，她的原话一个字都不少。
                _LOGGER.warning("voice_only_initiative_degraded (dispatcher_failed)")
            delivered = await self.sender.send(
                text_outcome,
                event_id=outbound_id,
                conversation_id=conversation_id,
                owner_qq=self.owner_qq,
                occurred_at_utc=send_at,
                dispatch_guard=initiative_dispatch_guard,
                sent_commit_hook=consume_topic_evidence,
                generation_metadata=self._response_observations.pop(initiative_event.event_id, None),
            )
            if delivered is None:
                self.events.update_status(initiative_event.event_id, "cancelled")
                self._record_cancelled_trace(
                    initiative_event, "initiative", "dispatch_guard_rejected"
                )
                return None
            self._record_delivery_trace(
                initiative_event,
                "initiative",
                delivered,
                outcome,
            )
            self.events.update_status(
                initiative_event.event_id,
                "processed" if delivered.status == "sent" else delivered.status,
            )
            if voice_part is not None and self._voice_available:
                # 文字组发完之后才交给后台 —— 与对话热路径同一套时序。
                self._schedule_voice_delivery(
                    persisted=initiative_event,
                    group_event_id=outbound_id,
                    part_index=voice_part[0],
                    part_count=voice_part[1],
                    part_text=voice_part[2],
                    parts=reply_parts,
                    voice_index=getattr(outcome, "voice_part_index", None),
                    initiative=True,
                )
            return delivered

    def _persist_and_advance(
        self, event: ConversationEvent, raw_payload: object
    ) -> tuple[ConversationEvent, int | None]:
        with self.database.transaction() as connection:
            persisted, created = self.events.insert_in_transaction(
                connection, event, raw_payload=raw_payload
            )
            if not created:
                return persisted, None
            timestamp = persisted.received_at_utc.isoformat()
            row = connection.execute(
                "INSERT INTO conversation_cursors "
                "(conversation_id, context_version, last_user_activity_utc) VALUES (?, 1, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "context_version = conversation_cursors.context_version + 1, "
                "last_user_activity_utc = excluded.last_user_activity_utc "
                "RETURNING context_version",
                (persisted.conversation_id, timestamp),
            ).fetchone()
        return persisted, int(row["context_version"])

    def _sensitive_admitted(self, record, quoted_event_ids, explicit_recall: bool) -> bool:
        """Admit one sensitive record: a quote that lands on its own evidence counts."""
        by_quote = any(evidence.event_id in quoted_event_ids for evidence in record.memory_evidence)
        return sensitive_detail_allowed(
            privacy_class=record.privacy_class,
            recall_policy=record.recall_policy,
            explicit_recall=explicit_recall or by_quote,
        )

    async def _current_images(
        self, current: ConversationEvent, *, skip: bool = False
    ) -> tuple[tuple[ModelImage, ...], dict[str, object]]:
        """Fetch this turn's pictures; never guess what any of them shows.

        The switch is the whole gate: with vision off nothing is downloaded and
        the facts stay exactly as they were.  Once it is on, a picture that
        cannot be obtained is reported as such, so the model is told nothing was
        seen instead of being left to imagine it.
        """

        segments = image_segments(current.message_segments)
        details: dict[str, object] = {
            "image_segments": len(segments),
            "images_attached": 0,
            "images_unexpanded": 0,
            "images_unavailable": 0,
        }
        if not segments:
            return (), details
        settings = self.vision
        if skip:
            details["vision_skipped"] = "disabled_for_this_turn"
            details["images_unavailable"] = len(segments)
            return (), details
        if settings is None or not settings.enabled:
            details["vision_skipped"] = "switch_off"
            return (), details
        if self._get_image_async is None or self._media_root is None:
            # Switched on without a way to fetch: say so rather than let the
            # envelope claim a picture was provided.
            details["vision_skipped"] = "transport_unavailable"
            details["images_unavailable"] = min(len(segments), settings.max_images_per_turn)
            return (), details
        result = await fetch_inbound_images(
            current.message_segments,
            event_id=current.event_id,
            data_root=self._media_root,
            get_image=self._get_image_async,
            max_images=settings.max_images_per_turn,
            max_bytes=settings.max_image_bytes,
            timeout_seconds=settings.download_timeout_seconds,
        )
        images = tuple(
            ModelImage(path=str(item.path), content_type=item.content_type)
            for item in result.outcomes
            if item.status == STORED and item.path is not None and item.content_type is not None
        )
        details["images_attached"] = len(images)
        details["images_unexpanded"] = result.overflow
        details["images_unavailable"] = len(result.failures)
        # Statuses are names, never content: the trace must not carry what a
        # picture showed.
        details["image_statuses"] = [item.status for item in result.outcomes]
        if images and not settings.keep_original_images:
            self._turn_media[current.event_id] = tuple(Path(image.path) for image in images)
        return images, details

    def _discard_turn_media(self, event_id: str) -> None:
        """Drop originals the configuration says not to keep; never the ledger."""

        paths = self._turn_media.pop(event_id, ())
        if self.vision is not None and self.vision.keep_original_images:
            return
        for path in paths:
            try:
                path.unlink()
            except OSError:
                continue

    def _turn_tool_decision(
        self, current: ConversationEvent, *, source: str, fresh: int,
        carried_paths: tuple[Path, ...],
    ) -> tuple[tuple[Mapping[str, Any], ...], Path | None]:
        """这一轮声明哪些工具（纯策略见 net.tools.tool_plan），并记在这次事件名下。"""

        from qichi.net import tool_plan  # 局部导入：没开联网时不引入这条链路

        enabled, with_image = tool_plan(
            enabled=self._tool_runner is not None, source=source,
            fresh_images=fresh, carried_images=len(carried_paths),
        )
        declared: tuple[Mapping[str, Any], ...] = (
            self._tool_runner.declares(with_image=with_image) if enabled else ()
        )
        image_path = carried_paths[0] if (with_image and carried_paths) else None
        self._turn_tools[current.event_id] = (declared, image_path)
        return declared, image_path

    async def _build_input(
        self,
        current: ConversationEvent,
        context_version: int,
        *,
        source: str = "dialogue",
        skip_images: bool = False,
        now: datetime | None = None,
        recent_events: tuple[ConversationEvent, ...] | None = None,
        current_event_handle: str | None = None,
        retrieval_query: str | None = None,
        initiative_attempt_index: int = 1,
        initiative_previous_at: datetime | None = None,
        # 足迹窗口（2026-09-15）：默认沿用常驻值；主动开口那一套传更宽的窗口，
        # 因为它的素材只有「他没回」，素材不变就会复读（见 _INITIATIVE_FOOTPRINT_*）。
        footprint_days: int = _FOOTPRINT_DAYS,
        footprint_per_fragment: int = _FOOTPRINT_PER_FRAGMENT,
        footprint_limit: int = _FOOTPRINT_LIMIT,
    ) -> DialogueInput:
        context_started = time.monotonic()
        quoted = None
        quoted_chain: tuple[ConversationEvent, ...] = ()
        quote_resolution_status = "none"
        if current.reply_to_platform_message_id is not None:
            quote_resolution_status = "unavailable"
            if self._get_msg_async is not None:
                quoted = await self.quotes.resolve_async(current.event_id, self._get_msg_async)
            else:
                quoted = self.quotes.resolve(current.event_id, self._get_msg)
            if quoted is not None:
                quoted_chain = transitive_quote_chain(self.events.get(quoted.event_id), self.events.get)
                quote_resolution_status = "resolved"

        now = _aware_utc(self._clock() if now is None else now, "clock result")
        previous = self._previous_event(current)
        seconds_since_last = None
        if previous is not None:
            seconds_since_last = max(0, int((now - previous.occurred_at_utc).total_seconds()))
        fresh_images, image_details = await self._current_images(current, skip=skip_images)
        # 宽限窗口（2026-09-13 用户裁定「图片可以多挂一轮」）：上一轮带图，这一轮再从
        # 本地存档挂一次。图不在这一轮重新下载，也不改变「新图到达轮不给工具」那条规则——
        # 她要先问用户要不要去查，用户同意后的那一轮才拿得到图搜工具。
        carried_images, carried_from_event_id = _advance_image_carry(
            self._carried_images,
            conversation_id=current.conversation_id,
            event_id=current.event_id,
            fresh=tuple(fresh_images),
            carry_turns=self._image_carry_turns,
        )
        # 宽限重挂：把图的来源事件取出来，作为事实交给模型（见 capability_manifest）。
        # 注意 carried_images 是「这一轮要挂的图」，新图轮里它就是刚到的那些；
        # carried_count 才是「其中来自更早那条消息的」——只有它才需要说明来源。
        carried_source = (
            self.events.get(carried_from_event_id)
            if carried_from_event_id is not None
            else None
        )
        carried_count = len(carried_images) if carried_from_event_id is not None else 0
        if carried_count:
            image_details = {**dict(image_details), "images_carried": carried_count}
        # _advance_image_carry 的返回值就是「这一轮该挂的图」：新图轮是这一轮新到的，
        # 宽限轮是存档里那张，过期/同事件重装都是空——不要再把 fresh 加一遍。
        current_images = tuple(carried_images)
        tools, image_path = self._turn_tool_decision(
            current, source=source, fresh=len(fresh_images),
            carried_paths=tuple(Path(image.path) for image in carried_images),
        )
        history = self._events_before(current) if recent_events is None else recent_events
        # The user's last message before this one: if it carried a picture, the
        # pixels are gone this turn but the fact must survive.
        previous_image_message = False
        for event in reversed(history):
            if event.direction == "inbound" and event.actor == "mumo":
                previous_image_message = any(
                    segment.type == "image" for segment in event.message_segments
                )
                break
        facts = RuntimeFacts(
            current_time=now,
            seconds_since_last_message=seconds_since_last,
            current_source=self._source_fact(current),
            quoted_source=self._source_fact(quoted_chain[0]) if quoted_chain else None,
            quote_resolution_status=quote_resolution_status,
            received_media=self._received_media(current),
            available_actions=(
                ("text", "reply")
                + (("qq_face",) if self._available_face_keys else ())
                + (
                    ("reaction",)
                    if self._available_reaction_keys
                    and source != "initiative"
                    and current.kind != "poke"
                    and current.platform_message_id is not None
                    else ()
                )
                + (("voice",) if self._voice_available else ())
            ),
            voice_available=self._voice_available,
            voice_max_chars=self._voice_max_chars,
            available_qq_face_keys=self._available_face_keys,
            available_reaction_keys=tuple(
                self._available_reaction_keys
                if self._available_reaction_keys
                and source != "initiative"
                and current.kind != "poke"
                and current.platform_message_id is not None
                else ()
            ),
            vision_available=bool(current_images),
            images_attached=len(current_images),
            images_carried=carried_count,
            carried_image_source=(
                self._source_fact(carried_source) if carried_source is not None else None
            ),
            images_unexpanded=int(image_details["images_unexpanded"]),
            images_unavailable=int(image_details["images_unavailable"]),
            # 图真的挂在这一轮时，「上一条是图片消息、本轮你看不到那张图」这句就是假的
            # （宽限窗口内像素一直在手上）——两句事实不能互相打架，重挂那一轮由上面那行说话。
            previous_image_message=previous_image_message and not carried_count,
            external_tools_available=False,
            initiative_attempt=source == "initiative",
            initiative_attempt_index=initiative_attempt_index,
            initiative_previous_at=initiative_previous_at,
        )
        active_memories = self.memories.list_active(current.conversation_id, now)
        base_query = (current.text or "") if retrieval_query is None else retrieval_query
        retrieval_terms = self._memory_query_terms(base_query, quoted_chain)
        retrieved = self.memory_retriever.retrieve(
            current.conversation_id,
            "\n".join(retrieval_terms),
            now,
            query_terms=retrieval_terms or None,
            include_confirmation=True,
            # Sensitive records may come back as candidates so that topic_only
            # ones can match a real topic, but finding them is not permission to
            # use them: admission happens per record below, and ContextBuilder
            # keeps its own gate as defence in depth.
            include_sensitive=True,
        )
        # A lexical hit is not a request.  Sensitive evidence and detail open
        # one record at a time: either the user asked explicitly (frozen narrow
        # vocabulary) or the quote chain landed on that record's own evidence.
        quoted_event_ids = {item.event_id for item in quoted_chain}
        # 2026-09-12 T2：授权与定位分开。「钥匙」只说这一轮指没指、用什么指的；它能
        # 打开什么由计划 §2.1 决定。三字窗口命中不再是钥匙（问题冻结 P1）。
        pointer = self._episode_pointer(current, history, quoted_event_ids, now)
        sensitive_recall = pointer.key in _SENSITIVE_RECALL_KEYS
        detail_recall = sensitive_recall or pointer.key in _ORDINARY_RECALL_KEYS
        retrieved_sensitive = tuple(
            record
            for record in (*retrieved.candidates, *retrieved.confirmation_candidates)
            if record.privacy_class != "ordinary" or record.recall_policy != "daily_safe"
        )

        def _admitted(record) -> bool:
            return self._sensitive_admitted(record, quoted_event_ids, sensitive_recall)
        allow_sensitive_memory = any(_admitted(record) for record in retrieved_sensitive)
        blocked_sensitive_events = {
            evidence.event_id
            for record in retrieved_sensitive
            if not _admitted(record)
            for evidence in record.memory_evidence
        }
        sensitive_ids = {
            record.memory_id
            for record in (*retrieved.context_candidates, *retrieved.confirmation_candidates)
            if (record.privacy_class != "ordinary" or record.recall_policy != "daily_safe")
            and _admitted(record)
        }
        context_active_memories = tuple(
            record
            for record in active_memories
            if record.privacy_class == "ordinary" or record.memory_id in sensitive_ids
        )
        relationship_state = tuple(
            record
            for record in context_active_memories
            if record.type in self.always_include_memory_types
        ) + self._remembered_episodes(context_active_memories)
        relationship_ids = {record.memory_id for record in relationship_state}
        memory_candidates = tuple(
            record
            for record in retrieved.context_candidates
            if record.memory_id not in relationship_ids
            and (
                (record.privacy_class == "ordinary" and record.recall_policy == "daily_safe")
                or _admitted(record)
            )
        )
        # Confirmation candidates are a separate source and are capped to one
        # item; they never enter the active relationship-state block.
        confirmation = retrieved.confirmation_candidates[:1] if (
            source != "initiative" and current.direction == "inbound" and
            current.actor == "mumo" and current.kind == "text"
        ) else ()
        # A confirmation candidate is still a sensitive record: it must pass the
        # same per-record admission as everything else, otherwise admitting one
        # record would smuggle a different, unadmitted one past the gate.
        confirmation = tuple(
            record
            for record in confirmation
            if (record.privacy_class == "ordinary" and record.recall_policy == "daily_safe")
            or _admitted(record)
        )
        memory_candidates = tuple(record for record in memory_candidates if record.status == "active")
        evidence_events = {
            event_id: item
            for event_id, item in retrieved.evidence_events.items()
            if event_id not in blocked_sensitive_events
        }
        for record in active_memories:
            # Rebuilding the evidence pool must not undo the block above: a
            # sensitive record that was not admitted contributes no evidence.
            if (
                record.privacy_class != "ordinary" or record.recall_policy != "daily_safe"
            ) and not _admitted(record):
                continue
            for evidence in record.memory_evidence:
                evidence_events[evidence.event_id] = self.events.get(evidence.event_id)
        sensitive_event_ids = tuple(
            evidence.event_id
            for record in (*retrieved.context_candidates, *retrieved.confirmation_candidates)
            if (record.privacy_class != "ordinary" or record.recall_policy != "daily_safe")
            and _admitted(record)
            for evidence in record.memory_evidence
        )
        # Details are verbatim and therefore only ever open on an explicit request
        # or a quote that lands on the episode's own evidence -- never on a lexical
        # coincidence.  Which episode opens is decided here: the quoted episode, the
        # episode the running conversation names, and only as a last resort the most
        # recent one, which the block then has to declare as a guess.
        detail_fragments, detail_note, detail_reason = self._detail_targets(pointer)
        # 点了日期但那天什么都没存：把「没有」当成一条事实交给她，而不是让她在
        # 空白里自己圆（计划 §2.3）。措辞由模型决定，代码只给这条事实。
        recall_note = (
            describe_missing_day(pointer.days[0])
            if pointer.key in _EMPTY_RECALL_KEYS and pointer.days
            else ""
        )
        # 明细按证据事件兜底查询只属于引用这一条路：引用点到某条记录的原始消息，
        # 而那条消息不落在任何片段里。别的钥匙不许借这条路（否则「指向不唯一」的
        # 那一次会被 topic_only 的记录顺手填满）。
        quote_fallback = sensitive_event_ids if pointer.key == "quote" else ()
        if detail_recall and (detail_fragments or quote_fallback):
            memory_details = self.memory_details.list_details(
                current.conversation_id,
                query="",
                explicit_request=True,
                event_ids=None if detail_fragments else (quote_fallback or None),
                fragment_ids=detail_fragments or None,
            )
        else:
            memory_details = ()
        # 逐条准入用的是**敏感钥匙**，不是「能展开」这件事本身：点到今天或原样
        # 复述一句话只放行普通明细（计划 §2.1）。同一条消息点了两天时，权限按片段
        # 分天算——今天的那些片段即使跟在过去的日子后面，也只有普通明细（T8）。
        ordinary_only = set(pointer.ordinary_only)
        memory_details = tuple(
            detail
            for detail in memory_details
            if sensitive_detail_allowed(
                privacy_class=detail.privacy_class,
                recall_policy=detail.recall_policy,
                explicit_recall=(
                    sensitive_recall and detail.fragment_id not in ordinary_only
                ),
            )
        )
        memory_details, detail_note = self._cap_details(memory_details, detail_note)
        # 「什么都没展开」不等于「没有指向」：点了空日子（day_missing）和指向不唯一
        # （ambiguous）都必须留在轨迹里，否则这两条规则在面板上永远看不见（T4）。
        if not memory_details and detail_reason not in {"day_missing", "ambiguous"}:
            detail_reason = "none"
        # The index is built last so it can tell the truth about this turn: a
        # fragment whose quotes are in the context must not still be labelled
        # "细节未展开" (2026-09-11: she read the label and denied the content that
        # was sitting right below it).
        expanded_details: dict[str, int] = {}
        privacy_by_fragment: dict[str, list[str]] = {}
        for detail in memory_details:
            expanded_details[detail.fragment_id] = expanded_details.get(detail.fragment_id, 0) + 1
            privacy_by_fragment.setdefault(detail.fragment_id, []).append(detail.privacy_class)
        detail_labels: dict[str, str] = {}
        if expanded_details:
            seats = tuple(expanded_details)
            placeholders = ",".join("?" for _ in seats)
            for row in self.database.connection.execute(
                "SELECT fragment_id, started_at_utc, ended_at_utc FROM memory_fragments "
                f"WHERE fragment_id IN ({placeholders})",
                seats,
            ):
                detail_labels[row["fragment_id"]] = describe_fragment(
                    row["started_at_utc"],
                    row["ended_at_utc"],
                    privacy_by_fragment.get(row["fragment_id"], ()),
                    self.local_zone,
                )
        # 裁定 A（T6）：一轮的明细块有 token 上限，超出时保留最近的那几段，并把
        # 「更早的没展开」写成事实。顺序放在标签之后——预算按真实渲染量算。
        memory_details, detail_note = self._budget_details(
            memory_details, detail_note, detail_labels, priority=detail_fragments
        )
        expanded_details = {}
        privacy_by_fragment = {}
        for detail in memory_details:
            expanded_details[detail.fragment_id] = expanded_details.get(detail.fragment_id, 0) + 1
            privacy_by_fragment.setdefault(detail.fragment_id, []).append(detail.privacy_class)
        if not memory_details and detail_reason not in {"day_missing", "ambiguous"}:
            detail_reason = "none"
        pinned = tuple(
            dict.fromkeys((*expanded_details, *self._named_day_fragments(current, history, now)))
        )
        index_objects = build_index_line_objects(
            self.database.connection,
            conversation_id=current.conversation_id,
            now=now,
            local_zone=self.local_zone,
            expanded_details=expanded_details,
            pinned_fragment_ids=pinned,
        )
        index_lines = tuple(line.text for line in index_objects)
        # 2026-09-12 T1：这一轮摊开的每一段，索引里是不是真的有它那一行？
        # 说 false 就是审计里那条缺陷（2c7ca819 展开后 82 字符超过 80 的行上限，
        # 整行被丢，而它的 32 条原文照常注入）。
        indexed_fragments = {line.fragment_id for line in index_objects}
        memory_detail_indexed = set(expanded_details) <= indexed_fragments
        for detail in memory_details:
            for evidence in detail.evidence:
                evidence_events[evidence.event_id] = self.events.get(evidence.event_id)
        # 最近原文足迹（2026-09-12 用户反馈「没法找到原文字段」）：常驻一段「最近几天
        # 说过的逐字原话」。她的常驻层此前只有改写（工作集）。隐私门与召回门一致：
        # 只看 ordinary + daily_safe 的片段与明细，成人与非日常内容不进日常上下文。
        footprint_details = self.memory_details.footprint_details(
            current.conversation_id,
            since=now - timedelta(days=footprint_days),
            per_fragment=footprint_per_fragment,
            limit=footprint_limit,
        )
        footprint_labels: dict[str, str] = {}
        if footprint_details:
            seats = tuple(dict.fromkeys(item.fragment_id for item in footprint_details))
            placeholders = ",".join("?" for _ in seats)
            for row in self.database.connection.execute(
                "SELECT fragment_id, started_at_utc, ended_at_utc FROM memory_fragments "
                f"WHERE fragment_id IN ({placeholders})",
                seats,
            ):
                footprint_labels[row["fragment_id"]] = describe_fragment(
                    row["started_at_utc"], row["ended_at_utc"], ("ordinary",), self.local_zone
                )
        built = self.context_builder.build(
            ContextBuildRequest(
                role_core=self.role_core,
                runtime_facts=facts,
                current_event=current,
                quoted_chain=quoted_chain,
                recent_events=history,
                relationship_state=relationship_state,
                memory_working_set=context_active_memories,
                memory_candidates=tuple(record for record in memory_candidates if record.status == "active"),
                confirmation_candidates=confirmation,
                earlier_events=(),
                evidence_events=evidence_events,
                memory_details=memory_details,
                allow_sensitive_memory=allow_sensitive_memory,
                memory_index=index_lines,
                memory_recall_note=recall_note,
                memory_detail_note=detail_note,
                allow_sensitive_details=sensitive_recall,
                current_images=current_images,
                memory_detail_labels=detail_labels,
                memory_footprint=footprint_details,
                memory_footprint_labels=footprint_labels,
            )
        )
        if confirmation and confirmation[0].memory_id not in built.metrics.selected_memory_ids:
            confirmation = ()
            built = self.context_builder.build(ContextBuildRequest(
                role_core=self.role_core, runtime_facts=facts, current_event=current,
                quoted_chain=quoted_chain, recent_events=history,
                relationship_state=relationship_state, memory_working_set=context_active_memories,
                memory_candidates=memory_candidates,
                confirmation_candidates=(), earlier_events=(), evidence_events=evidence_events,
                memory_details=memory_details,
                allow_sensitive_memory=allow_sensitive_memory,
                memory_index=index_lines,
                memory_recall_note=recall_note,
                memory_detail_note=detail_note,
                allow_sensitive_details=sensitive_recall,
                current_images=current_images,
                memory_detail_labels=detail_labels,
                memory_footprint=footprint_details,
                memory_footprint_labels=footprint_labels,
            ))
        elif confirmation and source != "initiative" and current.direction == "inbound" and current.actor == "mumo" and current.kind == "text":
            fragment_key = hashlib.sha256((history[0].event_id if history else current.event_id).encode()).hexdigest()
            if not self.memories.try_record_confirmation_presentation(
                conversation_id=current.conversation_id, memory_id=confirmation[0].memory_id,
                fragment_key=fragment_key, trigger_event_id=current.event_id,
                context_version=context_version, presented_at_utc=now,
            ):
                confirmation = ()
                built = self.context_builder.build(ContextBuildRequest(
                    role_core=self.role_core, runtime_facts=facts, current_event=current,
                    quoted_chain=quoted_chain, recent_events=history,
                    relationship_state=relationship_state, memory_working_set=context_active_memories,
                    memory_candidates=memory_candidates,
                    confirmation_candidates=(), earlier_events=(), evidence_events=evidence_events,
                    memory_details=memory_details,
                    allow_sensitive_memory=allow_sensitive_memory,
                    memory_index=index_lines,
                memory_recall_note=recall_note,
                    memory_detail_note=detail_note,
                    allow_sensitive_details=sensitive_recall,
                    current_images=current_images,
                    memory_detail_labels=detail_labels,
                    memory_footprint=footprint_details,
                    memory_footprint_labels=footprint_labels,
                ))
        self._append_trace(
            current,
            source=source,
            phase="context",
            details={
                "context_version": context_version,
                "context_ms": round(max(0.0, (time.monotonic() - context_started) * 1000), 3),
                "input_tokens": built.metrics.input_tokens,
                "input_budget_tokens": built.metrics.input_budget_tokens,
                "window_tokens": built.metrics.window_tokens,
                "expanded": built.metrics.expanded,
                "category_tokens": dict(built.metrics.category_tokens),
                "omitted_counts": dict(built.metrics.omitted_counts),
                "selected_history_event_ids": list(built.metrics.selected_history_event_ids),
                "working_set_memory_ids": list(
                    built.metrics.selected_working_memory_ids
                ),
                "selected_memory_ids": list(built.metrics.selected_memory_ids),
                "relationship_memory_ids": sorted(relationship_ids),
                "retrieval_candidate_ids": [record.memory_id for record in retrieved.candidates],
                "memory_scores": {
                    record.memory_id: retrieved.scores[record.memory_id]
                    for record in memory_candidates if record.memory_id in retrieved.scores
                },
                "memory_reasons": {
                    record.memory_id: retrieved.reasons[record.memory_id]
                    for record in memory_candidates if record.memory_id in retrieved.reasons
                },
                "confirmation_memory_ids": [record.memory_id for record in confirmation],
                "retrieval_mode": retrieved.search_mode,
                "retrieval_degraded": retrieved.degraded_reason is not None,
                "retrieval_query_source_count": len(retrieval_terms),
                "quote_resolution_status": quote_resolution_status,
                "quoted_event_id": quoted_chain[0].event_id if quoted_chain else None,
                # Which rule picked the episode, and what it picked.  This is the
                # difference between "nothing was retrieved" and "something was
                # retrieved and not used", which is otherwise invisible after the
                # fact (2026-09-11: the detail block was filled with the wrong
                # episode for three turns before anyone could see it).
                "memory_detail_reason": detail_reason,
                # 这一轮**实际摊开**的片段（预算裁过之后），不是钥匙点到的那一批：
                # 面板上要回答的是「她这回看到了哪几段」（T6 加预算之后两者会不同）。
                "memory_detail_fragments": list(expanded_details),
                "memory_detail_count": len(memory_details),
                # T1（2026-09-12）：冻结计划 §2.1 的钥匙**本该**是什么，写在运行中的
                # 判据旁边。这里只记录、不做决定——两套判据不一致的地方要先在轨迹里
                # 看得见，T2 才会把其中任何一条变成真的。
                "memory_detail_key": pointer.key,
                # 这一轮有没有把「那天没有记录」当成事实写进上下文（T4）。只记种类，
                # 不记措辞——措辞归模型。
                "memory_recall_note": "missing_day" if recall_note else None,
                # K3（逐字原话 ≥8 字）命中了几个片段；0 表示这条钥匙没响。
                "memory_detail_match_count": pointer.match_count,
                "memory_detail_indexed": memory_detail_indexed,
                # What happened to this turn's pictures, in names only.  This is
                # the difference between "there was no image" and "the image was
                # not obtained", which decides whether she may describe anything.
                "image_segments": image_details["image_segments"],
                "images_attached": image_details["images_attached"],
                # 2026-09-16：面板的字段白名单里一直有 images_carried，但这里从没写过它，
                # 于是「这一轮的图是新到的还是重挂的」在轨迹里查不出来——我据此误判过一次。
                "images_carried": image_details.get("images_carried", 0),
                "images_unexpanded": image_details["images_unexpanded"],
                "images_unavailable": image_details["images_unavailable"],
                "image_statuses": image_details.get("image_statuses", []),
                "vision_skipped": image_details.get("vision_skipped"),
            },
        )
        # 改口路径要的是「上一轮摊开过什么」（T6）：这一轮结束后记下来。
        self._last_detail_fragments[current.conversation_id] = tuple(expanded_details)
        self._response_observations[current.event_id] = {
            "source": source, "context_version": context_version,
            "relationship_memory_ids": tuple(sorted(record.memory_id for record in relationship_state)),
            "working_set_memory_ids": tuple(built.metrics.selected_working_memory_ids),
            "retrieved_memory_ids": tuple(sorted(record.memory_id for record in memory_candidates)),
            "candidate_memory_ids": tuple(sorted(record.memory_id for record in memory_candidates)),
            "quoted_event_id": quoted_chain[0].event_id if quoted_chain else None,
            "history_event_count": len(history),
        }
        return DialogueInput(
            conversation_id=current.conversation_id,
            trigger_event_ids=(current.event_id,),
            current_event_handle=current_event_handle or current.visible_handle,
            quoted_target=quoted,
            context_version=context_version,
            role_messages=built.messages,
            current_time=now,
            platform_capabilities=build_capability_manifest(facts),
            source=source,
        )

    async def _generate_turn(
        self,
        current: ConversationEvent,
        context_version: int,
        *,
        source: str,
        dialogue_input: DialogueInput,
    ) -> tuple[DialogueInput, DialogueOutcome, DialogueGenerationObservation | None]:
        """Generate once, and once more without pictures if the endpoint refuses them.

        Measured 2026-09-11: the endpoint accepts image input, so this is a
        defensive path.  The rebuilt turn re-derives its facts, which means the
        envelope truthfully says nothing was seen instead of the model being
        told about a picture it never received.
        """

        tools, image_path = self._turn_tools.pop(current.event_id, ((), None))
        try:
            outcome, observation = await self._generate(
                dialogue_input, tools=tools, image_path=image_path, event_id=current.event_id
            )
        except LLMImageRejectedError as rejection:
            self._append_trace(
                current,
                source=source,
                phase="generation",
                details={"vision_degraded": "endpoint_rejected"},
            )
            try:
                fallback = await self._build_input(
                    current, context_version, source=source, skip_images=True
                )
            except Exception:
                raise rejection
            outcome, observation = await self._generate(
                fallback, tools=tools, image_path=None, event_id=current.event_id
            )
            return fallback, outcome, observation
        return dialogue_input, outcome, observation

    async def _generate(
        self,
        dialogue_input: DialogueInput,
        *,
        tools: tuple[Mapping[str, Any], ...] = (),
        image_path: Path | None = None,
        event_id: str | None = None,
        thinking: Mapping[str, str] | None = None,
    ) -> tuple[DialogueOutcome, DialogueGenerationObservation | None]:
        observed_generate = getattr(self.dialogue_engine, "generate_with_observation", None)
        if not callable(observed_generate):
            return await self.dialogue_engine.generate(dialogue_input), None
        # 没给就不传：热路径的调用形状与以前逐字一致（也给测试里的假引擎留活路）。
        options: dict[str, Any] = {"thinking": dict(thinking)} if thinking is not None else {}
        if not tools or self._tool_runner is None:
            value = await observed_generate(dialogue_input, **options)
        else:
            async def runner(call: Any) -> str:
                result = await self._tool_runner.run(
                    call, image_path=image_path, now=self._clock()
                )
                if event_id is not None:
                    # 只保留最近一次：失败路径不一定走到 trace，别让这个字典无限长。
                    self._turn_tool_usage.clear()
                    self._turn_tool_usage[event_id] = {
                        "tool_name": result.name,
                        "tool_query": result.query,
                        "tool_ok": result.ok,
                        "tool_degraded": result.degraded_reason,
                        "tool_elapsed_ms": result.elapsed_ms,
                        "tool_result_chars": len(result.block),
                    }
                return result.block

            value = await observed_generate(
                dialogue_input, tools=tools, tool_runner=runner, **options
            )
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[1], DialogueGenerationObservation)
        ):
            raise TypeError("generate_with_observation returned an invalid contract")
        return value

    def _remembered_episodes(
        self, records: tuple[MemoryRecord, ...]
    ) -> tuple[MemoryRecord, ...]:
        """常驻层用的「最近三条普通 episode」，按时间倒序。

        它并进 relationship_state 一起装配与渲染（同一处渲染、同一族标签、一个装配点），
        目的是让那三条**每轮都在场、不靠话题检索**——他之前抱怨的「记下来了不会自己提」
        就出在这里：原来的常驻集合只有 agreement/correction，episode 一律靠话题检索。
        """

        if _PINNED_EPISODE_LIMIT <= 0:
            return ()
        candidates = [
            record
            for record in records
            if record.type == "episode"
            and record.status == "active"
            and record.privacy_class == "ordinary"
            and record.recall_policy == "daily_safe"
        ]
        candidates.sort(key=lambda record: (record.valid_from_utc, record.memory_id), reverse=True)
        return tuple(candidates[:_PINNED_EPISODE_LIMIT])

    def _episode_pointer(
        self,
        current: ConversationEvent,
        history: tuple[ConversationEvent, ...],
        quoted_event_ids: set[str],
        now: datetime,
    ) -> "_EpisodePointer":
        """Which episode does this turn point at, and with which key?

        2026-09-12 T2（历史修复计划 §2.1）。授权与定位
        从这一版开始分开：这里只回答「有没有指，用什么指的」，_detail_targets 回答
        「指到哪一段」。三字窗口命中**不再是钥匙**——它只说明「像」，不说明「你要」，
        而正是它让两个字（"不够"）打开了 32 条成人原文（问题冻结 P1）。
        """

        conversation_id = current.conversation_id
        text = current.text or ""
        today = now.astimezone(self.local_zone).date()
        days = referenced_dates((text,), now=now, local_zone=self.local_zone) if text else ()
        # 过去的日子和「今天 + 时段词」是可以同时出现的，而且**都算**：
        # 「准确来说不是昨天晚上，而是今天的凌晨」里两个日子都在，过去的那天不许
        # 独占地把今天挤掉（2026-09-12 真机：那句话三轮都锁在昨天那一段，用户
        # 的更正等于没说）。今天只拿到普通明细的权限，过去的日子照旧。
        past_days = _past_days(days, today)
        today_named = any(day == today for day in days) and names_a_time_of_day(text)
        if past_days or today_named:
            wanted = past_days + ((today,) if today_named else ())
            # 时段词参与选片段：点「凌晨」就先给凌晨那几段（T10）。
            prefer_hours = time_of_day_hours(text)
            anchors = time_of_day_anchors(text)
            stored = self.memory_details.fragments_on_dates(
                conversation_id, wanted, local_zone=self.local_zone,
                limit=_DAY_FRAGMENT_LIMIT, prefer_hours=prefer_hours, anchors=anchors,
            )
            if not stored:
                return _EpisodePointer("date_missing", days=wanted)
            # 2026-09-17：这一天/这个时段里，把他**这一轮说的话对得上的那几段**排到最前。
            # 指纹按「钟点距离 + 最新优先」取样，而明细块预算只装得下最前面那几段——真机上
            # 他要核对的那段排第三，于是永远进不来（她只能说「我这儿翻不着」）。这里只重排
            # 已经由这把钥匙授权的片段：不新开片段、不改授权集合（P1 的成因是词面凭空开门，
            # 与排序无关）。
            if text:
                ranked = self.memory_details.fragments_ranked_by_text(
                    conversation_id, (text,), within=stored, limit=_DAY_FRAGMENT_LIMIT
                )
                if ranked:
                    matched = tuple(fragment_id for fragment_id, _run, _hits in ranked)
                    stored = matched + tuple(item for item in stored if item not in matched)
            ordinary_only: tuple[str, ...] = ()
            if today_named:
                on_today = set(
                    self.memory_details.fragments_on_dates(
                        conversation_id, (today,), local_zone=self.local_zone, limit=_DAY_FRAGMENT_LIMIT
                    )
                )
                ordinary_only = tuple(item for item in stored if item in on_today)
            return _EpisodePointer(
                "date_now" if past_days else "today",
                days=wanted,
                fragments=stored,
                ordinary_only=ordinary_only,
            )
        if quoted_event_ids:
            quoted = self.memory_details.fragments_for_events(
                conversation_id, tuple(sorted(quoted_event_ids))
            )
            if quoted:
                return _EpisodePointer("quote", fragments=quoted)
        if text:
            matched = self.memory_details.fragments_quoting(conversation_id, text)
            if matched:
                return _EpisodePointer("verbatim", match_count=len(matched), fragments=matched)
        # 最新的那次点名说了算：它说今天（或将来）时继承链就到此为止——今天不
        # 继承，将来的日子本来也不是回顾。
        inherited_pointer: _EpisodePointer | None = None
        for earlier in reversed(_spoken_text(history)[-_DAY_LOOKBACK:]):
            earlier_days = referenced_dates((earlier,), now=now, local_zone=self.local_zone)
            if earlier_days:
                inherited_days = _past_days(earlier_days, today)
                if not inherited_days:
                    inherited_pointer = _EpisodePointer("none")
                    break
                stored = self.memory_details.fragments_on_dates(
                    conversation_id, inherited_days, local_zone=self.local_zone
                )
                inherited_pointer = (
                    _EpisodePointer("date_inherited", days=inherited_days, fragments=stored)
                    if stored
                    else _EpisodePointer("date_missing", days=inherited_days)
                )
                break
        # 改口（T6）：上一轮真的摊开过某一段，这一轮用词面指了**另一段**。词面本身
        # 仍然不是钥匙，能开门是因为上一轮的门还没关——而且只开普通明细。
        previous = self._last_detail_fragments.get(conversation_id, ())
        correction_pointer: _EpisodePointer | None = None
        if previous and text:
            identified = tuple(
                fragment
                for fragment in self.memory_details.fragments_matching(conversation_id, (text,))
                if fragment not in previous
            )
            if len(identified) == 1:
                correction_pointer = _EpisodePointer("correction", match_count=1, fragments=identified)
            elif len(identified) > 1:
                correction_pointer = _EpisodePointer("ambiguous", match_count=len(identified))
        # 继承下来的恰好就是上一轮已经摊开的那一段时，再摊一次没有意义：让改口说话。
        # 反过来，如果这一轮没有别的指向，继承照旧生效（同一段连问两轮不会被吞掉）。
        if (
            correction_pointer is not None
            and inherited_pointer is not None
            and inherited_pointer.fragments
            and set(inherited_pointer.fragments) == set(previous)
        ):
            return correction_pointer
        if inherited_pointer is not None:
            return inherited_pointer
        if correction_pointer is not None:
            return correction_pointer
        return _EpisodePointer("none")

    def _named_days(
        self, text: str | None, spoken: tuple[str, ...], now: datetime
    ) -> tuple[date, ...]:
        """The **past** day the user named, newest mention first.

        A day named in this message decides alone.  Otherwise the newest earlier
        message that names one decides: people say "九号那天" once and then keep
        saying "那天", so the mention is routinely several turns back -- and a
        conversation that merely talks about that day is always newer than the day
        itself, so word overlap must never get the chance to answer for it.

        2026-09-12 T3：这里和钥匙用同一个窗口、同一条「只认过去」的规则。最新的
        那次点名说了算——它说的是今天时，回看链就到此为止，不再往更早的消息里找
        一个过去的日子来顶替。
        """

        today = now.astimezone(self.local_zone).date()
        days = _past_days(
            referenced_dates((text or "",), now=now, local_zone=self.local_zone), today
        )
        if days:
            return days
        for earlier in reversed(spoken[-_DAY_LOOKBACK:]):
            days = referenced_dates((earlier,), now=now, local_zone=self.local_zone)
            if days:
                return _past_days(days, today)
        return ()

    def _named_day_fragments(
        self, current: ConversationEvent, history: tuple[ConversationEvent, ...], now: datetime
    ) -> tuple[str, ...]:
        """Fragments stored on a day this conversation actually named.

        These keep their index line even when the recall gate stays shut for the
        details.  Answering "那块没存进来" about an episode that is stored and
        merely sealed is a false statement (measured 2026-09-11).
        """

        days = self._named_days(current.text, _spoken_text(history), now)
        if not days:
            return ()
        return self.memory_details.fragments_on_dates(
            current.conversation_id, days, local_zone=self.local_zone
        )

    @staticmethod
    def _detail_targets(pointer: "_EpisodePointer") -> tuple[tuple[str, ...], str, str]:
        """Which episodes the key just opened, and what to call the rule.

        The pointer already resolved both halves (plan §2.1/§2.2); this only maps
        a key onto the reason the trace, the detail block and the panel speak in.
        Nothing is guessed here: an episode the key did not name is not unfolded,
        and a key that names nothing says so instead of borrowing a neighbour.
        """

        if pointer.key in {"date_now", "date_inherited", "today"}:
            return pointer.fragments, "", "day"
        if pointer.key == "quote":
            return pointer.fragments, "", "quote"
        if pointer.key == "verbatim":
            if len(pointer.fragments) > 1:
                # 计划 §2.2：指向不唯一就不猜——她可以凭索引问用户「你指哪一次」。
                return (), "", "ambiguous"
            return pointer.fragments, "", "verbatim"
        if pointer.key == "date_missing":
            # 计划 §2.3：点了日期但那天什么都没存。不展开，也不许拿别的段顶上。
            return (), "", "day_missing"
        if pointer.key == "correction":
            return pointer.fragments, "", "correction"
        if pointer.key == "ambiguous":
            return (), "", "ambiguous"
        return (), "", "none"

    def _budget_details(
        self,
        details: tuple[MemoryDetailRecord, ...],
        note: str,
        labels: Mapping[str, str],
        budget: int = _DETAIL_TOKEN_BUDGET,
        priority: tuple[str, ...] = (),
    ) -> tuple[tuple[MemoryDetailRecord, ...], str]:
        """Keep one turn's detail block inside the token budget (T6, 裁定 A).

        The newest episodes win: when a named day holds more than the budget
        allows, the older parts are stated as not unfolded instead of being
        dropped in silence.  One episode is always kept whole, even alone over
        budget -- an empty block would contradict the index line that says this
        episode was expanded.
        """

        if not details:
            return details, note
        present: list[str] = []
        for detail in details:
            if detail.fragment_id not in present:
                present.append(detail.fragment_id)
        # 先花在定位规则点名的那几段上（点「凌晨」就是凌晨那几段），剩下的按时间
        # 从新到旧——list_details 给的是从旧到新。没有点名顺序时退回原来的行为。
        order: list[str] = [item for item in priority if item in present]
        order += [item for item in reversed(present) if item not in order]
        selected: list[str] = []
        for fragment_id in order:
            trial = tuple(item for item in details if item.fragment_id in {*selected, fragment_id})
            if selected and self.context_builder.detail_block_tokens(trial, labels) > budget:
                break
            selected.append(fragment_id)
        kept = tuple(item for item in details if item.fragment_id in set(selected))
        dropped = len(details) - len(kept)
        if dropped <= 0:
            return details, note
        extra = _DETAIL_BUDGET_NOTE.format(kept=len(kept), dropped=dropped)
        return kept, f"{note}\n{extra}" if note else extra

    @staticmethod
    def _cap_details(
        details: tuple[MemoryDetailRecord, ...], note: str
    ) -> tuple[tuple[MemoryDetailRecord, ...], str]:
        """Keep one episode's timeline inside the recorded per-fragment budget."""

        kept: list[MemoryDetailRecord] = []
        per_fragment: dict[str, int] = {}
        for item in details:
            seen = per_fragment.get(item.fragment_id, 0)
            if seen >= MAX_DETAILS_PER_FRAGMENT:
                continue
            per_fragment[item.fragment_id] = seen + 1
            kept.append(item)
        dropped = len(details) - len(kept)
        if dropped <= 0:
            return details, note
        remainder = _DETAIL_REMAINDER_NOTE.format(count=dropped)
        return tuple(kept), f"{note}\n{remainder}" if note else remainder

    @staticmethod
    def _memory_query_terms(
        primary_query: str,
        quoted_chain: tuple[ConversationEvent, ...],
    ) -> tuple[str, ...]:
        sources = [primary_query]
        sources.extend(event.text for event in quoted_chain if event.text is not None)
        terms: list[str] = []
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, str):
                raise TypeError("memory retrieval sources must be text")
            normalized = source.strip()
            if not normalized:
                continue
            if len(normalized) > _MEMORY_QUERY_SOURCE_LIMIT:
                half = _MEMORY_QUERY_SOURCE_LIMIT // 2
                normalized = normalized[:half] + normalized[-half:]
            if normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
        return tuple(terms)

    def _append_trace(
        self,
        event: ConversationEvent,
        *,
        source: str,
        phase: str,
        details: Mapping[str, object],
        occurred_at_utc: datetime | None = None,
    ) -> None:
        try:
            self.turn_traces.append_once(
                trigger_event_id=event.event_id,
                conversation_id=event.conversation_id,
                source=source,
                phase=phase,
                occurred_at_utc=_aware_utc(
                    self._clock() if occurred_at_utc is None else occurred_at_utc,
                    "trace occurred_at_utc",
                ),
                details=details,
            )
        except Exception as error:
            _LOGGER.error(
                "turn trace append failed (%s)",
                type(error).__name__,
                extra={"trace_phase": phase},
            )

    @staticmethod
    def _generation_details(
        outcome: DialogueOutcome | None,
        started_at: float,
        observation: DialogueGenerationObservation | None,
    ) -> dict[str, object]:
        elapsed_ms = round(max(0.0, (time.monotonic() - started_at) * 1000), 3)
        if observation is None:
            return {
                "result_type": (
                    "failure"
                    if outcome is None
                    else "skip"
                    if isinstance(outcome, DialogueSkip)
                    else "reply"
                ),
                "model_route": getattr(outcome, "model_route", "primary"),
                "generation_ms": elapsed_ms,
                "attempt_count": 1,
                "retry_count": 0,
            }
        finish_reason = observation.finish_reason
        if finish_reason not in {None, "stop", "length", "tool_calls", "content_filter"}:
            finish_reason = "other"
        return {
            "result_type": observation.result_type,
            "model_id": observation.model_id,
            "model_route": observation.model_route,
            "model_tier": observation.model_tier,
            "generation_ms": elapsed_ms,
            "provider_latency_ms": round(observation.latency_ms, 3),
            "attempt_count": observation.attempt_count,
            "retry_count": observation.retry_count,
            "input_tokens": observation.input_tokens,
            "output_tokens": observation.output_tokens,
            "reasoning_tokens": observation.reasoning_tokens,
            "cache_hit_tokens": observation.cache_hit_tokens,
            "finish_reason": finish_reason,
            # 2026-09-15：重试过就要说清为什么（真机出现过 attempt=2 却查不出原因）。
            # 没重试时不写这个键，轨迹保持干净。
            **(
                {"structural_retry_reasons": list(observation.retry_reasons)}
                if observation.retry_reasons
                else {}
            ),
        }

    def _record_generation_trace(
        self,
        event: ConversationEvent,
        source: str,
        outcome: DialogueOutcome,
        started_at: float,
        observation: DialogueGenerationObservation | None,
    ) -> None:
        details = self._generation_details(outcome, started_at, observation)
        usage = self._turn_tool_usage.pop(event.event_id, None)
        if usage:
            details.update(usage)
        if isinstance(outcome, DialogueResult):
            details.update(
                {
                    "message_part_count": len(outcome.message_parts),
                    "reply_target_handle": outcome.reply_target,
                    "expression_kind": (
                        outcome.expression_intent.kind if outcome.expression_intent else None
                    ),
                    "expression_key": (
                        outcome.expression_intent.key if outcome.expression_intent else None
                    ),
                }
            )
        self._append_trace(event, source=source, phase="generation", details=details)

    def _generation_failure_details(
        self,
        failure_category: str,
        started_at: float,
        observation: DialogueGenerationObservation | None,
    ) -> dict[str, object]:
        details = self._generation_details(None, started_at, observation)
        details.update({"stage": "generation", "failure_category": failure_category})
        return details

    def _record_cancelled_trace(
        self, event: ConversationEvent, source: str, cancellation_category: str
    ) -> None:
        self._append_trace(
            event,
            source=source,
            phase="cancelled",
            details={"cancellation_category": cancellation_category},
        )

    def _record_delivery_trace(
        self,
        trigger: ConversationEvent,
        source: str,
        delivered: ConversationEvent,
        outcome: DialogueResult,
    ) -> None:
        outbox_rows = self.database.connection.execute(
            "SELECT operation_key,status FROM outbox WHERE event_id=? ORDER BY operation_key",
            (delivered.event_id,),
        ).fetchall()
        self._append_trace(
            trigger,
            source=source,
            phase="delivery",
            details={
                "status": delivered.status,
                "outbound_event_id": delivered.event_id,
                "platform_message_id_present": delivered.platform_message_id is not None,
                "message_part_count": len(outcome.message_parts),
                "reply_target_handle": outcome.reply_target,
                "expression_kind": (
                    outcome.expression_intent.kind if outcome.expression_intent else None
                ),
                "expression_key": (
                    outcome.expression_intent.key if outcome.expression_intent else None
                ),
                "outbox_statuses": {
                    row["operation_key"]: row["status"] for row in outbox_rows
                },
            },
        )

    def _events_before(self, current: ConversationEvent) -> tuple[ConversationEvent, ...]:
        rows = self.database.connection.execute(
            "SELECT event_id FROM conversation_events "
            "WHERE conversation_id = ? AND sequence < ? AND "
            "((direction = 'inbound' AND status = 'received') OR "
            "(direction = 'outbound' AND status = 'sent')) "
            "ORDER BY sequence, event_id",
            (current.conversation_id, current.sequence),
        ).fetchall()
        history = tuple(self.events.get(row["event_id"]) for row in rows)
        # Include the current user event for the boundary decision, then keep
        # it out of the returned history.  A delayed bot send must not itself
        # split a session; only a user's new message after a real idle gap can
        # start one.
        with_current = self._session_history((*history, current))
        return tuple(event for event in with_current if event.event_id != current.event_id)

    @staticmethod
    def _session_history(
        events: tuple[ConversationEvent, ...],
        *,
        boundary_at_utc: datetime | None = None,
    ) -> tuple[ConversationEvent, ...]:
        """Keep the latest conversational fragment after a real idle gap.

        Durable history remains complete in SQLite.  The model only needs the
        latest continuous fragment to understand the current turn; older
        assistant prose is evidence for explicit retrieval, not a style
        template for a new conversation. An initiative has no user message of
        its own, so its attempt time can be supplied as a boundary while the
        latest user event remains available as the unconsumed topic anchor.
        """
        if not events:
            return ()
        if boundary_at_utc is not None:
            boundary_at_utc = _aware_utc(boundary_at_utc, "boundary_at_utc")
        boundary = 0
        last_user_at: datetime | None = None
        last_user_index: int | None = None
        for index, event in enumerate(events):
            if event.actor != "mumo":
                continue
            if last_user_at is not None:
                gap = event.occurred_at_utc - last_user_at
                if timedelta(0) <= gap > _SESSION_GAP:
                    boundary = index
            last_user_at = event.occurred_at_utc
            last_user_index = index
        if (
            boundary_at_utc is not None
            and last_user_at is not None
            and last_user_index is not None
        ):
            gap = boundary_at_utc - last_user_at
            if timedelta(0) <= gap > _SESSION_GAP:
                # Keep the latest user topic and any sent reply after it, but
                # drop stale pre-idle history for an initiative attempt. A
                # normal current user event is appended by _events_before and
                # removed by that caller.
                boundary = max(boundary, last_user_index)
        return events[boundary:]

    def _previous_event(self, current: ConversationEvent) -> ConversationEvent | None:
        row = self.database.connection.execute(
            "SELECT event_id FROM conversation_events "
            "WHERE conversation_id = ? AND sequence < ? AND "
            "((direction = 'inbound' AND status = 'received') OR "
            "(direction = 'outbound' AND status = 'sent')) "
            "ORDER BY sequence DESC LIMIT 1",
            (current.conversation_id, current.sequence),
        ).fetchone()
        return None if row is None else self.events.get(row["event_id"])

    @staticmethod
    def _source_fact(event: ConversationEvent) -> MessageSourceFact:
        return MessageSourceFact(
            actor=event.actor,
            handle=event.visible_handle,
            kind=event.kind,
            occurred_at_utc=event.occurred_at_utc,
        )

    @staticmethod
    def _received_media(event: ConversationEvent) -> tuple[str, ...]:
        media: list[str] = []
        if event.kind == "poke":
            media.append("poke")
        if event.text is not None:
            media.append("text")
        for segment in event.message_segments:
            if segment.type in {"image", "face"} and segment.type not in media:
                media.append(segment.type)
        return tuple(media)

    def _context_version(self, conversation_id: str) -> int:
        row = self.database.connection.execute(
            "SELECT context_version FROM conversation_cursors WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return int(row["context_version"])

    @staticmethod
    def _poke_identity(raw_payload: object) -> str:
        if not isinstance(raw_payload, Mapping):
            raise ValueError("poke payload must be a mapping")
        identity = {
            key: raw_payload.get(key)
            for key in ("self_id", "user_id", "sender_id", "target_id", "time")
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"poke:{digest}"

    def _mark_processed(self, conversation_id: str, sequence: int) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE conversation_cursors SET last_processed_sequence = "
                "CASE WHEN last_processed_sequence IS NULL OR last_processed_sequence < ? "
                "THEN ? ELSE last_processed_sequence END WHERE conversation_id = ?",
                (sequence, sequence, conversation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(conversation_id)

    def _usable_reply_target(
        self, conversation_id: str, handle: str | None
    ) -> ConversationEvent | None:
        """Return a target the sender can actually address on QQ."""
        return self.sender._reply_target(conversation_id, handle)

    def _fail_inbound(self, event: ConversationEvent, reason: str) -> None:
        """Converge a terminal structural failure without retaining a blocker.

        The original inbound row remains immutable except for its lifecycle
        status.  Advancing the processed watermark in the same transaction
        prevents a known-bad event from blocking recovery after restart while
        keeping its exact payload available for offline diagnosis.
        """
        if event.direction != "inbound":
            raise ValueError("only inbound events can be failed")
        if not isinstance(reason, str) or not reason:
            raise ValueError("failure reason must be non-empty text")
        with self.database.transaction() as connection:
            updated = connection.execute(
                "UPDATE conversation_events SET status = 'failed' "
                "WHERE event_id = ? AND direction = 'inbound' AND status = 'received'",
                (event.event_id,),
            )
            if updated.rowcount != 1:
                raise ValueError("inbound event is not in received state")
            cursor = connection.execute(
                "UPDATE conversation_cursors SET last_processed_sequence = "
                "CASE WHEN last_processed_sequence IS NULL OR last_processed_sequence < ? "
                "THEN ? ELSE last_processed_sequence END WHERE conversation_id = ?",
                (event.sequence, event.sequence, event.conversation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(event.conversation_id)

    def _record_generation_failure(
        self,
        event: ConversationEvent,
        reason: str,
        started_at: float,
        *,
        source: str,
        observation: DialogueGenerationObservation | None = None,
    ) -> None:
        """Persist a terminal generation failure and emit safe diagnostics."""
        generation_ms = max(0.0, (time.monotonic() - started_at) * 1000)
        self._record_processing_failure(
            event,
            reason,
            stage="generation",
            source=source,
            trace_details=self._generation_failure_details(
                reason,
                started_at,
                observation,
            ),
        )
        _LOGGER.error(
            "dialogue generation failed for inbound event %s (%s)",
            event.event_id,
            reason,
            extra={
                "failure_type": reason,
                "generation_ms": round(generation_ms, 3),
            },
        )

    def _record_processing_failure(
        self,
        event: ConversationEvent,
        reason: str,
        *,
        stage: str,
        source: str,
        trace_details: Mapping[str, object] | None = None,
    ) -> None:
        """Converge a post-persistence failure without losing the input."""
        self._fail_inbound(event, reason)
        self._append_trace(
            event,
            source=source,
            phase="failure",
            details=(
                {"stage": stage, "failure_category": reason}
                if trace_details is None
                else trace_details
            ),
        )

    def _notify_memory_activity(self, conversation_id: str) -> None:
        """Tell the durable worker to rebuild its snapshot after a terminal event."""
        if self.memory_worker is None:
            return
        try:
            self.memory_worker.notify_reliable_activity(conversation_id)
        except Exception as error:
            # Memory consolidation is a side path.  A notification failure
            # must never turn a successfully delivered reply into a retry.
            _LOGGER.error(
                "memory activity notification failed (%s)", type(error).__name__
            )

    async def _emit_failure_notice(self, event: ConversationEvent, reason: str) -> None:
        """Send one idempotent, non-semantic status notice after a failed turn."""
        try:
            status = await self.sender.send_failure_notice(
                event,
                owner_qq=self.owner_qq,
                occurred_at_utc=_aware_utc(self._clock(), "clock result"),
                reason=reason,
            )
        except Exception as error:
            _LOGGER.error("failure notice could not be sent (%s)", type(error).__name__)
            return
        if status != "sent":
            _LOGGER.error("failure notice delivery ended as %s", status)

    @staticmethod
    def _new_id(factory: Callable[[], str], field: str) -> str:
        value = factory()
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must return non-empty text")
        return value
