"""Assemble evidence-backed dialogue context within verified token budgets."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import json
import re
from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence, TypeAlias
from zoneinfo import ZoneInfo

from qichi.domain.dialogue import ModelImage, ModelMessage
from qichi.domain.events import ConversationEvent, quote_is_verbatim
from qichi.domain.memory import Agreement, Correction, MemoryEvidence, MemoryRecord
from qichi.domain.memory_details import MemoryDetailRecord
from qichi.memory.index_digest import INDEX_HEADER as MEMORY_INDEX_HEADER

from .capability_manifest import (
    MessageSourceFact,
    RuntimeFacts,
    render_stable_facts,
    render_turn_facts,
)
from .model_capability import ModelCapability, ModelCapabilityError, ProviderCapabilityUnverifiedError
from .token_counter import TokenCounter


RelationshipState: TypeAlias = MemoryRecord | Agreement | Correction

# Pictures are billed by the provider in prompt tokens the local counter cannot
# see.  Measured 2026-09-11 against the production endpoint: about 600 prompt
# tokens for a one-megapixel image, with the second image marginally cheaper.
# The counter rounds up to keep the budget honest rather than exact.
_IMAGE_TOKEN_ESTIMATE = 700

_LOCAL_ZONE = ZoneInfo("Asia/Shanghai")

# 2026-09-18 呈现层：常驻的「她自己记着的事」结尾那句边界说明。用户裁定保留——
# 它是边界（她有权不提），不是行为要求（不写台词、不写「你应该……」）。
REMEMBERED_NOTE = "（这些是你本来就知道的事，不必当作清单逐条回应。）"
# 足迹块只放短原话；超长的整条不要（切一半的「原话」比没有更糟），长段落仍归明细块。
_FOOTPRINT_QUOTE_MAX_CHARS = 400
_LONG_GAP = timedelta(hours=1)
_CATEGORY_KEYS = (
    "role_core",
    "runtime_facts",
    "relationship_state",
    "memory_working_set",
    "recent_history",
    "memory_evidence",
    "memory_details",
    "memory_index",
    "memory_footprint",
    "earlier_history",
    "time_gaps",
    "direct_quotes",
    "current_input",
)


class ContextValidationError(ValueError):
    """Context inputs disagree with their durable source records."""


class ContextBudgetError(ValueError):
    """Mandatory context cannot fit within a verified input budget."""


@dataclass(frozen=True, slots=True)
class ContextBuildRequest:
    role_core: str
    runtime_facts: RuntimeFacts
    current_event: ConversationEvent
    quoted_chain: tuple[ConversationEvent, ...]
    recent_events: tuple[ConversationEvent, ...]
    relationship_state: tuple[RelationshipState, ...]
    memory_candidates: tuple[MemoryRecord, ...]
    earlier_events: tuple[ConversationEvent, ...]
    evidence_events: Mapping[str, ConversationEvent]
    memory_working_set: tuple[MemoryRecord, ...] = ()
    confirmation_candidates: tuple[MemoryRecord, ...] = ()
    memory_details: tuple[MemoryDetailRecord, ...] = ()
    # When the expansion target is inferred rather than identified by the user,
    # the block must carry that fact: the model has to confirm which episode is
    # meant instead of retelling a guess as if the user had named it.
    memory_detail_note: str = ""
    # Sensitive evidence is admitted only when the current query has
    # explicitly/lexically brought the topic into scope.  The default keeps
    # adult details out of ordinary daily context without deleting them.
    allow_sensitive_memory: bool = False
    # Verbatim details answer a different question than memory pieces do: they open
    # when the user asked to revisit an episode or quoted it, even when no memory
    # record was retrieved in this turn.  Keeping the flag separate means the record
    # gate above stays exactly as strict as before.
    allow_sensitive_details: bool = False
    # The always-on neutral index: dates, level and counts only.  It carries no
    # quote and no normalized text, so it can stay resident without opening the
    # recall matrix (frozen contract section 5.2).
    memory_index: tuple[str, ...] = ()
    # 最近原文足迹（2026-09-12）：最近几天每个片段的头几条逐字原话，常驻上下文。
    # 起因是用户反馈「角色没法找到原文字段」：她的常驻层只有改写（工作集/索引），
    # 逐字原话只在钥匙开门时才注入，转述式追问（不点日期、不引用、非逐字）永远不开门。
    memory_footprint: tuple[MemoryDetailRecord, ...] = ()
    # fragment_id -> 足迹里那一段的窗口标签，与索引、明细块用同一套说法。
    memory_footprint_labels: Mapping[str, str] = field(default_factory=dict)
    # One code-generated fact line for a day the user named that holds nothing.
    # It rides with the index because that is where "what exists" is stated; the
    # model decides how to say it (2026-09-12 T4).
    memory_recall_note: str = ""
    # Pictures of the user's current message.  They ride on the current user
    # turn only: history replay stays text, and the role core never carries one.
    current_images: tuple[ModelImage, ...] = ()
    # fragment_id -> the window label the index uses for the same episode.  The
    # detail block must name its episode in the same words, otherwise the model
    # cannot tell which index line it just opened (2026-09-11: 26 quotes of 09-09
    # were injected and she still answered "九号那块还是空的").
    memory_detail_labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role_core, str) or not self.role_core.strip():
            raise ValueError("role_core must be a non-empty string")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if not isinstance(self.current_event, ConversationEvent):
            raise TypeError("current_event must be ConversationEvent")
        _event_tuple(self.quoted_chain, "quoted_chain")
        _event_tuple(self.recent_events, "recent_events")
        _event_tuple(self.earlier_events, "earlier_events")
        if not isinstance(self.relationship_state, tuple) or not all(
            isinstance(item, (MemoryRecord, Agreement, Correction)) for item in self.relationship_state
        ):
            raise TypeError("relationship_state must be a tuple of memory relationship records")
        if not isinstance(self.memory_candidates, tuple) or not all(
            isinstance(item, MemoryRecord) for item in self.memory_candidates
        ):
            raise TypeError("memory_candidates must be a tuple of MemoryRecord")
        if not isinstance(self.memory_working_set, tuple) or not all(
            isinstance(item, MemoryRecord) for item in self.memory_working_set
        ):
            raise TypeError("memory_working_set must be a tuple of MemoryRecord")
        if not isinstance(self.memory_index, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.memory_index
        ):
            raise TypeError("memory_index must be a tuple of non-empty strings")
        if not isinstance(self.memory_recall_note, str):
            raise TypeError("memory_recall_note must be a string")
        if not isinstance(self.current_images, tuple) or not all(
            isinstance(item, ModelImage) for item in self.current_images
        ):
            raise TypeError("current_images must be a tuple of ModelImage")
        if len({item.path for item in self.current_images}) != len(self.current_images):
            raise ContextValidationError("duplicate current image path")
        if not isinstance(self.memory_detail_labels, Mapping) or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in self.memory_detail_labels.items()
        ):
            raise TypeError("memory_detail_labels must map fragment IDs to non-empty labels")
        object.__setattr__(self, "memory_detail_labels", MappingProxyType(dict(self.memory_detail_labels)))
        if not isinstance(self.memory_details, tuple) or not all(
            isinstance(item, MemoryDetailRecord) for item in self.memory_details
        ):
            raise TypeError("memory_details must be a tuple of MemoryDetailRecord")
        if not isinstance(self.memory_detail_note, str):
            raise TypeError("memory_detail_note must be a string")
        detail_ids = tuple(item.detail_id for item in self.memory_details)
        if len(set(detail_ids)) != len(detail_ids):
            raise ContextValidationError("duplicate memory detail ID")
        if type(self.allow_sensitive_memory) is not bool:
            raise TypeError("allow_sensitive_memory must be a bool")
        if type(self.allow_sensitive_details) is not bool:
            raise TypeError("allow_sensitive_details must be a bool")
        working_ids = tuple(item.memory_id for item in self.memory_working_set)
        if len(set(working_ids)) != len(working_ids):
            raise ContextValidationError("duplicate working set memory ID")
        if not isinstance(self.confirmation_candidates, tuple) or not all(
            isinstance(item, MemoryRecord) and item.status == "candidate"
            for item in self.confirmation_candidates
        ):
            raise TypeError("confirmation_candidates must contain candidate MemoryRecord values")
        if len(self.confirmation_candidates) > 1 or any(
            item.recall_scope != "confirmation" for item in self.confirmation_candidates
        ):
            raise ValueError("confirmation candidates must have confirmation recall scope")
        if any(item.memory_id in {record.memory_id for record in self.memory_candidates} for item in self.confirmation_candidates):
            raise ValueError("confirmation candidate overlaps active memory")
        if not isinstance(self.evidence_events, Mapping):
            raise TypeError("evidence_events must be a mapping")
        copied: dict[str, ConversationEvent] = {}
        for event_id, event in self.evidence_events.items():
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("evidence event keys must be non-empty strings")
            if not isinstance(event, ConversationEvent):
                raise TypeError("evidence event values must be ConversationEvent")
            if event_id != event.event_id:
                raise ContextValidationError("evidence event key does not match event ID")
            copied[event_id] = event
        object.__setattr__(self, "evidence_events", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class ContextTokenMetrics:
    input_tokens: int
    input_budget_tokens: int
    window_tokens: int
    expanded: bool
    category_tokens: Mapping[str, int]
    omitted_counts: Mapping[str, int]
    selected_history_event_ids: tuple[str, ...]
    selected_working_memory_ids: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]
    # 2026-09-22 成本观测：本轮提示的字符数，以及它与**上一轮**逐字相同的公共前缀字符数。
    # 只记数字，绝不落提示正文。与 provider 报的 cache_hit_tokens/input_tokens 并排看，
    # 就能判定"命中偏低"是提示结构问题还是供应商侧的命中判定差异。
    prompt_chars: int = 0
    prompt_prefix_chars: int = 0
    # 2026-09-22 成本观测之二：本轮提示是否**完整覆盖了上一轮的全部提示**。
    # DeepSeek 官方：缓存前缀单元落在"每次请求的用户输入结束位置"，后续请求**完整匹配**
    # 该单元才命中。所以这正是可验证的预测——为真时，命中至少应达到上一轮的输入 tokens。
    prompt_covers_previous: bool = False


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    messages: tuple[ModelMessage, ...]
    metrics: ContextTokenMetrics


@dataclass(frozen=True, slots=True)
class _Piece:
    category: str
    message: ModelMessage
    tokens: int


@dataclass(frozen=True, slots=True)
class _HistoryPiece:
    event: ConversationEvent
    pieces: tuple[_Piece, ...]

    @property
    def tokens(self) -> int:
        return sum(piece.tokens for piece in self.pieces)


@dataclass(frozen=True, slots=True)
class _MemoryPiece:
    memory_id: str
    piece: _Piece


@dataclass(frozen=True, slots=True)
class _MemoryWorkingSetPiece:
    memory_ids: tuple[str, ...]
    piece: _Piece


@dataclass(frozen=True, slots=True)
class _Prepared:
    role: _Piece
    # 稳定半：逐轮逐字一致，排在 prompt 最前面喂供应商的前缀缓存。
    facts: _Piece
    # 本轮半：带秒的钟、来源、媒体、尾标协议，排在历史之后、当前输入之前。
    facts_turn: _Piece
    relationships: tuple[_Piece, ...]
    relationship_omitted: int
    memory_working_set: _MemoryWorkingSetPiece | None
    memory_working_set_omitted: int
    recent: tuple[_HistoryPiece, ...]
    memories: tuple[_MemoryPiece, ...]
    memory_base_omitted: int
    memory_details: _Piece | None
    memory_details_omitted: int
    memory_index: _Piece | None
    memory_index_omitted: int
    # 最近原文足迹：最近几天每个片段的头几条逐字原话（常驻）。
    memory_footprint: _Piece | None
    earlier: tuple[_HistoryPiece, ...]
    quotes: tuple[_Piece, ...]
    current: tuple[_Piece, ...]
    current_event: ConversationEvent


@dataclass(frozen=True, slots=True)
class _Assembly:
    result: ContextBuildResult
    omitted_for_budget: bool


def _event_tuple(value: object, field: str) -> tuple[ConversationEvent, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, ConversationEvent) for item in value):
        raise TypeError(f"{field} must be a tuple of ConversationEvent")
    return value


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _fit_working_records(
    records: Sequence[Any],
    render: Any,
    tokens_of: Any,
    budget_tokens: int,
) -> tuple[tuple[Any, ...], int]:
    """常驻层按优先级保留前缀，返回 (保留的元组, 丢弃条数)。

    以前只有「整块装得下就全装、装不下就整块丢掉」这一条路（见 _assemble 里那道
    全局预算门），于是它一超过自己的预算就**整层消失**，没有优雅退化。这里改成从
    优先级最低的尾部开始丢：records 必须已按优先级排好序。装不下时按二分找最大的
    可容纳前缀，绝不越过预算；连第一条都装不下时如实全部丢弃。
    """
    if not records:
        return (), 0
    if tokens_of(render(records)) <= budget_tokens:
        return tuple(records), 0
    low, high = 1, len(records)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if tokens_of(render(records[:middle])) <= budget_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return tuple(records[:best]), len(records) - best


def _local_iso(value: datetime) -> str:
    return value.astimezone(_LOCAL_ZONE).isoformat(timespec="seconds")


def _event_role(event: ConversationEvent) -> str:
    if event.actor == "mumo":
        return "user"
    if event.actor == "qichi":
        return "assistant"
    return "system"


# Images ride only the turn they arrive in (frozen plan section 3).  A replayed
# image message therefore has no text and would look like an empty message -- on
# 2026-09-11 the character read that as proof she had invented her own (entirely
# correct) description of the picture.  The marker keeps the fact, not the pixels.
#
# 2026-09-16：旧措辞只说「图像不随历史重放」，没有把这条锚定成「你已经处理过的那一条」，
# 于是她把历史里的图片消息读成「又收到一张」（真机 18:04「这张跟上次那个一个模子刻出来的」
# 「这回托着腮又在思考什么人生大事」；副本复现 1/6）。现在把身份与新旧一起说清楚：
# 它仍然是「这里有一条图片消息」这个事实，但明确不是新图。
IMAGE_HISTORY_MARKER = "[图片消息 · 早前那一条，不是新收到的图；图像本身不重放]"


def _history_body(event: ConversationEvent) -> str:
    carried_image = any(segment.type == "image" for segment in event.message_segments)
    if not carried_image:
        return event.text if event.text is not None else "[无文本内容]"
    if event.text:
        return f"{event.text}\n{IMAGE_HISTORY_MARKER}"
    return IMAGE_HISTORY_MARKER


def _render_event(event: ConversationEvent, label: str) -> str:
    metadata = _render_event_metadata(event, label)
    return "\n".join((metadata, _history_body(event)))


def _render_event_metadata(event: ConversationEvent, label: str) -> str:
    handle = event.visible_handle if event.visible_handle is not None else "none"
    header = (
        f"[{label} | actor={event.actor}; handle={handle}; "
        f"time={_local_iso(event.occurred_at_utc)}; kind={event.kind}]"
    )
    metadata: list[str] = [header]
    visible_segments: list[dict[str, str]] = []
    for segment in event.message_segments:
        # 2026-09-16：image 段不再单独列一行——图片消息已经有自己的标记（IMAGE_HISTORY_MARKER），
        # 这一行只会再加一次「这里有个段」的暗示，而它正是「又收到一张图」那种误读的燃料。
        if segment.type in {"text", "reply", "image"}:
            continue
        visible: dict[str, str] = {"type": segment.type}
        if segment.type == "face":
            face_id = segment.data.get("id")
            if isinstance(face_id, (str, int)) and not isinstance(face_id, bool):
                visible["id"] = str(face_id)
        visible_segments.append(visible)
    if visible_segments:
        metadata.append(
            "[非文本消息段类型] "
            + json.dumps(visible_segments, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(metadata)


def _source_matches(source: MessageSourceFact, event: ConversationEvent) -> bool:
    return (
        source.actor == event.actor
        and source.handle == event.visible_handle
        and source.kind == event.kind
        and source.occurred_at_utc == event.occurred_at_utc
    )


def _duration_text(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分")
    if seconds and not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts) or "0秒"


def _valid_at(valid_from: datetime, valid_until: datetime | None, now: datetime) -> bool:
    return valid_from <= now and (valid_until is None or valid_until >= now)


def _is_emoji_scalar(character: str) -> bool:
    """Recognize common Unicode emoji blocks without a fixed emoji allowlist."""
    codepoint = ord(character)
    return 0x1F000 <= codepoint <= 0x1FAFF or 0x2600 <= codepoint <= 0x27BF


# 颜文字（(￣▽￣) 这种）不是 Unicode emoji，是普通字符——2026-09-14 真机发现她整晚只用
# 一个颜文字 41 次，而统计块看不见它。这里把它当作**一个表情单位**统计，规则与 emoji 一致。
_KAOMOJI_CHARS = frozenset(
    "￣▽△･・ωﾟ｡∀︿⌒ᴗ≧≦╯╰っ♡´｀∇゜ㅂㅅㅇヮ╥╭╮ε˘³ᵕ๑＊﹏◡"
)
_KAOMOJI_PATTERN = re.compile(r"[（(][^)）\n]{1,14}[)）]")


def _expression_tokens(text: str) -> list[str]:
    """Her expression tokens: Unicode emoji scalars plus whole kaomoji as one unit."""

    tokens = [character for character in text if _is_emoji_scalar(character)]
    for match in _KAOMOJI_PATTERN.finditer(text):
        piece = match.group(0)
        inner = piece[1:-1]
        # 括号里不能有汉字：那样是普通注释（（垂耳兔）），不是表情。
        if any("\u4e00" <= character <= "\u9fff" for character in inner):
            continue
        if any(character in _KAOMOJI_CHARS for character in inner):
            tokens.append(piece)
    return tokens


def _recent_emoji_hint(events: tuple[ConversationEvent, ...]) -> str | None:
    """Report only repeated historical expression tokens as a neutral observation.

    The model owns the current expression choice.  A one-off emoji is ordinary
    conversation evidence and should not become a prompt instruction.  When a
    continuous slice contains a repeated token, expose the count as an
    auditable fact, without telling the model to suppress, repeat, or replace
    it.  Tokens are Unicode emoji scalars **and** kaomoji: 2026-09-14 she used
    one single kaomoji 41 times in a night and this block could not see it,
    because ￣ and ▽ are ordinary characters.
    """
    usage: Counter[str] = Counter()
    for event in events:
        if event.actor != "qichi" or event.text is None:
            continue
        usage.update(_expression_tokens(event.text))
    repeated = {character: count for character, count in usage.items() if count >= 2}
    if not repeated:
        return None
    recent = sorted(repeated.items(), key=lambda item: (-item[1], item[0]))[:8]
    summary = ", ".join(f"{character}={count}" for character, count in recent)
    return (
        "[近期表情使用统计 | 来自角色已发送原文；不代表当前表达选择]\n"
        f"{summary}\n"
        # doc/新架构设计.md 18.3 原话：这条统计要「提醒模型表情可以省略且不能机械复用」。
        # 2026-09-14 之前只报了纯事实，副本 A/B 证明那样她不会改；这里把它补上。
        # 仍然**不指定替代表情**、不按轮数或随机数触发。
        "表情可以省略；同一个也不必机械复用。"
    )


# 复读提示（2026-09-13）：一次长会话把近期历史填到上限后，她自己上一轮的规则句与
# 口头禅成了最显眼的范本，同一套说法被一轮轮放大。这里只报**客观统计**：她自己在
# 最近若干条里重复用过的短语，不判断语义、不给词表。
#
# 门槛分两档（2026-09-13 下午，切 pro 关思考的四臂实测之后加严）：
#   长片段 >=8 字，出现 2 条不同消息就报——这正是「把自己上一轮的整句再抄一遍」的样子；
#   短片段 5-7 字，仍要 3 条才算——两次同用一个短说法属于正常措辞（「想你了」这类）。
#
# 2026-09-13 晚真机复检又发现两个毛病，这一版一并修掉：
#   1. 颜文字正好 5-7 个字符，越过了字数下限，于是「(￣▽￣)」×4 这种统计**每轮**都出现；
#      现在要求片段里至少有 _REPEAT_HINT_MIN_LETTERS 个字母类字符（中文算，符号不算）。
#   2. 一个片段一旦重复过，就会在 24 条窗口里连着十几轮被反复报出来；现在只报
#      **她最新这条消息里刚出现**的那些，说过一次就不再念叨。
# 2026-09-15（亲密长会话模板化，见 doc/诊断-20260915-亲密长会话模板化.md）：
# 真机那一场里她的**句式骨架**重复了 15-51 次（「我这边」51、「下面那只手」20、「跟着你的节奏走」15），
# 而提示只在 2-4 轮里提到过它们——因为旧口径要求片段 ≥5 字、且「长片段出现 2 条就报」，
# 于是日常段被长片段刷（17%），亲密段的短骨架反而进不了候选。
# 离线扫描（同一天三段真实样本）后改成：**下限 4 字 + 统一门槛 3 条不同消息**。
# 实测：亲密段命中真实模板的轮数 7%→11%，日常段噪声 14%→10%、下午段 17%→0%。
# 仍然只报「她最新那条里刚出现」的，说过一次就不再念叨（09-13 的教训不变）。
_REPEAT_HINT_WINDOW = 24
_REPEAT_HINT_MIN_MESSAGES = 3
_REPEAT_HINT_MIN_CHARS = 4
_REPEAT_HINT_MAX_CHARS = 12
_REPEAT_HINT_LIMIT = 3
_REPEAT_HINT_MIN_LETTERS = 3


def _repeat_pieces(text: str) -> set[str]:
    """Every 5-12 character fragment of one of her messages worth counting."""

    pieces: set[str] = set()
    for size in range(_REPEAT_HINT_MIN_CHARS, _REPEAT_HINT_MAX_CHARS + 1):
        for start in range(len(text) - size + 1):
            piece = text[start : start + size]
            if piece.strip() != piece or any(char.isdigit() for char in piece):
                continue
            # 颜文字与标点不算话：它们本身不是「说法」，只是语气。
            if sum(1 for char in piece if char.isalpha()) < _REPEAT_HINT_MIN_LETTERS:
                continue
            pieces.add(piece)
    return pieces


def _recent_repeat_hint(events: tuple[ConversationEvent, ...]) -> str | None:
    """Report the phrase she just repeated, as a neutral observation.

    Same shape as the emoji hint: only her own sent text, only counts, and no
    instruction to suppress anything — the model owns the current expression.
    A fragment counts when it is in her newest message and in an earlier one
    (short phrases need two earlier messages, a fragment of at least
    _REPEAT_HINT_LONG_CHARS characters needs one). Only the moment of the
    repeat is reported, and only fragments carrying real words: the first
    version announced kaomoji statistics on every single turn.
    """

    texts = [
        event.text
        for event in events
        if event.actor == "qichi" and isinstance(event.text, str) and event.text.strip()
    ][-_REPEAT_HINT_WINDOW:]
    if len(texts) < 2:
        return None
    newest = _repeat_pieces(texts[-1])
    if not newest:
        return None
    counts: Counter[str] = Counter()
    for text in texts[:-1]:
        counts.update(_repeat_pieces(text))
    repeated = [
        (piece, count + 1)
        for piece, count in counts.items()
        if piece in newest and count >= _REPEAT_HINT_MIN_MESSAGES - 1
    ]
    if not repeated:
        return None
    # 长的更具体：先列最长的，并跳过已经是已列短语子串的那些。
    repeated.sort(key=lambda item: (-len(item[0]), -item[1], item[0]))
    chosen: list[tuple[str, int]] = []
    for piece, count in repeated:
        if any(piece in picked for picked, _ in chosen):
            continue
        chosen.append((piece, count))
        if len(chosen) >= _REPEAT_HINT_LIMIT:
            break
    summary = "、".join(f"「{piece}」×{count}" for piece, count in chosen)
    return (
        "[近期重复表达统计 | 来自角色自己已发送的原文；不代表当前表达选择]\n"
        f"{summary}\n"
        "这些是你自己刚说过的话，不是当前要做的事；下一轮换一种说法，别接着念同一句。"
    )


class ContextBuilder:
    """Build model messages without semantic routing or lossy summaries."""

    def __init__(
        self,
        token_counter: TokenCounter,
        model_capability: ModelCapability,
        *,
        preferred_window_tokens: int = 262_144,
        max_window_tokens: int = 524_288,
        output_reserve_tokens: int = 4_096,
        recent_history_budget_tokens: int = 16_384,
        # 卡B④：常驻工作集自己的 token 上限。它以前只有全局预算那道门，超过就整层消失。
        working_set_max_tokens: int = 6_000,
    ) -> None:
        if not callable(getattr(token_counter, "count_text", None)):
            raise TypeError("token_counter must provide count_text")
        if not isinstance(model_capability, ModelCapability):
            raise TypeError("model_capability must be ModelCapability")
        preferred = _positive_int(preferred_window_tokens, "preferred_window_tokens")
        maximum = _positive_int(max_window_tokens, "max_window_tokens")
        if isinstance(output_reserve_tokens, bool) or not isinstance(output_reserve_tokens, int):
            raise TypeError("output_reserve_tokens must be an int")
        if output_reserve_tokens < 0:
            raise ValueError("output_reserve_tokens must not be negative")
        if preferred > maximum:
            raise ValueError("preferred_window_tokens must not exceed max_window_tokens")
        if output_reserve_tokens >= preferred:
            raise ValueError("output_reserve_tokens must leave a positive preferred input budget")
        if isinstance(recent_history_budget_tokens, bool) or not isinstance(recent_history_budget_tokens, int):
            raise TypeError("recent_history_budget_tokens must be an int")
        if recent_history_budget_tokens < 0:
            raise ValueError("recent_history_budget_tokens must not be negative")
        if isinstance(working_set_max_tokens, bool) or not isinstance(working_set_max_tokens, int):
            raise TypeError("working_set_max_tokens must be an int")
        if working_set_max_tokens < 0:
            raise ValueError("working_set_max_tokens must not be negative")
        self._token_counter = token_counter
        self._model_capability = model_capability
        self._preferred_window_tokens = preferred
        self._max_window_tokens = maximum
        self._output_reserve_tokens = output_reserve_tokens
        self._recent_history_budget_tokens = recent_history_budget_tokens
        self._working_set_max_tokens = working_set_max_tokens

    def build(self, request: ContextBuildRequest) -> ContextBuildResult:
        if not isinstance(request, ContextBuildRequest):
            raise TypeError("request must be ContextBuildRequest")
        self._assert_window_verified(self._preferred_window_tokens)
        prepared = self._prepare(request)
        gap_cache: dict[tuple[str, str], _Piece | None] = {}

        try:
            preferred = self._assemble(
                prepared, self._preferred_window_tokens, gap_cache, expanded=False
            )
        except ContextBudgetError:
            if not self._can_expand():
                raise
            return self._with_prefix_metrics(
                request,
                self._assemble(
                    prepared, self._max_window_tokens, gap_cache, expanded=True
                ).result,
            )

        if preferred.omitted_for_budget and self._can_expand():
            expanded = self._assemble(
                prepared, self._max_window_tokens, gap_cache, expanded=True
            )
            if self._selection_changed(preferred.result, expanded.result):
                return self._with_prefix_metrics(request, expanded.result)
        return self._with_prefix_metrics(request, preferred.result)

    def _with_prefix_metrics(
        self, request: "ContextBuildRequest", result: "ContextBuildResult"
    ) -> "ContextBuildResult":
        """补记"与上一轮的公共前缀"——**只记数字**，不落任何提示正文。

        2026-09-22：真机缓存命中只有 28%，而离线在库副本上构建相邻两轮真实上下文测得的
        公共前缀是 67~70%（稳定半与历史原文都在）。差额要么在供应商侧的命中判定，
        要么在真机/离线之间有我们没看到的差异。把这两个数字与 cache_hit_tokens 并排记，
        跑半天就能判定，不用再猜。
        """

        conversation = getattr(getattr(request, "current_event", None), "conversation_id", None)
        text = "\n".join(str(message) for message in result.messages)
        store = getattr(self, "_last_prompt", None)
        if store is None:
            store = {}
            self._last_prompt = store
        previous = store.get(conversation) if conversation is not None else None
        common = 0
        if isinstance(previous, str):
            limit = min(len(previous), len(text))
            while common < limit and previous[common] == text[common]:
                common += 1
        if conversation is not None:
            store[conversation] = text
        covers = isinstance(previous, str) and common == len(previous) and common > 0
        metrics = replace(
            result.metrics,
            prompt_chars=len(text),
            prompt_prefix_chars=common,
            prompt_covers_previous=covers,
        )
        return replace(result, metrics=metrics)

    def _assert_window_verified(self, window_tokens: int) -> None:
        supported = self._model_capability.supports_context(window_tokens)
        if supported is None:
            raise ProviderCapabilityUnverifiedError("provider context capacity is UNVERIFIED")
        if not supported:
            raise ModelCapabilityError("requested context window exceeds verified capability")

    def _can_expand(self) -> bool:
        return (
            self._max_window_tokens > self._preferred_window_tokens
            and self._model_capability.supports_context(self._max_window_tokens) is True
        )

    @staticmethod
    def _selection_changed(short: ContextBuildResult, large: ContextBuildResult) -> bool:
        return (
            short.metrics.selected_history_event_ids != large.metrics.selected_history_event_ids
            or short.metrics.selected_working_memory_ids
            != large.metrics.selected_working_memory_ids
            or short.metrics.selected_memory_ids != large.metrics.selected_memory_ids
            or short.metrics.input_tokens != large.metrics.input_tokens
        )

    def _count(self, text: str) -> int:
        count = self._token_counter.count_text(text)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TypeError("token counter must return a non-negative int")
        return count

    def _piece(
        self, role: str, content: str, category: str, images: tuple[ModelImage, ...] = ()
    ) -> _Piece:
        message = ModelMessage(role=role, content=content, images=images)
        return _Piece(category, message, self._count(content) + _IMAGE_TOKEN_ESTIMATE * len(images))

    def _event_pieces(
        self,
        event: ConversationEvent,
        label: str,
        category: str,
        images: tuple[ModelImage, ...] = (),
    ) -> tuple[_Piece, ...]:
        if images and event.actor == "platform":
            raise ContextValidationError("a platform message cannot carry images")
        metadata = _render_event_metadata(event, label)
        replayed = category in {"recent_history", "earlier_history"}
        if event.actor == "qichi" and replayed:
            # Historical assistant text is evidence of what Qichi previously
            # said, not a previous generation turn. Keeping it out of the
            # assistant role prevents old roleplay or formatting from becoming
            # an implicit writing template for the current reply.
            return (self._piece("system", f"{metadata}\n{_history_body(event)}", category),)
        pieces = [self._piece("system", metadata, category)]
        if event.actor != "platform":
            body = _history_body(event) if replayed else (event.text or "")
            pieces.append(self._piece(_event_role(event), body, category, images))
        return tuple(pieces)

    def _prepare(self, request: ContextBuildRequest) -> _Prepared:
        self._validate_sources(request)
        now = request.runtime_facts.current_time
        evidence_events = request.evidence_events

        relationship_pieces: list[_Piece] = []
        relationship_omitted = 0
        remembered_count = 0
        relationship_memory_ids: set[str] = set()
        relationship_ids: set[tuple[str, str]] = set()
        for item in request.relationship_state:
            if isinstance(item, MemoryRecord):
                key = ("memory", item.memory_id)
                if key in relationship_ids:
                    raise ContextValidationError("duplicate relationship memory ID")
                relationship_ids.add(key)
                relationship_memory_ids.add(item.memory_id)
                if item.privacy_class != "ordinary" and not request.allow_sensitive_memory:
                    relationship_omitted += 1
                    continue
                if item.status != "active":
                    raise ContextValidationError("relationship memory must be active")
                if not _valid_at(item.valid_from_utc, item.valid_until_utc, now):
                    relationship_omitted += 1
                    continue
                self._validate_evidence(item.evidence, evidence_events, request.current_event.conversation_id)
                # 2026-09-18 呈现层：常驻的「她自己记着的事」只写一行归一事实——不带
                # memory_id、不带证据、不带条数。目的是让它像本来就知道的事，而不是
                # 一份可核对的台账（doc/方案-20260918-常驻记忆与主动提起.md §5/§9）。
                if item.type == "episode":
                    remembered_count += 1
                    content = self._render_remembered(item)
                else:
                    content = self._render_memory(item, evidence_events, relationship=True)
            elif isinstance(item, Agreement):
                key = ("agreement", item.agreement_id)
                if key in relationship_ids:
                    raise ContextValidationError("duplicate agreement ID")
                relationship_ids.add(key)
                if item.status != "pending" or not _valid_at(item.valid_from_utc, item.valid_until_utc, now):
                    relationship_omitted += 1
                    continue
                self._validate_evidence(item.evidence, evidence_events, request.current_event.conversation_id)
                content = self._render_agreement(item, evidence_events)
            else:
                key = ("correction", item.correction_id)
                if key in relationship_ids:
                    raise ContextValidationError("duplicate correction ID")
                relationship_ids.add(key)
                self._validate_evidence(item.evidence, evidence_events, request.current_event.conversation_id)
                content = self._render_correction(item, evidence_events)
            relationship_pieces.append(self._piece("system", content, "relationship_state"))
        if remembered_count:
            # 用户 2026-09-18 裁定保留这句：它是边界（她有权不提），不是行为要求。
            relationship_pieces.append(
                self._piece("system", REMEMBERED_NOTE, "relationship_state")
            )

        working_records: list[MemoryRecord] = []
        working_omitted = 0
        for item in request.memory_working_set:
            if item.status != "active":
                raise ContextValidationError("working set memory must be active")
            if item.certainty not in {"explicit", "confirmed"} or item.temporal_scope == "unclassified":
                raise ContextValidationError("working set memory must be assessed")
            if item.privacy_class != "ordinary" and not request.allow_sensitive_memory:
                working_omitted += 1
                continue
            if not _valid_at(item.valid_from_utc, item.valid_until_utc, now):
                working_omitted += 1
                continue
            self._validate_evidence(
                item.evidence,
                evidence_events,
                request.current_event.conversation_id,
            )
            working_records.append(item)
        working_records.sort(key=self._working_set_sort_key)
        memory_working_set = None
        if working_records:
            # 卡B④：先按优先级截断到常驻层自己的预算，再渲染。装不下时丢尾部，
            # 而不是像以前那样整层丢掉（那种退化方式在预算吃紧时会一次全瞎）。
            kept, dropped = _fit_working_records(
                working_records,
                lambda subset: self._render_memory_working_set(tuple(subset), evidence_events),
                self._count,
                self._working_set_max_tokens,
            )
            working_omitted += dropped
            working_records = list(kept)
        if working_records:
            content = self._render_memory_working_set(tuple(working_records), evidence_events)
            memory_working_set = _MemoryWorkingSetPiece(
                tuple(item.memory_id for item in working_records),
                self._piece("system", content, "memory_working_set"),
            )

        memory_pieces: list[_MemoryPiece] = []
        memory_seen: set[str] = set()
        memory_base_omitted = 0
        for item in request.memory_candidates:
            if item.memory_id in memory_seen:
                raise ContextValidationError("duplicate memory candidate ID")
            memory_seen.add(item.memory_id)
            if item.status != "active":
                raise ContextValidationError("memory candidate must be active")
            if item.privacy_class != "ordinary" and not request.allow_sensitive_memory:
                memory_base_omitted += 1
                continue
            if item.memory_id in relationship_memory_ids or not _valid_at(
                item.valid_from_utc, item.valid_until_utc, now
            ):
                memory_base_omitted += 1
                continue
            self._validate_evidence(item.evidence, evidence_events, request.current_event.conversation_id)
            if len(memory_pieces) >= 12:
                memory_base_omitted += 1
                continue
            content = self._render_memory(item, evidence_events, relationship=False)
            memory_pieces.append(_MemoryPiece(item.memory_id, self._piece("system", content, "memory_evidence")))
        for item in request.confirmation_candidates[:1]:
            if item.privacy_class != "ordinary" and not request.allow_sensitive_memory:
                continue
            if not _valid_at(item.valid_from_utc, item.valid_until_utc, now):
                continue
            self._validate_evidence(item.evidence, evidence_events, request.current_event.conversation_id)
            content = "[待确认记忆，不是事实]\n" + self._render_memory(item, evidence_events, relationship=False)
            memory_pieces.append(_MemoryPiece(item.memory_id, self._piece("system", content, "memory_evidence")))

        detail_items: list[MemoryDetailRecord] = []
        detail_omitted = 0
        detail_seen: set[str] = set()
        for item in request.memory_details:
            if item.detail_id in detail_seen:
                raise ContextValidationError("duplicate memory detail ID")
            detail_seen.add(item.detail_id)
            if item.status not in {"candidate", "active"}:
                detail_omitted += 1
                continue
            if item.privacy_class != "ordinary" and not (
                request.allow_sensitive_details or request.allow_sensitive_memory
            ):
                detail_omitted += 1
                continue
            self._validate_detail(item, evidence_events, request.current_event.conversation_id)
            detail_items.append(item)
        detail_piece = None
        if detail_items:
            detail_piece = self._piece(
                "system",
                self._render_memory_details(
                    tuple(detail_items),
                    request.memory_detail_note,
                    request.memory_detail_labels,
                ),
                "memory_details",
            )

        footprint_piece = None
        if request.memory_footprint:
            rendered_footprint = self._render_memory_footprint(
                request.memory_footprint, request.memory_footprint_labels
            )
            if rendered_footprint:
                footprint_piece = self._piece("system", rendered_footprint, "memory_footprint")

        index_piece = None
        if request.memory_index or request.memory_recall_note:
            index_piece = self._piece(
                "system",
                self._render_memory_index(request.memory_index, request.memory_recall_note),
                "memory_index",
            )

        excluded_ids = {request.current_event.event_id, *(event.event_id for event in request.quoted_chain)}
        recent_events = self._history_events(request.recent_events, excluded_ids)
        recent_ids = {event.event_id for event in recent_events}
        earlier_events = self._history_events(request.earlier_events, excluded_ids | recent_ids)
        facts_content = render_stable_facts(request.runtime_facts)
        turn_facts_content = render_turn_facts(request.runtime_facts)
        emoji_hint = _recent_emoji_hint(recent_events)
        if emoji_hint is not None:
            turn_facts_content = f"{turn_facts_content}\n{emoji_hint}"
        repeat_hint = _recent_repeat_hint(recent_events)
        if repeat_hint is not None:
            turn_facts_content = f"{turn_facts_content}\n{repeat_hint}"

        recent = tuple(
            _HistoryPiece(
                event,
                self._event_pieces(event, "历史原文", "recent_history"),
            )
            for event in recent_events
        )
        earlier = tuple(
            _HistoryPiece(
                event,
                self._event_pieces(event, "更早原文", "earlier_history"),
            )
            for event in earlier_events
        )
        quotes = tuple(
            self._piece("system", _render_event(event, "直接引用"), "direct_quotes")
            for event in request.quoted_chain
        )
        current = self._event_pieces(
            request.current_event,
            "当前输入",
            "current_input",
            request.current_images,
        )
        return _Prepared(
            role=self._piece("system", request.role_core, "role_core"),
            facts=self._piece("system", facts_content, "runtime_facts"),
            facts_turn=self._piece("system", turn_facts_content, "runtime_facts"),
            relationships=tuple(relationship_pieces),
            relationship_omitted=relationship_omitted,
            memory_working_set=memory_working_set,
            memory_working_set_omitted=working_omitted,
            recent=recent,
            memories=tuple(memory_pieces),
            memory_base_omitted=memory_base_omitted,
            memory_details=detail_piece,
            memory_details_omitted=detail_omitted,
            memory_index=index_piece,
            memory_index_omitted=0,
            memory_footprint=footprint_piece,
            earlier=earlier,
            quotes=quotes,
            current=current,
            current_event=request.current_event,
        )

    def _render_memory_index(self, lines: tuple[str, ...], note: str = "") -> str:
        body = chr(10).join(f"- {line}" for line in lines)
        text = f"{MEMORY_INDEX_HEADER}{chr(10)}{body}" if lines else MEMORY_INDEX_HEADER
        return f"{text}{chr(10)}{note}" if note else text

    def _validate_sources(self, request: ContextBuildRequest) -> None:
        conversation_id = request.current_event.conversation_id
        initiative_shape = (
            request.current_event.direction == "internal"
            and request.current_event.actor == "platform"
            and request.current_event.kind == "initiative"
        )
        if (
            request.current_event.direction == "internal"
            or request.current_event.actor == "platform"
            or request.current_event.kind == "initiative"
        ) and not initiative_shape:
            raise ContextValidationError(
                "only internal initiative platform event may be current"
            )
        if not _source_matches(request.runtime_facts.current_source, request.current_event):
            raise ContextValidationError("runtime current source does not match current event")

        mandatory_ids = {request.current_event.event_id}
        for quoted in request.quoted_chain:
            if quoted.direction == "internal":
                raise ContextValidationError("quoted event must not be internal")
            if quoted.event_id in mandatory_ids:
                raise ContextValidationError("current and quoted events must be unique")
            mandatory_ids.add(quoted.event_id)
        expected_quote = request.quoted_chain[0] if request.quoted_chain else None
        actual_quote = request.runtime_facts.quoted_source
        if expected_quote is None:
            if actual_quote is not None:
                raise ContextValidationError("runtime quoted source has no quoted event")
        elif actual_quote is None or not _source_matches(actual_quote, expected_quote):
            raise ContextValidationError("runtime quoted source does not match direct quote")

        all_events = (
            (request.current_event,)
            + request.quoted_chain
            + request.recent_events
            + request.earlier_events
            + tuple(request.evidence_events.values())
        )
        by_id: dict[str, ConversationEvent] = {}
        for event in all_events:
            if event.conversation_id != conversation_id:
                raise ContextValidationError("all context events must belong to the current conversation")
            prior = by_id.get(event.event_id)
            if prior is not None and prior != event:
                raise ContextValidationError("event ID resolves to conflicting event records")
            by_id[event.event_id] = event

    @staticmethod
    def _history_events(
        events: tuple[ConversationEvent, ...], excluded_ids: set[str]
    ) -> tuple[ConversationEvent, ...]:
        unique: dict[str, ConversationEvent] = {}
        for event in events:
            if event.direction == "internal" or event.event_id in excluded_ids:
                continue
            unique.setdefault(event.event_id, event)
        return tuple(sorted(unique.values(), key=lambda event: (event.sequence, event.event_id)))

    @staticmethod
    def _validate_evidence(
        evidence_items: tuple[MemoryEvidence, ...],
        evidence_events: Mapping[str, ConversationEvent],
        conversation_id: str,
    ) -> None:
        for evidence in evidence_items:
            event = evidence_events.get(evidence.event_id)
            if event is None or event.conversation_id != conversation_id:
                raise ContextValidationError("memory evidence event is unavailable")
            if event.actor != evidence.actor:
                raise ContextValidationError("memory evidence actor does not match source event")
            if event.occurred_at_utc != evidence.occurred_at_utc:
                raise ContextValidationError("memory evidence time does not match source event")
            if not quote_is_verbatim(evidence.exact_quote, event.text):
                raise ContextValidationError("memory evidence exact quote is absent from source event")

    @staticmethod
    def _render_evidence(
        evidence_items: tuple[MemoryEvidence, ...], evidence_events: Mapping[str, ConversationEvent]
    ) -> str:
        lines: list[str] = []
        for evidence in evidence_items:
            source = evidence_events[evidence.event_id]
            handle = source.visible_handle if source.visible_handle is not None else "none"
            quote = json.dumps(evidence.exact_quote, ensure_ascii=False)
            lines.append(
                f"证据 event_id={evidence.event_id}; actor={evidence.actor}; handle={handle}; "
                f"time={_local_iso(evidence.occurred_at_utc)}; exact_quote={quote}"
            )
        return "\n".join(lines)

    def _render_remembered(self, record: MemoryRecord) -> str:
        """常驻层的一行：她自己记着的一件事。

        只有归一事实：不带 memory_id、不带证据、不带条数——这一层的目的是让记忆像
        「本来就知道的事」，而不是一份可核对的台账（2026-09-18 呈现层方案 §5/§9）。
        文字逐字来自库里的 normalized_fact，代码不生成新句子。
        """

        return f"- {record.normalized_fact}"

    def _render_memory(
        self,
        record: MemoryRecord,
        evidence_events: Mapping[str, ConversationEvent],
        *,
        relationship: bool,
    ) -> str:
        label = (
            "用户已确认"
            if relationship and all(item.actor == "mumo" for item in record.evidence)
            else "关系状态" if relationship else "过去背景"
        )
        header = f"[{label} | memory_id={record.memory_id}; type={record.type}; status={record.status}]"
        if record.privacy_class != "ordinary" or record.recall_policy != "daily_safe":
            header = (
                f"[{label} | memory_id={record.memory_id}; type={record.type}; status={record.status}; "
                f"privacy={record.privacy_class}; recall={record.recall_policy}]"
            )
        return "\n".join((header, f"归一事实: {record.normalized_fact}", self._render_evidence(record.evidence, evidence_events)))

    @staticmethod
    def _render_memory_footprint(
        details: tuple[MemoryDetailRecord, ...], labels: Mapping[str, str] | None = None
    ) -> str:
        """最近原文足迹：最近几天保存下来的逐字原话，常驻在上下文里。

        2026-09-12 真机：她被转述式追问往事时只能回答「原话我翻不到，不给你编」——
        因为常驻层只有改写（工作集只有归一事实、索引只有存在性），逐字原话只在钥匙
        开门时注入。这里把「最近的原文」变成常驻事实：她要能认出「这事确实说过」。
        成人与非 daily 的条目一律不进（隐私门与既有召回门一致）。
        """

        names = labels or {}
        keep = tuple(
            item
            for item in details
            if item.privacy_class == "ordinary"
            and item.recall_policy == "daily_safe"
            and len(item.exact_quote) <= _FOOTPRINT_QUOTE_MAX_CHARS
        )
        if not keep:
            return ""
        lines = [
            "[最近原文足迹 | 下面每一行都是过去确实说过的逐字原话：不是摘要、不是本轮话题、"
            "也不代表当前同意。被问到最近聊过什么、或需要引用原话时以它们为准；"
            "列表之外的内容不要编造，也不要把它们说成正在发生的事]",
        ]
        current_fragment: str | None = None
        for item in keep:
            if item.fragment_id != current_fragment:
                current_fragment = item.fragment_id
                lines.append(f"[片段 {names.get(current_fragment, current_fragment)}]")
            clock = item.occurred_at_utc.astimezone(_LOCAL_ZONE).strftime("%m-%d %H:%M")
            lines.append(
                f"- {clock} {item.actor}: {json.dumps(item.exact_quote, ensure_ascii=False)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _validate_detail(
        detail: MemoryDetailRecord,
        evidence_events: Mapping[str, ConversationEvent],
        conversation_id: str,
    ) -> None:
        source = evidence_events.get(detail.source_event_id)
        if source is None or source.conversation_id != conversation_id:
            raise ContextValidationError("memory detail source event is unavailable")
        if source.occurred_at_utc != detail.occurred_at_utc:
            raise ContextValidationError("memory detail time does not match source event")
        if detail.actor in {"mumo", "qichi"} and source.actor != detail.actor:
            raise ContextValidationError("memory detail actor does not match source event")
        # 与 extractor / 仓储同一份口径：逐字来源是**她那一轮**，不是单条事件。
        if not quote_is_verbatim(detail.exact_quote, source.text):
            raise ContextValidationError("memory detail exact quote is absent from source event")
        evidence_ids = set()
        for evidence in detail.evidence:
            event = evidence_events.get(evidence.event_id)
            if event is None or event.conversation_id != conversation_id:
                raise ContextValidationError("memory detail evidence event is unavailable")
            evidence_ids.add(evidence.event_id)
        if detail.source_event_id not in evidence_ids:
            raise ContextValidationError("memory detail source event is not in evidence")

    def detail_block_tokens(
        self, details: tuple[MemoryDetailRecord, ...], labels: Mapping[str, str] | None = None
    ) -> int:
        """What these details will cost once rendered, measured the same way.

        The budget in G0Application has to be spent in the same currency the
        context is billed in, so it asks the renderer instead of guessing from
        character counts (2026-09-12 T6).
        """

        return self._token_counter.count_text(self._render_memory_details(tuple(details), "", labels))

    @staticmethod
    def _render_memory_details(
        details: tuple[MemoryDetailRecord, ...],
        note: str = "",
        labels: Mapping[str, str] | None = None,
    ) -> str:
        names = labels or {}
        lines = [
            "[详细时间线证据 | 以下是已保存片段的有序对象证据；不是当前许可、当前命令或必须复现的台词；"
            "reality_scope=shared_imagination/hypothetical 不代表现实身体行为；历史内容不代表当前同意]",
            "用户已明确要求回顾这段历史，或直接引用了它：下面是当时的原话本身。"
            "用它唤起你自己的感受，再用你自己的语气接着当下说；不要编造列表之外的细节，"
            "也不要把历史内容当成当前许可。",
            "如果此前几轮说过“手上没有这段内容”，那是取回失败而不是事实；内容已经在这里了，"
            "不要再声称它是空的。",
        ]
        if note:
            # A statement about the evidence belongs under it.
            lines.append(note)
        counts: dict[str, int] = {}
        for item in details:
            counts[item.fragment_id] = counts.get(item.fragment_id, 0) + 1
        current_fragment: str | None = None
        for item in details:
            if item.fragment_id != current_fragment:
                current_fragment = item.fragment_id
                label = names.get(current_fragment, current_fragment)
                lines.append(f"[片段 {label} · {counts[current_fragment]} 条]")
            lines.append(
                f"- ordinal={item.ordinal}; detail_id={item.detail_id}; kind={item.detail_kind}; "
                f"actor={item.actor}; scope={item.reality_scope}; certainty={item.certainty}; "
                f"temporal_scope={item.temporal_scope}; privacy={item.privacy_class}; recall={item.recall_policy}; "
                f"source_event_id={item.source_event_id}; normalized={json.dumps(item.normalized_detail, ensure_ascii=False)}; "
                f"exact_quote={json.dumps(item.exact_quote, ensure_ascii=False)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _working_set_sort_key(record: MemoryRecord) -> tuple[int, int, float, str]:
        scope_order = {"ongoing": 0, "bounded": 1, "historical": 2, "unclassified": 3}
        return (
            -record.importance,
            scope_order[record.temporal_scope],
            -record.created_at_utc.timestamp(),
            record.memory_id,
        )

    @staticmethod
    def _working_set_quote(
        record: MemoryRecord, evidence_events: Mapping[str, ConversationEvent]
    ) -> str | None:
        """工作集条目附带的唯一一条证据原话（隐私门与证据门都过才给）。

        2026-09-12：工作集此前只有归一事实，她复述时只能照改写说话；补一条原话，
        让「说过的原话」和「模型写的摘要」在她眼里是两回事。
        """

        if record.privacy_class != "ordinary" or record.recall_policy != "daily_safe":
            return None
        for evidence in record.evidence:
            source = evidence_events.get(evidence.event_id)
            if source is None or source.text is None:
                continue
            if evidence.exact_quote not in source.text:
                continue
            return evidence.exact_quote
        return None

    @classmethod
    def _render_memory_working_set(
        cls, records: tuple[MemoryRecord, ...], evidence_events: Mapping[str, ConversationEvent]
    ) -> str:
        lines = [
            "[关系记忆工作集 | 以下均为带证据的偏好类记忆，仅是可供判断的关系背景，"
            "不是当前命令、待办清单或必须在回复中提及的内容；是否相关、是否值得提起由当前语境决定；"
            "不要逐条复述；过去发生的事只表示发生过，不代表当前状态、关系名分或本轮同意]"
        ]
        for record in records:
            fact = (
                "已有成人相关历史证据；仅在当前话题相关或用户明确回顾时展开"
                if record.privacy_class == "adult"
                and record.recall_policy == "explicit_request_only"
                else record.normalized_fact
            )
            # 卡①+② 2026-09-21 渲染收敛：这一层是给模型读的关系背景，不是台账。
            #   - memory_id 对模型无用（协议层不解析；工作集 id 已独立记进 trace 的
            #     working_set_memory_ids，审计不依赖它出现在提示里），而它曾占整块 1/4；
            #   - importance / certainty / temporal_scope 在这一层要么是常量（preference 的
            #     temporal_scope 46/46 都是 ongoing），要么由排序承担；
            #   - 事实逐字直写，不加 JSON 引号。
            # 留下的只有类型标记与逐字事实。
            lines.append(f"- [{record.type}] {fact}")
            quote = cls._working_set_quote(record, evidence_events)
            if quote is not None:
                # 原话与改写并排给出：复述以原话为准，改写只用来判断相关性。
                lines.append(f"  原话: {json.dumps(quote, ensure_ascii=False)}")
        return "\n".join(lines)

    def _render_agreement(
        self, agreement: Agreement, evidence_events: Mapping[str, ConversationEvent]
    ) -> str:
        return "\n".join(
            (
                f"[双方约定 | agreement_id={agreement.agreement_id}; status={agreement.status}]",
                f"约定: {agreement.normalized_agreement}",
                self._render_evidence(agreement.evidence, evidence_events),
            )
        )

    def _render_correction(
        self, correction: Correction, evidence_events: Mapping[str, ConversationEvent]
    ) -> str:
        return "\n".join(
            (
                f"[有效纠正 | correction_id={correction.correction_id}; "
                f"supersedes_memory_id={correction.supersedes_memory_id}]",
                f"纠正事实: {correction.normalized_fact}",
                self._render_evidence(correction.evidence, evidence_events),
            )
        )

    def _gap_piece(
        self,
        left: ConversationEvent,
        right: ConversationEvent,
        gap_cache: dict[tuple[str, str], _Piece | None],
    ) -> _Piece | None:
        key = (left.event_id, right.event_id)
        if key in gap_cache:
            return gap_cache[key]
        delta = right.occurred_at_utc - left.occurred_at_utc
        if delta < _LONG_GAP:
            gap_cache[key] = None
            return None
        content = (
            f"[时间经过: {_duration_text(delta)}; 从 {_local_iso(left.occurred_at_utc)} "
            f"到 {_local_iso(right.occurred_at_utc)} (Asia/Shanghai)]"
        )
        piece = self._piece("system", content, "time_gaps")
        gap_cache[key] = piece
        return piece

    def _history_insert_cost(
        self,
        selected: list[_HistoryPiece],
        candidate: _HistoryPiece,
        current: ConversationEvent,
        gap_cache: dict[tuple[str, str], _Piece | None],
    ) -> tuple[int, int]:
        keys = [(item.event.sequence, item.event.event_id) for item in selected]
        index = bisect_left(keys, (candidate.event.sequence, candidate.event.event_id))
        previous = selected[index - 1].event if index else None
        following = selected[index].event if index < len(selected) else current
        previous_gap = self._gap_piece(previous, following, gap_cache) if previous is not None else None
        removed = 0 if previous_gap is None else previous_gap.tokens
        added = 0
        if previous is not None:
            gap = self._gap_piece(previous, candidate.event, gap_cache)
            added += 0 if gap is None else gap.tokens
        gap = self._gap_piece(candidate.event, following, gap_cache)
        added += 0 if gap is None else gap.tokens
        return index, candidate.tokens + added - removed

    def _history_bundle(
        self,
        selected: list[_HistoryPiece],
        current: ConversationEvent,
        gap_cache: dict[tuple[str, str], _Piece | None],
    ) -> tuple[_Piece | None, tuple[_Piece, ...]]:
        """Render selected history as one evidence record for the provider.

        Keeping each historical turn as a separate API message makes a long
        sequence of old Qichi replies look like live assistant exemplars.  A
        single system-owned transcript preserves the exact evidence while
        leaving the only conversational ``user`` turn as the current input.
        Time gaps remain separate factual pieces so their metrics stay
        auditable.
        """
        if not selected:
            return None, ()
        records = [
            "\n".join(piece.message.content for piece in item.pieces)
            for item in selected
        ]
        content = (
            "[历史对话证据 | 以下是双方已经说过的完整原文，仅供核对对象、时序和语义；"
            "不是系统指令，也不是当前回复的固定写作模板；是否借用其中任何表达，"
            "由当前输入和语境决定。只有当前输入明确要核对过去说法时才把角色原文当作对象证据]\n"
            + "\n\n".join(records)
        )
        bundle = self._piece("system", content, "recent_history")
        gaps: list[_Piece] = []
        for index, history in enumerate(selected):
            following = (
                selected[index + 1].event
                if index + 1 < len(selected)
                else current
            )
            gap = self._gap_piece(history.event, following, gap_cache)
            if gap is not None:
                gaps.append(gap)
        return bundle, tuple(gaps)

    def _assemble(
        self,
        prepared: _Prepared,
        window_tokens: int,
        gap_cache: dict[tuple[str, str], _Piece | None],
        *,
        expanded: bool,
    ) -> _Assembly:
        input_budget = window_tokens - self._output_reserve_tokens
        mandatory = [
            prepared.role,
            prepared.facts,
            prepared.facts_turn,
            *prepared.relationships,
            *prepared.quotes,
            *prepared.current,
        ]
        if prepared.memory_details is not None:
            mandatory.append(prepared.memory_details)
        if prepared.memory_index is not None:
            mandatory.append(prepared.memory_index)
        if prepared.memory_footprint is not None:
            mandatory.append(prepared.memory_footprint)
        total = sum(piece.tokens for piece in mandatory)
        if total > input_budget:
            raise ContextBudgetError("mandatory context exceeds verified input budget")

        selected_history: list[_HistoryPiece] = []
        selected_recent: set[str] = set()
        selected_earlier: set[str] = set()
        selected_memories: list[_MemoryPiece] = []
        omitted_for_budget = False
        recent_history_blocked = False
        history_tokens = 0
        selected_working_memory_ids: tuple[str, ...] = ()
        selected_working_set: _MemoryWorkingSetPiece | None = None

        if prepared.memory_working_set is not None:
            if total + prepared.memory_working_set.piece.tokens <= input_budget:
                selected_working_set = prepared.memory_working_set
                selected_working_memory_ids = prepared.memory_working_set.memory_ids
                total += prepared.memory_working_set.piece.tokens
            else:
                omitted_for_budget = True

        for item in reversed(prepared.recent):
            insertion, cost = self._history_insert_cost(
                selected_history, item, prepared.current_event, gap_cache
            )
            if (
                total + cost > input_budget
                or history_tokens + cost > self._recent_history_budget_tokens
            ):
                omitted_for_budget = True
                recent_history_blocked = True
                break
            selected_history.insert(insertion, item)
            selected_recent.add(item.event.event_id)
            total += cost
            history_tokens += cost

        for item in prepared.memories:
            if total + item.piece.tokens > input_budget:
                omitted_for_budget = True
                break
            selected_memories.append(item)
            total += item.piece.tokens

        if not recent_history_blocked:
            for item in reversed(prepared.earlier):
                insertion, cost = self._history_insert_cost(
                    selected_history, item, prepared.current_event, gap_cache
                )
                if (
                    total + cost > input_budget
                    or history_tokens + cost > self._recent_history_budget_tokens
                ):
                    omitted_for_budget = True
                    break
                selected_history.insert(insertion, item)
                selected_earlier.add(item.event.event_id)
                total += cost
                history_tokens += cost

        # Selection uses per-event costs for efficient newest-first packing,
        # then this final pass accounts for the transcript header and joins
        # exactly.  If those boundaries consume the last few tokens, drop the
        # oldest selected event and recompute rather than exceeding the gate.
        history_bundle, history_gaps = self._history_bundle(
            selected_history, prepared.current_event, gap_cache
        )
        base_total = sum(piece.tokens for piece in mandatory) + (
            0 if selected_working_set is None else selected_working_set.piece.tokens
        ) + sum(
            item.piece.tokens for item in selected_memories
        )
        while history_bundle is not None:
            history_total = history_bundle.tokens + sum(piece.tokens for piece in history_gaps)
            if (
                history_total <= self._recent_history_budget_tokens
                and base_total + history_total <= input_budget
            ):
                break
            removed = selected_history.pop(0)
            selected_recent.discard(removed.event.event_id)
            selected_earlier.discard(removed.event.event_id)
            omitted_for_budget = True
            history_bundle, history_gaps = self._history_bundle(
                selected_history, prepared.current_event, gap_cache
            )

        total = base_total
        if history_bundle is not None:
            total += history_bundle.tokens + sum(piece.tokens for piece in history_gaps)

        # 顺序就是钱：供应商的前缀缓存只认从第一个 token 起完全一致的前缀单元，
        # 所以稳定的东西（角色核心、稳定事实、关系状态、纯追加的历史原文）排在
        # 前面，逐轮变化的检索块与本轮事实排在后面（2026-09-12 实测：钟在第二行
        # 时全轮命中只有 6.6%，把易变块后置后稳定前缀能吃下整段历史）。
        # 注意：prepared.facts 是**每轮逐字相同**的「稳定半」（见 render_stable_facts 的
        # docstring），必须留在最前——2026-09-12 实测：把每轮都变的那行（时钟）放在第二位，
        # 会把每轮命中率压到 6.6%；同理把稳定半本身挪到尾部也会让回复变短变平
        # （副本 15 轮：63 字 → 79 字）。所以这里保持 role / facts / relationships 的次序。
        pieces: list[_Piece] = [prepared.role, prepared.facts, *prepared.relationships]
        if history_bundle is not None:
            pieces.append(history_bundle)
            pieces.extend(history_gaps)
        if prepared.memory_index is not None:
            pieces.append(prepared.memory_index)
        if prepared.memory_footprint is not None:
            pieces.append(prepared.memory_footprint)
        # 2026-09-22 曾按「变化频率升序」把 memory_details（按查询取、最易每轮不同）从
        # working_set 之前挪到之后，理由是「它一变会把后面稳定的 working_set 一起踢出缓存」。
        #
        # 2026-09-24 **实测推翻了那条理由**（doc/待检测-20260923-缓存成本与常驻记忆.md §3.3）：
        # 相邻两轮的公共前缀在**历史原文块内部**就断了（历史排在检索块之前、且每轮追加），
        # 48 轮里「公共前缀 >= working_set 起点」的轮数 = **0**——两个块**都不在可缓存前缀里**，
        # 谁也踢不到谁；两臂逐轮 prompt_prefix_chars 完全相同（48/48），命中不可能有差别。
        # 按事先判决线属「命中不动 -> 回滚」：删掉 A/B 用的臂开关，**块序维持现状**
        # （两种序实测等价，改回去只是没有收益的改动）。
        if selected_working_set is not None:
            pieces.append(selected_working_set.piece)
        if prepared.memory_details is not None:
            pieces.append(prepared.memory_details)
        pieces.extend(item.piece for item in selected_memories)
        pieces.extend(prepared.quotes)
        pieces.append(prepared.facts_turn)
        pieces.extend(prepared.current)

        input_tokens = sum(piece.tokens for piece in pieces)
        if input_tokens != total:
            raise RuntimeError("context token accounting mismatch")
        if input_tokens > input_budget:
            raise ContextBudgetError("context exceeds verified input budget")

        category_tokens = {key: 0 for key in _CATEGORY_KEYS}
        for piece in pieces:
            category_tokens[piece.category] += piece.tokens
        omitted_counts = {
            "relationship_state": prepared.relationship_omitted,
            "memory_working_set": prepared.memory_working_set_omitted + (
                0
                if prepared.memory_working_set is None or selected_working_set is not None
                else len(prepared.memory_working_set.memory_ids)
            ),
            "recent_history": len(prepared.recent) - len(selected_recent),
            "memory_evidence": prepared.memory_base_omitted + len(prepared.memories) - len(selected_memories),
            "memory_details": prepared.memory_details_omitted,
            "memory_index": prepared.memory_index_omitted,
            "earlier_history": len(prepared.earlier) - len(selected_earlier),
        }
        metrics = ContextTokenMetrics(
            input_tokens=input_tokens,
            input_budget_tokens=input_budget,
            window_tokens=window_tokens,
            expanded=expanded,
            category_tokens=MappingProxyType(category_tokens),
            omitted_counts=MappingProxyType(omitted_counts),
            selected_history_event_ids=tuple(item.event.event_id for item in selected_history),
            selected_working_memory_ids=selected_working_memory_ids,
            selected_memory_ids=tuple(item.memory_id for item in selected_memories),
        )
        return _Assembly(ContextBuildResult(tuple(piece.message for piece in pieces), metrics), omitted_for_budget)
