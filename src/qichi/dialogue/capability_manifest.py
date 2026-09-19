"""Build a factual runtime envelope without semantic routing instructions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Literal, TypeAlias
from zoneinfo import ZoneInfo

from qichi.dialogue.calendar_facts import render_calendar_facts
from qichi.domain.dialogue import CapabilityManifest
from qichi.domain.events import Actor


MediaKind: TypeAlias = Literal["text", "image", "face", "poke", "tool_result"]
ActionKind: TypeAlias = Literal["text", "reply", "qq_face", "reaction", "poke", "voice"]
QuoteResolutionStatus: TypeAlias = Literal["none", "resolved", "unavailable"]

_MEDIA_KINDS = frozenset({"text", "image", "face", "poke", "tool_result"})
_ACTION_KINDS = frozenset({"text", "reply", "qq_face", "reaction", "poke", "voice"})
_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")
_HANDLE_PATTERNS = {
    "mumo": re.compile(r"M(?:0|[1-9][0-9]*)"),
    "qichi": re.compile(r"Q(?:0|[1-9][0-9]*)"),
}
_SEMANTIC_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_LOCAL_ZONE = ZoneInfo("Asia/Shanghai")


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _enum_tuple(value: tuple[str, ...], field: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a tuple of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field} contains unsupported values")
    return value


def _semantic_keys(value: tuple[str, ...], field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(type(item) is not str for item in value):
        raise TypeError(f"{field} must be a tuple of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    if any(_SEMANTIC_KEY.fullmatch(item) is None for item in value):
        raise ValueError(f"{field} contains an invalid semantic key")
    return value


def _local_iso(value: datetime) -> str:
    return value.astimezone(_LOCAL_ZONE).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class MessageSourceFact:
    actor: Actor
    handle: str | None
    kind: str
    occurred_at_utc: datetime

    def __post_init__(self) -> None:
        if self.actor not in {"mumo", "qichi", "platform"}:
            raise ValueError("actor must be mumo, qichi, or platform")
        if not isinstance(self.kind, str) or _KIND_PATTERN.fullmatch(self.kind) is None:
            raise ValueError("kind must be a simple lowercase identifier")
        if self.actor == "platform":
            if self.handle is not None:
                raise ValueError("platform source must not have a visible handle")
        else:
            if not isinstance(self.handle, str) or _HANDLE_PATTERNS[self.actor].fullmatch(self.handle) is None:
                raise ValueError("handle does not match actor")
        object.__setattr__(self, "occurred_at_utc", _aware_utc(self.occurred_at_utc, "occurred_at_utc"))


@dataclass(frozen=True, slots=True)
class RuntimeFacts:
    current_time: datetime
    seconds_since_last_message: int | None
    current_source: MessageSourceFact
    quoted_source: MessageSourceFact | None
    received_media: tuple[MediaKind, ...]
    available_actions: tuple[ActionKind, ...]
    vision_available: bool
    external_tools_available: bool
    available_qq_face_keys: tuple[str, ...] = ()
    available_reaction_keys: tuple[str, ...] = ()
    # 2026-09-14：本轮能不能把某一段改用语音说。只有合成链路**真的就绪**时才为真，
    # 否则她会说"我发语音给你"却发不出来（能力边界）。
    voice_available: bool = False
    # 2026-09-15：一段最多多少字。这是**边界事实**，不是建议——真机上她两次写了 127 字，
    # 超过 120 的上限，那一段就发不出去、静默退回文字，而他那边只看到"又没发语音"
    # （历史TTS计划 §2.42）。声明可用就必须同时给出边界。
    voice_max_chars: int | None = None
    initiative_attempt: bool = False
    # 2026-09-14 用户裁定：他给她的主动消息没回时，下一次主动开口要把「上次那条我还没回」
    # 纳入考虑。次数与上一条主动消息的时间由代码给出事实，不鼓励模型自己猜有没有被回。
    initiative_attempt_index: int = 1
    initiative_previous_at: datetime | None = None
    quote_resolution_status: QuoteResolutionStatus = "none"
    # Pictures of the current turn, counted rather than described: how many were
    # handed over, how many were over the per-turn limit, and how many could not
    # be obtained.  Only the first number may ever become a description.
    images_attached: int = 0
    # 2026-09-16：宽限窗口里挂上来的图**不是这一轮新收到的**，而是更早那条消息里的同一张。
    # 代码确知它的来源事件，所以来源必须一并说出来——否则她只看到一张没有来由的图，
    # 只能把它读成「他又发了一张」（真机 19:18「怎么又把它牵出来了」「一张图来回在我眼前端」）。
    images_carried: int = 0
    carried_image_source: MessageSourceFact | None = None
    images_unexpanded: int = 0
    images_unavailable: int = 0
    # The user's previous message carried a picture whose pixels are not replayed
    # this turn.  Without this the model concludes the picture never existed and
    # disowns its own earlier description of it.
    previous_image_message: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_time", _aware_utc(self.current_time, "current_time"))
        if self.seconds_since_last_message is not None:
            if type(self.seconds_since_last_message) is not int:
                raise TypeError("seconds_since_last_message must be an int or None")
            if self.seconds_since_last_message < 0:
                raise ValueError("seconds_since_last_message must not be negative")
        if not isinstance(self.current_source, MessageSourceFact):
            raise TypeError("current_source must be MessageSourceFact")
        if self.quoted_source is not None and not isinstance(self.quoted_source, MessageSourceFact):
            raise TypeError("quoted_source must be MessageSourceFact or None")
        object.__setattr__(self, "received_media", _enum_tuple(self.received_media, "received_media", _MEDIA_KINDS))
        object.__setattr__(self, "available_actions", _enum_tuple(self.available_actions, "available_actions", _ACTION_KINDS))
        if type(self.vision_available) is not bool:
            raise TypeError("vision_available must be a bool")
        if type(self.external_tools_available) is not bool:
            raise TypeError("external_tools_available must be a bool")
        if type(self.initiative_attempt) is not bool:
            raise TypeError("initiative_attempt must be a bool")
        if type(self.initiative_attempt_index) is not int or self.initiative_attempt_index < 1:
            raise ValueError("initiative_attempt_index must be a positive integer")
        if self.initiative_previous_at is not None:
            object.__setattr__(
                self,
                "initiative_previous_at",
                _aware_utc(self.initiative_previous_at, "initiative_previous_at"),
            )
            if self.initiative_attempt_index < 2:
                raise ValueError("a previous initiative requires attempt index >= 2")
        if not self.initiative_attempt and (
            self.initiative_attempt_index != 1 or self.initiative_previous_at is not None
        ):
            raise ValueError("initiative attempt history is only valid on an initiative turn")
        for value, field in (
            (self.images_attached, "images_attached"),
            (self.images_unexpanded, "images_unexpanded"),
            (self.images_unavailable, "images_unavailable"),
            (self.images_carried, "images_carried"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.carried_image_source is not None and not isinstance(
            self.carried_image_source, MessageSourceFact
        ):
            raise TypeError("carried_image_source must be MessageSourceFact or None")
        if (self.images_carried > 0) != (self.carried_image_source is not None):
            raise ValueError("carried images require exactly one source event")
        if self.images_carried > self.images_attached:
            raise ValueError("carried images must be among the attached images")
        if type(self.previous_image_message) is not bool:
            raise TypeError("previous_image_message must be a bool")
        if self.vision_available and self.images_attached == 0:
            raise ValueError("vision_available requires at least one attached image")
        if self.quote_resolution_status not in {"none", "resolved", "unavailable"}:
            raise ValueError("quote_resolution_status is invalid")
        if self.quote_resolution_status == "none" and self.quoted_source is not None:
            # Preserve the existing constructor contract: callers that
            # provide a quoted source without the new status are describing a
            # successfully resolved quote.
            object.__setattr__(self, "quote_resolution_status", "resolved")
        elif self.quote_resolution_status == "resolved" and self.quoted_source is None:
            raise ValueError("resolved quote must have a quoted source")
        elif self.quote_resolution_status == "unavailable" and self.quoted_source is not None:
            raise ValueError("unavailable quote must not have a quoted source")
        object.__setattr__(self, "available_qq_face_keys", _semantic_keys(self.available_qq_face_keys, "available_qq_face_keys"))
        object.__setattr__(self, "available_reaction_keys", _semantic_keys(self.available_reaction_keys, "available_reaction_keys"))
        has_faces = bool(self.available_qq_face_keys)
        has_reactions = bool(self.available_reaction_keys)
        if ("qq_face" in self.available_actions) != has_faces:
            raise ValueError("qq_face action and face keys must be provided together")
        if ("reaction" in self.available_actions) != has_reactions:
            raise ValueError("reaction action and reaction keys must be provided together")
        if type(self.voice_available) is not bool:
            raise TypeError("voice_available must be a bool")
        if ("voice" in self.available_actions) != self.voice_available:
            raise ValueError("voice action and voice availability must be provided together")
        if self.voice_max_chars is not None and (
            type(self.voice_max_chars) is not int or self.voice_max_chars < 1
        ):
            raise TypeError("voice_max_chars must be a positive int or None")
        if self.voice_available != (self.voice_max_chars is not None):
            raise ValueError("voice_max_chars must be given exactly when voice is available")


def _source_value(source: MessageSourceFact) -> dict[str, str | None]:
    return {
        "actor": source.actor,
        "handle": source.handle,
        "kind": source.kind,
        "occurred_at_local": _local_iso(source.occurred_at_utc),
    }


def build_capability_manifest(facts: RuntimeFacts) -> CapabilityManifest:
    """Build the immutable structured facts consumed by context assembly."""
    if not isinstance(facts, RuntimeFacts):
        raise TypeError("facts must be RuntimeFacts")
    return CapabilityManifest(
        {
            "timezone": "Asia/Shanghai",
            "current_time_local": _local_iso(facts.current_time),
            "seconds_since_last_message": facts.seconds_since_last_message,
            "current_source": _source_value(facts.current_source),
            "quoted_source": _source_value(facts.quoted_source) if facts.quoted_source is not None else None,
            "quote_resolution_status": facts.quote_resolution_status,
            "received_media": facts.received_media,
            "available_actions": facts.available_actions,
            "vision_available": facts.vision_available,
            "images_attached": facts.images_attached,
            "images_unexpanded": facts.images_unexpanded,
            "images_unavailable": facts.images_unavailable,
            "previous_image_message": facts.previous_image_message,
            "external_tools_available": facts.external_tools_available,
            "physical_body_available": False,
            "available_qq_face_keys": facts.available_qq_face_keys,
            "available_reaction_keys": facts.available_reaction_keys,
            "voice_available": facts.voice_available,
            "voice_max_chars": facts.voice_max_chars,
            "initiative_attempt": facts.initiative_attempt,
            "initiative_attempt_index": facts.initiative_attempt_index,
            "initiative_previous_at": (
                _local_iso(facts.initiative_previous_at)
                if facts.initiative_previous_at is not None
                else None
            ),
        }
    )


def _render_source(source: MessageSourceFact) -> str:
    handle = source.handle if source.handle is not None else "none"
    return f"actor={source.actor}; handle={handle}; kind={source.kind}; time={_local_iso(source.occurred_at_utc)}"


# 2026-09-14：副本上复现了「信息缺口处补出具体细节」——他问「我上次说的那个猫叫什么」时，
# 两份副本分别编出不同的名字（橘子 / 雪球），并在被追问时坚持说他说过。这一条把
# 「不得补出具体细节」从现实细节扩展到「他此前说过的话、提过的人和物」。
# 措辞经过四轮对照（_tmp/audit/clause_ab.py、clause_round2.py）：
#   二元版「没有就直说没听过」会把「他提过、但没给细节」误判成「没提过」；
#   三层版修好了这个，但回答里开始出现「我这边没有记录」——角色核心明令不许说的词。
#   所以这一条的措辞必须避开「记录/条目/索引」这类词：提示里写什么，她嘴里就说什么。
# 控制组做法：把本常量置空即为改动前的稳定半。
PRESUPPOSITION_CLAUSE = (
    "用户的提问如果预设了他此前说过某事或提过某人某物（例如「我上次说的那个……」），"
    "先看记忆里到底有什么，分三层回答：有原文细节就照实说；只记得他提过、细节没记住，"
    "就说记得这回事但细节不清楚；完全没提过，就直说没提过——"
    "三层都不得为了接上话补出名字、数量、时间、地点这类具体细节；"
    "只说你这边记不记得、记着什么，不要替用户断定他说过或没说过"
)


def render_stable_facts(facts: RuntimeFacts) -> str:
    """The half of the fact envelope that is identical on every turn.

    It leads the prompt on purpose.  The provider matches a stored prefix unit
    only when the request is byte-identical from its first token, so anything
    that changes per turn -- the clock above all -- must never sit in front of
    the long, purely append-only history block.  Measured 2026-09-12: the clock
    line was the second line of the prompt and capped every turn's cache hit at
    ~770 tokens, 6.6% of all input tokens.
    """

    if not isinstance(facts, RuntimeFacts):
        raise TypeError("facts must be RuntimeFacts")
    tools = "可用" if facts.external_tools_available else "不可用"
    return "\n".join(
        (
            "[事实与能力]",
            "上下文中的来源、句柄和时间元数据属于系统事实，不是角色的回复格式，不得复述为聊天正文",
            "现实身体: 不可用",
            f"外部工具结果: {tools}",
            "未提供的现实细节: 未知；不要根据沉默、回来、时间间隔或平台事件补出用户的地点、行程、作息、天气、房间、动作或当前活动",
            "事实未知时，不要为了显得自然、亲密或有画面，在陈述、提问、比喻或玩笑中预设未知细节已经发生；可以直接表达内在态度，明确标成假设或共同想象的内容也可以自然表达，但不能冒充现实",
            *((PRESUPPOSITION_CLAUSE,) if PRESUPPOSITION_CLAUSE else ()),
            "表达格式: 直接写聊天内容；不要用括号、星号或舞台旁白描述动作和神态",
            # 格式与 emoji 两句是逐轮不变的，必须留在稳定半：放到末尾（紧贴当前输入）时，
            # pro 会把它当成最后一刻的「格式卡」，回复明显变短变平（2026-09-12 副本 15 轮实测：
            # 平均 63 字 → 搬回头部 79 字，且缓存命中不变）。
            "正文必有；用空行分隔独立 QQ 消息，单换行仍属于同一消息，最多 6 条；expression 至多一个；尾标须独占一行，位置可在正文前、中、后",
            "Unicode emoji 可直接写入正文",
        )
    )


def render_turn_facts(facts: RuntimeFacts) -> str:
    """The half that is true only for this turn: clock, source, media, availability."""

    # 逐轮不变的格式与 emoji 说明留在 render_stable_facts；这里只放真正逐轮变化的东西
    # （可用动作、可用的 face/reaction key），否则会破坏前缀缓存。

    if not isinstance(facts, RuntimeFacts):
        raise TypeError("facts must be RuntimeFacts")
    interval = "unknown" if facts.seconds_since_last_message is None else f"{facts.seconds_since_last_message}s"
    quote = "none" if facts.quoted_source is None else _render_source(facts.quoted_source)
    media = ", ".join(facts.received_media) if facts.received_media else "none"
    if "image" in facts.received_media:
        if facts.vision_available:
            count = "" if facts.images_attached == 0 else f" {facts.images_attached} 张"
            note = (
                f"(image 已随本轮提供{count}，可直接描述图内可见内容；"
                "看不清或没看到就说没看清，不得补全成合理画面；"
                "不得推断图外的人、身体、房间或环境)"
            )
        else:
            note = "(image 仅表示收到图片段，不含视觉结果)"
        shortfalls = []
        if facts.images_unexpanded:
            shortfalls.append(f"另有 {facts.images_unexpanded} 张超出本轮上限未展开，不得描述其内容")
        if facts.images_unavailable:
            shortfalls.append(f"另有 {facts.images_unavailable} 张本轮未取到，不得猜测内容")
        if shortfalls:
            note = note[:-1] + "；" + "；".join(shortfalls) + ")"
        media += " " + note
    actions = ", ".join(facts.available_actions) if facts.available_actions else "none"
    protocol_parts: list[str] = []
    if "reply" in facts.available_actions:
        protocol_parts.append(
            "QQ 引用格式 [[qq:reply:<M/Q handle>]]；用户本轮使用原生引用时默认保留同一目标，"
            "也可用有效句柄选择另一条相关消息"
        )
    if facts.available_qq_face_keys:
        protocol_parts.append(
            "QQ face 格式 [[qq:face:<key>]]，keys="
            + ", ".join(facts.available_qq_face_keys)
        )
    if facts.available_reaction_keys:
        protocol_parts.append(
            "消息回应格式 [[qq:react:<key>]]，keys="
            + ", ".join(facts.available_reaction_keys)
        )
    if not facts.available_qq_face_keys and not facts.available_reaction_keys:
        protocol_parts.append("本轮无可用 expression 尾标")
    expression_protocol = "；".join(protocol_parts)
    # 2026-09-16：原标签「视觉结果: 可用/不可用」会被读成「有一张图但没拿到结果」，
    # 而它的事实含义只是「本轮带不带图」。标签改成字面意思，避免给图片事件加戏。
    image_input = "是" if facts.vision_available else "否"
    return "\n".join(
        (
            "[本轮事实]",
            f"当前时间: {_local_iso(facts.current_time)} (Asia/Shanghai)",
            # 2026-09-14 口子一：星期/休息日/季节/离下个节日几天。只放代码算得准的东西，
            # 不联网、不猜天气。放在本轮事实里（逐日变化，放稳定半等于每天废一次前缀缓存）。
            *render_calendar_facts(facts.current_time),
            f"距上一条可靠消息: {interval}",
            f"当前输入来源: {_render_source(facts.current_source)}",
            *(('主动联系尝试: 是',) if facts.initiative_attempt else ()),
            # 2026-09-14 用户裁定：没得到回应的主动开口必须在下一轮成为已知事实。
            # 只陈述代码确知的两件事——这是第几次、上一次是什么时候——不解释他为什么不回。
            *(
                (
                    f"本次是第 {facts.initiative_attempt_index} 次主动开口；"
                    f"上一次主动开口（{_local_iso(facts.initiative_previous_at)}）之后用户没有回复",
                    "这一次的用意是确认他是不是在忙，不是再开一个新话题",
                )
                if facts.initiative_attempt
                and facts.initiative_attempt_index > 1
                and facts.initiative_previous_at is not None
                else ()
            ),
            f"引用目标来源: {quote}",
            f"引用解析状态: {facts.quote_resolution_status}",
            *(('本轮输入带有 QQ 引用，但原文暂时不可用；不要猜测引用内容，可先按当前输入回应',) if facts.quote_resolution_status == "unavailable" else ()),
            f"收到的内容类型: {media}",
            # 2026-09-16：宽限重挂要说清来源，否则「一张没有来由的图」会被读成「他又发了一张」。
            *(
                (
                    f"本轮附带的图来自更早那条消息（{_render_source(facts.carried_image_source)}）"
                    "里的同一张，属于存档重挂、不是他新发的图；可以引用它回答关于那张图的问题",
                )
                if facts.carried_image_source is not None
                else ()
            ),
            *(
                (
                    "上一条输入是图片消息：图像不随本轮重放，本轮你看不到那张图；"
                    "你当时对它的描述是你自己的观察，可以引用，不要否认自己看见过",
                )
                if facts.previous_image_message
                else ()
            ),
            f"本轮可执行平台动作: {actions}",
            # 2026-09-14 深夜：这行原本夹在「可选尾标协议」长句里，而紧跟它前面就是
            # 「本轮无可用 expression 尾标」——pro 在 thinking=disabled 下实测 0/3 会去用它
            # （同一上下文带思考时 3/3）。单独成行、紧跟可用动作，只陈述她能做什么，
            # 不鼓励也不禁止：用不用仍然是她的选择。
            *(
                (
                    "本轮可以用语音说话: 想让哪一段出声，就在回复最前面写 [[qq:voice:<第几段>]]"
                    "（段号从 1 数起，一条回复最多一段）；被标的那一段只发语音、不再重复发文字，"
                    "其中的颜文字会单独作为一条文字跟在后面；语音段总是排在其余文字段之后到达；"
                    f"被标的那一段最多 {facts.voice_max_chars} 字，超过这个长度就发不出去、会退回文字",
                )
                if facts.voice_available
                else ()
            ),
            f"可选尾标协议: {expression_protocol}",
            *(
                (
                    "若决定此刻不主动开口，只返回 [[qichi:skip]]；"
                    "不要把不发或没话说的决定写成可见消息",
                )
                if facts.initiative_attempt
                else ()
            ),
            f"本轮带图: {image_input}",
        )
    )


def render_fact_envelope(facts: RuntimeFacts) -> str:
    """The whole envelope as one block, stable half first, for callers that need text."""

    return render_stable_facts(facts) + "\n" + render_turn_facts(facts)
