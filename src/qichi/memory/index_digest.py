"""Build the always-on neutral index of recent memory fragments.

The frozen contract (doc/Memory-Detailed-Record-Plan.md section 5.2) requires a
short, code-generated index so the character knows *that* something happened
recently without any content being disclosed.  Detail and verbatim text stay
behind the recall matrix and are only expanded on a native quote or the frozen
vocabulary.

Two rules come straight from checking the real data:

* The index never reads fragment_type, reality_scope or privacy_class from
  memory_fragments.  Those columns currently hold parser fallback values, so an
  index built from them labelled an adult evening as an ordinary chat.
* Every character in a line is assembled from stored fields.  No model writes
  any part of it, and no quote, normalized fact or detail text may appear.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Collection, Mapping, Sequence

INDEX_HEADER = (
    "[最近片段索引]（内部台账，只有存在性与数量；标为已展开的行，其原文已附在下方。"
    "这是给你自己看有没有这段用的，别把这里的标签、日期写法和条数念给用户听。）"
)
INDEX_FOOTER = "细节未展开"
# A line may only claim the details are closed while that is still true.  On
# 2026-09-11 the index said "细节未展开" for the very fragment whose 32 quotes had
# just been injected, and the character believed the index and told the user she
# could not open it -- two facts in one context contradicting each other.
INDEX_EXPANDED_TEMPLATE = "细节已在本轮展开（{count} 条已附在下方）"
# 用户点名了一个日子，而那天什么都没存。只写事实：那天没有记录。要怎么说，由她
# 在生成时决定（2026-09-12 T4）。不写「不要拿别的片段当它讲」这类指令——代码给
# 事实，措辞归模型。
INDEX_MISSING_DAY_TEMPLATE = "（{day} 没有存下来的记录：索引里没有这一天。）"
DEFAULT_DAYS = 7
DEFAULT_MAX_LINES = 8
DEFAULT_MAX_TOKENS = 600
DEFAULT_MAX_LINE_TOKENS = 80
# 2026-09-15 用户裁定（L1 放开权限，让他自己决定）：带成人/亲密标注的窗口不占
# 日常行的名额。他问「上次是什么时候」时，这些行是他唯一能对上的东西，而默认 8 行
# 的预算会把「09-13 凌晨 · 含成人内容」挤到第 10 行——**他能问到的历史被预算截断**。
# 常驻的仍然只是「哪几段、什么性质、有几条」；原文照旧只在点名或引用时展开，
# 由他决定打开哪一段。
DEFAULT_MAX_SENSITIVE_LINES = 8
DEFAULT_MAX_SENSITIVE_TOKENS = 400

_TYPE_LABELS = {"agreement": "条约定", "preference": "条偏好", "episode": "段经历",
                "correction": "条纠正", "self_expression": "条自述"}

from qichi.memory.dates import band_label


@dataclass(frozen=True, slots=True)
class IndexLine:
    fragment_id: str
    text: str
    tokens: int
    last_at_utc: str


def _band(hour: int) -> str:
    return band_label(hour)


def _stamp(value: str | None, local_zone) -> str:
    if not value:
        return "时间未知"
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(local_zone)
    except (TypeError, ValueError):
        return "时间未知"
    return f"{moment:%m-%d} {_band(moment.hour)}"



def _iso(value) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def describe_window(started_at_utc, ended_at_utc, local_zone) -> str:
    """Name one stored window exactly the way the index does.

    The detail block and the index must describe the same episode with the same
    words, otherwise the model cannot tell which index line it just opened.
    """

    span = _stamp(_iso(started_at_utc), local_zone)
    last = _stamp(_iso(ended_at_utc), local_zone)
    return span if span == last else f"{span} ~ {last}"


def describe_fragment(started_at_utc, ended_at_utc, privacy: Sequence[str], local_zone) -> str:
    """Name one fragment exactly the way its index line does.

    The detail block groups its quotes under this label, so the episode the user
    named and the block the model reads carry the same words.
    """

    return f"{describe_window(started_at_utc, ended_at_utc, local_zone)} · {_level(privacy)}"


def describe_missing_day(day) -> str:
    """Name a day the user pointed at that has nothing stored, in one line."""

    return INDEX_MISSING_DAY_TEMPLATE.format(day=f"{day.month}月{day.day}日")


# 2026-09-15 用户裁定：「讨论不等于有。后面关于成人内容可以做个区分，是讨论还是做了
# 可以标注出来，如果只概括会丢失信息。」所以敏感行不再只写一个「含成人内容」——
# 它同时给出**这些敏感明细各自落在哪种现实范围里**（事实分布），怎么理解归她。
# 注意：reality_scope 本身不可全信（真机上"真发生过"的那两段在库里全是共同想象），
# 所以这里**只呈现事实，绝不用它断言「发生过」**——否则会让她否认一件真发生过的事。
_SCOPE_LABELS = (
    ("conversation", "谈到"),
    ("shared_imagination", "想象"),
    ("claimed_real", "现实相关"),
    ("hypothetical", "假设"),
    ("mixed", "混合"),
    ("unknown", "未定"),
)


def _scope_note(privacy: Sequence[str], scopes: Sequence[str]) -> str:
    """敏感明细的现实范围分布，例如「（谈到 11 / 想象 24）」。

    只数非日常明细——它们才是这一行被标成敏感的原因。
    """

    tally: dict[str, int] = {}
    for level, scope in zip(privacy, scopes):
        if level == "ordinary":
            continue
        tally[scope] = tally.get(scope, 0) + 1
    if not tally:
        return ""
    ordered = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
    parts = ["%s %d" % (label, tally[key]) for key, label in _SCOPE_LABELS if tally.get(key)]
    labelled = {key for key, _ in _SCOPE_LABELS}
    extra = sum(count for key, count in ordered if key not in labelled)
    if extra:
        parts.append("其他 %d" % extra)
    return "（%s）" % " / ".join(parts)


def _level(privacy: Sequence[str]) -> str:
    """按"这段时间主要是什么"命名，而不是"里面有没有一条敏感的"。

    2026-09-15：原判据是"含一条敏感即标敏感"。09-14 那晚（TTS 与一次记忆核对）因为挂着
    7 条亲昵明细（"想你""揉一揉"），整段被标成「含成人内容」；而那一晚恰恰是用户纠正她
    "没有那回事"的一晚 —— 纠正这件事的对话，反过来变成了那件事的证据，她照着这行回答
    "上次是 9 月 14 号下午"。改为多数：少数几条亲昵不改变这一段时间的性质。
    """

    if not privacy:
        return "日常"
    total = len(privacy)
    if privacy.count("adult") * 2 > total:
        return "含成人内容"
    if (privacy.count("adult") + privacy.count("intimate")) * 2 > total:
        return "含亲密内容"
    return "日常"


def _counts(types: Sequence[str], details: int = 0) -> str:
    tally: dict[str, int] = {}
    for item in types:
        tally[item] = tally.get(item, 0) + 1
    parts = [f"{count} {_TYPE_LABELS.get(kind, kind)}" for kind, count in sorted(tally.items())]
    if details:
        parts.append(f"{details} 条原文记录")
    return " / ".join(parts) if parts else "无可索引条目"


def build_index_line_objects(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    now: datetime,
    local_zone,
    days: int = DEFAULT_DAYS,
    max_lines: int = DEFAULT_MAX_LINES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_line_tokens: int = DEFAULT_MAX_LINE_TOKENS,
    max_sensitive_lines: int = DEFAULT_MAX_SENSITIVE_LINES,
    max_sensitive_tokens: int = DEFAULT_MAX_SENSITIVE_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
    expanded_details: Mapping[str, int] | None = None,
    pinned_fragment_ids: Collection[str] = (),
) -> tuple[IndexLine, ...]:
    """Return the index lines with their fragment ids, newest first.

    Callers that must be able to prove which fragments actually kept a line --
    the context trace records exactly that (2026-09-12 T1) -- need the ids, not
    just the rendered text.

    A fragment whose details were expanded this turn is pinned: it keeps its line
    even when the window is full.  The detail block below names it, and an index
    without a line for it tells the model the episode does not exist -- measured
    2026-09-11, when "九号那块还是空的" was answered while that episode's 26 quotes
    sat in the same context.
    """
    measure = count_tokens or len
    since = (now - timedelta(days=days)).astimezone(timezone.utc).isoformat()
    rows = connection.execute(
        """
        SELECT f.fragment_id AS fragment_id,
               MIN(e.occurred_at_utc) AS first_at,
               MAX(e.occurred_at_utc) AS last_at,
               f.closed_at_utc AS closed_at
          FROM memory_fragments f
          JOIN memory_fragment_events fe ON fe.fragment_id = f.fragment_id
          JOIN conversation_events e ON e.event_id = fe.event_id
         WHERE f.conversation_id = ?
         GROUP BY f.fragment_id
         ORDER BY MAX(e.occurred_at_utc) DESC
        """,
        (conversation_id,),
    ).fetchall()

    def render(row, *, forced: bool) -> tuple[IndexLine | None, bool]:
        privacy: list[str] = []
        scopes: list[str] = []
        detail_count = 0
        for detail in connection.execute(
            "SELECT privacy_class, reality_scope FROM memory_detail_records WHERE fragment_id = ?",
            (row[0],),
        ):
            detail_count += 1
            privacy.append(detail[0])
            scopes.append(detail[1])
        linked = connection.execute(
            """
            SELECT DISTINCT m.type AS type, m.privacy_class AS privacy_class
              FROM memory_records m
              JOIN memory_evidence me ON me.memory_id = m.memory_id
              JOIN memory_fragment_events fe ON fe.event_id = me.event_id
             WHERE fe.fragment_id = ? AND m.status IN ('active', 'candidate')
            """,
            (row[0],),
        ).fetchall()
        if not linked and not detail_count:
            # Nothing to index in this window: a line saying so would be noise.
            return None, False
        for item in linked:
            privacy.append(item[1])
        span = _stamp(row[1], local_zone)
        last = _stamp(row[2], local_zone)
        window = span if span == last else f"{span} ~ {last}"
        state = "已结束" if row[3] else "进行中"
        injected = 0 if expanded_details is None else int(expanded_details.get(row[0], 0))
        footer = INDEX_FOOTER if injected <= 0 else INDEX_EXPANDED_TEMPLATE.format(count=injected)
        level = _level(privacy)
        note = _scope_note(privacy, scopes) if level != "日常" else ""
        text = f"{window} · {level}{note} · {state} · {_counts([item[0] for item in linked], detail_count)} · {footer}"
        tokens = measure(text)
        if tokens > max_line_tokens:
            # 太长时按重要性依次让位：先丢计数，再丢现实范围的补充；「哪一段 + 什么性质」
            # 永远保住。这一轮真的摊开了它，行就不能消失（2026-09-12 T5）：82 字符的那一条
            # 曾经整行被丢，32 条原文照常注入，于是「索引里没有这段」和「这段的原话
            # 就在下面」同时成立。
            text = f"{window} · {level}{note} · {state} · {footer}"
            tokens = measure(text)
        if tokens > max_line_tokens and note:
            text = f"{window} · {level} · {state} · {footer}"
            tokens = measure(text)
        if tokens > max_line_tokens and not forced:
            return None, False
        return IndexLine(row[0], text, tokens, str(row[2])), level != "日常"

    pinned = frozenset(pinned_fragment_ids)
    lines: list[IndexLine] = []
    for row in rows:
        if row[0] not in pinned:
            continue
        # 钉住的行不受 7 天窗口约束：窗口管的是常驻索引，而这一行是「这一轮真的
        # 摊开了它」。点到很久以前的某一天时，明细进来了、索引却没有行，是最坏的
        # 组合（2026-09-12 T5）。
        line, _sensitive = render(row, forced=True)
        if line is not None:
            lines.append(line)
    # 两笔预算分开算：日常行照旧（行数 + token），敏感行另有自己的额度，不抢日常的
    # 名额，也不被日常挤掉。这是 2026-09-15 的裁定：他能问到的历史不该被预算截断。
    sensitive_rows: list[tuple[int, IndexLine]] = []
    ordinary_rows: list[tuple[int, IndexLine]] = []
    for position, row in enumerate(rows):
        if row[0] in pinned:
            continue
        if str(row[2]) < since:
            continue
        line, sensitive = render(row, forced=False)
        if line is None:
            continue
        (sensitive_rows if sensitive else ordinary_rows).append((position, line))

    budget = max_tokens
    emitted = 0
    chosen: dict[int, IndexLine] = {}
    for position, line in ordinary_rows:
        if emitted >= max_lines:
            break
        if line.tokens > budget:
            break
        budget -= line.tokens
        emitted += 1
        chosen[position] = line
    sensitive_budget = max_sensitive_tokens
    sensitive_emitted = 0
    for position, line in sensitive_rows:
        if sensitive_emitted >= max_sensitive_lines:
            break
        if line.tokens > sensitive_budget:
            break
        sensitive_budget -= line.tokens
        sensitive_emitted += 1
        chosen[position] = line
    for position in sorted(chosen):
        lines.append(chosen[position])
    return tuple(lines)


def build_index_lines(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    now: datetime,
    local_zone,
    days: int = DEFAULT_DAYS,
    max_lines: int = DEFAULT_MAX_LINES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_line_tokens: int = DEFAULT_MAX_LINE_TOKENS,
    max_sensitive_lines: int = DEFAULT_MAX_SENSITIVE_LINES,
    max_sensitive_tokens: int = DEFAULT_MAX_SENSITIVE_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
    expanded_details: Mapping[str, int] | None = None,
    pinned_fragment_ids: Collection[str] = (),
) -> tuple[str, ...]:
    """The rendered index lines, newest first, bounded by lines and tokens."""

    return tuple(
        line.text
        for line in build_index_line_objects(
            connection,
            conversation_id=conversation_id,
            now=now,
            local_zone=local_zone,
            days=days,
            max_lines=max_lines,
            max_tokens=max_tokens,
            max_line_tokens=max_line_tokens,
            max_sensitive_lines=max_sensitive_lines,
            max_sensitive_tokens=max_sensitive_tokens,
            count_tokens=count_tokens,
            expanded_details=expanded_details,
            pinned_fragment_ids=pinned_fragment_ids,
        )
    )
