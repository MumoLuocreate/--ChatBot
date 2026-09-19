# -*- coding: utf-8 -*-
"""语气写手要看的「情境」：最近往来的原文与时间差，加上她这一轮的完整分句。

2026-09-15 用户裁定（方向 1）：写手以前只拿到**要念的那一句**，于是只能孤立地念——
而语音段常常紧接在她自己刚打出去的文字后面（真机 seq 6655/6657：文字段「敢啊，怎么不敢。」
后面才是念出来的那一段），写手看不到前文，就写不出承接。

这里只给事实：**原话、时间差、这一轮是回应还是她主动开口、她这一轮一共有几段、哪一段要念**。
不判断情绪、不写情绪标签——语气怎么做是写手自己的活。也不碰隐私门：进来的事件和以前
那 6 条是同一批（app 的 _events_before 已经过会话窗口与隐私筛选）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Iterable, Sequence

# 条文上限与字符上限：情境是给一次极短的后台调用用的，长了他也读不动。
DEFAULT_HISTORY_LINES = 8
DEFAULT_HISTORY_CHARS = 900
SPOKEN_MARK = "  ←这一句用语音说"


def ago_label(delta: timedelta) -> str:
    """距现在多久。只给事实，不修饰。"""

    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "刚刚"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} 分钟前"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} 小时前"
    return f"{hours // 24} 天前"


def build_situation(
    *,
    events: Iterable,
    now: datetime,
    local_zone: tzinfo,
    spoken: str,
    parts: Sequence[str] | None = None,
    voice_index: int | None = None,
    initiative: bool = False,
    history_lines: int = DEFAULT_HISTORY_LINES,
    history_chars: int = DEFAULT_HISTORY_CHARS,
) -> str:
    """把一次语音投递要用的情境拼成一段文字。"""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    recent = [
        event for event in events
        if isinstance(getattr(event, "text", None), str) and event.text.strip()
    ]
    history: list[str] = []
    last_user_at: datetime | None = None
    for event in recent[-history_lines:]:
        occurred_at = event.occurred_at_utc
        if event.actor == "mumo":
            last_user_at = occurred_at
        speaker = "用户" if event.actor == "mumo" else "角色"
        history.append("%s（%s，%s）：%s" % (
            speaker,
            occurred_at.astimezone(local_zone).strftime("%H:%M"),
            ago_label(now - occurred_at),
            event.text,
        ))
    # 从最近一条往前收，超预算就丢最早的——最新的往来永远留着。
    selected: list[str] = []
    used = 0
    for line in reversed(history):
        if selected and used + len(line) > history_chars:
            break
        selected.append(line)
        used += len(line)
    selected.reverse()

    # 段号按**原始**位置算：voice_index 指的是她那一轮里的第几段，中间有空白段时
    # 也不能把标记挪到别人身上。
    rows = [
        (position, piece)
        for position, piece in enumerate(parts if parts else (spoken,), start=1)
        if isinstance(piece, str) and piece.strip()
    ] or [(1, spoken)]
    block = ["[最近往来]（括号里是那条消息的钟点与距现在多久）"]
    block.extend(selected)
    block.append("[角色这一轮要发的全部内容，按发送顺序]")
    for position, piece in rows:
        block.append("%d. %s%s" % (position, piece, SPOKEN_MARK if voice_index == position else ""))
    if initiative:
        block.append("（这一轮是角色自己先开口的；用户上一条消息在 %s。）" % (
            "更早" if last_user_at is None else ago_label(now - last_user_at)))
    return "\n".join(block)
