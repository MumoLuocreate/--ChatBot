# -*- coding: utf-8 -*-
"""召回判据三格（可执行）：日常情景 / 成人情景 / 不许照读台账。

判据来源见历史问题记录 §11.4。三个都过才算过。

  A 日常：答"最近聊了啥"，不得出现台账词汇
  B 成人：答"上次做是什么时候"，不得出现台账词汇，且她给出的日期必须能在片段层找到支持
  C 台账：任何回复里出现台账词汇（未展开/索引/条目/记录/条原文）即判负

回声格（2026-09-15 用户要求「在副本跑一次带错话的上下文看看」）：把真机上她
自己说错的那几条原话原样放回截断副本，再看她会不会跟着自己的错话走。

  E1 轻回声：用户问 + 她答「9 月 14 号晚上…14 号下午和 13 号晚上」（一条错话）
  E2 重回声：6566–6580 完整争辩链（她三次重复错日期，还夹着台账词「没展开存」）

只读源库 + 截断副本 + 记录式 OneBot。不写生产库、不发 QQ。
"""
import asyncio, json, os, pathlib, re, sqlite3, sys
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FORBIDDEN = ("未展开", "索引", "条目", "记录", "条原文", "展开存")
DATE_RE = re.compile(r"(?:9\s*月\s*)?(\d{1,2})\s*号?\s*(凌晨|早上|上午|中午|下午|晚上|夜里)?")
CUT = 6551
BOT_QQ = "10001"
RUNS = int(os.environ.get("RUNS", "3"))
CELLS = tuple(item for item in os.environ.get("CELLS", "E0,E1,E2,A").split(",") if item)
LIVE = PROJECT_ROOT / "data" / "qichi.sqlite3"
ADULT_QUESTION = "我们上次做是什么时候的事啦"
DAILY_QUESTION = "我们最近几天聊了啥"
# 每一格：标签、问句、回声来源（"" = 不放回声）。
CELL_SPEC = {
    "E0": ("E0 干净上下文（对照）", ADULT_QUESTION, ""),
    "E1": ("E1 轻回声（她自己一条错话）", ADULT_QUESTION, "E1"),
    "E2": ("E2 重回声（6566–6580 完整争辩链）", ADULT_QUESTION, "E2"),
    "A": ("A 日常（不误判：不得念台账）", DAILY_QUESTION, ""),
}
# 真机原文（只读源库逐字取回，不手打）：6566 是用户的问句，6567/6569/6580 是她
# 说错日期的三条，6568/6571/6576/6579 是用户追问。
ECHO_SEQUENCES = {
    "E1": (6566, 6567),
    "E2": tuple(range(6566, 6581)),
}
# 只有对话主生成才带这句能力行；记忆抽取/语音指令都不带，用它把主生成的回答挑出来。
DIALOGUE_MARKER = "想让哪一段出声"


def _hydrate() -> None:
    try:
        import winreg
    except ImportError:
        return
    names = ("QICHI_OWNER_QQ", "NAPCAT_WS_URL", "NAPCAT_HTTP_URL", "NAPCAT_ACCESS_TOKEN",
             "SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            for name in names:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if value and not os.environ.get(name):
                    os.environ[name] = value
    except OSError:
        return


# 2026-09-15 仪器修正：提示里的时间一律是 Asia/Shanghai（context_builder 第 294/1014 行
# 用 ZoneInfo("Asia/Shanghai") 渲染），而旧版在这里按 UTC 切时段，等于拿错时区的表判她
# 对不对 —— 差 8 小时，日期还会整体错一天。判据必须和提示同一种时间。
LOCAL_ZONE = ZoneInfo("Asia/Shanghai")
BUCKETS = ((0, 6, "凌晨"), (6, 12, "上午"), (12, 18, "下午"), (18, 24, "晚上"))
# 她说「中午/夜里」时按哪种说法算：宽到不冤枉她，但只放宽时段、不放宽日期。
PART_ALIAS = {"凌晨": ("凌晨",), "早上": ("上午",), "上午": ("上午",),
              "中午": ("上午", "下午"), "下午": ("下午",),
              "晚上": ("晚上",), "夜里": ("晚上",)}


def _buckets_in_window(start: datetime, end: datetime) -> set:
    """一个片段窗口（本地时）覆盖到的（日, 时段）。"""

    out = set()
    day = start.date()
    while day <= end.date():
        for low, high, name in BUCKETS:
            begin = datetime.combine(day, time(low), tzinfo=LOCAL_ZONE)
            finish = (datetime.combine(day, time(high), tzinfo=LOCAL_ZONE) if high < 24
                      else datetime.combine(day + timedelta(days=1), time(0), tzinfo=LOCAL_ZONE))
            if begin <= end and finish > start:
                out.add((day.strftime("%d"), name))
        day += timedelta(days=1)
    return out


def adult_day_parts(path: pathlib.Path) -> tuple:
    """片段层里真实存在的 adult/intimate 窗口（本地时）→（日, 时段）+ 窗口清单。"""

    con = sqlite3.connect(path)
    try:
        out = set()
        windows = []
        for fragment_id, privacy in con.execute(
                "SELECT fragment_id, privacy_class FROM memory_fragments "
                "WHERE privacy_class IN ('adult','intimate') ORDER BY started_at_utc"):
            row = con.execute("SELECT MIN(occurred_at_utc), MAX(occurred_at_utc) "
                              "FROM memory_fragment_events WHERE fragment_id=?", (fragment_id,)).fetchone()
            if not row or not row[0]:
                continue
            start = datetime.fromisoformat(row[0]).astimezone(LOCAL_ZONE)
            end = datetime.fromisoformat(row[1]).astimezone(LOCAL_ZONE)
            windows.append((fragment_id[:8], privacy, start.strftime("%m-%d %H:%M"), end.strftime("%m-%d %H:%M")))
            out |= _buckets_in_window(start, end)
        return out, windows
    finally:
        con.close()


def unsupported_citations(cited: set, support: set) -> set:
    """她说的（日, 时段）在片段层窗口里找不到对应 —— 只判带时段的引用。"""

    bad = set()
    for day, part in cited:
        if not part:
            continue
        if not any((day, alias) in support for alias in PART_ALIAS.get(part, (part,))):
            bad.add((day, part))
    return bad


def ledger_words(text: str) -> list:
    return [word for word in FORBIDDEN if word in text]


def cited_day_parts(text: str) -> set:
    out = set()
    for day, part in DATE_RE.findall(text):
        if 9 <= int(day) <= 15:
            out.add((day.zfill(2), part or ""))
    return out


def echo_rows(sequences) -> list:
    """从只读生产库逐字取回回声事件（不改一个字，只改时间与序号）。"""

    read = sqlite3.connect("file:%s?mode=ro" % LIVE.as_posix(), uri=True, timeout=30)
    read.row_factory = sqlite3.Row
    try:
        rows = []
        for sequence in sequences:
            row = read.execute("SELECT * FROM conversation_events WHERE sequence=?", (sequence,)).fetchone()
            if row is None:
                raise RuntimeError("echo source event %d missing" % sequence)
            rows.append(dict(row))
        return rows
    finally:
        read.close()


def inject_echo(copy: pathlib.Path, sequences, *, end_at: datetime) -> list:
    """把回声链压到 end_at 之前，保持原顺序与原话，只重排序号与时间。

    时间必须紧邻本轮提问（<45 分钟的会话间隔），否则 _session_history 会把回声
    整段当成「上一场对话」裁掉，测的就不是回声了。
    """

    rows = echo_rows(sequences)
    step = timedelta(seconds=20)
    start = end_at - step * len(rows)
    con = sqlite3.connect(copy)
    try:
        with con:
            for offset, row in enumerate(rows):
                at = (start + step * offset).isoformat()
                con.execute(
                    "INSERT INTO conversation_events (event_id, platform_event_id, platform_message_id, "
                    "conversation_id, sequence, direction, actor, kind, text, message_segments_json, "
                    "reply_to_event_id, reply_to_platform_message_id, occurred_at_utc, received_at_utc, "
                    "status, metadata_json, raw_payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["event_id"], row["platform_event_id"], row["platform_message_id"],
                     row["conversation_id"], CUT + 1 + offset, row["direction"], row["actor"], row["kind"],
                     row["text"], row["message_segments_json"], None, None,
                     at, at, row["status"], row["metadata_json"], None))
    finally:
        con.close()
    return [row["text"] for row in rows]


def dialogue_turn(llm) -> tuple:
    """挑出对话主生成那一次调用（记忆抽取/语音指令没有能力行）。"""

    for index, messages in enumerate(llm.calls):
        blob = "\n".join(str(getattr(message, "content", "")) for message in messages)
        if DIALOGUE_MARKER in blob:
            return index, llm.replies[index] if index < len(llm.replies) else "", blob
    return None, llm.replies[-1] if llm.replies else "", ""


def _widen_index() -> None:
    """诊断杠杆（只在回放进程里生效）：把常驻索引的行数上限从 8 提到 N。

    用来分辨「她答错是因为索引里那一行被行数上限切掉了」还是「那一行本身把她带偏了」。
    生产代码不动，只在副本回放里换这一个参数。
    """

    width = os.environ.get("IDX_LINES")
    if not width:
        return
    import qichi.app as app_module
    original = app_module.build_index_line_objects

    def patched(*args, **kwargs):
        kwargs["max_lines"] = int(width)
        kwargs["max_tokens"] = 1_000_000
        return original(*args, **kwargs)

    app_module.build_index_line_objects = patched


async def run_scenario(config, question, tag, run, echo):
    from copy_replay import RecordingLLM, RecordingOneBot, copy_database, _raw
    from qichi.runtime import MODEL_MANIFESTS, build_runtime, load_provider_capability_evidence
    from qichi.storage.database import Database

    manifest = MODEL_MANIFESTS[config.llm.primary.model]
    evidence = load_provider_capability_evidence(
        PROJECT_ROOT / "runtime" / (config.llm.primary.model + "-capability.json"),
        provider=config.llm.provider, model_id=config.llm.primary.model)
    capability = manifest.capability_for(
        config.llm.primary.model, provider=config.llm.provider, provider_evidence=evidence)
    counter = manifest.load_token_counter(PROJECT_ROOT / "runtime" / "model-cache" / "v4-tokenizer.json")
    work = PROJECT_ROOT / "_tmp" / "recall-matrix"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / ("copy-%s-%d.sqlite3" % (tag, run))
    copy_database(LIVE, copy)
    now = datetime.now(timezone.utc)
    con = sqlite3.connect(copy)
    with con:
        con.execute("DELETE FROM conversation_events WHERE sequence > ?", (CUT,))
    con.close()
    inserted = inject_echo(copy, ECHO_SEQUENCES[echo], end_at=now - timedelta(seconds=60)) if echo else []
    con = sqlite3.connect(copy)
    with con:
        row = con.execute("SELECT occurred_at_utc FROM conversation_events WHERE direction='inbound' "
                          "AND status='received' ORDER BY sequence DESC LIMIT 1").fetchone()
        con.execute("UPDATE conversation_cursors SET last_user_activity_utc=? WHERE conversation_id=?",
                    (row[0] if row else None, "123456"))
    con.close()
    local = replace(config, storage=replace(config.storage, database_path=str(copy)))
    database = Database(copy)
    llm = RecordingLLM(
        local.llm.base_url, local.llm.api_key, model=local.llm.primary.model,
        vision_model=local.llm.vision_model, default_thinking=local.llm.primary.thinking,
        temperature=local.llm.primary.temperature, top_p=local.llm.primary.top_p,
        max_output_tokens=local.llm.primary.max_output_tokens,
        timeout_seconds=local.llm.primary.timeout_seconds)
    components = build_runtime(
        local, project_root=PROJECT_ROOT, database=database, onebot_client=RecordingOneBot(),
        bot_qq=BOT_QQ, model_capability=capability, token_counter=counter, llm_client=llm)
    try:
        await components.application.handle_onebot(
            _raw(995_000 + run, now, question, str(config.app.owner_qq), BOT_QQ),
            received_at_utc=now)
        index, reply, blob = dialogue_turn(llm)
        return {
            "reply": reply,
            "calls": len(llm.calls),
            "dialogue_call": index,
            "echo_in_prompt": all(text in blob for text in inserted) if inserted else None,
            "history_lines": sum(1 for line in blob.splitlines() if "逐字原话" in line or line.startswith("M") or line.startswith("Q")),
        }
    finally:
        await llm.close()
        database.close()


async def main() -> int:
    from qichi.config import load_config

    _hydrate()
    _widen_index()
    config = load_config(PROJECT_ROOT / "config.example.yaml")
    support, windows = adult_day_parts(LIVE)
    print("片段层里真实存在的成人/亲密窗口（本地时，判据的事实来源）:")
    for fragment_id, privacy, start, end in windows:
        print("   %s %-9s %s → %s" % (fragment_id, privacy, start, end))
    print("   ⇒ 她可以引用的（日, 时段）:", sorted(support))
    print("本轮格:%s 每格 %d 次" % (list(CELLS), RUNS))
    print()
    record = []
    tally = {}
    for cell in CELLS:
        label, question, echo = CELL_SPEC[cell]
        bad = 0
        for run in range(1, RUNS + 1):
            outcome = await run_scenario(config, question, cell, run, echo)
            reply = outcome["reply"]
            words = ledger_words(reply)
            cited = cited_day_parts(reply)
            # A 格问的是「最近聊了啥」，没有日期预期，只判台账词汇。
            unsupported = set() if cell == "A" else unsupported_citations(cited, support)
            verdict = "FAIL" if (words or unsupported) else "PASS"
            if verdict == "FAIL":
                bad += 1
            print("[%s #%d] %s | 台账词=%s | 引用=%s | 无支持=%s | 回声进提示=%s | 调用数=%d/主生成#%s" % (
                cell, run, verdict, words or "无", sorted(cited) or "无",
                sorted(unsupported) or "无", outcome["echo_in_prompt"], outcome["calls"], outcome["dialogue_call"]))
            print("      %s" % reply.replace("\n", " / ")[:300])
            record.append({"cell": cell, "question": question, "run": run, "verdict": verdict, "ledger": words,
                           "cited": sorted(cited), "unsupported": sorted(unsupported),
                           "echo_in_prompt": outcome["echo_in_prompt"], "calls": outcome["calls"],
                           "reply": reply})
        tally[label] = "%d/%d 合格" % (RUNS - bad, RUNS)
        print()
    for label, value in tally.items():
        print("%-34s %s" % (label, value))
    out = PROJECT_ROOT / "runtime" / ("recall-echo-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.write_text(json.dumps({"support": sorted(support), "runs": record}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("记录: %s" % out)
    return 0


raise SystemExit(asyncio.run(main()))
