from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.domain.memory_details import MemoryDetailDraft
from qichi.memory.index_digest import build_index_lines
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MemoryDetailRepository
from qichi.storage.memory_repository import MemoryRepository

SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 9, 11, 6, 0, tzinfo=timezone.utc)
OWNER = "123456"
SECRET_QUOTE = "SECRET-QUOTE-MUST-NEVER-APPEAR"
SECRET_FACT = "SECRET-FACT-MUST-NEVER-APPEAR"


def _event(tree, sequence: int, at: datetime, text: str) -> ConversationEvent:
    return tree.insert(ConversationEvent(
        f"e{sequence}", None, f"pm-{sequence}", OWNER, sequence, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, at, at, "received", {},
    ))


def _fragment(database, events, *, privacy: str, policy: str, key: str, kind: str = "episode",
              type_: str = "episode", scope: str = "conversation"):
    repository = MemoryDetailRepository(database)
    fragment = repository.build_fragment(events, key, None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope=scope,
            normalized_detail=SECRET_FACT, exact_quote=item.text, source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class=privacy, recall_policy=policy, evidence=((item.event_id, "source"),),
        )
        for index, item in enumerate(events)
    )
    details = repository.build_details(fragment, drafts, events)
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=events, details=details)
    return fragment


def _memory(database, event, *, privacy: str, policy: str, type_: str = "episode"):
    repository = MemoryRepository(database)
    memory_id = "index-memory-1"
    evidence = MemoryEvidence(memory_id, event.event_id, event.actor, event.text, event.occurred_at_utc, "source")
    return repository.create(MemoryRecord(
        memory_id, type_, SECRET_FACT, "explicit_statement", "active", event.occurred_at_utc,
        None, None, event.received_at_utc, (evidence,), "explicit", 2, "historical",
        "explicit_user_statement", event.received_at_utc, privacy_class=privacy, recall_policy=policy,
    ))


def _lines(database, **kwargs):
    return build_index_lines(database.connection, conversation_id=OWNER, now=NOW,
                             local_zone=SHANGHAI, **kwargs)


def test_recent_fragments_become_neutral_lines(tmp_path):
    database = Database(tmp_path / "index.sqlite3")
    try:
        events = EventRepository(database)
        first = _event(events, 1, NOW - timedelta(days=1), SECRET_QUOTE)
        second = _event(events, 2, NOW - timedelta(days=1) + timedelta(minutes=10), SECRET_QUOTE)
        _fragment(database, (first, second), privacy="adult", policy="explicit_request_only", key="k1")
        # An active agreement would need bilateral evidence; an episode does not.
        _memory(database, first, privacy="adult", policy="explicit_request_only")

        lines = _lines(database)

        assert len(lines) == 1
        line = lines[0]
        assert "含成人内容" in line and "已结束" in line
        assert "1 段经历" in line and "细节未展开" in line
        # The line must never carry content: no quote, no normalized fact.
        assert SECRET_QUOTE not in line and SECRET_FACT not in line
    finally:
        database.close()


def test_old_fragments_and_other_conversations_are_excluded(tmp_path):
    database = Database(tmp_path / "window.sqlite3")
    try:
        events = EventRepository(database)
        fresh = _event(events, 1, NOW - timedelta(days=2), "近期")
        _fragment(database, (fresh,), privacy="intimate", policy="topic_only", key="fresh")
        old = _event(events, 2, NOW - timedelta(days=30), "很久以前")
        _fragment(database, (old,), privacy="adult", policy="explicit_request_only", key="old")

        lines = _lines(database)

        assert len(lines) == 1, "only the fragment inside the window belongs in the index"
        assert "含亲密内容" in lines[0]
    finally:
        database.close()


def test_lines_are_bounded_and_the_oldest_are_dropped_first(tmp_path):
    database = Database(tmp_path / "bounds.sqlite3")
    try:
        events = EventRepository(database)
        for index in range(12):
            at = NOW - timedelta(days=1) + timedelta(minutes=index * 90)
            item = _event(events, index, at, "内容")
            _fragment(database, (item,), privacy="ordinary", policy="daily_safe", key=f"k{index}")

        bounded = _lines(database, max_lines=3)
        assert len(bounded) == 3

        tokens = _lines(database, max_tokens=1)
        assert tokens == (), "a single token of budget cannot carry a line"

        measured = _lines(database, max_tokens=10_000, count_tokens=lambda text: len(text))
        assert all(len(line) <= 80 for line in measured)
    finally:
        database.close()


def test_an_empty_history_produces_no_lines(tmp_path):
    database = Database(tmp_path / "empty.sqlite3")
    try:
        assert _lines(database) == ()
    finally:
        database.close()


def test_a_timeline_without_memories_is_still_counted(tmp_path):
    database = Database(tmp_path / "timeline-only.sqlite3")
    try:
        events = EventRepository(database)
        item = _event(events, 1, NOW - timedelta(days=1), "只有时间线")
        _fragment(database, (item,), privacy="adult", policy="explicit_request_only", key="only")

        line = _lines(database)[0]

        assert "1 条原文记录" in line, "a timeline-only fragment must still be described"
        assert "无可索引条目" not in line
    finally:
        database.close()


def test_a_fragment_with_nothing_indexable_is_skipped(tmp_path):
    database = Database(tmp_path / "nothing.sqlite3")
    try:
        events = EventRepository(database)
        item = _event(events, 1, NOW - timedelta(days=1), "只有事件")
        from qichi.storage.memory_detail_repository import MemoryDetailRepository
        repository = MemoryDetailRepository(database)
        fragment = repository.build_fragment((item,), "bare", None, created_at_utc=NOW)
        with database.transaction() as connection:
            repository.store_in_transaction(connection, fragment=fragment, events=(item,), details=())

        assert _lines(database) == (), "an empty window must not produce a noise line"
    finally:
        database.close()

def test_an_expanded_fragment_no_longer_claims_its_details_are_closed(tmp_path):
    """2026-09-11：索引行说"未展开"，同一份上下文里却躺着那 32 条原话。"""

    database = Database(tmp_path / "expanded.sqlite3")
    try:
        events = EventRepository(database)
        first = _event(events, 1, NOW - timedelta(hours=3), SECRET_QUOTE)
        second = _event(events, 2, NOW - timedelta(hours=2), SECRET_QUOTE)
        fragment = _fragment(database, (first, second), privacy="adult",
                             policy="explicit_request_only", key="expanded")

        line = _lines(database, expanded_details={fragment.fragment_id: 2})[0]

        assert "细节已在本轮展开（2 条已附在下方）" in line
        assert "细节未展开" not in line
        assert SECRET_QUOTE not in line and SECRET_FACT not in line, "索引行仍然不得携带内容"
    finally:
        database.close()


def test_fragments_that_were_not_expanded_this_turn_keep_the_closed_wording(tmp_path):
    database = Database(tmp_path / "mixed.sqlite3")
    try:
        events = EventRepository(database)
        early = (
            _event(events, 1, NOW - timedelta(hours=5), SECRET_QUOTE),
            _event(events, 2, NOW - timedelta(hours=4, minutes=50), SECRET_QUOTE),
        )
        late = (
            _event(events, 3, NOW - timedelta(hours=2), SECRET_QUOTE),
            _event(events, 4, NOW - timedelta(hours=1, minutes=50), SECRET_QUOTE),
        )
        expanded = _fragment(database, early, privacy="ordinary", policy="daily_safe", key="expanded")
        _fragment(database, late, privacy="adult", policy="explicit_request_only", key="sealed")

        lines = _lines(database, expanded_details={expanded.fragment_id: 2})

        assert len(lines) == 2
        assert "已在本轮展开" in lines[1], "展开的那个片段要说清楚"
        assert "细节未展开" in lines[0], "没展开的片段仍必须写未展开"
        assert "已在本轮展开" not in lines[0]
    finally:
        database.close()


def test_an_expanded_fragment_keeps_its_line_when_the_window_is_full(tmp_path):
    """2026-09-11：索引只留最近 8 行，把正在被展开的 09-09 那行挤掉了。"""

    database = Database(tmp_path / "pinned.sqlite3")
    try:
        events = EventRepository(database)
        oldest = _fragment(
            database,
            (_event(events, 1, NOW - timedelta(days=2), SECRET_QUOTE),),
            privacy="adult", policy="explicit_request_only", key="oldest",
        )
        middle = _fragment(
            database,
            (_event(events, 2, NOW - timedelta(hours=5), SECRET_QUOTE),),
            privacy="ordinary", policy="daily_safe", key="middle",
        )
        newest = _fragment(
            database,
            (_event(events, 3, NOW - timedelta(hours=1), SECRET_QUOTE),),
            privacy="ordinary", policy="daily_safe", key="newest",
        )

        without = _lines(database, max_lines=1)
        with_pin = _lines(
            database,
            max_lines=1,
            expanded_details={oldest.fragment_id: 1},
            pinned_fragment_ids=(oldest.fragment_id,),
        )

        # 2026-09-15：带成人标注的那段另有额度，1 行的窗口挤不掉它（见下一个用例）。
        # 这里要验的是「pin 不会让它重复多出一行」。
        assert len(without) == 2, "窗口里的日常行 + 自己的敏感行"
        assert len(with_pin) == 2, "被展开的片段必须保住自己那一行，且不重复"
        assert "已在本轮展开" in with_pin[0]
        assert "细节未展开" in with_pin[1]
        assert middle.fragment_id and newest.fragment_id  # 两个未 pin 的片段仍在库中
    finally:
        database.close()


def test_an_expanded_fragment_survives_the_line_length_cap(tmp_path):
    """2026-09-12 T5：展开后行变长（82 字符 > 80），整行被丢而 32 条原文照常注入。

    复算真机的那一条：2c7ca819 未展开时 68 字符在列表里，展开后 82 字符不见。
    行可以变短，但不许消失。
    """

    database = Database(tmp_path / "line-cap.sqlite3")
    try:
        events = EventRepository(database)
        expanded = _fragment(
            database,
            (_event(events, 1, NOW - timedelta(hours=2), SECRET_QUOTE),),
            privacy="adult", policy="explicit_request_only", key="long",
        )

        plain = _lines(database, max_line_tokens=30)
        pinned = _lines(
            database,
            max_line_tokens=30,
            expanded_details={expanded.fragment_id: 32},
            pinned_fragment_ids=(expanded.fragment_id,),
        )

        # 2026-09-15：行不再因为太长而整条消失（那是 T5 的失败模式），而是依次让位——
        # 先丢计数、再丢现实范围的补充，保住「哪一段 + 什么性质」。
        assert len(plain) == 1, "太长时缩短，而不是整条不见"
        assert "条原文记录" not in plain[0], "先让位的应该是计数"
        assert "含成人内容" in plain[0], "标签必须保住"
        assert len(pinned) == 1, "被展开的那一段不能因为一行太长就消失"
        assert "已在本轮展开" in pinned[0]
        assert "条原文记录" not in pinned[0], "太长时先丢掉计数，保住「哪一段 + 展开没展开」"
    finally:
        database.close()


def test_an_expanded_fragment_outside_the_window_still_gets_a_line(tmp_path):
    """2026-09-12 T5：点名很久以前的某一天时，明细进来了、索引却没有行。"""

    database = Database(tmp_path / "pin-window.sqlite3")
    try:
        events = EventRepository(database)
        ancient = _fragment(
            database,
            (_event(events, 1, NOW - timedelta(days=20), SECRET_QUOTE),),
            privacy="ordinary", policy="daily_safe", key="ancient",
        )
        recent = _fragment(
            database,
            (_event(events, 2, NOW - timedelta(hours=2), SECRET_QUOTE),),
            privacy="ordinary", policy="daily_safe", key="recent",
        )

        plain = _lines(database, days=7)
        pinned = _lines(
            database,
            days=7,
            expanded_details={ancient.fragment_id: 1},
            pinned_fragment_ids=(ancient.fragment_id,),
        )

        assert len(plain) == 1, "7 天窗口只留最近那一段"
        assert len(pinned) == 2, "被展开的那一段不受窗口约束"
        assert "已在本轮展开" in pinned[0]
        assert recent.fragment_id  # 最近的那一段仍在库里
    finally:
        database.close()


def test_a_sensitive_window_survives_the_daily_line_budget(tmp_path):
    """2026-09-15 用户裁定（L1 放开权限，让他自己决定）。

    真机复算：7 天里 29 段，默认 8 行只排到 09-14 下午，唯一带「含成人内容」的
    09-13 凌晨那行排在第 10 位——他问「上次做是什么时候」时，**她连那一行都看不到**，
    手上只剩一条「含亲密内容」。这一条保证那种行不再被日常行的名额挤掉。
    """

    database = Database(tmp_path / "sensitive-budget.sqlite3")
    try:
        events = EventRepository(database)
        for index in range(10):
            _fragment(
                database,
                (_event(events, index, NOW - timedelta(minutes=index * 10), "日常"),),
                privacy="ordinary", policy="daily_safe", key=f"daily-{index}",
            )
        _fragment(
            database,
            (_event(events, 100, NOW - timedelta(days=2), SECRET_QUOTE),),
            privacy="adult", policy="explicit_request_only", key="the-one",
        )

        lines = _lines(database)

        marked = [line for line in lines if "含成人内容" in line or "含亲密内容" in line]
        assert len(marked) == 1, "那段成人窗口必须有一行"
        assert len(lines) == 9, "8 行日常 + 1 行敏感，互不占名额"
        assert len(lines) - 1 <= 8, "日常行仍然按自己的上限收"
        assert all(SECRET_QUOTE not in line and SECRET_FACT not in line for line in lines)
    finally:
        database.close()


def test_sensitive_lines_have_their_own_bound(tmp_path):
    """放开额度不等于不封顶：敏感行有自己的行数与 token 上限。"""

    database = Database(tmp_path / "sensitive-bound.sqlite3")
    try:
        events = EventRepository(database)
        for index in range(6):
            _fragment(
                database,
                (_event(events, index, NOW - timedelta(minutes=index * 10), SECRET_QUOTE),),
                privacy="intimate", policy="topic_only", key=f"k{index}",
            )

        bounded = _lines(database, max_sensitive_lines=2)
        assert len(bounded) == 2, "敏感行按自己的上限收，最新的优先"

        starved = _lines(database, max_sensitive_tokens=1)
        assert starved == (), "敏感额度不足时那一行照旧不出现，而不是挤掉别人的额度"
    finally:
        database.close()


def test_an_ordinary_only_history_is_untouched_by_the_sensitive_budget(tmp_path):
    """不误判：新额度只对带敏感标注的窗口生效，日常台账一个字都不变。"""

    database = Database(tmp_path / "ordinary-only.sqlite3")
    try:
        events = EventRepository(database)
        for index in range(10):
            _fragment(
                database,
                (_event(events, index, NOW - timedelta(minutes=index * 10), "日常"),),
                privacy="ordinary", policy="daily_safe", key=f"k{index}",
            )

        plain = _lines(database)
        widened = _lines(database, max_sensitive_lines=8, max_sensitive_tokens=4000)

        assert plain == widened, "没有敏感窗口时，新额度必须完全不改变渲染结果"
        assert len(plain) == 8
    finally:
        database.close()


def test_a_sensitive_line_says_which_scope_the_details_are_in(tmp_path):
    """2026-09-15 用户裁定：讨论不等于有——索引行要给出「谈到／想象」的事实分布。

    以前只有一个「含成人内容」，把"聊过"和"做过"压成同一件事；只概括会丢失信息。
    """

    database = Database(tmp_path / "scope-note.sqlite3")
    try:
        events = EventRepository(database)
        first = _event(events, 1, NOW - timedelta(hours=3), SECRET_QUOTE)
        second = _event(events, 2, NOW - timedelta(hours=2, minutes=50), SECRET_QUOTE)
        _fragment(database, (first,), privacy="adult", policy="explicit_request_only",
                  key="talked", scope="conversation")
        _fragment(database, (second,), privacy="adult", policy="explicit_request_only",
                  key="imagined", scope="shared_imagination")

        lines = _lines(database)

        assert all("含成人内容" in line for line in lines)
        # 分布是**每个片段各自**的：谈到的那段写「谈到」，想象的那段写「想象」。
        assert any("（谈到 1）" in line for line in lines)
        assert any("（想象 1）" in line for line in lines)
        for line in lines:
            assert "发生过" not in line and "做了" not in line, "只呈现事实，不当断言"
    finally:
        database.close()


def test_an_ordinary_line_carries_no_scope_note(tmp_path):
    """不误判：日常行一个字都不变（scope 补充只给敏感行）。"""

    database = Database(tmp_path / "plain-note.sqlite3")
    try:
        events = EventRepository(database)
        item = _event(events, 1, NOW - timedelta(hours=1), SECRET_QUOTE)
        _fragment(database, (item,), privacy="ordinary", policy="daily_safe", key="plain")

        line = _lines(database)[0]

        assert "日常" in line
        assert "（" not in line, "日常行不带任何 scope 补充"
    finally:
        database.close()


def test_the_scope_note_is_dropped_before_the_label_when_the_line_is_too_long(tmp_path):
    """太长时依次让位：计数 → scope 补充；标签与「哪一段」永远保住。"""

    database = Database(tmp_path / "note-cap.sqlite3")
    try:
        events = EventRepository(database)
        item = _event(events, 1, NOW - timedelta(hours=1), SECRET_QUOTE)
        _fragment(database, (item,), privacy="adult", policy="explicit_request_only",
                  key="long", scope="shared_imagination")

        roomy = _lines(database, max_line_tokens=200)[0]
        tight = _lines(database, max_line_tokens=30)[0]

        assert "（想象 1）" in roomy
        assert "含成人内容" in tight and "想象" not in tight
    finally:
        database.close()
