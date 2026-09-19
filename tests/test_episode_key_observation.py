"""T1（观测先行）：新的判据只被记录，不被执行。

2026-09-12 冻结的计划把「钥匙（授权）」和「定位」拆开
（历史修复计划 §2）。T1 只把新判据**观测下来**：
真正的开关仍然是 _points_at_an_episode，所以两套判据不一致的地方必须先能在
轨迹里看见，再谈改行为。T2 才会翻转其中任何一条。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
from pathlib import Path
import sys

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory_details import MemoryDetailDraft
from qichi.memory.verbatim import VERBATIM_MIN_CHARS, shares_run
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MemoryDetailRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import NOW, OWNER, FakeLLM, FakeNapCat, app, seed_event  # noqa: E402

EVERYDAY_WORD = "不够"
VERBATIM_QUOTE = "垂耳白兔子那晚你笑得很开心"
# 只重合前七个字：比门槛少一个字，任何情况下都不许当成指向。
SHORT_REPEAT = "你还记得垂耳白兔子那晚的事吗"


def _fragment(database, events, *, key: str, texts, at=None, privacy="ordinary", policy="daily_safe"):
    moment = NOW - timedelta(days=2) if at is None else at
    repository = MemoryDetailRepository(database)
    seeded = tuple(
        seed_event(events, key + "-" + str(index), text, at=moment + timedelta(minutes=index))
        for index, text in enumerate(texts)
    )
    fragment = repository.build_fragment(seeded, key, None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="记录一条原文。", exact_quote=item.text, source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class=privacy, recall_policy=policy, evidence=((item.event_id, "source"),),
        )
        for index, item in enumerate(seeded)
    )
    details = repository.build_details(fragment, drafts, seeded)
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=seeded, details=details)
    return fragment.fragment_id, seeded


def _turn(application, events, text: str, event_id: str):
    event = events.insert(ConversationEvent(
        event_id, "pe-" + event_id, "pm-" + event_id, OWNER, 903, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    ))
    built = asyncio.run(application._build_input(event, 1))
    rendered = "\n".join(str(getattr(message, "content", message)) for message in built.role_messages)
    return rendered, event.event_id


def _trace(database, event_id: str) -> dict:
    row = database.connection.execute(
        "SELECT details_json FROM turn_trace_events WHERE trigger_event_id=? AND phase='context'",
        (event_id,),
    ).fetchone()
    assert row is not None, "每一轮都必须留下上下文轨迹"
    return json.loads(row[0])


def test_the_verbatim_floor_is_eight_characters():
    assert VERBATIM_MIN_CHARS == 8
    assert shares_run(SHORT_REPEAT, VERBATIM_QUOTE) is False, "七个字重合不算指向"
    assert shares_run("你还记得" + VERBATIM_QUOTE + "吗", VERBATIM_QUOTE) is True
    assert shares_run("今天天气不错", VERBATIM_QUOTE) is False
    assert shares_run("短", VERBATIM_QUOTE) is False


def test_a_two_character_word_records_no_key(tmp_path):
    database = Database(tmp_path / "key-everyday.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="everyday", texts=("那样还不够，再多一点", "这一句也是原话"))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn(application, events, EVERYDAY_WORD, "key-1")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "none", "两个字不是指向"
        # T2 翻转过来了：词面命中不再开门，所以这一轮什么都不展开。
        assert details["memory_detail_count"] == 0
        assert details["memory_detail_reason"] == "none"
        # 甲之后原话会常驻足迹；明细块本身仍然不许打开。
        assert "[详细时间线证据" not in rendered
    finally:
        database.close()


def test_a_past_date_named_this_turn_is_the_key(tmp_path):
    database = Database(tmp_path / "key-date.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="date", texts=("那天中午我们说过的话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, "细说26号那天", "key-2")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "date_now"
    finally:
        database.close()


def test_a_past_date_with_nothing_stored_is_recorded_as_missing(tmp_path):
    database = Database(tmp_path / "key-missing.sqlite3")
    try:
        events = EventRepository(database)
        # 只有 26 号有片段；25 号点得出来，但那天什么都没有。
        _fragment(database, events, key="missing", texts=("那天中午我们说过的话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, "细说25号那天", "key-2b")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "date_missing", (
            "点了过去的日期但那天是空的，要单独记一档，不能悄悄变成别的规则"
        )
    finally:
        database.close()


def test_today_named_this_turn_is_its_own_key(tmp_path):
    database = Database(tmp_path / "key-today.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="today", texts=("今天中午说过的话",), at=NOW - timedelta(hours=1))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, "今天中午我们聊了什么", "key-3")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "today", "今天单独记一档，它只开普通明细"
    finally:
        database.close()


def test_a_day_named_three_messages_back_is_the_inherited_key(tmp_path):
    database = Database(tmp_path / "key-inherited.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="inherited", texts=("那天中午我们说过的话",))
        application = app(database, FakeLLM(["回复"] * 4), FakeNapCat())

        _turn(application, events, "就是26号那天呐", "key-4a")
        _turn(application, events, "你还记得吗", "key-4b")
        _turn(application, events, "再想想", "key-4c")
        _, event_id = _turn(application, events, "你在回忆一下那天中午？为了我再试试呗", "key-4d")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "date_inherited", "点名过的那天还在回看窗口里"
    finally:
        database.close()


def test_today_is_never_inherited_from_earlier_messages(tmp_path):
    database = Database(tmp_path / "key-today-inherit.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="today-inherit", texts=("今天中午说过的话",))
        application = app(database, FakeLLM(["回复"] * 4), FakeNapCat())

        _turn(application, events, "今天中午我们聊了什么", "key-5a")
        _turn(application, events, "你还记得吗", "key-5b")
        _turn(application, events, "再想想", "key-5c")
        _, event_id = _turn(application, events, "那你说说看", "key-5d")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "none", "今天不从回看窗口继承"
    finally:
        database.close()


def test_a_long_verbatim_repeat_is_the_key(tmp_path):
    database = Database(tmp_path / "key-verbatim.sqlite3")
    try:
        events = EventRepository(database)
        fragment_id, _ = _fragment(database, events, key="verbatim", texts=(VERBATIM_QUOTE,))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, "你还记得吗，" + VERBATIM_QUOTE + "，是真的吗", "key-6")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "verbatim"
        assert details["memory_detail_match_count"] == 1
        assert details["memory_detail_fragments"] == [fragment_id]
    finally:
        database.close()


def test_a_short_repeat_never_becomes_the_key(tmp_path):
    database = Database(tmp_path / "key-short.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="short", texts=(VERBATIM_QUOTE,))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, SHORT_REPEAT, "key-7")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "none", "差一个字就不认"
    finally:
        database.close()


def test_a_quote_that_lands_on_the_episode_is_the_key(tmp_path):
    database = Database(tmp_path / "key-quote.sqlite3")
    try:
        events = EventRepository(database)
        _, seeded = _fragment(database, events, key="quote", texts=("被引用的那一句原话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        current = seed_event(events, "quote-turn", "引用一下这句", at=NOW)

        pointer = application._episode_pointer(current, (), {seeded[0].event_id}, NOW)

        assert pointer.key == "quote"
        assert pointer.match_count == 0
        assert pointer.fragments
    finally:
        database.close()


def test_the_trace_records_the_observation_fields(tmp_path):
    database = Database(tmp_path / "key-trace.sqlite3")
    try:
        events = EventRepository(database)
        fragment_id, _ = _fragment(database, events, key="trace", texts=("那天中午我们说过的话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, "细说26号那天", "key-8")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "date_now"
        assert details["memory_detail_match_count"] == 0
        assert details["memory_detail_indexed"] is True, "展开的片段必须有索引行"
        assert details["memory_detail_fragments"] == [fragment_id]
    finally:
        database.close()


def test_an_ordinary_turn_reports_no_key_and_a_clean_index(tmp_path):
    database = Database(tmp_path / "key-none.sqlite3")
    try:
        events = EventRepository(database)
        _fragment(database, events, key="none", texts=("不会被摊开的原话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn(application, events, "今天天气不错", "key-9")
        details = _trace(database, event_id)

        assert details["memory_detail_key"] == "none"
        assert details["memory_detail_match_count"] == 0
        assert details["memory_detail_indexed"] is True, "没有展开时索引一致性天然成立"
        assert details["memory_detail_count"] == 0
    finally:
        database.close()


def _two_same_day_fragments(database, events):
    """同一天两段：老的那段装着那句话，新的那段无关。"""

    repository = MemoryDetailRepository(database)
    old_event = seed_event(events, "rank-old", "兔子你如果会的话，什么时候会想要我呀", at=NOW - timedelta(hours=2))
    new_event = seed_event(events, "rank-new", "今天上课很累", at=NOW)
    fragments = []
    for name, event in (("rank-old-frag", old_event), ("rank-new-frag", new_event)):
        fragment = repository.build_fragment((event,), name, None, created_at_utc=NOW)
        drafts = (MemoryDetailDraft(
            ordinal=0, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="记录一条原文。", exact_quote=event.text, source_event_id=event.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class="ordinary", recall_policy="daily_safe", evidence=((event.event_id, "source"),),
        ),)
        details = repository.build_details(fragment, drafts, (event,))
        with database.transaction() as connection:
            repository.store_in_transaction(connection, fragment=fragment, events=(event,), details=details)
        fragments.append(fragment.fragment_id)
    return tuple(fragments)


def test_the_named_days_fragments_are_reordered_by_this_turns_words(tmp_path):
    """2026-09-17：他点名「今天凌晨」又提了那句话，装那句话的那段必须排最前。

    真机那次：指针按「钟点距离 + 最新优先」取样，装着那句话的那段排第三，明细块预算只装得下
    最前面一段 → 她只能说「我这儿翻不着」。排序只重排**这把钥匙已经授权的片段**。
    """

    database = Database(tmp_path / "key-rank.sqlite3")
    try:
        events = EventRepository(database)
        old_fragment, new_fragment = _two_same_day_fragments(database, events)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        current = seed_event(events, "rank-turn", "今天凌晨我还问过你，什么时候想要我这件事", at=NOW)

        pointer = application._episode_pointer(current, (), set(), NOW)

        assert pointer.key == "today"
        assert set(pointer.fragments) == {old_fragment, new_fragment}
        assert pointer.fragments[0] == old_fragment, "对得上这轮话的那段排最前，才装得进预算"
    finally:
        database.close()


def test_without_a_content_hit_the_days_own_order_is_kept(tmp_path):
    """不误判：这轮说的话跟两段都对不上时，顺序仍是取样给的（最新优先）。"""

    database = Database(tmp_path / "key-rank-plain.sqlite3")
    try:
        events = EventRepository(database)
        old_fragment, new_fragment = _two_same_day_fragments(database, events)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        current = seed_event(events, "rank-plain", "今天凌晨外面下雨了", at=NOW)

        pointer = application._episode_pointer(current, (), set(), NOW)

        assert pointer.fragments == (new_fragment, old_fragment)
    finally:
        database.close()
