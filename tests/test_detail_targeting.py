from __future__ import annotations

import asyncio
import json
from datetime import timedelta, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from qichi.app import G0Application
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory_details import MemoryDetailDraft, MemoryDetailEvidence, MemoryDetailRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MemoryDetailRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import NOW, OWNER, FakeLLM, FakeNapCat, app, seed_event  # noqa: E402

RABBIT_MARK = "垂耳白兔子"


def _fragment(database, events, *, privacy: str, policy: str, key: str):
    repository = MemoryDetailRepository(database)
    fragment = repository.build_fragment(events, key, None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="记录一条原文。", exact_quote=item.text, source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class=privacy, recall_policy=policy, evidence=((item.event_id, "source"),),
        )
        for index, item in enumerate(events)
    )
    details = repository.build_details(fragment, drafts, events)
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=events, details=details)
    return fragment.fragment_id


def _rabbit_fragment(database, events):
    first = seed_event(events, "rabbit-1", RABBIT_MARK + "那次你笑得很开心", at=NOW - timedelta(days=3))
    second = seed_event(events, "rabbit-2", "我说下次还带" + RABBIT_MARK + "来", at=NOW - timedelta(days=3) + timedelta(minutes=4))
    seeded = (first, second)
    return _fragment(database, seeded, privacy="ordinary", policy="daily_safe", key="rabbit"), seeded


def _plain_fragment(database, events):
    first = seed_event(events, "plain-1", "今天天气不错，晚上想吃火锅", at=NOW - timedelta(days=1))
    second = seed_event(events, "plain-2", "那就订那家老店，我等你", at=NOW - timedelta(days=1) + timedelta(minutes=4))
    seeded = (first, second)
    return _fragment(database, seeded, privacy="ordinary", policy="daily_safe", key="plain"), seeded


def _turn(application, events, text: str, event_id: str) -> str:
    event = events.insert(ConversationEvent(
        event_id, "pe-" + event_id, "pm-" + event_id, OWNER, 901, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    ))
    built = asyncio.run(application._build_input(event, 1))
    return "\n".join(str(getattr(message, "content", message)) for message in built.role_messages)


def test_a_request_that_points_at_nothing_unfolds_nothing(tmp_path):
    database = Database(tmp_path / "target-newest.sqlite3")
    try:
        events = EventRepository(database)
        _rabbit_fragment(database, events)
        _plain_fragment(database, events)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "细说那次", "target-1")

        # 2026-09-11 用户裁定 B1′：话里没有任何指向时什么都不展开，也不再猜最近一段。
        # 2026-09-12 甲：最近 2 天的普通原话会常驻「最近原文足迹」（有条数上限），
        # 所以判据从「一个字都不许出现」收窄为「明细块没有打开、也没有拿别的段顶上」。
        assert "[详细时间线证据" not in rendered
        assert "今天天气不错" not in _detail_block(rendered)
        assert RABBIT_MARK not in rendered
    finally:
        database.close()


def test_a_content_clue_alone_no_longer_opens_an_episode(tmp_path):
    """2026-09-12 T2：词面命中只说明「像」，不说明「你要」。

    这一条原本断言「话里点到的那段必须打开」。审计发现同一个判据让两个字打开了
    32 条成人原文，因此「钥匙」收紧成日期／引用／逐字原话（计划 §2.1）。改口路径
    （上一轮摊错、这一轮换一段）由 T6 单独放行，不靠词面自己开门。
    """

    database = Database(tmp_path / "target-clue.sqlite3")
    try:
        events = EventRepository(database)
        _rabbit_fragment(database, events)
        _plain_fragment(database, events)
        application = app(database, FakeLLM(["回复", "回复"]), FakeNapCat())

        corrected = _turn(application, events, "不是，我说的是" + RABBIT_MARK + "那次", "target-2b")

        assert "[详细时间线证据" not in corrected, "词面点一下不算请求"
        assert "今天天气不错" not in _detail_block(corrected)
    finally:
        database.close()


def _detail_block(rendered: str) -> str:
    """Just the detail block, so an assertion cannot be met by the user's own words."""

    marker = "[详细时间线证据"
    start = rendered.find(marker)
    return "" if start < 0 else rendered[start:]


def test_history_words_alone_never_open_an_episode(tmp_path):
    """2026-09-12 T7：这条以前是假绿。

    原断言是 RABBIT_MARK in rendered，而**上一轮用户自己**说过「垂耳白兔子」——
    那句话本来就会作为最近对话重放，所以明细块开不开它都成立。现在只对明细块本身
    断言，并把「重放里确实有那句话」单独写出来，说明当初为什么会绿。
    """

    database = Database(tmp_path / "target-history.sqlite3")
    try:
        events = EventRepository(database)
        _rabbit_fragment(database, events)
        _plain_fragment(database, events)
        application = app(database, FakeLLM(["回复", "回复"]), FakeNapCat())

        opened = _turn(application, events, "我想聊聊" + RABBIT_MARK + "的事", "target-3a")
        rendered = _turn(application, events, "细说那次", "target-3b")

        assert _detail_block(opened) == "", "词面不是钥匙：第一轮什么都不展开"
        assert _detail_block(rendered) == "", "上一轮说过的话也不会替这一轮开门"
        assert RABBIT_MARK in rendered, "上一轮原话会被重放——原断言正是被这一句满足的"
        assert "今天天气不错" not in _detail_block(rendered)
    finally:
        database.close()


def test_an_ordinary_turn_opens_no_detail_block(tmp_path):
    database = Database(tmp_path / "target-daily.sqlite3")
    try:
        events = EventRepository(database)
        _rabbit_fragment(database, events)
        _plain_fragment(database, events)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "今天有点累，随便聊聊", "target-4")

        assert "[最近片段索引]" in rendered
        assert "[详细时间线证据" not in rendered
        assert RABBIT_MARK not in rendered
    finally:
        database.close()


def test_a_verbatim_repeat_opens_ordinary_details_only(tmp_path):
    """2026-09-12 T2：原样复述是一把钥匙，但只开普通明细（计划 §2.1 的 K3）。

    成人明细要的是日期或引用（K1/K2）。这一条同时守住两侧：复述不越权，点名那天
    才把成人明细放出来。
    """

    database = Database(tmp_path / "target-adult.sqlite3")
    try:
        events = EventRepository(database)
        first = seed_event(events, "adult-1", "只属于我们两个人的夜晚", at=NOW - timedelta(hours=20))
        second = seed_event(events, "adult-2", "我记得你答应过我的事", at=NOW - timedelta(hours=20) + timedelta(minutes=3))
        _fragment(database, (first, second), privacy="adult", policy="explicit_request_only", key="adult")
        application = app(database, FakeLLM(["回复", "回复"]), FakeNapCat())

        daily = _turn(application, events, "今天有点累", "target-5a")
        repeated = _turn(application, events, "只属于我们两个人的夜晚", "target-5b")
        by_day = _turn(application, events, "细说27号那天", "target-5c")

        assert "[详细时间线证据" not in daily
        assert "只属于我们两个人的夜晚" not in daily
        assert "[详细时间线证据" not in repeated, "复述一句话不开成人的门"
        assert "[详细时间线证据" in by_day, "点名那天才把成人明细放出来"
        assert "只属于我们两个人的夜晚" in by_day
    finally:
        database.close()


def test_a_named_day_with_nothing_stored_says_so(tmp_path):
    """2026-09-12 T4：点名了一个空日子时，把「没有」当成事实交给她。

    审计里的 P3：那天没有记录，代码却继续往下掉到词面，把别的段当成用户要的那段
    讲出来。这条守住两件事——什么都不展开，以及上下文里确实写着「没有」。
    """

    database = Database(tmp_path / "missing-day.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="keep", at=NOW - timedelta(days=2), texts=("那天中午我们说过的话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(application, events, "回忆一下25号那天", "missing-1")
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "date_missing"
        assert details["memory_detail_reason"] == "day_missing"
        assert details["memory_detail_count"] == 0, "空日子不展开任何原文"
        assert "那天中午我们说过的话" not in _detail_block(rendered), "也不许拿别的段顶上"
        assert "8月25日 没有存下来的记录" in rendered, "「没有」要作为事实写进上下文"
    finally:
        database.close()


def test_today_opens_ordinary_details_and_leaves_adult_closed(tmp_path):
    """2026-09-12 T2：点到今天只开普通明细（计划 §2.1 的 K1′）。

    今天是最容易被顺手点到的一天，所以它拿到的是最窄的一把钥匙：同一段里只要有
    成人明细，那部分仍然关着。
    """

    database = Database(tmp_path / "target-today.sqlite3")
    try:
        events = EventRepository(database)
        adult_first = seed_event(events, "today-adult-1", "只属于我们两个人的夜晚", at=NOW - timedelta(minutes=50))
        adult_second = seed_event(events, "today-adult-2", "我记得你答应过我的事", at=NOW - timedelta(minutes=47))
        _fragment(database, (adult_first, adult_second), privacy="adult", policy="explicit_request_only", key="today-adult")
        plain_first = seed_event(events, "today-plain-1", "中午我们说的是晚饭吃什么", at=NOW - timedelta(minutes=90))
        plain_second = seed_event(events, "today-plain-2", "我说想吃那家老店", at=NOW - timedelta(minutes=87))
        _fragment(database, (plain_first, plain_second), privacy="ordinary", policy="daily_safe", key="today-plain")
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(application, events, "细说今天中午", "today-1")
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "today"
        assert "中午我们说的是晚饭吃什么" in rendered, "今天的普通明细照常打开"
        assert "只属于我们两个人的夜晚" not in rendered, "今天的成人明细仍然关着"
    finally:
        database.close()


def test_expanded_details_are_verbatim_and_in_order(tmp_path):
    database = Database(tmp_path / "target-order.sqlite3")
    try:
        events = EventRepository(database)
        seeded = tuple(
            seed_event(events, "order-" + str(index), "第" + str(index) + "句原文留在账本里",
                       at=NOW - timedelta(days=1) + timedelta(minutes=index))
            for index in range(5)
        )
        _fragment(database, seeded, privacy="ordinary", policy="daily_safe", key="order")
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "第0句原文留在账本里", "target-6")

        ordinals = [segment.split(";")[0].strip() for segment in rendered.split("- ordinal=")[1:]]
        assert ordinals == ["0", "1", "2", "3", "4"], "the timeline must stay in stored order"
        for event in seeded:
            assert event.text in rendered, "every expanded line is the stored original text"
    finally:
        database.close()


def _stored_detail(fragment_id: str, ordinal: int) -> MemoryDetailRecord:
    event_id = fragment_id + "-evt-" + str(ordinal)
    return MemoryDetailRecord(
        detail_id=fragment_id + "-detail-" + str(ordinal), fragment_id=fragment_id, ordinal=ordinal,
        detail_kind="message", actor="mumo", reality_scope="conversation",
        normalized_detail="记录一条原文。", exact_quote="第" + str(ordinal) + "条很短的原话",
        source_event_id=event_id, occurred_at_utc=NOW, certainty="explicit",
        temporal_scope="historical", status="active", privacy_class="ordinary",
        recall_policy="daily_safe", evidence=(MemoryDetailEvidence(event_id, "source"),),
    )


def test_expansion_stays_inside_the_per_fragment_budget():
    crowded = tuple(_stored_detail("frag-a", index) for index in range(40))
    sparse = tuple(_stored_detail("frag-b", index) for index in range(5))

    kept, note = G0Application._cap_details(crowded + sparse, "")

    assert len(kept) == 37, "each episode carries its own budget"
    assert {item.fragment_id for item in kept} == {"frag-a", "frag-b"}
    assert "8 条原文未展开" in note, "the remainder is stated, never silently dropped"


def _episode(database, events, *, key: str, texts, at):
    seeded = tuple(
        seed_event(events, key + "-" + str(index), text, at=at + timedelta(minutes=index))
        for index, text in enumerate(texts)
    )
    return _fragment(database, seeded, privacy="ordinary", policy="daily_safe", key=key), seeded


def test_a_named_date_opens_that_day_not_the_talk_about_it(tmp_path):
    database = Database(tmp_path / "date-vs-talk.sqlite3")
    try:
        events = EventRepository(database)
        # The episode the user means: two days before NOW, local date 08-26.
        episode, seeded = _episode(
            database, events, key="day", at=NOW - timedelta(days=2),
            texts=("那天下午我们聊了很久", "说的都是只属于我们的事"),
        )
        # A much newer conversation that merely talks about that date.
        talk, _ = _episode(
            database, events, key="talk", at=NOW - timedelta(hours=1),
            texts=("26号那天我这边确实没有记录", "不是不认账，是没证据我不编"),
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "细说26号那天", "date-1")

        assert "那天下午我们聊了很久" in rendered, "the named day must open its own episode"
        assert "26号那天我这边确实没有记录" not in _detail_block(rendered), (
            "a newer conversation that only mentions the date must not win"
        )
        assert "目标为推定" not in rendered, "a named date is an identified target"
    finally:
        database.close()


def test_a_chinese_numeral_date_is_understood(tmp_path):
    database = Database(tmp_path / "date-chinese.sqlite3")
    try:
        events = EventRepository(database)
        episode, _ = _episode(
            database, events, key="cn", at=NOW - timedelta(days=2),
            texts=("那天下午我们聊了很久",),
        )
        talk, _ = _episode(
            database, events, key="cntalk", at=NOW - timedelta(hours=1),
            texts=("26号那天我这边确实没有记录",),
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "那先回忆一下二十六号那天的吧", "date-2")

        assert "那天下午我们聊了很久" in rendered, "written-out numerals are the normal way to say it"
        assert "26号那天我这边确实没有记录" not in _detail_block(rendered)
    finally:
        database.close()


def test_yesterday_opens_yesterdays_episode(tmp_path):
    database = Database(tmp_path / "date-yesterday.sqlite3")
    try:
        events = EventRepository(database)
        yesterday, _ = _episode(
            database, events, key="y", at=NOW - timedelta(days=1),
            texts=("昨天傍晚我跟你说过一句话",),
        )
        older, _ = _episode(
            database, events, key="o", at=NOW - timedelta(days=3),
            texts=("这是更早那天的事",),
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "细说昨天那次", "date-3")

        assert "昨天傍晚我跟你说过一句话" in rendered
        assert "这是更早那天的事" not in rendered
    finally:
        database.close()


def test_last_night_opens_yesterdays_episode(tmp_path):
    """2026-09-22 真机回归：「昨晚」此前不在任何日期词表里，指针落到 none，展开永不发生。

    用户 09-22 早上连续 5 轮用「昨晚」指前一夜（「昨晚你抓住我的手腕…，这个你忘啦」），
    五轮全部 reason=none/day_missing、0 条明细，她只能回答「我这边没摊开细节」。
    """

    database = Database(tmp_path / "date-last-night.sqlite3")
    try:
        events = EventRepository(database)
        last_night, _ = _episode(
            database, events, key="n", at=NOW - timedelta(days=1),
            texts=("我抓住你的手腕往你腿间带，问你今晚还撑不撑得住",),
        )
        older, _ = _episode(
            database, events, key="o", at=NOW - timedelta(days=3),
            texts=("这是更早那天的事",),
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        # 问句只带日期词、不逐字复述那一段，好把这一条锁在日期判据上。
        rendered, event_id = _turn_with_event(
            application, events, "昨晚那段你还记得吗，具体说说", "date-ln"
        )
        details = _context_trace(database, event_id)

        # 指针键是 date_now（过去那天），规则名 day——修复前这里是 none、什么都不展开。
        assert details["memory_detail_key"] == "date_now", "「昨晚」要解析成过去那一天，不是 none"
        assert details["memory_detail_reason"] == "day"
        assert last_night in details["memory_detail_fragments"]
        # 只看展开块：常用足迹本来就会常驻最近几天的原话，不能用它判展开与否。
        block = _detail_block(rendered)
        assert "我抓住你的手腕往你腿间带" in block, "点名的那一段必须被展开"
        assert "这是更早那天的事" not in block, "不许把更早的段顶上来"
    finally:
        database.close()


def test_this_morning_points_at_today_not_yesterday(tmp_path):
    """不误判：合体写法要落到它自己那一天——今早是今天，不是昨天。"""

    database = Database(tmp_path / "date-this-morning.sqlite3")
    try:
        events = EventRepository(database)
        this_morning, _ = _episode(
            database, events, key="m", at=NOW - timedelta(minutes=30),
            texts=("今早我又赖了一会儿床",),
        )
        yesterday, _ = _episode(
            database, events, key="y", at=NOW - timedelta(days=1),
            texts=("昨天傍晚我跟你说过一句话",),
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(application, events, "细说今早那次", "date-tm")
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "today"
        assert this_morning in details["memory_detail_fragments"]
        block = _detail_block(rendered)
        assert "今早我又赖了一会儿床" in block
        assert "昨天傍晚我跟你说过一句话" not in block, "今早不该把昨天那一段也摊开"
    finally:
        database.close()


def test_the_verbatim_check_ignores_edge_punctuation_only():
    """2026-09-22 真机：模型抄她的话时在末尾补了一个「。」（她那条以颜文字结尾、原文没有
    句号），判失败 → 整个整合任务失败 → 记忆水位钉住 8 小时。两端句读不算内容。"""

    from qichi.domain.events import quote_is_verbatim

    source = "主人，腿间那点水已经流到腿根了，手往下挪挪，别只顾着上头那两点 (´-ω`)"
    assert quote_is_verbatim(source + "。", source) is True, "末尾补句读要放过"
    assert quote_is_verbatim("  " + source + "  ", source) is True, "两端空白要放过"
    assert quote_is_verbatim("「" + source + "」", source) is True, "两端引号要放过"


def test_the_verbatim_check_still_rejects_any_change_in_the_middle():
    """不误判：只放过两端句读——中间改一个字、或只有句读，都必须被拒。"""

    from qichi.domain.events import quote_is_verbatim

    source = "主人，腿间那点水已经流到腿根了，手往下挪挪"
    assert quote_is_verbatim("主人，腿间那点水已经流到腿根了，手往下动动", source) is False, "中间改字要拒"
    assert quote_is_verbatim("主人腿上那点水已经流到腿根了，手往下挪挪", source) is False, "中间少字要拒"
    assert quote_is_verbatim("。。", source) is False, "只有句读的引文不算数"
    assert quote_is_verbatim("", source) is False, "空引文不算数"
    assert quote_is_verbatim(source, None) is False, "没有来源就没有逐字"


def test_a_day_with_no_episode_unfolds_nothing(tmp_path):
    database = Database(tmp_path / "date-missing.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="recent", at=NOW - timedelta(hours=2), texts=("最近这段的原话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "细说15号那天", "date-4")

        assert "最近这段的原话" not in _detail_block(rendered), "点了一天但那天没有记录时，不许拿别的片段顶上"
        assert "[详细时间线证据" not in rendered
    finally:
        database.close()


def test_the_recall_block_tells_the_model_it_may_recount(tmp_path):
    database = Database(tmp_path / "date-recount.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="r", at=NOW - timedelta(days=1), texts=("当晚我对你说过的话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered = _turn(application, events, "当晚我对你说过的话", "date-5")

        assert "用户已明确要求回顾" in rendered, (
            "holding the lines is not the same as being allowed to retell them"
        )
    finally:
        database.close()


def test_a_day_named_a_couple_of_turns_back_still_decides(tmp_path):
    """2026-09-12 T2/T3：回看窗口从 8 条收到 3 条（计划 §2.1 的 K4）。

    窗口内：点名过的那天仍然自己说了算。窗口外：不再继承——她凭常驻索引问用户
    「你指哪一天」，而不是替他猜一段。
    """

    database = Database(tmp_path / "date-far-back.sqlite3")
    try:
        events = EventRepository(database)
        episode, _ = _episode(
            database, events, key="far", at=NOW - timedelta(days=2),
            texts=("那天中午我们说过的话",),
        )
        talk, _ = _episode(
            database, events, key="fartalk", at=NOW - timedelta(hours=1),
            texts=("26号那天我这边确实没有记录", "里面是空的，不是我不肯摊"),
        )
        application = app(database, FakeLLM(["回复"] * 8), FakeNapCat())

        _turn(application, events, "就是26号那天呐", "far-1")
        _turn(application, events, "你还记得吗", "far-2")
        inside = _turn(application, events, "你在回忆一下那天中午？为了我再试试呗", "far-3")

        assert "那天中午我们说过的话" in inside, "窗口内，点名过的那天仍然说了算"
        assert "里面是空的，不是我不肯摊" not in _detail_block(inside), (
            "a newer conversation about the missing record must not answer for the day"
        )

        # 再往后走三条，那次点名已经出了 3 条窗口：不继承，也不许拿别的段顶上。
        _turn(application, events, "再想想", "far-4")
        _turn(application, events, "试试看", "far-5")
        outside = _turn(application, events, "那次我们再聊聊？", "far-6")

        # 甲之后那两句原话可能出现在「最近原文足迹」里；这里断言明细块不继承、不顶替。
        assert "那天中午我们说过的话" not in _detail_block(outside), "窗口外不再继承"
        assert "里面是空的，不是我不肯摊" not in _detail_block(outside)
    finally:
        database.close()


def _turn_with_event(application, events, text: str, event_id: str):
    event = events.insert(ConversationEvent(
        event_id, "pe-" + event_id, "pm-" + event_id, OWNER, 902, "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),), None, None, NOW, NOW, "received", {},
    ))
    built = asyncio.run(application._build_input(event, 1))
    rendered = "\n".join(str(getattr(message, "content", message)) for message in built.role_messages)
    return rendered, event.event_id


def _context_trace(database, event_id: str) -> dict:
    row = database.connection.execute(
        "SELECT details_json FROM turn_trace_events WHERE trigger_event_id=? AND phase='context'",
        (event_id,),
    ).fetchone()
    assert row is not None, "每一轮都必须留下上下文轨迹"
    return json.loads(row[0])


def test_the_trace_records_which_rule_picked_the_episode(tmp_path):
    database = Database(tmp_path / "trace-day.sqlite3")
    try:
        events = EventRepository(database)
        episode, _ = _episode(database, events, key="trace", at=NOW - timedelta(days=2), texts=("那天中午的原话",))
        _episode(database, events, key="tracetalk", at=NOW - timedelta(hours=1), texts=("26号那天我这边没有记录",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn_with_event(application, events, "细说26号那天", "trace-1")
        details = _context_trace(database, event_id)

        assert details["memory_detail_reason"] == "day", "面板要能说出是哪条规则选中了这一轮"
        assert details["memory_detail_fragments"] == [episode]
        assert details["memory_detail_count"] >= 1
        assert details["category_tokens"]["memory_details"] > 0, "明细 token 不能在被投影时丢掉"
        assert details["category_tokens"]["memory_index"] > 0, "索引 token 同上"
    finally:
        database.close()


def test_the_trace_records_that_nothing_was_pointed_at(tmp_path):
    database = Database(tmp_path / "trace-guess.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="g", at=NOW - timedelta(hours=2), texts=("最近这段的原话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn_with_event(application, events, "细说那次", "trace-2")
        details = _context_trace(database, event_id)

        assert details["memory_detail_reason"] == "none"
        assert details["memory_detail_count"] == 0
    finally:
        database.close()


def test_an_ordinary_turn_records_no_targeting_at_all(tmp_path):
    database = Database(tmp_path / "trace-none.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="n", at=NOW - timedelta(hours=2), texts=("不会展开的原话",))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        _, event_id = _turn_with_event(application, events, "今天天气不错", "trace-3")
        details = _context_trace(database, event_id)

        assert details["memory_detail_reason"] == "none"
        assert details["memory_detail_count"] == 0
        assert details["memory_detail_fragments"] == []
        assert details["category_tokens"].get("memory_details", 0) == 0
    finally:
        database.close()


def test_within_budget_details_pass_through_untouched():
    details = tuple(_stored_detail("frag-a", index) for index in range(32))

    kept, note = G0Application._cap_details(details, "目标为推定")

    assert kept == details and note == "目标为推定"


def test_list_details_shares_the_lexical_rule_with_the_retriever(tmp_path):
    database = Database(tmp_path / "target-lexical.sqlite3")
    try:
        events = EventRepository(database)
        _rabbit_fragment(database, events)
        _plain_fragment(database, events)
        repository = MemoryDetailRepository(database)

        long_clue = repository.list_details(OWNER, query="不是，我说的是" + RABBIT_MARK + "那次")
        short_clue = repository.list_details(OWNER, query=RABBIT_MARK)
        unrelated = repository.list_details(OWNER, query="完全无关的另一件事情")

        assert len(long_clue) == 2, "a sentence-length clue must reach the stored lines"
        assert len(short_clue) == 2, "a short clue keeps the substring rule"
        assert unrelated == ()
    finally:
        database.close()


def test_fragment_filter_restricts_detail_listing(tmp_path):
    database = Database(tmp_path / "target-filter.sqlite3")
    try:
        events = EventRepository(database)
        rabbit, rabbit_events = _rabbit_fragment(database, events)
        plain, _ = _plain_fragment(database, events)
        repository = MemoryDetailRepository(database)

        selected = repository.list_details(OWNER, fragment_ids=(rabbit,))
        by_event = repository.fragments_for_events(OWNER, (rabbit_events[0].event_id,))
        on_day = repository.fragments_on_dates(
            OWNER, (rabbit_events[0].occurred_at_utc.date(),), local_zone=timezone.utc, limit=4
        )

        assert {item.fragment_id for item in selected} == {rabbit}
        assert by_event == (rabbit,)
        # 2026-09-12 T7：原来这里断言的是 newest_fragment()（「猜最近一段」那条路的
        # 遗留函数，已经没有调用者）。换成按日期取片段——那是现在真正在用的入口。
        assert on_day == (rabbit,)
        assert plain != rabbit
    finally:
        database.close()


def test_a_named_day_expands_the_newest_episodes_first(tmp_path):
    """2026-09-12 T6（裁定 A）：同一天有多段时，先给最近的那几段。

    复算过：09-11 那天有 6 段，旧顺序（最早优先）截断后留下的是那天的上午，
    而人嘴里的「那天」通常是最近发生的那部分。
    """

    database = Database(tmp_path / "day-order.sqlite3")
    try:
        zone = ZoneInfo("Asia/Shanghai")
        events = EventRepository(database)
        repository = MemoryDetailRepository(database)
        moments = [
            seed_event(events, "order-" + str(index), "第" + str(index) + "段原话",
                       at=NOW - timedelta(days=2) + timedelta(hours=index * 4 + 1))
            for index in range(3)
        ]
        fragments = []
        for index, event in enumerate(moments):
            fragment = repository.build_fragment((event,), "day-" + str(index), None, created_at_utc=NOW)
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
        day = moments[0].occurred_at_utc.astimezone(zone).date()

        newest_first = repository.fragments_on_dates(OWNER, (day,), local_zone=zone, limit=2)

        assert newest_first == (fragments[2], fragments[1]), "限额先花在最近的那几段上"
    finally:
        database.close()


def test_the_detail_budget_keeps_the_newest_episode_and_says_what_it_dropped(tmp_path):
    database = Database(tmp_path / "budget.sqlite3")
    try:
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        older = tuple(_stored_detail("frag-old", index) for index in range(4))
        newer = tuple(_stored_detail("frag-new", index) for index in range(4))

        kept, note = application._budget_details(older + newer, "", {}, budget=40)

        assert {item.fragment_id for item in kept} == {"frag-new"}, "预算先给最近的那一段"
        assert "4 条没有展开" in note, "丢掉的部分要如实说出来"
    finally:
        database.close()


def test_a_correction_can_reach_the_other_episode(tmp_path):
    """2026-09-12 T6：上一轮摊错了段，这一轮用词面改口——只开普通明细。"""

    database = Database(tmp_path / "correction.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="day", at=NOW - timedelta(days=2), texts=("那天中午我们说过的话",))
        rabbit, _ = _rabbit_fragment(database, events)
        application = app(database, FakeLLM(["回复", "回复"]), FakeNapCat())

        _turn(application, events, "细说26号那天", "corr-1")
        corrected, event_id = _turn_with_event(
            application, events, "不是，我说的是" + RABBIT_MARK + "那次", "corr-2"
        )
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "correction"
        assert details["memory_detail_reason"] == "correction"
        assert details["memory_detail_fragments"] == [rabbit]
        assert "我说下次还带" in corrected, "要真的换成另一段的内容，而不是重复上一段"
        assert "那天中午我们说过的话" not in _detail_block(corrected)
    finally:
        database.close()


def test_a_correction_needs_something_to_correct(tmp_path):
    database = Database(tmp_path / "correction-none.sqlite3")
    try:
        events = EventRepository(database)
        _episode(database, events, key="day", at=NOW - timedelta(days=2), texts=("那天中午我们说过的话",))
        _rabbit_fragment(database, events)
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(
            application, events, "不是，我说的是" + RABBIT_MARK + "那次", "corr-3"
        )
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "none", "上一轮什么都没摊开，就没有可改口的上文"
        assert details["memory_detail_count"] == 0
        assert "我说下次还带" not in rendered
    finally:
        database.close()



def _store_fragment(database, seeded, specs, *, key):
    """Store one fragment whose details may carry different privacy classes."""

    repository = MemoryDetailRepository(database)
    fragment = repository.build_fragment(seeded, key, None, created_at_utc=NOW)
    drafts = tuple(
        MemoryDetailDraft(
            ordinal=index, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="记录一条原文。", exact_quote=item.text, source_event_id=item.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class=privacy, recall_policy=policy, evidence=((item.event_id, "source"),),
        )
        for index, (item, (privacy, policy)) in enumerate(zip(seeded, specs))
    )
    details = repository.build_details(fragment, drafts, seeded)
    with database.transaction() as connection:
        repository.store_in_transaction(connection, fragment=fragment, events=seeded, details=details)
    return fragment.fragment_id


def test_a_message_that_names_two_days_opens_both(tmp_path):
    """2026-09-12 T8：真机上「准确来说不是昨天晚上，而是今天的凌晨」三轮都锁在昨天。

    过去的日子不再独占：它和「今天 + 时段词」一起打开。权限按片段分天算——今天
    那一段只有普通明细，过去那段照旧完整。
    """

    database = Database(tmp_path / "two-days.sqlite3")
    try:
        events = EventRepository(database)
        yesterday_seeded = (
            seed_event(events, "two-y1", "昨天晚上的那句原话", at=NOW - timedelta(days=1)),
        )
        yesterday = _store_fragment(
            database, yesterday_seeded, (("adult", "explicit_request_only"),), key="two-y"
        )
        today_seeded = (
            seed_event(events, "two-t1", "今天凌晨的那句普通原话", at=NOW - timedelta(hours=3)),
            seed_event(events, "two-t2", "今天凌晨的那句成人原话",
                       at=NOW - timedelta(hours=3) + timedelta(minutes=1)),
        )
        today = _store_fragment(
            database, today_seeded,
            (("ordinary", "daily_safe"), ("adult", "explicit_request_only")), key="two-t",
        )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(
            application, events, "准确来说不是昨天，而是今天的凌晨，零点过后那段时间", "two-1"
        )
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "date_now"
        assert set(details["memory_detail_fragments"]) == {yesterday, today}
        assert "昨天晚上的那句原话" in rendered, "过去的那天照旧打开"
        assert "今天凌晨的那句普通原话" in rendered, "今天也要打开"
        assert "今天凌晨的那句成人原话" not in rendered, "今天只给普通明细"
    finally:
        database.close()


def test_only_today_named_still_opens_today_alone(tmp_path):
    database = Database(tmp_path / "today-alone.sqlite3")
    try:
        events = EventRepository(database)
        yesterday_seeded = (
            seed_event(events, "alone-y1", "昨天晚上的那句原话", at=NOW - timedelta(days=1)),
        )
        _store_fragment(database, yesterday_seeded, (("ordinary", "daily_safe"),), key="alone-y")
        today_seeded = (
            seed_event(events, "alone-t1", "今天凌晨的那句原话", at=NOW - timedelta(hours=3)),
        )
        _store_fragment(database, today_seeded, (("ordinary", "daily_safe"),), key="alone-t")
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(application, events, "细说今天凌晨", "alone-1")
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "today"
        assert "今天凌晨的那句原话" in rendered
        # 甲之后：昨晚那条可能出现在「最近原文足迹」里；明细块里仍然不许带出来。
        assert "昨天晚上的那句原话" not in _detail_block(rendered), "没点昨天就不许顺手带出来"
    finally:
        database.close()



def test_the_time_of_day_word_decides_which_episode_comes_first(tmp_path):
    """2026-09-12 T10：问「今天凌晨」，当天下午那段不许把凌晨那段挤掉。"""

    database = Database(tmp_path / "band-order.sqlite3")
    try:
        events = EventRepository(database)
        moments = {
            "dawn": NOW - timedelta(hours=9),       # 本地 01:00
            "morning": NOW - timedelta(hours=2),    # 本地 08:00
            "afternoon": NOW + timedelta(hours=5),  # 本地 15:00 —— 当天最新
        }
        fragments = {}
        for key, moment in moments.items():
            seeded = (seed_event(events, "band-" + key, key + " 的原话", at=moment),)
            fragments[key] = _store_fragment(
                database, seeded, (("ordinary", "daily_safe"),), key="band-" + key
            )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(application, events, "细说今天凌晨", "band-1")
        details = _context_trace(database, event_id)

        assert details["memory_detail_key"] == "today"
        assert fragments["dawn"] in details["memory_detail_fragments"], "点名的时段要排在前面"
        assert "dawn 的原话" in rendered
    finally:
        database.close()


def test_the_hour_windows_come_from_the_words_the_user_used():
    from qichi.memory.dates import band_label, time_of_day_hours

    assert time_of_day_hours("今天的凌晨，零点过后") == ((0, 5),)
    assert time_of_day_hours("昨天下午") == ((13, 17),)
    assert time_of_day_hours("今天中午和九号中午") == ((11, 13),)
    assert time_of_day_hours("今天天气不错") == ()
    # 2026-09-22：合体写法（哪一天 + 哪个时段）此前一个都没被认出来。
    assert time_of_day_hours("昨晚") == ((17, 24),)
    assert time_of_day_hours("昨夜") == ((17, 24),)
    assert time_of_day_hours("今晚我要加班") == ((17, 24),)
    assert [band_label(hour) for hour in (1, 7, 9, 12, 15, 21)] == [
        "凌晨", "早上", "上午", "中午", "下午", "晚上"
    ]


def test_clock_words_are_read_as_hours_only_when_they_are_hours():
    from qichi.memory.dates import time_of_day_anchors

    assert time_of_day_anchors("今天的凌晨，零点过后那段时间") == (0,)
    assert time_of_day_anchors("凌晨一点多的时候") == (1,)
    assert time_of_day_anchors("23点前后") == (23,)
    assert time_of_day_anchors("昨天晚上十点半") == (10,)
    assert time_of_day_anchors("我今天有点失落") == (), "「有点」不是钟点"
    assert time_of_day_anchors("再等一点点就好") == (), "「一点点」不是钟点"
    assert time_of_day_anchors("随便聊聊") == ()


def test_the_clock_word_decides_which_of_the_band_comes_first(tmp_path):
    """点了「零点过后」，就该先给最靠近零点的那一段，而不是同段里最新的那一段。"""

    database = Database(tmp_path / "anchor-order.sqlite3")
    try:
        events = EventRepository(database)
        moments = {
            "just-after-midnight": NOW - timedelta(hours=9, minutes=30),   # 本地 00:30
            "one-am": NOW - timedelta(hours=8),                            # 本地 02:00
            "three-am": NOW - timedelta(hours=6),                          # 本地 04:00
            "afternoon": NOW + timedelta(hours=5),                         # 本地 15:00
        }
        fragments = {}
        for key, moment in moments.items():
            seeded = (seed_event(events, "anchor-" + key, key + " 的原话", at=moment),)
            fragments[key] = _store_fragment(
                database, seeded, (("ordinary", "daily_safe"),), key="anchor-" + key
            )
        application = app(database, FakeLLM(["回复"]), FakeNapCat())

        rendered, event_id = _turn_with_event(
            application, events, "今天的凌晨，零点过后那段时间", "anchor-1"
        )
        details = _context_trace(database, event_id)

        assert fragments["just-after-midnight"] in details["memory_detail_fragments"], (
            "最靠近零点的那一段要排在前面"
        )
        assert "just-after-midnight 的原话" in rendered
    finally:
        database.close()


def _same_day_rank_fixtures(database, events):
    """同一天两段：old 装着那句话，new 无关。"""

    repository = MemoryDetailRepository(database)
    seeded = []
    for name, text, offset in (("rank-old", "兔子你如果会的话，什么时候会想要我呀", 2), ("rank-new", "今天上课很累", 0)):
        event = seed_event(events, name, text, at=NOW - timedelta(hours=offset))
        fragment = repository.build_fragment((event,), name + "-frag", None, created_at_utc=NOW)
        drafts = (MemoryDetailDraft(
            ordinal=0, detail_kind="message", actor="mumo", reality_scope="conversation",
            normalized_detail="记录一条原文。", exact_quote=text, source_event_id=event.event_id,
            certainty="explicit", temporal_scope="historical", status="active",
            privacy_class="ordinary", recall_policy="daily_safe", evidence=((event.event_id, "source"),),
        ),)
        details = repository.build_details(fragment, drafts, (event,))
        with database.transaction() as connection:
            repository.store_in_transaction(connection, fragment=fragment, events=(event,), details=details)
        seeded.append(fragment.fragment_id)
    return repository, tuple(seeded)


def test_ranking_orders_the_fragment_it_matches_first(tmp_path):
    """命中：装着那句话的那段排最前，即使它不是最新的一段。"""

    database = Database(tmp_path / "rank-hit.sqlite3")
    try:
        events = EventRepository(database)
        repository, (old_fragment, new_fragment) = _same_day_rank_fixtures(database, events)
        ranked = repository.fragments_ranked_by_text(
            OWNER, ("今天凌晨我还问过你，什么时候想要我这件事",),
            within=(new_fragment, old_fragment), limit=8,
        )

        assert ranked and ranked[0][0] == old_fragment, "最新的那段不该因为新就排前面"
        assert ranked[0][1] >= 3, "最长公共连续串（「想要我」/「什么时候」）必须被认出来"
    finally:
        database.close()


def test_ranking_never_reaches_outside_the_fragments_it_was_given(tmp_path):
    """不越界：排序只允许重排给定片段，绝不把别的片段带进来。"""

    database = Database(tmp_path / "rank-scope.sqlite3")
    try:
        events = EventRepository(database)
        repository, (old_fragment, new_fragment) = _same_day_rank_fixtures(database, events)
        ranked = repository.fragments_ranked_by_text(
            OWNER, ("今天凌晨我还问过你，什么时候想要我这件事",),
            within=(new_fragment,), limit=8,
        )

        assert all(fragment_id == new_fragment for fragment_id, *_ in ranked), "给定的范围之外一个都不许出现"
    finally:
        database.close()


def test_ranking_says_nothing_when_the_turn_matches_nothing(tmp_path):
    """不误判：这轮说的话跟哪段都对不上时，返回空，调用方保持原顺序。"""

    database = Database(tmp_path / "rank-none.sqlite3")
    try:
        events = EventRepository(database)
        repository, (old_fragment, new_fragment) = _same_day_rank_fixtures(database, events)
        ranked = repository.fragments_ranked_by_text(
            OWNER, ("外面下雨了记得带伞",), within=(old_fragment, new_fragment), limit=8,
        )

        assert ranked == ()
    finally:
        database.close()

