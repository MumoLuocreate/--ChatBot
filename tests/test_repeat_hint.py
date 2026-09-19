"""复读统计提示：只报她自己反复用过的短语，且够不到门槛就不出现。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qichi.dialogue.context_builder import _recent_repeat_hint
from qichi.domain.events import ConversationEvent, MessageSegment


UTC = timezone.utc
BASE = datetime(2026, 9, 13, 1, 0, tzinfo=UTC)


def event(sequence: int, actor: str, text: str) -> ConversationEvent:
    occurred = BASE + timedelta(minutes=sequence)
    return ConversationEvent(
        event_id=f"e-{sequence}-{actor}",
        platform_event_id=None,
        platform_message_id=f"pm-{sequence}",
        conversation_id="owner-1",
        sequence=sequence,
        direction="inbound" if actor == "mumo" else "outbound",
        actor=actor,
        kind="text",
        text=text,
        message_segments=(MessageSegment("text", {"text": text}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=occurred,
        received_at_utc=occurred,
        status="sent",
        metadata={},
    )


def test_a_phrase_she_repeated_in_three_messages_is_reported():
    events = (
        event(1, "qichi", "说好一起早睡，别光我一个人"),
        event(2, "mumo", "好"),
        event(3, "qichi", "记住，一起早睡，谁也别偷懒"),
        event(4, "qichi", "一起早睡这句今晚作数"),
    )

    hint = _recent_repeat_hint(events)

    assert hint is not None
    assert "一起早睡" in hint
    assert "×3" in hint


def test_two_occurrences_of_a_short_phrase_stay_silent():
    """两次仍属正常措辞：门槛统一是 3 条不同消息（2026-09-15 起没有长片段例外）。"""

    events = (
        event(1, "qichi", "说好一起早睡"),
        event(3, "qichi", "一起早睡，别光我一个"),
    )

    assert _recent_repeat_hint(events) is None


def test_a_long_fragment_needs_three_messages_too():
    """2026-09-15：长片段不再「两次就报」——那条例外正是日常段被刷的原因。

    离线扫描（同一天三段样本）：去掉例外并把下限降到 4 字后，
    命中模板的轮数 7%→11%，噪声 14%→10%、特定时段 17%→0%。
    """

    twice = (
        event(1, "qichi", "这话你已经说过一遍了，我都记住了"),
        event(2, "mumo", "唔"),
        event(3, "qichi", "这话你已经说过一遍了，我记着呢"),
    )
    thrice = twice + (event(4, "qichi", "这话你已经说过一遍了——我数着呢"),)

    assert _recent_repeat_hint(twice) is None
    hint = _recent_repeat_hint(thrice)
    assert hint is not None
    assert "这话你已经说过一遍了" in hint
    assert "×3" in hint


def test_a_four_character_skeleton_is_now_counted():
    """命中：4 字骨架（旧口径 ≥5 字，进不了候选）现在会被报出来。

    真机那一场里「早点休息」20 次、「跟着你的作息走」15 次，提示几乎没提过它们。
    """

    events = (
        event(1, "qichi", "你别不服，早点休息这句话我说过"),
        event(2, "mumo", "嗯"),
        event(3, "qichi", "别停，早点休息，我快说烦了"),
        event(4, "qichi", "你越这样，我越把早点休息挂嘴边"),
    )

    hint = _recent_repeat_hint(events)

    assert hint is not None
    assert "早点休息" in hint


def test_three_character_fillers_stay_out():
    """不误判：3 字口头填充（「我这边」这类）仍在候选之外，不然每轮都会响。"""

    events = (
        event(1, "qichi", "我这边没事，你放心去忙"),
        event(2, "mumo", "好"),
        event(3, "qichi", "我这边已经缓过来了，别惦记"),
        event(4, "qichi", "我这边就这样，你忙你的"),
    )

    hint = _recent_repeat_hint(events)

    assert hint is None or "我这边" not in hint


def test_two_unrelated_long_messages_stay_silent():
    """长消息之间没有共同片段时什么都不报。"""

    events = (
        event(1, "qichi", "今天风大，你别站在外面等我"),
        event(3, "qichi", "锅里炖了汤，你回来就能喝上"),
    )

    assert _recent_repeat_hint(events) is None


def test_kaomoji_repetition_is_not_a_phrase():
    """颜文字只是语气，不是「说法」：真机复检里它曾每轮都被报出来。"""

    events = (
        event(1, "qichi", "又来这一套，早说过了(￣▽￣)"),
        event(2, "qichi", "第三下了，凑齐。说吧(￣▽￣)"),
        event(3, "qichi", "你倒挺会挑时候(￣▽￣)"),
        event(4, "qichi", "我不念了，道理你懂就行(｡･ω･｡)"),
        event(5, "qichi", "这话你说得实在，记着这份情了(｡･ω･｡)"),
    )

    assert _recent_repeat_hint(events) is None


def test_a_repeat_is_reported_only_on_the_message_that_repeats_it():
    """只报刚发生的那一次：她换了说法以后，旧的重复不再挂着。"""

    repeated = (
        event(1, "qichi", "说好一起早睡，谁也别偷懒"),
        event(2, "mumo", "好"),
        event(3, "qichi", "一起早睡，谁也别偷懒，这话我说过"),
        event(4, "qichi", "一起早睡，谁也别偷懒——我数着呢"),
    )

    assert _recent_repeat_hint(repeated) is not None
    assert _recent_repeat_hint(repeated + (event(5, "qichi", "我去泡杯茶，你忙你的"),)) is None


def test_only_her_own_words_are_counted():
    """用户反复说的话不算她复读。"""

    events = tuple(event(index, "mumo", "一起早睡就对了") for index in range(1, 5))

    assert _recent_repeat_hint(events) is None


def test_too_little_history_stays_silent():
    assert _recent_repeat_hint(()) is None
    assert _recent_repeat_hint((event(1, "qichi", "在呢"),)) is None


def test_the_longest_phrases_win_and_substrings_are_not_repeated_back():
    events = (
        event(1, "qichi", "这事今晚就算定了，你别绕"),
        event(2, "qichi", "这事今晚就算定了，我再说一次"),
        event(3, "qichi", "这事今晚就算定了"),
    )

    hint = _recent_repeat_hint(events)

    assert hint is not None
    assert "这事今晚就算定了" in hint
    # 子串不该在同一个提示里重复列出
    assert hint.count("这事今晚就算定了") == 1


def test_the_hint_states_a_fact_and_does_not_issue_a_word_list():
    events = tuple(event(index, "qichi", "接着往下说，别停在那儿") for index in range(1, 5))

    hint = _recent_repeat_hint(events)

    assert hint is not None
    assert "来自角色自己已发送的原文" in hint
    assert "不是当前要做的事" in hint
