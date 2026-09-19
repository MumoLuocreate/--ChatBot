"""宽限窗口状态机：新图到达后能挂几轮，什么时候收回，以及「挂的是哪条事件的图」。"""

from __future__ import annotations

from qichi.app import _advance_image_carry


def advance(state, *, event_id, fresh=(), carry_turns, conversation_id="c"):
    return _advance_image_carry(
        state,
        conversation_id=conversation_id,
        event_id=event_id,
        fresh=fresh,
        carry_turns=carry_turns,
    )


def test_a_fresh_picture_is_returned_and_opens_the_window():
    state: dict[str, tuple] = {}

    images, source = advance(state, event_id="e1", fresh=("img-a",), carry_turns=3)

    assert images == ("img-a",)
    # 新图轮的来源就是本轮，调用方手上已经有它——不需要额外回一个事件 id。
    assert source is None
    assert state["c"] == ("e1", ("img-a",), 3)


def test_the_window_carries_the_picture_and_names_its_source_event():
    state: dict[str, tuple] = {}
    advance(state, event_id="e1", fresh=("img-a",), carry_turns=3)

    carried = [
        advance(state, event_id=f"e{index}", carry_turns=3) for index in range(2, 6)
    ]

    # 三轮之内还挂着，之后收回（2026-09-14 真机：一轮太短，用户隔了六轮才说「你查一下」）。
    assert [images for images, _ in carried] == [("img-a",), ("img-a",), ("img-a",), ()]
    # 2026-09-16：重挂必须带回来源事件 id，那一轮要把它作为事实说出来。
    assert [source for _, source in carried] == ["e1", "e1", "e1", None]
    assert "c" not in state


def test_one_turn_is_still_exactly_one_turn():
    state: dict[str, tuple] = {}
    advance(state, event_id="e1", fresh=("img-a",), carry_turns=1)

    assert advance(state, event_id="e2", carry_turns=1) == (("img-a",), "e1")
    assert advance(state, event_id="e3", carry_turns=1) == ((), None)


def test_rebuilding_the_same_event_neither_carries_nor_consumes():
    """视觉档被拒后的文字回退会二次装配同一条事件——那时不能把图又带上。"""

    state: dict[str, tuple] = {}
    advance(state, event_id="e1", fresh=("img-a",), carry_turns=2)

    assert advance(state, event_id="e1", carry_turns=2) == ((), None)
    # 窗口没被吃掉：下一轮照挂。
    assert advance(state, event_id="e2", carry_turns=2) == (("img-a",), "e1")


def test_a_newer_picture_resets_the_window():
    state: dict[str, tuple] = {}
    advance(state, event_id="e1", fresh=("img-a",), carry_turns=5)
    advance(state, event_id="e2", carry_turns=5)

    images, source = advance(state, event_id="e3", fresh=("img-b",), carry_turns=5)

    assert images == ("img-b",) and source is None
    assert state["c"] == ("e3", ("img-b",), 5)


def test_windows_are_per_conversation():
    state: dict[str, tuple] = {}
    advance(state, event_id="e1", fresh=("img-a",), carry_turns=2, conversation_id="c1")

    assert advance(state, event_id="e9", carry_turns=2, conversation_id="c2") == ((), None)
    assert advance(state, event_id="e2", carry_turns=2, conversation_id="c1") == (("img-a",), "e1")
