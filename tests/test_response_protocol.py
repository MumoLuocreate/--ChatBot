import logging

import pytest

from qichi.dialogue.response_protocol import ResponseProtocol, ResponseProtocolError
from qichi.domain.dialogue import DialogueResult, DialogueSkip


@pytest.fixture
def protocol():
    return ResponseProtocol(face_keys={"shy", "smile"}, reaction_keys={"heart"})


def parse(protocol, text, source="dialogue"):
    return protocol.parse(text, source=source, current_event_handle="M1760", context_version=7)


def test_plain_text_and_unicode_are_unchanged(protocol):
    result = parse(protocol, "好呀 🙂\n[普通方括号]")
    assert result == DialogueResult("好呀 🙂\n[普通方括号]", None, None, "primary", 7)

    trailing = parse(protocol, "保留换行\r\n")
    assert trailing.text == "保留换行\r\n"


@pytest.mark.parametrize(
    ("raw", "expected_parts"),
    [
        ("同一条第一行\n同一条第二行", ("同一条第一行\n同一条第二行",)),
        ("第一条\n\n第二条", ("第一条", "第二条")),
        ("第一条\r\n\r\n第二条", ("第一条", "第二条")),
        ("第一条\n \t\n第二条\n\n\n第三条", ("第一条", "第二条", "第三条")),
    ],
)
def test_blank_lines_are_explicit_qq_message_boundaries(protocol, raw, expected_parts):
    result = parse(protocol, raw)
    assert result.text == raw
    assert result.message_parts == expected_parts


def test_control_lines_are_removed_before_message_boundaries_are_derived(protocol):
    result = parse(protocol, "第一条\n\n[[qq:reply:M1]]\n\n第二条")
    assert result.text == "第一条\n\n第二条"
    assert result.message_parts == ("第一条", "第二条")
    assert result.reply_target == "M1"


@pytest.mark.parametrize("source", ["dialogue", "interaction"])
def test_skip_only_belongs_to_initiative(protocol, source):
    with pytest.raises(ResponseProtocolError):
        parse(protocol, "[[qichi:skip]]", source)


def test_initiative_skip(protocol):
    assert parse(protocol, "[[qichi:skip]]", "initiative") == DialogueSkip("primary", 7)


@pytest.mark.parametrize("framing", ["\n", "\r\n", "\r", "\n\n"])
def test_initiative_skip_allows_terminal_line_framing(protocol, framing):
    assert parse(protocol, "[[qichi:skip]]" + framing, "initiative") == DialogueSkip("primary", 7)


def test_initiative_skip_does_not_allow_body_whitespace(protocol):
    with pytest.raises(ResponseProtocolError):
        parse(protocol, "[[qichi:skip]] \n", "initiative")


def test_reply_face_and_reaction(protocol):
    reply = parse(protocol, "回你\n[[qq:reply:Q0]]")
    assert reply.reply_target == "Q0"
    face = parse(protocol, "害羞\n[[qq:face:shy]]")
    assert (face.expression_intent.kind, face.expression_intent.key) == ("face", "shy")
    reaction = parse(protocol, "收到\n[[qq:react:heart]]")
    assert reaction.expression_intent.target_event_handle == "M1760"


def test_reply_and_expression_are_independent(protocol):
    result = parse(protocol, "行\n[[qq:reply:M1760]]\n[[qq:face:shy]]")
    assert result.reply_target == "M1760"
    assert result.expression_intent.kind == "face"

    result = parse(protocol, "行\n[[qq:react:heart]]\n[[qq:reply:Q0]]")
    assert result.reply_target == "Q0"
    assert result.expression_intent.kind == "reaction"


@pytest.mark.parametrize(
    ("raw", "expected_text"),
    [
        ("[[qq:reply:M1]]\n行呀 🙂", "行呀 🙂"),
        ("第一行\n[[qq:reply:M1]]\n第二行 🙂", "第一行\n第二行 🙂"),
        ("第一行\r\n[[qq:face:shy]]\r\n第二行", "第一行\r\n第二行"),
    ],
)
def test_standalone_control_lines_do_not_depend_on_body_order(protocol, raw, expected_text):
    result = parse(protocol, raw)
    assert result.text == expected_text
    if "reply" in raw:
        assert result.reply_target == "M1"
    else:
        assert result.expression_intent is not None
        assert result.expression_intent.key == "shy"


@pytest.mark.parametrize(
    ("raw", "expected_text"),
    [
        (
            "第一段\n\n[[qq:reply:M1]]\n\n第二段",
            "第一段\n\n第二段",
        ),
        (
            "第一段\n\n[[qq:reply:M1]]\n第二段",
            "第一段\n\n第二段",
        ),
        (
            "第一段\n[[qq:reply:M1]]\n\n第二段",
            "第一段\n\n第二段",
        ),
        (
            "第一段\n\n\n[[qq:reply:M1]]\n\n第二段",
            "第一段\n\n\n第二段",
        ),
    ],
)
def test_control_boundary_whitespace_is_not_added_together(protocol, raw, expected_text):
    result = parse(protocol, raw)
    assert result.text == expected_text
    assert result.reply_target == "M1"


@pytest.mark.parametrize(
    ("raw", "expected_text"),
    [
        ("第一段\n\n\n第二段", "第一段\n\n\n第二段"),
        ("[[qq:reply:M1]]\n\n第一段", "第一段"),
        ("第一段\n\n[[qq:reply:M1]]", "第一段"),
        (
            "第一段\n\n[[qq:reply:M1]]\n\n[[qq:face:shy]]\n\n第二段",
            "第一段\n\n第二段",
        ),
        (
            "第一段\n[[qq:reply:M1]]\n[[qq:reply:M1]]\n第二段",
            "第一段\n第二段",
        ),
        (
            "第一段\r\n\r\n[[qq:reply:M1]]\r\n\r\n第二段",
            "第一段\r\n\r\n第二段",
        ),
        (
            "第一段\r\r[[qq:reply:M1]]\r\r第二段",
            "第一段\r\r第二段",
        ),
    ],
)
def test_control_boundary_cleanup_preserves_body_layout(protocol, raw, expected_text):
    result = parse(protocol, raw)
    assert result.text == expected_text


def test_identical_control_actions_are_idempotently_collapsed(protocol):
    result = parse(protocol, "行\n[[qq:reply:M1]]\n[[qq:reply:M1]]")
    assert result.text == "行"
    assert result.reply_target == "M1"

    result = parse(protocol, "收到\n[[qq:face:shy]]\n[[qq:face:shy]]")
    assert result.text == "收到"
    assert result.expression_intent is not None
    assert result.expression_intent.key == "shy"


def test_conflicting_duplicate_control_actions_still_fail(protocol):
    with pytest.raises(ResponseProtocolError, match="duplicate control action"):
        parse(protocol, "x\n[[qq:reply:M1]]\n[[qq:reply:Q2]]")

    with pytest.raises(ResponseProtocolError, match="duplicate control action"):
        parse(protocol, "x\n[[qq:face:shy]]\n[[qq:face:smile]]")


def test_controlled_body_preserves_original_line_endings(protocol):
    result = parse(protocol, "第一行\r\n第二行\r\n[[qq:face:shy]]\r\n")
    assert result.text == "第一行\r\n第二行"

    result = parse(protocol, "第一行\r第二行\r[[qq:reply:M0]]\n")
    assert result.text == "第一行\r第二行"


@pytest.mark.parametrize(
    "text",
    [
        "x\n[[qq:reply:M01]]",
        "x\n[[qq:face:unknown]]",
        "x\n[[qq:react:unknown]]",
    ],
)
def test_invalid_action_is_dropped_without_rewriting_body(protocol, text):
    result = parse(protocol, text)
    assert result.text == "x"
    assert result.reply_target is None
    assert result.expression_intent is None


def test_invalid_action_does_not_drop_other_valid_action(protocol):
    result = parse(protocol, "x\n[[qq:reply:not-a-handle]]\n[[qq:face:shy]]")
    assert result.text == "x"
    assert result.reply_target is None
    assert result.expression_intent.key == "shy"


@pytest.mark.parametrize(
    "text",
    [
        "x\n[[qq:reply:M1]]\n[[qq:reply:Q2]]",
        "x\n[[qq:face:shy]]\n[[qq:react:heart]]",
        "x\n[[qq:face:shy]]\n[[qq:face:smile]]",
        "x\n[[qq:reply:M1]]\n[[qq:face:shy]]\n[[qq:react:heart]]",
        "x\n[[qq:reply:]]",
        "x\n[[qq:face:shy]",
        "x [[qq:reply:M1]]",
        "x\nprefix [[qq:reply:M1]]\ny",
        "[[qichi:skip]]\ntext",
    ],
)
def test_invalid_structure_fails_and_never_leaks_controls(protocol, text):
    with pytest.raises(ResponseProtocolError):
        parse(protocol, text)


def test_empty_body_fails(protocol):
    with pytest.raises(ResponseProtocolError):
        parse(protocol, " \n[[qq:face:shy]]")


def test_constructor_and_inputs_are_strict(protocol):
    with pytest.raises((TypeError, ValueError)):
        ResponseProtocol(face_keys=["shy", 1], reaction_keys=[])
    with pytest.raises((TypeError, ValueError)):
        protocol.parse(1, source="dialogue", current_event_handle="M1", context_version=1)
    with pytest.raises((TypeError, ValueError)):
        protocol.parse("x", source="bad", current_event_handle="M1", context_version=1)
    with pytest.raises((TypeError, ValueError)):
        protocol.parse("x", source="dialogue", current_event_handle="M01", context_version=1)
    with pytest.raises((TypeError, ValueError)):
        protocol.parse("x", source="dialogue", current_event_handle="M1", context_version=True)

# --- 语音控制行 [[qq:voice:N]]（2026-09-14，见 doc/TTS-实施计划-20260914.md §3.3）---


def test_voice_marker_selects_a_part_and_never_leaks_into_the_body(protocol):
    result = parse(protocol, "第一条\n\n第二条\n\n[[qq:voice:2]]")

    assert result.text == "第一条\n\n第二条"
    assert result.message_parts == ("第一条", "第二条")
    assert result.voice_part_index == 2


@pytest.mark.parametrize("position", ["before", "middle", "after"])
def test_voice_marker_position_carries_no_meaning(protocol, position):
    bodies = {
        "before": "[[qq:voice:1]]\n\n第一条\n\n第二条",
        "middle": "第一条\n\n[[qq:voice:1]]\n\n第二条",
        "after": "第一条\n\n第二条\n\n[[qq:voice:1]]",
    }
    result = parse(protocol, bodies[position])

    assert result.text == "第一条\n\n第二条"
    assert result.message_parts == ("第一条", "第二条")
    assert result.voice_part_index == 1


def test_voice_can_share_a_turn_with_reply_and_face(protocol):
    result = parse(protocol, "就这一句\n[[qq:reply:M1]]\n[[qq:face:shy]]\n[[qq:voice:1]]")

    assert result.text == "就这一句"
    assert result.reply_target == "M1"
    assert result.expression_intent is not None and result.expression_intent.key == "shy"
    assert result.voice_part_index == 1


def test_repeated_identical_voice_marker_is_collapsed(protocol):
    result = parse(protocol, "就这一句\n[[qq:voice:1]]\n[[qq:voice:1]]")

    assert result.voice_part_index == 1


def test_out_of_range_voice_index_drops_the_action_but_keeps_the_body(protocol, caplog):
    with caplog.at_level(logging.WARNING):
        result = parse(protocol, "第一条\n\n第二条\n\n[[qq:voice:5]]")

    assert result.text == "第一条\n\n第二条"
    assert result.message_parts == ("第一条", "第二条")
    assert result.voice_part_index is None
    # 动作被丢弃时不能一声不吭：否则「她试了」与「她没选」在账本上分不出来。
    assert "voice_part_out_of_range" in caplog.text


@pytest.mark.parametrize("marker", ["[[qq:voice:abc]]", "[[qq:voice:0]]", "[[qq:voice:-1]]", "[[qq:voice:1.0]]"])
def test_malformed_voice_index_fails_closed(protocol, marker):
    with pytest.raises(ResponseProtocolError):
        parse(protocol, "正文\n" + marker)


def test_bare_voice_marker_is_not_a_valid_control_line(protocol):
    with pytest.raises(ResponseProtocolError):
        parse(protocol, "正文\n[[qq:voice]]")


def test_duplicate_voice_targets_are_rejected(protocol):
    with pytest.raises(ResponseProtocolError, match="duplicate"):
        parse(protocol, "第一条\n\n第二条\n[[qq:voice:1]]\n[[qq:voice:2]]")


def test_four_control_lines_are_rejected(protocol):
    with pytest.raises(ResponseProtocolError, match="too many"):
        parse(protocol, "正文\n[[qq:reply:M1]]\n[[qq:face:shy]]\n[[qq:voice:1]]\n[[qq:react:heart]]")


def test_plain_text_reply_has_no_voice_part(protocol):
    assert parse(protocol, "就是普通一句话").voice_part_index is None

