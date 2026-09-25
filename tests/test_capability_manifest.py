from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from qichi.dialogue.capability_manifest import (
    MessageSourceFact,
    RuntimeFacts,
    build_capability_manifest,
    render_fact_envelope,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 27, 10, 30, 45, tzinfo=UTC)
CURRENT_AT = datetime(2026, 8, 27, 10, 30, 40, tzinfo=UTC)
QUOTED_AT = datetime(2026, 8, 26, 23, 5, tzinfo=UTC)


def test_envelope_states_whether_an_image_is_actually_visible():
    """信封必须说清「看到的是图还是只有一个图片段」——这是 2026-08-16 那次幻觉的护栏。"""

    without = render_fact_envelope(facts(vision_available=False))
    with_vision = render_fact_envelope(facts(vision_available=True, images_attached=2))
    short = render_fact_envelope(
        facts(vision_available=True, images_attached=2, images_unexpanded=1, images_unavailable=1)
    )

    assert "image 仅表示收到图片段，不含视觉结果" in without
    assert "已随本轮提供 2 张" in with_vision
    assert "不得推断图外" in with_vision
    assert "看不清或没看到就说没看清，不得补全成合理画面" in with_vision
    assert "另有 1 张超出本轮上限未展开，不得描述其内容" in short
    assert "另有 1 张本轮未取到，不得猜测内容" in short


def test_the_envelope_remembers_that_the_previous_message_was_a_picture():
    rendered = render_fact_envelope(facts(previous_image_message=True))
    plain = render_fact_envelope(facts())

    assert "上一条输入是图片消息" in rendered
    assert "不要否认自己看见过" in rendered
    assert "上一条输入是图片消息" not in plain


def test_a_carried_picture_names_the_earlier_message_and_never_claims_it_is_new():
    """2026-09-16 真机：宽限重挂的图没有来由，她读成「他又发了一张」。"""

    source = MessageSourceFact(
        actor="mumo", handle="M9", kind="text", occurred_at_utc=QUOTED_AT
    )
    carried = render_fact_envelope(
        facts(
            vision_available=True,
            images_attached=1,
            images_carried=1,
            carried_image_source=source,
        )
    )
    plain = render_fact_envelope(facts(vision_available=True, images_attached=1))

    assert "存档重挂、不是他新发的图" in carried
    assert "handle=M9" in carried
    assert "存档重挂" not in plain


def test_carried_pictures_must_name_exactly_one_source_and_stay_within_the_attached_ones():
    source = MessageSourceFact(
        actor="mumo", handle="M9", kind="text", occurred_at_utc=QUOTED_AT
    )

    with pytest.raises(ValueError, match="exactly one source event"):
        facts(images_carried=1)
    with pytest.raises(ValueError, match="exactly one source event"):
        facts(carried_image_source=source)
    with pytest.raises(ValueError, match="among the attached images"):
        facts(
            vision_available=True,
            images_attached=1,
            images_carried=2,
            carried_image_source=source,
        )
    with pytest.raises(ValueError):
        facts(images_carried=-1)


def test_claiming_vision_without_a_picture_is_refused():
    """信封不许在没有图的情况下说「已随本轮提供」。"""

    with pytest.raises(ValueError, match="at least one attached image"):
        facts(vision_available=True)
    with pytest.raises(ValueError):
        facts(images_attached=-1)


def facts(**overrides) -> RuntimeFacts:
    values = {
        "current_time": NOW,
        "seconds_since_last_message": 5,
        "current_source": MessageSourceFact("mumo", "M12", "text", CURRENT_AT),
        "quoted_source": MessageSourceFact("qichi", "Q7", "text", QUOTED_AT),
        "received_media": ("text", "image"),
        "available_actions": ("text", "reply", "qq_face"),
        "vision_available": False,
        "external_tools_available": False,
        "available_qq_face_keys": ("shy",),
    }
    values.update(overrides)
    return RuntimeFacts(**values)


def test_manifest_contains_only_runtime_facts_and_is_immutable():
    manifest = build_capability_manifest(facts())
    values = manifest.capabilities

    assert values["timezone"] == "Asia/Shanghai"
    assert values["current_time_local"] == "2026-08-27T18:30:45+08:00"
    assert values["seconds_since_last_message"] == 5
    assert values["current_source"] == {
        "actor": "mumo",
        "handle": "M12",
        "kind": "text",
        "occurred_at_local": "2026-08-27T18:30:40+08:00",
    }
    assert values["quoted_source"] == {
        "actor": "qichi",
        "handle": "Q7",
        "kind": "text",
        "occurred_at_local": "2026-08-27T07:05:00+08:00",
    }
    assert values["quote_resolution_status"] == "resolved"
    assert values["received_media"] == ("text", "image")
    assert values["available_actions"] == ("text", "reply", "qq_face")
    assert values["vision_available"] is False
    assert values["external_tools_available"] is False
    assert values["physical_body_available"] is False
    assert values["available_qq_face_keys"] == ("shy",)
    assert values["available_reaction_keys"] == ()
    with pytest.raises(TypeError):
        values["vision_available"] = True


def test_envelope_is_factual_and_explicit_about_unavailable_capabilities():
    rendered = render_fact_envelope(facts())

    assert rendered.startswith("[事实与能力]\n")
    assert "当前时间: 2026-08-27T18:30:45+08:00 (Asia/Shanghai)" in rendered
    assert "当前输入来源: actor=mumo; handle=M12; kind=text; time=2026-08-27T18:30:40+08:00" in rendered
    assert "引用目标来源: actor=qichi; handle=Q7; kind=text; time=2026-08-27T07:05:00+08:00" in rendered
    assert "收到的内容类型: text, image (image 仅表示收到图片段，不含视觉结果)" in rendered
    assert "本轮可执行平台动作: text, reply, qq_face" in rendered
    assert "Unicode emoji 可直接写入正文" in rendered
    assert "QQ 引用格式 [[qq:reply:<M/Q handle>]]；用户本轮使用原生引用时默认保留同一目标" in rendered
    assert "QQ face 格式 [[qq:face:<key>]]，keys=shy" in rendered
    assert "[[qq:face:<key>]]" in rendered
    assert "[[qq:react:<key>]]" not in rendered
    assert "本轮带图: 否" in rendered
    assert "外部工具结果: 不可用" in rendered
    assert "现实身体: 不可用" in rendered
    assert "不要为了显得自然、亲密或有画面" in rendered
    assert "提问、比喻或玩笑中预设未知细节已经发生" in rendered
    assert "明确标成假设或共同想象" in rendered
    for persona_instruction in ("成人亲密互动", "直白或露骨表达", "SM 倾向"):
        assert persona_instruction not in rendered
    for semantic_instruction in ("情绪分类", "回复模板", "必须温柔", "安慰", "拌嘴"):
        assert semantic_instruction not in rendered


def test_absent_quote_and_unknown_interval_are_explicit():
    rendered = render_fact_envelope(facts(quoted_source=None, seconds_since_last_message=None))

    assert "距上一条可靠消息: unknown" in rendered
    assert "引用目标来源: none" in rendered
    assert "引用解析状态: none" in rendered


def test_unavailable_quote_is_explicit_without_fabricating_target_content():
    rendered = render_fact_envelope(
        facts(quoted_source=None, quote_resolution_status="unavailable")
    )

    assert "引用解析状态: unavailable" in rendered
    assert "原文暂时不可用" in rendered
    assert "引用目标来源: none" in rendered


def test_reply_protocol_is_not_advertised_when_reply_action_is_unavailable():
    rendered = render_fact_envelope(
        facts(
            available_actions=("text",),
            quoted_source=None,
            available_qq_face_keys=(),
        )
    )

    assert "Unicode emoji 可直接写入正文" in rendered
    assert "QQ 引用格式 [[qq:reply:<M/Q handle>]]" not in rendered


def test_platform_event_has_no_fake_visible_handle():
    source = MessageSourceFact("platform", None, "poke", CURRENT_AT)
    rendered = render_fact_envelope(
        facts(current_source=source, quoted_source=None, received_media=("poke",), available_actions=("text", "poke"), available_qq_face_keys=())
    )

    assert "actor=platform; handle=none; kind=poke" in rendered
    assert "收到的内容类型: poke" in rendered


def test_initiative_skip_is_an_invisible_protocol_choice_not_a_visible_refusal():
    rendered = render_fact_envelope(
        facts(
            initiative_attempt=True,
            current_source=MessageSourceFact("platform", None, "initiative", CURRENT_AT),
            quoted_source=None,
            received_media=(),
            available_actions=("text",),
            available_qq_face_keys=(),
        )
    )

    assert "只返回 [[qichi:skip]]" in rendered
    assert "不要把不发或没话说的决定写成可见消息" in rendered


@pytest.mark.parametrize(
    "source",
    [
        MessageSourceFact("mumo", "M0", "text", CURRENT_AT),
        MessageSourceFact("qichi", "Q0", "text", CURRENT_AT),
        MessageSourceFact("mumo", "M1", "text", CURRENT_AT),
        MessageSourceFact("qichi", "Q2", "text", CURRENT_AT),
        MessageSourceFact("platform", None, "poke", CURRENT_AT),
    ],
)
def test_valid_source_shapes_are_frozen(source: MessageSourceFact):
    with pytest.raises(FrozenInstanceError):
        source.kind = "changed"


@pytest.mark.parametrize(
    "args",
    [
        ("unknown", "M1", "text", CURRENT_AT),
        ("mumo", "Q1", "text", CURRENT_AT),
        ("qichi", None, "text", CURRENT_AT),
        ("platform", "M1", "poke", CURRENT_AT),
        ("mumo", "M00", "text", CURRENT_AT),
        ("qichi", "Q01", "text", CURRENT_AT),
        ("mumo", "M1\ninjected", "text", CURRENT_AT),
        ("mumo", "M1", "text\ninjected", CURRENT_AT),
        ("mumo", "M1", "text", datetime(2026, 8, 27, 10, 0)),
    ],
)
def test_invalid_or_injectable_sources_are_rejected(args):
    with pytest.raises((TypeError, ValueError)):
        MessageSourceFact(*args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_time", datetime(2026, 8, 27, 10, 0)),
        ("seconds_since_last_message", -1),
        ("seconds_since_last_message", True),
        ("received_media", ("text", "emotion")),
        ("received_media", ("text", "text")),
        ("available_actions", ("text", "browse_web")),
        ("available_actions", ("text", "text")),
        ("vision_available", 1),
        ("external_tools_available", "no"),
    ],
)
def test_runtime_facts_fail_closed_on_invalid_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        facts(**{field: value})


def test_expression_keys_are_exposed_in_sorted_order_and_repeated_keys_fail():
    value = facts(
        available_actions=("text", "reply", "qq_face", "reaction"),
        available_qq_face_keys=("smile", "shy"),
        available_reaction_keys=("heart",),
    )
    manifest = build_capability_manifest(value).capabilities
    assert manifest["available_qq_face_keys"] == ("smile", "shy")
    assert manifest["available_reaction_keys"] == ("heart",)
    rendered = render_fact_envelope(value)
    assert "QQ face 格式 [[qq:face:<key>]]，keys=smile, shy" in rendered
    assert "消息回应格式 [[qq:react:<key>]]，keys=heart" in rendered
    with pytest.raises(ValueError):
        facts(available_qq_face_keys=("shy", "shy"))


@pytest.mark.parametrize("actions, face_keys, reaction_keys", [
    (("text", "reply", "qq_face"), (), ()),
    (("text", "reply"), ("shy",), ()),
    (("text", "reply", "reaction"), (), ()),
    (("text", "reply"), (), ("heart",)),
])
def test_expression_action_and_key_sets_must_match(actions, face_keys, reaction_keys):
    with pytest.raises(ValueError):
        facts(
            available_actions=actions,
            available_qq_face_keys=face_keys,
            available_reaction_keys=reaction_keys,
        )


def test_expression_action_and_key_sets_can_be_enabled_together():
    value = facts(
        available_actions=("text", "reply", "qq_face", "reaction"),
        available_qq_face_keys=("shy",),
        available_reaction_keys=("heart",),
    )
    assert value.available_qq_face_keys == ("shy",)

# --- 语音能力声明（TTS P1-3，见 doc/TTS-实施计划-20260914.md §3.4）---


def test_voice_is_declared_only_when_the_channel_is_actually_ready():
    """能力边界：链路没就绪就不许声明语音，否则她会说出做不到的事。"""

    off = render_fact_envelope(facts())
    on = render_fact_envelope(
        facts(available_actions=("text", "reply", "qq_face", "voice"),
              voice_available=True, voice_max_chars=120)
    )

    assert "[[qq:voice:" not in off
    assert "[[qq:voice:" in on
    assert "不再重复发文字" in on
    assert "颜文字会单独作为一条文字跟在后面" in on
    # 2026-09-14 深夜：这条事实必须**单独成行**，不能夹在「可选尾标协议」长句里 ——
    # pro 在 thinking=disabled 下实测 0/3 会去用夹在里面的那版（带思考时 3/3）。
    assert any(line.startswith("本轮可以用语音说话") for line in on.splitlines())
    assert build_capability_manifest(facts()).capabilities["voice_available"] is False
    manifest = build_capability_manifest(
        facts(available_actions=("text", "reply", "qq_face", "voice"),
              voice_available=True, voice_max_chars=120)
    )
    assert manifest.capabilities["voice_available"] is True


def test_voice_action_and_availability_must_agree():
    with pytest.raises(ValueError, match="voice"):
        facts(available_actions=("text", "voice"), available_qq_face_keys=(), voice_available=False)
    with pytest.raises(ValueError, match="voice"):
        facts(voice_available=True)
    with pytest.raises(TypeError, match="voice_available"):
        facts(voice_available=1)
    # 声明可用就必须同时给出边界（2026-09-15 §2.42）：不许出现"能说语音但不知道能说多长"。
    with pytest.raises(ValueError, match="voice_max_chars"):
        facts(available_actions=("text", "voice"), available_qq_face_keys=(), voice_available=True)
    with pytest.raises(ValueError, match="voice_max_chars"):
        facts(voice_max_chars=120)
    with pytest.raises(TypeError, match="voice_max_chars"):
        facts(available_actions=("text", "voice"), available_qq_face_keys=(),
              voice_available=True, voice_max_chars=0)


def test_the_voice_line_states_how_long_one_segment_may_be():
    """2026-09-15 §2.42：她两次写了 127 字、超过上限就静默退回文字。

    边界必须作为**事实**写在能力行里（不是建议、不是"请写短一点"），而且数字跟着配置走。
    """

    on = render_fact_envelope(
        facts(available_actions=("text", "voice"), available_qq_face_keys=(),
              voice_available=True, voice_max_chars=90)
    )
    line = next(item for item in on.splitlines() if item.startswith("本轮可以用语音说话"))

    assert "最多 90 字" in line
    assert "超过这个长度就发不出去" in line and "会退回文字" in line
    assert "建议" not in line and "请写短" not in line, "边界是事实，不是写作建议"
    assert "120" not in line, "数字必须来自配置，不许写死"

    off = render_fact_envelope(facts())
    assert "超过这个长度就发不出去" not in off, "没有语音能力就不带这条边界"
    assert not any(item.startswith("本轮可以用语音说话") for item in off.splitlines())


def test_voice_off_keeps_the_expression_protocol_text_unchanged():
    """没开语音时，逐轮事实里不能出现任何语音相关的字。"""

    assert "语音" not in render_fact_envelope(facts())

