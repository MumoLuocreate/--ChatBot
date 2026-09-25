"""共同想象场景状态：协议标记、状态持久化、事实注入、端到端。

2026-09-20 的实验依据：在同一情景里多注入一行场景状态事实，长度中位 65→51、
场景内"我没有身体/抬不了头"式撇清 3→0、提出请求 18→22，且她不再替他定方向。
所以这次改动落在**状态与事实**上，不动角色核心、不往提示词堆规则。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import json

import pytest

from qichi.app import G0Application
from qichi.dialogue.context_builder import ContextBuilder
from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.dialogue.output_guard import OutputGuard

from qichi.dialogue.capability_manifest import (
    MessageSourceFact,
    RuntimeFacts,
    render_stable_facts,
    render_turn_facts,
)
from qichi.dialogue.response_protocol import ResponseProtocol, ResponseProtocolError
from qichi.domain.dialogue import DialogueResult, split_voice_part
from qichi.app import SCENE_STATE_TTL, _SceneMarks
from qichi.storage.database import Database

UTC = timezone.utc
NOW = datetime(2026, 8, 28, 2, 0, tzinfo=UTC)
SCENE_FACT = "[当前场景 | 共同想象进行中]"


def protocol(**overrides) -> ResponseProtocol:
    values = {"face_keys": ("shy",), "reaction_keys": ("like",)}
    values.update(overrides)
    return ResponseProtocol(**values)


def parse(text: str, **overrides):
    return protocol(**overrides).parse(
        text, source="dialogue", current_event_handle="M9", context_version=3
    )


# --- 协议 ---------------------------------------------------------------

def test_scene_on_and_off_are_parsed_and_stripped_from_the_body():
    on = parse("在呢。\n[[qq:scene:on]]")
    off = parse("好。\n[[qq:scene:off]]")

    assert on.scene_mark == "on"
    assert off.scene_mark == "off"
    assert "scene" not in on.text
    assert "scene" not in off.text
    assert on.text.strip() == "在呢。"
    assert off.text.strip() == "好。"


def test_a_turn_without_a_scene_mark_says_nothing_about_scenes():
    assert parse("在呢。").scene_mark is None


@pytest.mark.parametrize("marker", ["[[qq:scene]]", "[[qq:scene:maybe]]", "[[qq:scene:ON]]"])
def test_an_invalid_scene_mark_drops_the_action_and_keeps_the_body(marker):
    """非法形态不是协议错误：丢掉这个动作、正文照发（与语音越界同一风格）。"""

    result = parse(f"在呢。\n{marker}")

    assert result.scene_mark is None
    assert "在呢。" in result.text
    assert "qq:scene" not in result.text


def test_a_scene_mark_that_is_not_standalone_is_a_protocol_error():
    with pytest.raises(ResponseProtocolError, match="standalone"):
        parse("他说[[qq:scene:on]]")


def test_conflicting_scene_marks_are_an_ambiguity():
    with pytest.raises(ResponseProtocolError, match="duplicate control action"):
        parse("在呢。\n[[qq:scene:on]]\n[[qq:scene:off]]")


def test_a_repeated_identical_scene_mark_collapses():
    assert parse("在呢。\n[[qq:scene:on]]\n[[qq:scene:on]]").scene_mark == "on"


def test_a_scene_mark_does_not_spend_the_native_action_budget():
    """场景标记是模式状态，不是原生动作：三个原生动作 + 场景标记仍应合法。"""

    result = parse(
        "在呢。\n[[qq:reply:M9]]\n[[qq:face:shy]]\n[[qq:voice:1]]\n[[qq:scene:on]]"
    )

    assert result.scene_mark == "on"
    assert result.reply_target == "M9"
    assert result.expression_intent is not None


# --- 结果类型 -----------------------------------------------------------

def test_dialogue_result_round_trips_the_scene_mark():
    result = DialogueResult("在呢。", None, None, "primary", 1, ("在呢。",), None, "on")

    assert result.to_dict()["scene_mark"] == "on"
    assert DialogueResult.from_dict(result.to_dict()) == result
    assert DialogueResult("在呢。", None, None, "primary", 1).scene_mark is None


def test_a_scene_mark_must_be_on_or_off():
    with pytest.raises(ValueError, match="scene_mark"):
        DialogueResult("在呢。", None, None, "primary", 1, scene_mark="maybe")


def test_splitting_the_voice_part_keeps_this_turns_scene_mark():
    result = DialogueResult(
        "第一段\n\n第二段", None, None, "primary", 1, ("第一段", "第二段"), 2, "off"
    )

    remaining, spoken = split_voice_part(result)

    assert spoken == "第二段"
    assert remaining is not None and remaining.scene_mark == "off"


# --- 状态（进程内存，刻意不落库）----------------------------------------

def marks() -> _SceneMarks:
    return _SceneMarks()


def test_marking_on_then_off_switches_the_state():
    states = marks()
    assert states.is_active("owner-private", now=NOW) is False

    started = states.apply("owner-private", "on", now=NOW)
    assert started.active is True
    assert started.started_at_utc == NOW
    assert states.is_active("owner-private", now=NOW) is True

    stopped = states.apply("owner-private", "off", now=NOW + timedelta(minutes=30))
    assert stopped.active is False
    assert stopped.started_at_utc is None
    assert states.is_active("owner-private", now=NOW + timedelta(minutes=30)) is False


def test_the_state_lives_only_in_the_process_by_design():
    """**刻意不落库**：这是取舍，不是遗漏。

    schema 版本是单向的——迁移只向前，readiness 又要求库里的 schema 等于代码里的
    SCHEMA_VERSION。为一个几小时就过期的场景标记把 schema 抬到 7，会让"回退代码"
    变成"启动被拒"，等于拿整个回滚能力换一个临时状态。

    所以场景状态只活在进程内存里：同一进程内跨多轮有效，进程重建后回到"不在场景里"。
    因此这里**不断言跨进程持久化**——持久化正是我们不要的行为。
    """

    states = marks()
    states.apply("owner-private", "on", now=NOW)
    # 同一进程里跨时点仍然有效（不是一次性标记）
    assert states.is_active("owner-private", now=NOW + timedelta(minutes=30)) is True

    # 进程重建 = 新实例：状态为空，这是**预期行为**
    assert marks().is_active("owner-private", now=NOW) is False


def test_a_stale_mark_stops_counting_as_active():
    """兜底：她标记 on 之后一直没再标记，第二天不能还算在场景里。"""

    states = marks()
    states.apply("owner-private", "on", now=NOW)

    assert states.is_active("owner-private", now=NOW + SCENE_STATE_TTL) is True
    assert states.is_active("owner-private", now=NOW + SCENE_STATE_TTL + timedelta(seconds=1)) is False


def test_marking_on_twice_does_not_move_the_start():
    states = marks()
    states.apply("owner-private", "on", now=NOW)
    again = states.apply("owner-private", "on", now=NOW + timedelta(minutes=5))

    assert again.started_at_utc == NOW
    assert again.updated_at_utc == NOW + timedelta(minutes=5)


def test_the_state_is_per_conversation():
    states = marks()
    states.apply("owner-private", "on", now=NOW)

    assert states.is_active("owner-private", now=NOW) is True
    assert states.is_active("another-conversation", now=NOW) is False


@pytest.mark.parametrize("mark", ["maybe", "", None, "ON"])
def test_only_on_and_off_can_be_recorded(mark):
    with pytest.raises(ValueError, match="mark"):
        marks().apply("owner-private", mark, now=NOW)


# --- 事实注入 -----------------------------------------------------------

def facts(**overrides) -> RuntimeFacts:
    values = {
        "current_time": NOW,
        "seconds_since_last_message": 5,
        "current_source": MessageSourceFact("mumo", "M12", "text", NOW - timedelta(seconds=5)),
        "quoted_source": None,
        "received_media": ("text",),
        "available_actions": ("text", "reply"),
        "vision_available": False,
        "external_tools_available": False,
    }
    values.update(overrides)
    return RuntimeFacts(**values)


def test_the_scene_line_appears_only_while_the_state_is_active():
    active = render_turn_facts(facts(scene_active=True))
    inactive = render_turn_facts(facts(scene_active=False))

    assert SCENE_FACT in active
    assert "方向与姿势由用户定" in active
    assert "能推进的是她自己的状态" in active
    assert "她可以提出请求" in active
    # 能力目录那一行本来就一直写着「共同想象」这个机制名，只有状态事实行该消失。
    assert "共同想象进行中" not in inactive


def test_the_scene_line_never_lands_in_the_stable_half():
    """放稳定半等于每轮都把这一行塞进前缀缓存，也等于让状态变成常驻设定。"""

    assert SCENE_FACT not in render_stable_facts(facts(scene_active=True))
    assert SCENE_FACT not in render_stable_facts(facts(scene_active=False))


def test_the_scene_fact_does_not_claim_an_agreement_code_cannot_verify():
    """代码只能陈述她标记过这一件事，不能替她说「双方已明确同意」。"""

    assert "同意" not in render_turn_facts(facts(scene_active=True))


def test_scene_active_must_be_a_bool():
    with pytest.raises(TypeError, match="scene_active"):
        facts(scene_active="yes")


# --- 端到端 -------------------------------------------------------------

OWNER, BOT = "10001", "20001"


class Counter:
    def count_text(self, text):
        return len(text)


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    async def generate(self, messages, **kwargs):
        self.calls.append(tuple(messages))
        return LLMGeneration(self.outputs.pop(0), "primary", "fake-v4", 1, 1, 1.0)


class FakeNapCat:
    def __init__(self, start=700):
        # 两个 app 实例共用一个库时，出站平台消息 id 必须错开，否则会撞
        # platform_message_map 的唯一约束（同一个平台消息只能映射一次）。
        self.sent, self.next_id = [], start

    async def send_private_msg(self, user_id, message):
        self.sent.append((str(user_id), message))
        self.next_id += 1
        return {"message_id": self.next_id}


def raw(message_id, text, at=NOW):
    return {
        "post_type": "message", "message_type": "private", "sub_type": "friend",
        "self_id": int(BOT), "user_id": OWNER, "target_id": OWNER,
        "sender": {"user_id": OWNER}, "message_id": message_id, "time": at.timestamp(),
        "message": [{"type": "text", "data": {"text": text}}],
    }


def application(database, llm, napcat):
    evidence = ProviderCapabilityEvidence("fake", "fake-v4", 262_144, "local", NOW)
    capability = ModelCapability("fake-v4", 262_144, "fake", evidence)
    engine = DialogueEngine(
        llm,
        OutputGuard(Counter(), 2048),
        ResponseProtocol(face_keys=(), reaction_keys=()),
    )
    return G0Application(
        database,
        ContextBuilder(Counter(), capability, output_reserve_tokens=1024),
        engine,
        napcat,
        owner_qq=OWNER,
        bot_qq=BOT,
        role_core="你是角色。",
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_the_scene_fact_reaches_the_next_turn_and_stops_after_the_off_mark(tmp_path):
    """她标记 on → 下一轮上下文里有那一行；标记 off → 之后不再有。"""

    database = Database(tmp_path / "scene-e2e.sqlite3")
    llm = FakeLLM([
        "在呢。\n[[qq:scene:on]]",   # 第 1 轮：她开始
        "还在这儿。",                  # 第 2 轮：这一轮的上下文该带场景事实
        "好。\n[[qq:scene:off]]",     # 第 3 轮：她结束
        "嗯，晚安。",                  # 第 4 轮：这一轮不该再带
    ])
    napcat = FakeNapCat()
    app = application(database, llm, napcat)
    try:
        at = NOW
        for message_id, text in ((101, "开始吧"), (102, "在吗"), (103, "结束吧"), (104, "睡了吗")):
            await app.handle_onebot(raw(message_id, text, at), received_at_utc=at)
            at += timedelta(minutes=1)

        prompts = ["\n".join(message.content for message in call) for call in llm.calls]

        assert len(prompts) == 4
        # 标记只是状态，绝不能作为文字发出去。
        blobs = [json.dumps(entry[1], ensure_ascii=False) for entry in napcat.sent]
        assert all("qq:scene" not in blob for blob in blobs)
        assert "在呢。" in blobs[0]
        # 第 1 轮生成时状态还没落库，所以它自己那一轮不该带这一行。
        assert "共同想象进行中" not in prompts[0]
        # 第 2、3 轮在场景里。
        assert "共同想象进行中" in prompts[1]
        assert "共同想象进行中" in prompts[2]
        # 第 4 轮不该再有。
        assert "共同想象进行中" not in prompts[3]
    finally:
        database.close()


# --- 卡⑥ 2026-09-21：场景状态必须能从盘上复查 ---------------------------
# 判据见 doc/问题冻结-20260920-场景内推进与用词.md 第 10 节。
# 在这之前 [[qq:scene:on]] 只活在进程内存：标没标、有没有在闲聊里误标，
# 事后在事件元数据 / trace / outbox 三处都查不到，这个特性因此不可复查。

def _scene_turns(database):
    """按 inbound 顺序取每轮的 context / generation details。"""

    ids = [
        row["event_id"]
        for row in database.connection.execute(
            "SELECT event_id FROM conversation_events WHERE direction='inbound' ORDER BY sequence"
        )
    ]
    turns = []
    for event_id in ids:
        turn = {}
        for row in database.connection.execute(
            "SELECT phase, details_json FROM turn_trace_events WHERE trigger_event_id=?",
            (event_id,),
        ):
            turn[row["phase"]] = json.loads(row["details_json"])
        turns.append(turn)
    return turns


@pytest.mark.asyncio
async def test_the_scene_state_is_auditable_from_disk(tmp_path):
    """命中：每一轮都能回答「注入了没有」（context.scene_active）和「她标了什么」（generation.scene_mark）。"""

    database = Database(tmp_path / "scene-audit.sqlite3")
    llm = FakeLLM([
        "在呢。\n[[qq:scene:on]]",
        "还在这儿。",
        "好。\n[[qq:scene:off]]",
        "嗯，晚安。",
    ])
    app = application(database, llm, FakeNapCat())
    try:
        at = NOW
        for message_id, text in ((101, "开始吧"), (102, "在吗"), (103, "结束吧"), (104, "睡了吗")):
            await app.handle_onebot(raw(message_id, text, at), received_at_utc=at)
            at += timedelta(minutes=1)

        turns = _scene_turns(database)
        assert len(turns) == 4

        # 第 1 轮：标记在这一轮结束之后才生效，所以本轮事实里没有；她标了 on。
        assert turns[0]["context"]["scene_active"] is False
        assert turns[0]["generation"]["scene_mark"] == "on"
        # 第 2 轮：在场景里；她这一轮没标。
        assert turns[1]["context"]["scene_active"] is True
        assert turns[1]["generation"]["scene_mark"] is None
        # 第 3 轮：结束那一轮仍在场景里；她标了 off。
        assert turns[2]["context"]["scene_active"] is True
        assert turns[2]["generation"]["scene_mark"] == "off"
        # 第 4 轮：场景外。
        assert turns[3]["context"]["scene_active"] is False
        assert turns[3]["generation"]["scene_mark"] is None

        # 命中：字段每轮都在场——缺席和 False 分不开就没法复查。
        for turn in turns:
            assert isinstance(turn["context"]["scene_active"], bool)
            assert "scene_mark" in turn["generation"]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_the_scene_trace_stays_content_free(tmp_path):
    """不误判：新增的是代码事实（布尔 / on / off / None），不许把正文或场景事实原文写进 trace。"""

    database = Database(tmp_path / "scene-audit-clean.sqlite3")
    llm = FakeLLM(["在呢。\n[[qq:scene:on]]", "还在这儿。"])
    app = application(database, llm, FakeNapCat())
    try:
        at = NOW
        for message_id, text in ((101, "开始吧"), (102, "在吗")):
            await app.handle_onebot(raw(message_id, text, at), received_at_utc=at)
            at += timedelta(minutes=1)

        rows = list(database.connection.execute("SELECT phase, details_json FROM turn_trace_events"))
        blob = "\n".join(row["details_json"] for row in rows)

        assert "方向与姿势由用户定" not in blob, "场景事实原文不许进 trace"
        assert "还在这儿" not in blob and "开始吧" not in blob, "正文不许进 trace"
        marks = {
            json.loads(row["details_json"]).get("scene_mark")
            for row in rows
            if "scene_mark" in row["details_json"]
        }
        assert marks, "scene_mark 字段必须在场，否则这条不误判是空转"
        assert marks <= {None, "on", "off"}, "只允许协议里的三个取值"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_a_new_process_starts_outside_the_scene_by_design(tmp_path):
    """进程重建后不在场景里——这是我们选的取舍（状态不落库），端到端确认一次。"""

    database = Database(tmp_path / "scene-e2e-restart.sqlite3")
    try:
        first = application(database, FakeLLM(["在呢。\n[[qq:scene:on]]"]), FakeNapCat())
        await first.handle_onebot(raw(101, "开始吧"), received_at_utc=NOW)

        second_llm = FakeLLM(["还在这儿。"])
        second = application(database, second_llm, FakeNapCat(start=900))
        await second.handle_onebot(raw(102, "在吗"), received_at_utc=NOW + timedelta(minutes=1))

        prompt = "\n".join(message.content for message in second_llm.calls[0])
        assert "共同想象进行中" not in prompt
    finally:
        database.close()
