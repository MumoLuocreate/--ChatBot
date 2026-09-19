"""版式要让供应商的前缀缓存命中：稳定块在前，易变块在最后。

2026-09-12 实测（副本重建相邻轮 prompt 后发真机读 usage）：命中只有 640~768 tokens，
因为 prompt 第二段第二行就是带秒的「当前时间」，第一处不同字符落在 token ~815。
缓存只认从头开始完全一致的前缀单元，所以任何易变内容只要排在稳定内容前面，就会
把它后面的一切（尤其是历史原文，占输入 token 的 61.6%）一起作废。

这组测试钉住版式：稳定事实不带钟；本轮事实和检索块排在历史原文之后。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qichi.dialogue.capability_manifest import (
    MessageSourceFact,
    RuntimeFacts,
    render_fact_envelope,
    render_stable_facts,
    render_turn_facts,
)
from qichi.dialogue.context_builder import ContextBuildRequest, ContextBuilder
from qichi.dialogue.model_capability import (
    ModelCapability,
    ProviderCapabilityEvidence,
)
from qichi.domain.events import ConversationEvent, MessageSegment


UTC = timezone.utc
BASE = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)


class UnitTokenCounter:
    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError
        return 0 if not text else 1


def capability(tokens: int) -> ModelCapability:
    return ModelCapability(
        "test/model",
        tokens,
        provider_evidence=ProviderCapabilityEvidence("SiliconFlow", "test/model", tokens, "evidence", BASE),
    )


def event(sequence: int, actor: str, text: str, *, at: datetime | None = None) -> ConversationEvent:
    direction = "inbound" if actor == "mumo" else "outbound"
    occurred = at or BASE + timedelta(minutes=sequence)
    return ConversationEvent(
        event_id=f"event-{sequence}-{actor}",
        platform_event_id=None,
        platform_message_id=f"platform-{sequence}-{actor}",
        conversation_id="owner-1",
        sequence=sequence,
        direction=direction,
        actor=actor,
        kind="text",
        text=text,
        message_segments=(MessageSegment("text", {"text": text}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=occurred,
        received_at_utc=occurred,
        status="received",
        metadata={},
    )


def facts(current: ConversationEvent, *, now: datetime | None = None) -> RuntimeFacts:
    return RuntimeFacts(
        current_time=now or current.occurred_at_utc + timedelta(seconds=5),
        seconds_since_last_message=5,
        current_source=MessageSourceFact(current.actor, current.visible_handle, current.kind, current.occurred_at_utc),
        quoted_source=None,
        received_media=("text",),
        available_actions=("text", "reply", "qq_face"),
        available_qq_face_keys=("shy",),
        vision_available=False,
        external_tools_available=False,
    )


def request(**overrides) -> ContextBuildRequest:
    current = overrides.pop("current_event", event(20, "mumo", "current"))
    values = {
        "role_core": "role-core",
        "runtime_facts": overrides.pop("runtime_facts", facts(current)),
        "current_event": current,
        "quoted_chain": (),
        "recent_events": (),
        "relationship_state": (),
        "memory_candidates": (),
        "memory_working_set": (),
        "earlier_events": (),
        "evidence_events": {},
    }
    values.update(overrides)
    return ContextBuildRequest(**values)


def builder() -> ContextBuilder:
    return ContextBuilder(
        UnitTokenCounter(),
        capability(400),
        preferred_window_tokens=60,
        max_window_tokens=120,
        output_reserve_tokens=2,
        recent_history_budget_tokens=40,
    )


def contents(result) -> list[str]:
    return [message.content for message in result.messages]


def index_of(lines: list[str], marker: str) -> int:
    for position, line in enumerate(lines):
        if line.startswith(marker):
            return position
    raise AssertionError(f"没有找到 {marker}")


def test_stable_facts_carry_no_per_turn_value_and_the_clock_moves_to_the_tail():
    current = event(1, "mumo", "在吗")
    envelope = facts(current)

    stable = render_stable_facts(envelope)
    turn = render_turn_facts(envelope)

    for per_turn in ("当前时间", "距上一条可靠消息", "当前输入来源", "收到的内容类型",
                     "本轮可执行平台动作", "可选尾标协议", "引用解析状态"):
        assert per_turn not in stable, per_turn
    # 格式与 emoji 说明逐轮不变，必须留在稳定半：它们若紧贴当前输入，pro 会把它们当成
    # 最后一刻的「格式卡」，回复变短变平（2026-09-12 副本 15 轮实测）。
    for steady in ("未提供的现实细节", "事实未知时", "预设了他此前说过", "表达格式", "现实身体",
                   "外部工具结果", "正文必有", "Unicode emoji"):
        assert steady in stable, steady
    assert "正文必有" not in turn
    for per_turn in ("当前时间", "距上一条可靠消息", "当前输入来源", "收到的内容类型",
                     "本轮可执行平台动作", "可选尾标协议"):
        assert per_turn in turn, per_turn
    assert stable.startswith("[事实与能力]")
    assert turn.startswith("[本轮事实]")
    # 整封信封仍然是两半按顺序拼起来，别的地方读到的内容不变。
    assert render_fact_envelope(envelope) == stable + "\n" + turn


def test_turn_facts_ride_just_before_the_current_input():
    previous = event(1, "mumo", "上一句")
    reply = event(2, "qichi", "上一句的回答")
    current = event(3, "mumo", "这一句")
    result = builder().build(request(current_event=current, recent_events=(previous, reply)))
    lines = contents(result)

    assert "当前时间" not in lines[1]
    assert lines[1].startswith("[事实与能力]")
    turn_index = index_of(lines, "[本轮事实]")
    assert "当前时间" in lines[turn_index]
    # 本轮事实之后只能剩下当前输入本身（它可能是「[当前输入 …] 头 + 正文」两段）
    tail = lines[turn_index + 1:]
    assert tail, "本轮事实后面必须有当前输入"
    assert all(line.startswith("[当前输入") or line == current.text for line in tail), tail
    assert index_of(lines, "[历史对话证据") < turn_index


def test_retrieval_blocks_and_the_clock_follow_the_history():
    first = event(1, "mumo", "第一句")
    second = event(2, "qichi", "第二句")
    current = event(3, "mumo", "第三句")
    result = builder().build(
        request(
            current_event=current,
            recent_events=(first, second),
            memory_index=("2026-08-27 有一段",),
        )
    )
    lines = contents(result)

    history = index_of(lines, "[历史对话证据")
    assert history > index_of(lines, "[事实与能力]")
    assert index_of(lines, "[最近片段索引") > history
    assert index_of(lines, "[本轮事实]") > history


def test_two_consecutive_turns_share_a_byte_identical_prefix():
    first = event(1, "mumo", "第一句")
    reply = event(2, "qichi", "第二句")
    current = event(3, "mumo", "第三句")
    earlier_turn = builder().build(request(current_event=current, recent_events=(first, reply)))

    later = event(4, "mumo", "第四句")
    later_turn = builder().build(request(current_event=later, recent_events=(first, reply, current)))

    before = contents(earlier_turn)
    after = contents(later_turn)
    # 稳定段逐字一致：role_core + 稳定事实
    assert before[0] == after[0]
    assert before[1] == after[1]
    # 历史段是纯追加：上一轮的历史原文是这一轮历史原文的前缀
    before_history = before[index_of(before, "[历史对话证据")]
    after_history = after[index_of(after, "[历史对话证据")]
    assert after_history.startswith(before_history)
    assert len(after_history) > len(before_history)
