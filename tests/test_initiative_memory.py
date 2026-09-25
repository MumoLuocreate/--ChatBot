"""卡⑤ 2026-09-21：主动开口进记忆——准入，以及它的结构边界。

用户裁定：「主动消息进记忆，但一般主动消息很少有有效信息，除非她找我本身
也是一件值得记录的事情。」这条裁定在代码里对应两半：

  ① 准入：MemoryWorker._is_reliable 认 initiative，所以整合窗口完整（不再出现
     「他的回复没有前因」的残片段），她的原话也能作为证据；
  ② 边界：记录所有权由取证角色矩阵保证——关于用户的记录必须以用户的话为证据
     （memory_repository._validate_active_evidence），关于她自己的记录走
     self_expression（只认 actor=qichi）。

判据：命中——主动文本可信；以她的主动话为证据的 self_expression 能建库、激活、
进工作集。不误判——形态不放宽（残段/未知态仍不可信）；关于他的事实不能由她的
话作证；场景结构兜底的门槛不放宽。
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.memory.extractor import MemoryExtractor
from qichi.memory.worker import MemoryWorker
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import NOW, OWNER, FakeLLM, FakeNapCat, app, raw  # noqa: E402

CONVERSATION = OWNER


def _outbound(sequence, text, *, source=None, kind="text", status="sent"):
    metadata = {}
    if source is not None:
        metadata["generation_metadata"] = {"source": source}
    at = NOW - timedelta(hours=1) + timedelta(seconds=sequence)
    return ConversationEvent(
        "ev-%d" % sequence, None, "pm-%d" % sequence, CONVERSATION, sequence,
        "outbound", "qichi", kind, text,
        (MessageSegment("text", {"text": text}),) if text else (),
        None, None, at, at, status, metadata,
    )


def _inbound(sequence, text):
    at = NOW - timedelta(hours=1) + timedelta(seconds=sequence)
    return ConversationEvent(
        "in-%d" % sequence, None, "pm-in-%d" % sequence, CONVERSATION, sequence,
        "inbound", "mumo", "text", text,
        (MessageSegment("text", {"text": text}),),
        None, None, at, at, "received", {},
    )


def _self_record(memory_id, fact, source_event):
    return MemoryRecord(
        memory_id, "self_expression", fact, "explicit_statement", "active",
        source_event.occurred_at_utc, None, None, source_event.received_at_utc,
        (MemoryEvidence(memory_id, source_event.event_id, "qichi", source_event.text,
                        source_event.occurred_at_utc, "source"),),
        "explicit", 2, "ongoing", "explicit_user_statement", source_event.received_at_utc,
        privacy_class="ordinary", recall_policy="daily_safe",
    )


def test_an_initiative_text_event_is_reliable():
    """命中：主动开口的文本事件与对话回复同判据。"""

    assert MemoryWorker._is_reliable(_outbound(1, "冒个头", source="initiative")) is True


@pytest.mark.parametrize("source", ["dialogue", "interaction"])
def test_the_original_reliable_sources_are_unchanged(source):
    """不误判：原有两种来源不受影响。"""

    assert MemoryWorker._is_reliable(_outbound(2, "在", source=source)) is True


@pytest.mark.parametrize(
    "event",
    [
        _outbound(3, "在", source=None),                      # 旧形态：没有 generation_metadata
        _outbound(4, "在", source="initiative", status="unknown"),  # 送达未知
        _outbound(5, "在", source="initiative", kind="image"),      # 非文本
    ],
)
def test_admission_widens_the_source_only(event):
    """不误判：准入只放宽来源，不放宽形态（残段/未知态/非文本仍不可信）。"""

    assert MemoryWorker._is_reliable(event) is False


def test_her_initiative_words_can_ground_her_own_record(tmp_path):
    """命中：以她的主动话为证据的 self_expression 能建库、激活、进常驻集合。"""

    database = Database(tmp_path / "initiative-self.sqlite3")
    try:
        events = EventRepository(database)
        source = events.insert(_outbound(10, "还有四天就中秋了，我这两天开始盼那个晚上了",
                                         source="initiative"))
        memories = MemoryRepository(database)
        record = memories.create(_self_record("self-init-1", "她说过开始盼中秋那天晚上", source))

        assert record.status == "active"
        active = memories.list_active(OWNER, NOW + timedelta(hours=1))
        assert [item.memory_id for item in active] == ["self-init-1"]
    finally:
        database.close()


def test_a_fact_about_him_cannot_be_grounded_in_her_own_words(tmp_path):
    """不误判：以她的主动话为唯一证据的 preference 必须被所有权矩阵拒绝。"""

    database = Database(tmp_path / "initiative-preference.sqlite3")
    try:
        events = EventRepository(database)
        source = events.insert(_outbound(11, "兔子下午有点想你", source="initiative"))
        bad = MemoryRecord(
            "pref-bad", "preference", "用户喜欢在下午被想念", "explicit_statement", "active",
            source.occurred_at_utc, None, None, source.received_at_utc,
            (MemoryEvidence("pref-bad", source.event_id, "qichi", source.text,
                            source.occurred_at_utc, "source"),),
            "explicit", 2, "ongoing", "explicit_user_statement", source.received_at_utc,
            privacy_class="ordinary", recall_policy="daily_safe",
        )
        # 候选路径（提取器走这条）与激活路径都按取证角色拒绝。
        with pytest.raises(ValueError, match="requires mumo evidence"):
            MemoryRepository._validate_candidate_ownership(bad)
        with pytest.raises(ValueError, match="requires mumo evidence"):
            MemoryRepository._validate_active_evidence(bad)
        # 端到端：建不进去，库里不会出现这条关于他的记录。
        with pytest.raises(ValueError):
            MemoryRepository(database).create(bad)
        assert MemoryRepository(database).list_active(OWNER, NOW + timedelta(hours=1)) == ()
    finally:
        database.close()


def test_the_scene_fallback_still_ignores_initiative():
    """不误判：场景结构兜底的「双边交换够长」门槛仍只认对话/互动来源。"""

    mumo = (_inbound(20, "第一句"), _inbound(21, "第二句"))
    her_initiative = (_outbound(22, "冒个头", source="initiative"),
                      _outbound(23, "再冒个头", source="initiative"))
    her_dialogue = (_outbound(24, "在呢", source="dialogue"),
                    _outbound(25, "嗯", source="dialogue"))

    only_initiative = mumo + her_initiative
    only_dialogue = mumo + her_dialogue

    assert MemoryExtractor._structural_sensitive_episode(
        only_initiative, frozenset(item.event_id for item in only_initiative), NOW
    ) is None, "主动话不能满足场景兜底的双边门槛"
    assert MemoryExtractor._structural_sensitive_episode(
        only_dialogue, frozenset(item.event_id for item in only_dialogue), NOW
    ) is not None, "对话来源仍必须能触发兜底"


@pytest.mark.asyncio
async def test_her_reaching_out_becomes_visible_in_the_resident_context(tmp_path):
    """命中（用户可见的结果）：她开口说的那句，最终出现在她每轮看到的关系背景里。"""

    database = Database(tmp_path / "initiative-context.sqlite3")
    try:
        events = EventRepository(database)
        source = events.insert(_outbound(30, "还有四天就中秋了，我这两天开始盼那个晚上了",
                                         source="initiative"))
        MemoryRepository(database).create(_self_record("self-init-2", "她说过开始盼中秋那天晚上", source))

        llm = FakeLLM(["回复"])
        application = app(database, llm, FakeNapCat())
        await application.handle_onebot(raw(101, "在忙吗"), received_at_utc=NOW)

        prompt = "\n".join(message.content for message in llm.calls[0])
        assert "- [self_expression] 她说过开始盼中秋那天晚上" in prompt
    finally:
        database.close()
