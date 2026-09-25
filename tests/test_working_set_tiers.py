"""卡② 2026-09-21：工作集分层——常驻层已有着落的不再重复渲染。

判据见 doc/优化批次-20260921-上下文成本与记忆边界.md 的卡② 一节：
  ① 命中：约定/纠正（已在常驻关系状态里带证据完整渲染）与 episode（最近三条由
     _remembered_episodes 钉住、其余靠话题检索）都不再出现在工作集里；偏好以
     「类型标记 + 逐字事实」的紧凑行出现。
  ② 不误判：不得因此丢掉任何一条事实（约定仍要到场、最近三条 episode 仍要在场）、
     不得改写或截断事实文字、其他类型不得被当成偏好。
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

import pytest

from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import (  # noqa: E402
    NOW,
    FakeLLM,
    FakeNapCat,
    app,
    raw,
    seed_event,
    seed_relationships,
)


def _episode(database, *, memory_id, fact, days_ago):
    """与 test_pinned_episodes 同一配方。"""

    source = seed_event(
        EventRepository(database), memory_id + "-source", fact + "（原话）",
        at=NOW - timedelta(days=days_ago),
    )
    MemoryRepository(database).create(MemoryRecord(
        memory_id, "episode", fact, "explicit_statement", "active", source.occurred_at_utc,
        None, None, source.received_at_utc,
        (MemoryEvidence(memory_id, source.event_id, source.actor, source.text,
                        source.occurred_at_utc, "source"),),
        "explicit", 2, "historical", "explicit_user_statement", source.received_at_utc,
        privacy_class="ordinary", recall_policy="daily_safe",
    ))


def _preference(database, *, memory_id, fact):
    source = seed_event(EventRepository(database), memory_id + "-source", fact + "（原话）", at=NOW)
    MemoryRepository(database).create(MemoryRecord(
        memory_id, "preference", fact, "explicit_statement", "active", source.occurred_at_utc,
        None, None, source.received_at_utc,
        (MemoryEvidence(memory_id, source.event_id, source.actor, source.text,
                        source.occurred_at_utc, "source"),),
        "explicit", 3, "ongoing", "explicit_user_statement", source.received_at_utc,
        privacy_class="ordinary", recall_policy="daily_safe",
    ))


async def _prompt(database, text="在忙吗"):
    llm = FakeLLM(["回复"])
    application = app(database, llm, FakeNapCat())
    await application.handle_onebot(raw(101, text), received_at_utc=NOW)
    assert llm.calls, "至少要有一次生成"
    return "\n".join(message.content for message in llm.calls[0])


@pytest.mark.asyncio
async def test_the_working_set_holds_only_what_is_not_rendered_elsewhere(tmp_path):
    """命中：工作集里不再出现 agreement/correction/episode 的行，只留偏好那一层。"""

    database = Database(tmp_path / "tiers.sqlite3")
    try:
        seed_relationships(database)
        _episode(database, memory_id="ep-a", fact="某天他忙到很晚", days_ago=1)

        prompt = await _prompt(database)

        assert "[关系记忆工作集" in prompt
        assert "- [preference] 用户喜欢雨声" in prompt
        assert "- [agreement]" not in prompt, "约定已在常驻关系状态完整渲染，不重复进工作集"
        assert "- [correction]" not in prompt
        # episode 仍然常驻（只换成紧凑行）——副本对照证明整段移出会让「我最近都在忙什么」塌掉。
        assert "- [episode] 某天他忙到很晚" in prompt
        # 注：self_expression 既不是约定/纠正也不是 episode，它按设计留在工作集里
        # （见 test_other_types_are_not_mislabelled_as_preferences）。生产库当前没有这个类型。
    finally:
        database.close()


@pytest.mark.asyncio
async def test_tiering_does_not_drop_the_agreement_or_the_newest_episodes(tmp_path):
    """不误判：分层不得把约定和最近三条 episode 一起弄丢。"""

    database = Database(tmp_path / "tiers-keep.sqlite3")
    try:
        seed_relationships(database)
        for index, days in enumerate((5, 3, 1)):
            _episode(database, memory_id="ep-%d" % index, fact="第%d件事" % index, days_ago=days)

        prompt = await _prompt(database)

        assert "周末一起聊书" in prompt, "约定仍要在场"
        assert "第2件事" in prompt and "第1件事" in prompt and "第0件事" in prompt
    finally:
        database.close()


@pytest.mark.asyncio
async def test_every_episode_stays_resident_after_compaction(tmp_path):
    """命中：episode 只被压缩、不被移出——最老那条也必须还在（副本回归教训）。"""

    database = Database(tmp_path / "tiers-old.sqlite3")
    try:
        seed_relationships(database)
        for index, days in enumerate((9, 5, 3, 1)):
            _episode(database, memory_id="ep-%d" % index, fact="第%d件事" % index, days_ago=days)

        prompt = await _prompt(database)

        for index in range(4):
            assert ("- [episode] 第%d件事" % index) in prompt, "每一条 episode 都必须还在"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_the_compact_line_keeps_the_fact_verbatim(tmp_path):
    """不误判：紧凑行只去掉台账字段，事实必须逐字、完整（代码不生成新句子）。"""

    database = Database(tmp_path / "tiers-verbatim.sqlite3")
    try:
        fact = "用户不喜欢别人替他决定，但允许她把想要的讲出来"
        _preference(database, memory_id="pref-v", fact=fact)

        prompt = await _prompt(database)

        assert ("- [preference] " + fact) in prompt, "事实必须逐字、完整"
        assert "fact=" not in prompt
        assert "importance=" not in prompt and "certainty=" not in prompt
    finally:
        database.close()


@pytest.mark.asyncio
async def test_other_types_are_not_mislabelled_as_preferences(tmp_path):
    """不误判：工作集里出现别的类型时，标记必须是它自己的类型，不能冒充偏好。"""

    database = Database(tmp_path / "tiers-other.sqlite3")
    try:
        seed_relationships(database)

        prompt = await _prompt(database)

        assert "- [self_expression] 角色表达过在意用户" in prompt
        assert "- [preference] 角色表达过在意用户" not in prompt
    finally:
        database.close()
