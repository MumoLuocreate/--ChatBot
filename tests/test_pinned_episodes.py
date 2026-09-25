"""常驻层的 episode 判据。

2026-09-18 引入「钉最新三条」；2026-09-22 用户看维护面板后裁定关闭
（app._PINNED_EPISODE_LIMIT = 0）：那三条是已经过去的时点事件，不该每轮挂在
「约定与纠正」那一块里。机制保留为开关，本文件的隐私边界断言继续有效。
"""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import (  # noqa: E402
    NOW, OWNER, FakeLLM, FakeNapCat, app, raw, seed_event,
)


def _episode(database, *, memory_id, fact, days_ago, privacy="ordinary", policy="daily_safe"):
    """造一条合法的 episode（含来源事件与证据），与 test_sensitive_recall_gate 同一配方。"""

    events, memories = EventRepository(database), MemoryRepository(database)
    source = seed_event(events, memory_id + "-source", fact + "（原话）", at=NOW - timedelta(days=days_ago))
    evidence = MemoryEvidence(memory_id, source.event_id, source.actor, source.text,
                             source.occurred_at_utc, "source")
    memories.create(MemoryRecord(
        memory_id, "episode", fact, "explicit_statement", "active", source.occurred_at_utc,
        None, None, source.received_at_utc, (evidence,), "explicit", 2, "historical",
        "explicit_user_statement", source.received_at_utc, privacy_class=privacy, recall_policy=policy,
    ))
    return source


def test_no_episode_is_pinned_after_the_2026_09_22_ruling(tmp_path):
    """命中（2026-09-22 用户裁定）：常驻层**不再钉任何 episode**。

    他在维护面板看到「后三条很明显不该在最近的对话里挂着」——那正是这里原先钉的
    「最新三条 ordinary+daily_safe episode」（9-20 看 CS 决赛、9-20 养活自己、9-21 兔子图），
    都是已经过去的时点事件，却每轮挂在标题写着「偏好与约定」的那块里。

    关闭后 episode 仍留在工作集里（紧凑行），所以「记下来了不会自己提」的顾虑由工作集承担；
    针对性检查（问「我最近都在忙些什么」）已确认她仍能自己说出 文档AI／团建／校区／接班。
    """

    database = Database(tmp_path / "pinned.sqlite3")
    try:
        for index, days in enumerate((9, 5, 3, 1)):
            _episode(database, memory_id="ep-%d" % index, fact="某件普通的事 %d" % index, days_ago=days)
        # 一条非 episode（偏好）不该进常驻层
        source = seed_event(EventRepository(database), "pref-source", "偏好原话", at=NOW - timedelta(days=0))
        MemoryRepository(database).create(MemoryRecord(
            "pref-1", "preference", "他喜欢这样", "explicit_statement", "active", source.occurred_at_utc,
            None, None, source.received_at_utc,
            (MemoryEvidence("pref-1", source.event_id, source.actor, source.text, source.occurred_at_utc, "source"),),
            "explicit", 3, "ongoing", "explicit_user_statement", source.received_at_utc,
            privacy_class="ordinary", recall_policy="daily_safe",
        ))
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        records = MemoryRepository(database).list_active(OWNER, NOW)

        pinned = application._remembered_episodes(tuple(records))

        assert pinned == (), "常驻层不许再钉 episode（偏好与近事都归工作集）"
    finally:
        database.close()


def test_sensitive_episodes_never_enter_the_pinned_layer(tmp_path):
    """隐私硬边界：亲密/成人 episode 再新也不进常驻层（延续 09-14 裁定）。"""

    database = Database(tmp_path / "pinned-sensitive.sqlite3")
    try:
        _episode(database, memory_id="ordinary-old", fact="普通旧事", days_ago=8)
        _episode(database, memory_id="intimate-new", fact="亲密新事", days_ago=0,
                 privacy="intimate", policy="topic_only")
        _episode(database, memory_id="adult-new", fact="成人新事", days_ago=0,
                 privacy="adult", policy="explicit_request_only")
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        records = MemoryRepository(database).list_active(OWNER, NOW)

        pinned = application._remembered_episodes(tuple(records))

        # 今天钉住机制已归零，所以常驻层本来就是空的；这条断言留着当**契约**：
        # 将来若有人把 _PINNED_EPISODE_LIMIT 改回正数，亲密/成人 episode 仍然一个都不许进。
        assert not {"intimate-new", "adult-new"} & {record.memory_id for record in pinned}
        assert pinned == (), "开关归零期间，常驻层不许有任何 episode"
    finally:
        database.close()


def test_without_ordinary_episodes_nothing_is_pinned(tmp_path):
    """不误判：库里没有普通 episode 时，常驻层为空（不塞别的东西顶替）。"""

    database = Database(tmp_path / "pinned-empty.sqlite3")
    try:
        _episode(database, memory_id="intimate-only", fact="只有亲密的事", days_ago=1,
                 privacy="intimate", policy="topic_only")
        application = app(database, FakeLLM(["回复"]), FakeNapCat())
        records = MemoryRepository(database).list_active(OWNER, NOW)

        assert application._remembered_episodes(tuple(records)) == ()
    finally:
        database.close()

@pytest.mark.asyncio
async def test_age_gate_retires_only_old_episodes_from_the_resident_layer(tmp_path):
    """卡B④ 的接缝测试：年龄门在真实一轮里的效果。

    命中：30 天前的 episode 不再进常驻工作集。
    不误判：2 天前的 episode 仍在场；**400 天前的偏好仍在场**（冻结不变式：
    偏好、约定、纠正是常驻关系状态，年龄门不许碰）。
    """

    database = Database(tmp_path / "working-set-age.sqlite3")
    try:
        _episode(database, memory_id="ep-old", fact="很久以前的一件小事", days_ago=30)
        _episode(database, memory_id="ep-young", fact="前两天的另一件小事", days_ago=2)
        pref_source = seed_event(
            EventRepository(database), "pref-old-source", "偏好原话",
            at=NOW - timedelta(days=400),
        )
        MemoryRepository(database).create(MemoryRecord(
            "pref-old", "preference", "他很早就说过的一件长久偏好", "explicit_statement",
            "active", pref_source.occurred_at_utc, None, None, pref_source.received_at_utc,
            (MemoryEvidence("pref-old", pref_source.event_id, pref_source.actor,
                            pref_source.text, pref_source.occurred_at_utc, "source"),),
            "explicit", 3, "ongoing", "explicit_user_statement", pref_source.received_at_utc,
            privacy_class="ordinary", recall_policy="daily_safe",
        ))
        llm = FakeLLM(["在的。"])
        await app(database, llm, FakeNapCat(), clock=lambda: NOW + timedelta(hours=1)).handle_onebot(
            raw(203, "在干嘛", NOW), received_at_utc=NOW,
        )
        trace = database.connection.execute(
            "SELECT details_json FROM turn_trace_events WHERE phase='context'"
        ).fetchone()
        ids = json.loads(trace[0])["working_set_memory_ids"]

        assert "ep-young" in ids, "两周内的经历仍该每轮在场"
        assert "pref-old" in ids, "偏好是常驻关系状态，年龄门不许退它"
        assert "ep-old" not in ids, "超过年龄门的 episode 不该再常驻"
    finally:
        database.close()
