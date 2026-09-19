"""2026-09-18 呈现层：常驻三条「她自己记着的事」的选取判据。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_g1_application import NOW, OWNER, FakeLLM, FakeNapCat, app, seed_event  # noqa: E402


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


def test_the_three_newest_ordinary_episodes_are_pinned(tmp_path):
    """命中：取最近三条、新的在前；非 episode 一律不进。"""

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

        assert [record.memory_id for record in pinned] == ["ep-3", "ep-2", "ep-1"], "最近三条，新的在前"
        assert all(record.type == "episode" for record in pinned), "偏好不进常驻层"
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

        assert [record.memory_id for record in pinned] == ["ordinary-old"]
        assert not {"intimate-new", "adult-new"} & {record.memory_id for record in pinned}
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