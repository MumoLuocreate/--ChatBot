"""卡B③：人工「恢复」——移出必须可逆，且语义要准确（faithful 路线）。

用户 2026-09-22 裁定 R1。判据见 doc/方案-20260922-写入侧作用范围与移出可逆.md 第 4 节。
要点：expire/reject 会把 certainty 降成 unsupported、importance 置 0、scope 置 unclassified，
**只翻状态会造出一条 active 却 unsupported 的记录**（违反激活矩阵，context_builder 会拒绝它
进工作集），所以恢复必须把移除前的评级一并还原。
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qichi.domain.memory import MemoryEvidence, MemoryReview  # noqa: E402
from test_memory_repository import NOW, event, memory, repositories  # noqa: E402


def test_schema_7_allows_restore_and_still_rejects_bogus_actions(repositories):
    """迁移：audit 的 action CHECK 加宽到含 restore，但没有被放宽成「什么都收」。"""

    database, _, _ = repositories
    c = database.connection
    assert c.execute(
        "SELECT value_json FROM runtime_meta WHERE key='schema_version'"
    ).fetchone()[0] == "7"
    c.execute(
        "INSERT INTO memory_records (memory_id,type,normalized_fact,modality,status,valid_from_utc,"
        "valid_until_utc,supersedes_id,created_at_utc,certainty,importance,temporal_scope,"
        "assessment_reason_code,assessed_at_utc,privacy_class,recall_policy) "
        "VALUES ('m-check','preference','x','explicit_statement','active',?,NULL,NULL,?,"
        "'explicit',1,'ongoing','explicit_user_statement',?,'ordinary','daily_safe')",
        (NOW.isoformat(), NOW.isoformat(), NOW.isoformat()),
    )
    c.execute(
        "INSERT INTO memory_audit_events (audit_event_id,memory_id,action,assessment_reason_code,"
        "occurred_at_utc) VALUES ('a-ok','m-check','restore','legacy_manual_review',?)",
        (NOW.isoformat(),),
    )
    with pytest.raises(Exception):
        c.execute(
            "INSERT INTO memory_audit_events (audit_event_id,memory_id,action,"
            "assessment_reason_code,occurred_at_utc) "
            "VALUES ('a-bogus','m-check','rollback','legacy_manual_review',?)",
            (NOW.isoformat(),),
        )
    with pytest.raises(Exception, match="append-only"):
        c.execute("UPDATE memory_audit_events SET action='create' WHERE audit_event_id='a-ok'")


def test_restore_brings_back_the_grading_that_was_in_force(repositories):
    """命中：走**真实 review 路径**移出（它会把评级降级），恢复必须拿回移除前的评级。

    注意 ``expire()`` 那条手动路径只翻状态、**不**降级（下面第二个测试锁住这点）；
    生产上的 12 次移出全是 review 路径，也就是会降级的那种，所以主用例走 review。
    """

    database, events, memories = repositories
    source = events.insert(event("e-restore-1", "我喜欢窗外细碎雨声", conversation_id="conversation-a"))
    memories.create(memory("m-restore-1", source, status="active", certainty="explicit",
                           importance=2, temporal_scope="ongoing"))
    later = events.insert(event(
        "e-restore-1-later", "这条不成立了",
        occurred_at_utc=NOW + timedelta(minutes=1),
        received_at_utc=NOW + timedelta(minutes=1),
    ))
    review = MemoryReview(
        "m-restore-1", "expire", "unsupported", 0, "unclassified", "expired_or_completed",
        (MemoryEvidence("m-restore-1", later.event_id, "mumo", "这条不成立了",
                        later.occurred_at_utc, "counterevidence"),),
    )
    with database.transaction() as connection:
        memories.apply_consolidation_in_transaction(
            connection, conversation_id="conversation-a", session_job_id="restore-job",
            candidates=(), reviews=(review,), assessed_at_utc=NOW + timedelta(minutes=2),
        )
    expired = memories.get("m-restore-1")
    assert (expired.status, expired.certainty, expired.importance) == ("expired", "unsupported", 0)
    assert expired.temporal_scope == "unclassified"
    evidence_before = len(expired.memory_evidence)

    restored = memories.restore("m-restore-1", at_utc=NOW + timedelta(minutes=3))

    assert restored.status == "active"
    assert restored.certainty == "explicit"
    assert restored.importance == 2
    assert restored.temporal_scope == "ongoing"
    # 本用例是直接用 status="active" 建的记录，项目把这种直接激活记为
    # legacy_manual_review；恢复忠实取回**该状态当初被建立时写下的原因码**。
    # 生产记录走 batch 激活，取回的是真实原因——已在生产副本上验证：
    # d026872a 恢复后 reason=bilateral_agreement、certainty=confirmed、importance=3、scope=bounded。
    assert restored.assessment_reason_code == "legacy_manual_review"
    assert len(restored.memory_evidence) == evidence_before, "恢复不许动证据"
    audit = database.connection.execute(
        "SELECT action, before_status, after_status, before_certainty, after_certainty, "
        "before_importance, after_importance, before_temporal_scope, after_temporal_scope "
        "FROM memory_audit_events WHERE memory_id='m-restore-1' "
        "ORDER BY occurred_at_utc DESC, audit_event_id DESC LIMIT 1"
    ).fetchone()
    assert tuple(audit) == (
        "restore", "expired", "active", "unsupported", "explicit", 0, 2, "unclassified", "ongoing",
    )
    # 时序必须可读：恢复的时刻晚于移出，审计链不许出现"先恢复后移出"。
    timeline = database.connection.execute(
        "SELECT action, occurred_at_utc FROM memory_audit_events WHERE memory_id='m-restore-1' "
        "ORDER BY occurred_at_utc, audit_event_id"
    ).fetchall()
    assert [row[0] for row in timeline] == ["create", "activate", "expire", "restore"]


def test_restored_memory_is_visible_to_list_active_again(repositories):
    """恢复的用意就是让她重新看得见它——必须真的回到 list_active。"""

    _, events, memories = repositories
    source = events.insert(event("e-restore-2", "他喜欢我认真听他讲", conversation_id="conversation-a"))
    memories.create(memory("m-restore-2", source, status="active", importance=3))
    memories.expire("m-restore-2")
    assert "m-restore-2" not in {r.memory_id for r in memories.list_active("conversation-a", NOW)}

    memories.restore("m-restore-2")

    assert "m-restore-2" in {r.memory_id for r in memories.list_active("conversation-a", NOW)}


def test_restore_keeps_bounded_scope_and_deadline(repositories):
    """不误判：bounded 记录恢复后仍是 bounded，且 valid_until 原样保留。"""

    _, events, memories = repositories
    deadline = NOW + timedelta(hours=6)
    source = events.insert(event("e-restore-3", "今晚陪我", conversation_id="conversation-a"))
    memories.create(memory("m-restore-3", source, status="active", temporal_scope="bounded",
                           valid_until=deadline))

    memories.expire("m-restore-3")
    restored = memories.restore("m-restore-3")

    assert restored.temporal_scope == "bounded"
    assert restored.valid_until_utc == deadline


@pytest.mark.parametrize("status", ["candidate", "active"])
def test_restore_refuses_targets_that_were_never_removed(repositories, status):
    """不误判：没被移出过的记录不许「恢复」（candidate/active 都不是终态）。"""

    _, events, memories = repositories
    source = events.insert(event(f"e-restore-never-{status}", "原话", conversation_id="conversation-a"))
    memories.create(memory(f"m-never-{status}", source, status=status, certainty="explicit"))
    with pytest.raises(ValueError, match=f"cannot restore from {status}"):
        memories.restore(f"m-never-{status}")


def test_restore_cannot_be_repeated(repositories):
    _, events, memories = repositories
    source = events.insert(event("e-restore-twice", "原话", conversation_id="conversation-a"))
    memories.create(memory("m-twice", source, status="active"))
    memories.expire("m-twice")
    memories.restore("m-twice")
    with pytest.raises(ValueError, match="cannot restore from active"):
        memories.restore("m-twice")


def test_restore_of_unknown_memory_is_a_key_error(repositories):
    _, _, memories = repositories
    with pytest.raises(KeyError):
        memories.restore("no-such-memory")


def test_removal_without_an_audit_trail_fails_closed(repositories):
    """遗留数据（迁移前就没有移除事件）必须 fail closed，不许猜一个评级塞回去。

    生产上这类记录有 44 条：它们的状态在一次性的 assess 事件里就已经是终态
    （before/after 都是 rejected），没有任何 before_* 可依据。
    """

    database, events, memories = repositories
    source = events.insert(event("e-legacy", "原话", conversation_id="conversation-a"))
    memories.create(memory("m-legacy", source, status="active"))
    database.connection.execute(
        "UPDATE memory_records SET status='rejected', certainty='unsupported', importance=0, "
        "temporal_scope='unclassified', assessment_reason_code='legacy_manual_review' "
        "WHERE memory_id='m-legacy'"
    )
    with pytest.raises(ValueError, match="no removal audit event"):
        memories.restore("m-legacy")


def test_restore_is_never_offered_to_the_extractor(repositories):
    """模型不能撤销自己的判决：allowed_review_actions 任何状态下都不含 restore。"""

    from qichi.storage.memory_repository import MemoryRepository

    _, events, memories = repositories
    source = events.insert(event("e-not-offered", "原话", conversation_id="conversation-a"))
    for status in ("candidate", "active", "expired", "rejected", "superseded"):
        record = memory(f"m-offer-{status}", source, status=status)
        assert "restore" not in MemoryRepository.allowed_review_actions(record)

def test_manual_expire_only_flips_status_and_restore_still_works(repositories):
    """手动 expire() 只把状态置为 expired，**不**降级评级（与 review 路径不同）。

    这是 2026-09-22 写这张卡时实测出来的：降级来自 review 的合同，不来自状态本身。
    记录在这里，免得以后有人以为 expired 一定意味着 unsupported。
    """

    _, events, memories = repositories
    source = events.insert(event("e-manual-expire", "原话", conversation_id="conversation-a"))
    memories.create(memory("m-manual", source, status="active", certainty="explicit", importance=2))
    expired = memories.expire("m-manual")
    assert (expired.certainty, expired.importance) == ("explicit", 2)

    restored = memories.restore("m-manual")

    assert (restored.status, restored.certainty, restored.importance) == ("active", "explicit", 2)

