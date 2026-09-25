-- 卡B③（2026-09-22 用户裁定，faithful 路线）：把人工「恢复」变成一个可审计的动作。
--
-- 背景：rejected / expired 是终局——allowed_review_actions 对它们返回空，
-- _validate_review 又要求目标处于 {candidate, active}，代码里没有任何 restore 路径。
-- 于是模型一次判错的移出，等于那条偏好永远不再出现在她眼前（行与证据还在库里，
-- 只能人工改库）。用户裁定：移出必须可逆，且语义要准确。
--
-- SQLite 不能原地修改 CHECK，所以这里是表重建：新建 → 拷行 → 删旧 → 改名 →
-- 重建索引与触发器。唯一语义变化是 action 的 CHECK 多了 'restore'，
-- 其余列定义与 0003/0006 逐字一致（含 append-only 的两个触发器）。

CREATE TABLE memory_audit_events_rebuilt (
    audit_event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_records(memory_id),
    session_job_id TEXT,
    action TEXT NOT NULL
        CHECK(action IN ('create', 'assess', 'support', 'activate', 'confirm', 'reject', 'expire', 'supersede', 'restore')),
    before_status TEXT
        CHECK(before_status IS NULL OR before_status IN ('candidate', 'active', 'superseded', 'rejected', 'expired')),
    after_status TEXT
        CHECK(after_status IS NULL OR after_status IN ('candidate', 'active', 'superseded', 'rejected', 'expired')),
    before_certainty TEXT
        CHECK(before_certainty IS NULL OR before_certainty IN ('unsupported', 'ambiguous', 'explicit', 'confirmed', 'unassessed')),
    after_certainty TEXT
        CHECK(after_certainty IS NULL OR after_certainty IN ('unsupported', 'ambiguous', 'explicit', 'confirmed', 'unassessed')),
    before_importance INTEGER
        CHECK(before_importance IS NULL OR (typeof(before_importance) = 'integer' AND before_importance BETWEEN 0 AND 3)),
    after_importance INTEGER
        CHECK(after_importance IS NULL OR (typeof(after_importance) = 'integer' AND after_importance BETWEEN 0 AND 3)),
    before_temporal_scope TEXT
        CHECK(before_temporal_scope IS NULL OR before_temporal_scope IN ('ongoing', 'bounded', 'historical', 'unclassified')),
    after_temporal_scope TEXT
        CHECK(after_temporal_scope IS NULL OR after_temporal_scope IN ('ongoing', 'bounded', 'historical', 'unclassified')),
    assessment_reason_code TEXT NOT NULL
        CHECK(assessment_reason_code IN (
            'explicit_user_statement', 'bilateral_agreement', 'later_user_confirmation',
            'user_correction', 'ambiguous_scope', 'historical_event',
            'contradicted_by_user', 'expired_or_completed', 'legacy_manual_review',
            'unsupported_or_transient'
        )),
    occurred_at_utc TEXT NOT NULL
        CHECK(length(occurred_at_utc) >= 20 AND (
            substr(occurred_at_utc, -1) = 'Z'
            OR substr(occurred_at_utc, -6, 1) IN ('+', '-')
        ))
);

INSERT INTO memory_audit_events_rebuilt (
    audit_event_id, memory_id, session_job_id, action,
    before_status, after_status, before_certainty, after_certainty,
    before_importance, after_importance, before_temporal_scope, after_temporal_scope,
    assessment_reason_code, occurred_at_utc
)
SELECT
    audit_event_id, memory_id, session_job_id, action,
    before_status, after_status, before_certainty, after_certainty,
    before_importance, after_importance, before_temporal_scope, after_temporal_scope,
    assessment_reason_code, occurred_at_utc
FROM memory_audit_events;

DROP TABLE memory_audit_events;

ALTER TABLE memory_audit_events_rebuilt RENAME TO memory_audit_events;

CREATE INDEX ix_memory_audit_memory_time
ON memory_audit_events(memory_id, occurred_at_utc, audit_event_id);

CREATE TRIGGER immutable_memory_audit_update
BEFORE UPDATE ON memory_audit_events
BEGIN
    SELECT RAISE(ABORT, 'memory audit events are append-only');
END;

CREATE TRIGGER immutable_memory_audit_delete
BEFORE DELETE ON memory_audit_events
BEGIN
    SELECT RAISE(ABORT, 'memory audit events are append-only');
END;
