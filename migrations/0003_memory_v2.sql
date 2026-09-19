ALTER TABLE memory_records
ADD COLUMN certainty TEXT NOT NULL DEFAULT 'unassessed'
CHECK(certainty IN ('unsupported', 'ambiguous', 'explicit', 'confirmed', 'unassessed'));

ALTER TABLE memory_records
ADD COLUMN importance INTEGER NOT NULL DEFAULT 0
CHECK(typeof(importance) = 'integer' AND importance BETWEEN 0 AND 3);

ALTER TABLE memory_records
ADD COLUMN temporal_scope TEXT NOT NULL DEFAULT 'unclassified'
CHECK(temporal_scope IN ('ongoing', 'bounded', 'historical', 'unclassified'));

ALTER TABLE memory_records
ADD COLUMN assessment_reason_code TEXT
CHECK(
    assessment_reason_code IS NULL OR assessment_reason_code IN (
        'explicit_user_statement', 'bilateral_agreement', 'later_user_confirmation',
        'user_correction', 'ambiguous_scope', 'historical_event',
        'contradicted_by_user', 'expired_or_completed', 'legacy_manual_review',
        'unsupported_or_transient'
    )
);

ALTER TABLE memory_records
ADD COLUMN assessed_at_utc TEXT
CHECK(
    assessed_at_utc IS NULL OR (
        length(assessed_at_utc) >= 20 AND (
            substr(assessed_at_utc, -1) = 'Z'
            OR substr(assessed_at_utc, -6, 1) IN ('+', '-')
        )
    )
);

ALTER TABLE memory_evidence
ADD COLUMN evidence_role TEXT NOT NULL DEFAULT 'source'
CHECK(evidence_role IN ('source', 'proposal', 'acceptance', 'confirmation', 'correction', 'counterevidence'));

CREATE TABLE memory_audit_events (
    audit_event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_records(memory_id),
    session_job_id TEXT,
    action TEXT NOT NULL
        CHECK(action IN ('create', 'assess', 'support', 'activate', 'confirm', 'reject', 'expire', 'supersede')),
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

CREATE TABLE memory_session_jobs (
    job_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL UNIQUE,
    revision INTEGER NOT NULL CHECK(typeof(revision) = 'integer' AND revision >= 1),
    fragment_key TEXT NOT NULL CHECK(length(fragment_key) > 0),
    start_sequence INTEGER NOT NULL CHECK(typeof(start_sequence) = 'integer' AND start_sequence >= 0),
    end_sequence INTEGER NOT NULL CHECK(typeof(end_sequence) = 'integer' AND end_sequence >= start_sequence),
    anchor_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    anchor_sequence INTEGER NOT NULL
        CHECK(typeof(anchor_sequence) = 'integer' AND anchor_sequence BETWEEN start_sequence AND end_sequence),
    anchor_received_at_utc TEXT NOT NULL,
    deadline_utc TEXT NOT NULL,
    context_version INTEGER NOT NULL CHECK(typeof(context_version) = 'integer' AND context_version >= 0),
    status TEXT NOT NULL CHECK(status IN ('pending', 'claimed', 'retry', 'failed')),
    claim_token TEXT,
    claim_owner TEXT,
    claim_lease_until_utc TEXT,
    next_retry_at_utc TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK(typeof(attempt_count) = 'integer' AND attempt_count >= 0),
    failure_category TEXT CHECK(failure_category IS NULL OR length(failure_category) > 0),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    CHECK(
        (status = 'claimed' AND claim_token IS NOT NULL AND claim_owner IS NOT NULL AND claim_lease_until_utc IS NOT NULL)
        OR
        (status <> 'claimed' AND claim_token IS NULL AND claim_owner IS NULL AND claim_lease_until_utc IS NULL)
    ),
    CHECK(
        (status = 'retry' AND next_retry_at_utc IS NOT NULL AND failure_category IS NOT NULL)
        OR
        (status <> 'retry' AND next_retry_at_utc IS NULL)
    ),
    CHECK(status <> 'failed' OR failure_category IS NOT NULL)
);

CREATE INDEX ix_memory_session_jobs_due
ON memory_session_jobs(status, deadline_utc, next_retry_at_utc, conversation_id);

CREATE TABLE memory_confirmation_presentations (
    presentation_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memory_records(memory_id),
    fragment_key TEXT NOT NULL CHECK(length(fragment_key) > 0),
    trigger_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    context_version INTEGER NOT NULL CHECK(typeof(context_version) = 'integer' AND context_version >= 0),
    presented_at_utc TEXT NOT NULL,
    UNIQUE(conversation_id, fragment_key)
);

CREATE INDEX ix_memory_confirmation_memory_time
ON memory_confirmation_presentations(memory_id, presented_at_utc, presentation_id);

UPDATE memory_records
SET certainty = CASE status
        WHEN 'active' THEN 'explicit'
        WHEN 'candidate' THEN 'ambiguous'
        WHEN 'expired' THEN 'explicit'
        WHEN 'rejected' THEN 'unsupported'
        ELSE 'unassessed'
    END,
    importance = CASE
        WHEN memory_id IN (
            '04e2296c-ba1c-503b-abf4-d61fe590ef44',
            '8bac2494-9f8b-5fa2-a882-ef5c96c77ceb',
            'b7aca90e-9216-59e0-99ed-990dc11efe65',
            'de392c29-a61b-5e25-9131-5a637f7132c4',
            '71cf09df-f555-54b4-93e5-56d147feb789',
            '9bd8e649-0444-5978-9b6e-0403ab70dd35',
            '6fc794ad-5548-5f92-a5b4-5bd1879a7d59'
        ) THEN 3
        WHEN status IN ('active', 'candidate') THEN 2
        ELSE 0
    END,
    temporal_scope = CASE
        WHEN status = 'active' AND type = 'episode' THEN 'historical'
        WHEN status = 'active' THEN 'ongoing'
        ELSE 'unclassified'
    END,
    assessment_reason_code = CASE
        WHEN status IN ('active', 'candidate', 'expired', 'rejected') THEN 'legacy_manual_review'
        ELSE NULL
    END,
    assessed_at_utc = CASE
        WHEN status IN ('active', 'candidate', 'expired', 'rejected')
        THEN strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
        ELSE NULL
    END;

INSERT INTO memory_audit_events (
    audit_event_id, memory_id, session_job_id, action,
    before_status, after_status, before_certainty, after_certainty,
    before_importance, after_importance, before_temporal_scope, after_temporal_scope,
    assessment_reason_code, occurred_at_utc
)
SELECT
    'legacy-v3:' || memory_id,
    memory_id,
    NULL,
    'assess',
    status,
    status,
    'unassessed',
    certainty,
    0,
    importance,
    'unclassified',
    temporal_scope,
    'legacy_manual_review',
    assessed_at_utc
FROM memory_records;

CREATE TRIGGER memory_v2_record_insert_guard
BEFORE INSERT ON memory_records
WHEN
    (NEW.temporal_scope = 'bounded' AND NEW.valid_until_utc IS NULL)
    OR (NEW.certainty = 'unassessed' AND (
        NEW.importance <> 0 OR NEW.temporal_scope <> 'unclassified'
        OR NEW.assessment_reason_code IS NOT NULL OR NEW.assessed_at_utc IS NOT NULL
    ))
    OR (NEW.certainty <> 'unassessed' AND (
        NEW.assessment_reason_code IS NULL OR NEW.assessed_at_utc IS NULL
    ))
BEGIN
    SELECT RAISE(ABORT, 'invalid memory v2 grading fields');
END;

CREATE TRIGGER memory_v2_record_update_guard
BEFORE UPDATE OF certainty, importance, temporal_scope, assessment_reason_code, assessed_at_utc, valid_until_utc
ON memory_records
WHEN
    (NEW.temporal_scope = 'bounded' AND NEW.valid_until_utc IS NULL)
    OR (NEW.certainty = 'unassessed' AND (
        NEW.importance <> 0 OR NEW.temporal_scope <> 'unclassified'
        OR NEW.assessment_reason_code IS NOT NULL OR NEW.assessed_at_utc IS NOT NULL
    ))
    OR (NEW.certainty <> 'unassessed' AND (
        NEW.assessment_reason_code IS NULL OR NEW.assessed_at_utc IS NULL
    ))
BEGIN
    SELECT RAISE(ABORT, 'invalid memory v2 grading fields');
END;
