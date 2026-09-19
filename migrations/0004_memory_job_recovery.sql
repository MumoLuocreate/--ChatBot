-- Durable recovery history for Memory V2 session jobs.
-- This table intentionally does not reference memory_session_jobs: a job row is
-- a mutable one-row-per-conversation cursor and is deleted after success, while
-- its public lifecycle history must remain available for diagnosis.
CREATE TABLE memory_job_events (
    job_event_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(typeof(revision) = 'integer' AND revision >= 1),
    fragment_key TEXT NOT NULL CHECK(length(fragment_key) > 0),
    start_sequence INTEGER NOT NULL CHECK(typeof(start_sequence) = 'integer' AND start_sequence >= 0),
    end_sequence INTEGER NOT NULL CHECK(typeof(end_sequence) = 'integer' AND end_sequence >= start_sequence),
    action TEXT NOT NULL CHECK(action IN ('retry_scheduled', 'failed', 'reopened', 'quarantined', 'completed')),
    failure_category TEXT CHECK(failure_category IS NULL OR length(failure_category) > 0),
    attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK(typeof(attempt_count) = 'integer' AND attempt_count >= 0),
    next_retry_at_utc TEXT,
    occurred_at_utc TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
        CHECK(json_valid(details_json))
);

CREATE INDEX ix_memory_job_events_job_time
ON memory_job_events(job_id, occurred_at_utc, job_event_id);

CREATE INDEX ix_memory_job_events_conversation_time
ON memory_job_events(conversation_id, occurred_at_utc, job_event_id);

CREATE TRIGGER immutable_memory_job_event_update
BEFORE UPDATE ON memory_job_events
BEGIN
    SELECT RAISE(ABORT, 'memory job events are append-only');
END;

CREATE TRIGGER immutable_memory_job_event_delete
BEFORE DELETE ON memory_job_events
BEGIN
    SELECT RAISE(ABORT, 'memory job events are append-only');
END;

-- Preserve an observable failure fact for jobs that were already terminal when
-- this migration is first applied. No message text or model output is copied.
INSERT INTO memory_job_events (
    job_event_id, job_id, conversation_id, revision, fragment_key,
    start_sequence, end_sequence, action, failure_category, attempt_count,
    next_retry_at_utc, occurred_at_utc, details_json
)
SELECT
    'legacy-v4:' || job_id,
    job_id,
    conversation_id,
    revision,
    fragment_key,
    start_sequence,
    end_sequence,
    'failed',
    failure_category,
    attempt_count,
    next_retry_at_utc,
    updated_at_utc,
    '{}'
FROM memory_session_jobs
WHERE status = 'failed';
