-- Detailed fragment index and evidence timeline.  The immutable conversation
-- event ledger remains the source of truth; these tables are derived, bounded
-- indexes and may be rebuilt from event IDs.
CREATE TABLE memory_fragments (
    fragment_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    fragment_key TEXT NOT NULL,
    start_sequence INTEGER NOT NULL CHECK(typeof(start_sequence) = 'integer' AND start_sequence >= 0),
    end_sequence INTEGER NOT NULL CHECK(typeof(end_sequence) = 'integer' AND end_sequence >= start_sequence),
    start_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    end_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT NOT NULL,
    fragment_type TEXT NOT NULL CHECK(fragment_type IN ('daily', 'intimate', 'adult', 'mixed', 'unknown')),
    reality_scope TEXT NOT NULL CHECK(reality_scope IN ('conversation', 'shared_imagination', 'hypothetical', 'claimed_real', 'mixed', 'unknown')),
    summary TEXT NOT NULL CHECK(length(summary) > 0 AND length(summary) <= 1024),
    privacy_class TEXT NOT NULL CHECK(privacy_class IN ('ordinary', 'intimate', 'adult')),
    recall_policy TEXT NOT NULL CHECK(recall_policy IN ('daily_safe', 'topic_only', 'explicit_request_only')),
    status TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'superseded', 'rejected')),
    closed_at_utc TEXT,
    created_at_utc TEXT NOT NULL,
    UNIQUE(conversation_id, fragment_key),
    CHECK(NOT (privacy_class = 'adult' AND recall_policy = 'daily_safe'))
);

CREATE INDEX ix_memory_fragments_recall
ON memory_fragments(conversation_id, status, privacy_class, recall_policy, started_at_utc, ended_at_utc);

CREATE TABLE memory_fragment_events (
    fragment_id TEXT NOT NULL REFERENCES memory_fragments(fragment_id),
    event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal >= 0),
    sequence INTEGER NOT NULL CHECK(typeof(sequence) = 'integer' AND sequence >= 0),
    actor TEXT NOT NULL CHECK(actor IN ('mumo', 'qichi', 'platform')),
    occurred_at_utc TEXT NOT NULL,
    PRIMARY KEY(fragment_id, event_id),
    UNIQUE(fragment_id, ordinal)
);

CREATE INDEX ix_memory_fragment_events_event
ON memory_fragment_events(event_id, fragment_id);

CREATE TABLE memory_detail_records (
    detail_id TEXT PRIMARY KEY,
    fragment_id TEXT NOT NULL REFERENCES memory_fragments(fragment_id),
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal >= 0),
    detail_kind TEXT NOT NULL CHECK(detail_kind IN ('message', 'statement', 'proposal', 'acceptance', 'boundary', 'choice', 'agreement', 'plan', 'uncertainty', 'correction', 'closure')),
    actor TEXT NOT NULL CHECK(actor IN ('mumo', 'qichi', 'joint', 'unknown')),
    reality_scope TEXT NOT NULL CHECK(reality_scope IN ('conversation', 'shared_imagination', 'hypothetical', 'claimed_real', 'mixed', 'unknown')),
    normalized_detail TEXT NOT NULL CHECK(length(normalized_detail) > 0 AND length(normalized_detail) <= 1024),
    exact_quote TEXT NOT NULL CHECK(length(exact_quote) > 0 AND length(exact_quote) <= 2048),
    source_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    occurred_at_utc TEXT NOT NULL,
    certainty TEXT NOT NULL CHECK(certainty IN ('explicit', 'confirmed', 'ambiguous', 'unsupported')),
    temporal_scope TEXT NOT NULL CHECK(temporal_scope IN ('historical', 'ongoing', 'future_plan', 'unclassified')),
    status TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'superseded', 'rejected')),
    privacy_class TEXT NOT NULL CHECK(privacy_class IN ('ordinary', 'intimate', 'adult')),
    recall_policy TEXT NOT NULL CHECK(recall_policy IN ('daily_safe', 'topic_only', 'explicit_request_only')),
    UNIQUE(fragment_id, ordinal),
    CHECK(NOT (privacy_class = 'adult' AND recall_policy = 'daily_safe'))
);

CREATE INDEX ix_memory_detail_recall
ON memory_detail_records(fragment_id, status, privacy_class, recall_policy, ordinal);

CREATE TABLE memory_detail_evidence (
    detail_id TEXT NOT NULL REFERENCES memory_detail_records(detail_id),
    event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    evidence_role TEXT NOT NULL CHECK(evidence_role IN ('source', 'proposal', 'acceptance', 'correction', 'context')),
    ordinal INTEGER NOT NULL CHECK(typeof(ordinal) = 'integer' AND ordinal >= 0),
    PRIMARY KEY(detail_id, event_id),
    UNIQUE(detail_id, ordinal)
);

CREATE INDEX ix_memory_detail_evidence_event
ON memory_detail_evidence(event_id, detail_id);

CREATE TRIGGER immutable_memory_fragment_event_update
BEFORE UPDATE ON memory_fragment_events
BEGIN
    SELECT RAISE(ABORT, 'memory fragment events are append-only');
END;

CREATE TRIGGER immutable_memory_fragment_event_delete
BEFORE DELETE ON memory_fragment_events
BEGIN
    SELECT RAISE(ABORT, 'memory fragment events are append-only');
END;

CREATE TRIGGER immutable_memory_detail_evidence_update
BEFORE UPDATE ON memory_detail_evidence
BEGIN
    SELECT RAISE(ABORT, 'memory detail evidence is append-only');
END;

CREATE TRIGGER immutable_memory_detail_evidence_delete
BEFORE DELETE ON memory_detail_evidence
BEGIN
    SELECT RAISE(ABORT, 'memory detail evidence is append-only');
END;

