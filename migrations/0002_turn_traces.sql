CREATE TABLE turn_trace_events (
    trace_event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    trigger_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    source TEXT NOT NULL CHECK(source IN ('dialogue', 'interaction', 'initiative')),
    phase TEXT NOT NULL CHECK(phase IN ('received', 'context', 'generation', 'delivery', 'failure', 'cancelled')),
    occurred_at_utc TEXT NOT NULL,
    details_json TEXT NOT NULL,
    UNIQUE(trace_id, phase)
);

CREATE INDEX ix_turn_trace_conversation_time
ON turn_trace_events(conversation_id, occurred_at_utc, trace_id);

CREATE TRIGGER immutable_turn_trace_update
BEFORE UPDATE ON turn_trace_events
BEGIN
    SELECT RAISE(ABORT, 'turn trace events are append-only');
END;

CREATE TRIGGER immutable_turn_trace_delete
BEFORE DELETE ON turn_trace_events
BEGIN
    SELECT RAISE(ABORT, 'turn trace events are append-only');
END;
