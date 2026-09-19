CREATE TABLE IF NOT EXISTS conversation_events (
    event_id TEXT PRIMARY KEY, platform_event_id TEXT, platform_message_id TEXT,
    conversation_id TEXT NOT NULL, sequence INTEGER NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('inbound', 'outbound', 'internal')),
    actor TEXT NOT NULL CHECK(actor IN ('mumo', 'qichi', 'platform')),
    kind TEXT NOT NULL, text TEXT, message_segments_json TEXT NOT NULL,
    reply_to_event_id TEXT, reply_to_platform_message_id TEXT,
    occurred_at_utc TEXT NOT NULL, received_at_utc TEXT NOT NULL,
    status TEXT NOT NULL CHECK(length(status) > 0), metadata_json TEXT NOT NULL, raw_payload_json TEXT,
    UNIQUE(conversation_id, sequence),
    FOREIGN KEY(reply_to_event_id) REFERENCES conversation_events(event_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_platform_event_id ON conversation_events(platform_event_id) WHERE platform_event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_platform_message_id ON conversation_events(platform_message_id) WHERE platform_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS platform_message_map (
    platform_message_id TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE REFERENCES conversation_events(event_id),
    source TEXT NOT NULL, created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message_links (
    link_id TEXT PRIMARY KEY, source_event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    target_event_id TEXT REFERENCES conversation_events(event_id), relation TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    operation_key TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    payload_json TEXT NOT NULL, status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
    response_json TEXT, created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_records (
    memory_id TEXT PRIMARY KEY, type TEXT NOT NULL, normalized_fact TEXT NOT NULL, modality TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'superseded', 'rejected', 'expired')),
    valid_from_utc TEXT NOT NULL, valid_until_utc TEXT,
    supersedes_id TEXT REFERENCES memory_records(memory_id), created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_evidence (
    memory_id TEXT NOT NULL REFERENCES memory_records(memory_id),
    event_id TEXT NOT NULL REFERENCES conversation_events(event_id),
    actor TEXT NOT NULL CHECK(actor IN ('mumo', 'qichi', 'platform')),
    exact_quote TEXT NOT NULL, occurred_at_utc TEXT NOT NULL,
    PRIMARY KEY(memory_id, event_id, exact_quote)
);
CREATE TABLE IF NOT EXISTS conversation_cursors (
    conversation_id TEXT PRIMARY KEY, context_version INTEGER NOT NULL DEFAULT 0,
    last_processed_sequence INTEGER, last_user_activity_utc TEXT, presence_topic_cursor INTEGER
);
CREATE TABLE IF NOT EXISTS runtime_meta (
    key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at_utc TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS immutable_event_fields
BEFORE UPDATE ON conversation_events
WHEN NEW.event_id IS NOT OLD.event_id
  OR NEW.platform_event_id IS NOT OLD.platform_event_id
  OR NEW.platform_message_id IS NOT OLD.platform_message_id
  OR NEW.conversation_id IS NOT OLD.conversation_id
  OR NEW.sequence IS NOT OLD.sequence
  OR NEW.direction IS NOT OLD.direction
  OR NEW.actor IS NOT OLD.actor
  OR NEW.kind IS NOT OLD.kind
  OR NEW.text IS NOT OLD.text
  OR NEW.message_segments_json IS NOT OLD.message_segments_json
  OR NEW.reply_to_event_id IS NOT OLD.reply_to_event_id
  OR NEW.reply_to_platform_message_id IS NOT OLD.reply_to_platform_message_id
  OR NEW.occurred_at_utc IS NOT OLD.occurred_at_utc
  OR NEW.received_at_utc IS NOT OLD.received_at_utc
  OR NEW.metadata_json IS NOT OLD.metadata_json
  OR NEW.raw_payload_json IS NOT OLD.raw_payload_json
BEGIN
    SELECT RAISE(ABORT, 'conversation events are immutable');
END;
