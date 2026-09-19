-- Privacy and recall are orthogonal to memory type and lifecycle.
-- Existing records receive conservative ordinary/daily_safe defaults because
-- their historical evidence is not reinterpreted during migration.
ALTER TABLE memory_records
ADD COLUMN privacy_class TEXT NOT NULL DEFAULT 'ordinary'
CHECK(privacy_class IN ('ordinary', 'intimate', 'adult'));

ALTER TABLE memory_records
ADD COLUMN recall_policy TEXT NOT NULL DEFAULT 'daily_safe'
CHECK(recall_policy IN ('daily_safe', 'topic_only', 'explicit_request_only'));

CREATE INDEX ix_memory_privacy_recall
ON memory_records(status, privacy_class, recall_policy, valid_from_utc, valid_until_utc);

CREATE TRIGGER memory_privacy_insert_guard
BEFORE INSERT ON memory_records
WHEN NEW.privacy_class = 'adult' AND NEW.recall_policy = 'daily_safe'
BEGIN
    SELECT RAISE(ABORT, 'adult memory cannot use daily_safe recall policy');
END;

CREATE TRIGGER memory_privacy_update_guard
BEFORE UPDATE OF privacy_class, recall_policy ON memory_records
WHEN NEW.privacy_class = 'adult' AND NEW.recall_policy = 'daily_safe'
BEGIN
    SELECT RAISE(ABORT, 'adult memory cannot use daily_safe recall policy');
END;
