from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

from qichi.domain.memory import MemoryEvidence, MemoryRecord, MemoryReview
from qichi.memory.lexical import match_count, query_fragments

from .database import Database


_STATUSES = frozenset({"candidate", "active", "superseded", "rejected", "expired"})
_INITIAL_STATUSES = frozenset({"candidate", "active"})
_BATCH_TYPES = frozenset(
    {"preference", "agreement", "correction", "episode", "self_expression"}
)
_SUPPORT_REASONS = frozenset(
    {
        "explicit_user_statement",
        "bilateral_agreement",
        "user_correction",
        "ambiguous_scope",
        "historical_event",
    }
)
_RELIABLE_EVENT_SHAPES = {
    ("mumo", "inbound", "text", "received"),
}
_CANDIDATE_ROLE_SETS = {
    "preference": frozenset({"source"}),
    "episode": frozenset({"source"}),
    "agreement": frozenset({"proposal", "acceptance"}),
    "correction": frozenset({"correction"}),
    "self_expression": frozenset({"source"}),
}
_MAX_BATCH_ITEMS = 12
_CONFIRMATION_COOLDOWN = timedelta(days=7)
_LEGACY_API_SESSION_JOB = "legacy-public-api"


def _outbound_source(metadata_json: object) -> bool:
    try:
        value = json.loads(metadata_json) if isinstance(metadata_json, str) else None
    except (TypeError, ValueError):
        return False
    if not isinstance(value, dict):
        return False
    generation = value.get("generation_metadata")
    return isinstance(generation, dict) and generation.get("source") in {"dialogue", "interaction"}


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be stored as text")
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error


def _stored_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be stored as text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} is invalid") from error
    try:
        offset = parsed.utcoffset()
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error
    if parsed.tzinfo is None or offset != timedelta(0):
        raise ValueError(f"{field} must be stored in UTC")
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error


@dataclass(frozen=True, slots=True)
class MemoryBatchResult:
    created_memory_ids: tuple[str, ...] = ()
    reviewed_memory_ids: tuple[str, ...] = ()
    activated_memory_ids: tuple[str, ...] = ()
    superseded_memory_ids: tuple[str, ...] = ()
    rejected_memory_ids: tuple[str, ...] = ()
    expired_memory_ids: tuple[str, ...] = ()
    # 同一条偏好被换句话又说了一遍：不重复建卡，但要把「跳过了几条」说出来
    # （2026-09-12，见 _is_restated_preference）。
    restated_memory_ids: tuple[str, ...] = ()


class MemoryRepository:
    """Store evidence-backed memories and deterministic lifecycle decisions."""

    def __init__(self, database: Database):
        self.database = database

    def create(self, record: MemoryRecord) -> MemoryRecord:
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        if record.status not in _INITIAL_STATUSES:
            raise ValueError("new memory status must be candidate or active")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memory_records WHERE memory_id = ?", (record.memory_id,)
            ).fetchone()
            if row is not None:
                restored = self._record_from_row(connection, row)
                if not self._same_identity(restored, record):
                    raise ValueError("memory_id immutable identity conflict")
                if restored.status == "candidate" and record.status == "active":
                    raise ValueError("memory status changes must use activate()")
                return restored

            conversation_id = self._validate_evidence(connection, record.memory_evidence)
            self._validate_correction_shape(record)
            if record.supersedes_id is not None:
                target = self._get(connection, record.supersedes_id)
                if self._validate_evidence(connection, target.memory_evidence) != conversation_id:
                    raise ValueError(
                        "correction and superseded memory must belong to the same conversation"
                    )

            # The public API is retained for compatibility with the pre-V2
            # worker/admin callers.  It is deliberately marked as a legacy
            # manual operation in the audit log, but an active result still
            # has to pass the same evidence/ownership matrix as V2.
            requested_status = record.status
            if requested_status == "active":
                self._validate_public_active_matrix(
                    connection, record, conversation_id
                )
            inserted = replace(record, status="candidate" if requested_status == "active" else requested_status)
            self._insert_record(connection, inserted)
            persisted = self._get(connection, record.memory_id)
            self._write_audit(
                connection,
                record.memory_id,
                _LEGACY_API_SESSION_JOB,
                "create",
                None,
                persisted,
                "legacy_manual_review",
                record.assessed_at_utc or record.created_at_utc,
            )
            if requested_status == "active":
                self._activate_with_audit(
                    connection,
                    record.memory_id,
                    conversation_id,
                    _LEGACY_API_SESSION_JOB,
                    record.assessed_at_utc or record.created_at_utc,
                    allow_manual_self_expression=True,
                )
            return self._get(connection, record.memory_id)

    def get(self, memory_id: str) -> MemoryRecord:
        return self._get(self.database.connection, _nonempty_text(memory_id, "memory_id"))

    def count(self) -> int:
        value = self.database.connection.execute(
            "SELECT COUNT(*) FROM memory_records"
        ).fetchone()[0]
        if type(value) is not int or value < 0:
            raise ValueError("persisted memory count is invalid")
        return value

    @staticmethod
    def allowed_review_actions(record: MemoryRecord) -> tuple[str, ...]:
        """Return lifecycle actions that can apply to a model-visible target."""
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        if record.type == "self_expression":
            return ()
        if record.status == "candidate":
            return ("support", "confirm", "reject", "expire")
        if record.status == "active":
            return ("confirm", "reject", "expire")
        return ()

    def filter_applicable_reviews_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        conversation_id: str,
        reviews: tuple[MemoryReview, ...],
    ) -> tuple[tuple[MemoryReview, ...], int]:
        """Isolate lifecycle-impossible model reviews before atomic apply.

        Full evidence and assessment validation still runs in the batch path.
        This gate only handles missing/ineligible targets and actions that can
        never apply to the target's current lifecycle state.
        """
        if connection is not self.database.connection or not connection.in_transaction:
            raise ValueError("connection must be this repository's active transaction")
        conversation_id = _nonempty_text(conversation_id, "conversation_id")
        if not isinstance(reviews, tuple) or not all(
            isinstance(item, MemoryReview) for item in reviews
        ):
            raise TypeError("reviews must be a tuple of MemoryReview")
        accepted: list[MemoryReview] = []
        dropped = 0
        for review in reviews:
            try:
                target = self._get(connection, review.memory_id)
            except KeyError:
                dropped += 1
                continue
            if self._validate_evidence(connection, target.memory_evidence) != conversation_id:
                dropped += 1
                continue
            if review.action not in self.allowed_review_actions(target):
                dropped += 1
                continue
            accepted.append(review)
        return tuple(accepted), dropped

    def activate(self, memory_id: str) -> MemoryRecord:
        memory_id = _nonempty_text(memory_id, "memory_id")
        with self.database.transaction() as connection:
            record = self._get(connection, memory_id)
            if record.status != "candidate":
                raise ValueError(f"cannot activate from {record.status}")
            conversation_id = self._validate_evidence(connection, record.memory_evidence)
            self._validate_public_active_matrix(connection, record, conversation_id)
            self._activate_with_audit(
                connection,
                memory_id,
                conversation_id,
                _LEGACY_API_SESSION_JOB,
                record.assessed_at_utc or record.created_at_utc,
                allow_manual_self_expression=True,
            )
            return self._get(connection, memory_id)

    def reject(self, memory_id: str) -> MemoryRecord:
        return self._transition(
            memory_id,
            "candidate",
            "rejected",
            "reject",
            reason="legacy_manual_review",
        )

    def expire(self, memory_id: str) -> MemoryRecord:
        memory_id = _nonempty_text(memory_id, "memory_id")
        with self.database.transaction() as connection:
            record = self._get(connection, memory_id)
            if record.status not in {"candidate", "active"}:
                raise ValueError(f"cannot expire from {record.status}")
            cursor = connection.execute(
                "UPDATE memory_records SET status = 'expired' "
                "WHERE memory_id = ? AND status IN ('candidate', 'active')",
                (memory_id,),
            )
            if cursor.rowcount != 1:
                self._raise_transition_error(connection, memory_id, "expire")
            after = self._get(connection, memory_id)
            self._write_audit(
                connection,
                memory_id,
                _LEGACY_API_SESSION_JOB,
                "expire",
                record,
                after,
                "legacy_manual_review",
                record.assessed_at_utc or record.created_at_utc,
            )
            return after

    def list_active(
        self, conversation_id: str, at_utc: datetime
    ) -> tuple[MemoryRecord, ...]:
        return self._list_by_statuses(
            conversation_id, at_utc, ("active",), require_current_validity=True
        )

    def list_reviewable(
        self, conversation_id: str, at_utc: datetime
    ) -> tuple[MemoryRecord, ...]:
        return self._list_by_statuses(
            conversation_id,
            at_utc,
            ("active", "candidate"),
            require_current_validity=False,
        )

    # 同一条偏好被复述时，两边的三字窗口互相命中多少才算「同一句话」。实测
    # （2026-09-12，118 条记录两两比对）：复述落在 0.61~0.70，不相干的偏好低于 0.3；
    # 而不同夜晚的片段因为共用同一个模板开头排在 0.53——所以这个判据只用于
    # preference，绝不碰 episode。
    _PREFERENCE_RESTATEMENT = 0.6

    def _is_restated_preference(
        self, connection: sqlite3.Connection, candidate: MemoryRecord, conversation_id: str
    ) -> bool:
        """Has this same preference already been stored, in almost the same words?"""

        if candidate.type != "preference" or candidate.status not in _INITIAL_STATUSES:
            return False
        text = candidate.normalized_fact or ""
        windows = query_fragments((text,))
        if not windows:
            return False
        rows = connection.execute(
            "SELECT DISTINCT m.normalized_fact FROM memory_records AS m "
            "JOIN memory_evidence AS me ON me.memory_id = m.memory_id "
            "JOIN conversation_events AS e ON e.event_id = me.event_id "
            "WHERE e.conversation_id=? AND m.type='preference' "
            "AND m.privacy_class=? AND m.recall_policy=? AND m.status IN ('active','candidate') "
            "AND m.memory_id<>?",
            (conversation_id, candidate.privacy_class, candidate.recall_policy, candidate.memory_id),
        ).fetchall()
        for row in rows:
            existing = row[0] or ""
            other = query_fragments((existing,))
            if not other:
                continue
            forward = match_count(existing, windows) / len(windows)
            backward = match_count(text, other) / len(other)
            if min(forward, backward) >= self._PREFERENCE_RESTATEMENT:
                return True
        return False

    def apply_consolidation_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        conversation_id: str,
        session_job_id: str,
        candidates: tuple[MemoryRecord, ...],
        reviews: tuple[MemoryReview, ...],
        assessed_at_utc: datetime,
    ) -> MemoryBatchResult:
        """Apply one extractor response without owning the outer transaction."""

        if connection is not self.database.connection or not connection.in_transaction:
            raise ValueError("connection must be this repository's active transaction")
        conversation_id = _nonempty_text(conversation_id, "conversation_id")
        session_job_id = _nonempty_text(session_job_id, "session_job_id")
        assessed_at_utc = _utc(assessed_at_utc, "assessed_at_utc")
        if not isinstance(candidates, tuple) or not all(
            isinstance(item, MemoryRecord) for item in candidates
        ):
            raise TypeError("candidates must be a tuple of MemoryRecord")
        if not isinstance(reviews, tuple) or not all(
            isinstance(item, MemoryReview) for item in reviews
        ):
            raise TypeError("reviews must be a tuple of MemoryReview")
        if len(candidates) > _MAX_BATCH_ITEMS or len(reviews) > _MAX_BATCH_ITEMS:
            raise ValueError("memory consolidation accepts at most 12 candidates and 12 reviews")

        candidate_ids = [item.memory_id for item in candidates]
        review_ids = [item.memory_id for item in reviews]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("duplicate candidate memory_id in batch")
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("duplicate review memory_id in batch")
        if set(candidate_ids) & set(review_ids):
            raise ValueError("a batch cannot create and review the same memory_id")
        targets = [item.supersedes_id for item in candidates if item.supersedes_id is not None]
        if len(targets) != len(set(targets)):
            raise ValueError("multiple corrections cannot supersede the same memory in one batch")
        if set(targets) & set(review_ids):
            raise ValueError("a batch cannot review a memory that it also supersedes")

        normalized = tuple(
            self._prepare_batch_candidate(connection, item, conversation_id, assessed_at_utc)
            for item in candidates
        )
        for review in reviews:
            self._validate_review(connection, review, conversation_id)

        savepoint = f"memory_batch_{uuid4().hex}"
        connection.execute(f"SAVEPOINT {savepoint}")
        try:
            created: list[str] = []
            reviewed: list[str] = []
            activated: list[str] = []
            superseded: list[str] = []
            rejected: list[str] = []
            expired: list[str] = []
            restated: list[str] = []
            for candidate in normalized:
                if self._is_restated_preference(connection, candidate, conversation_id):
                    restated.append(candidate.memory_id)
                    continue
                persisted, was_created = self._create_batch_candidate(
                    connection, candidate, session_job_id, assessed_at_utc
                )
                if was_created:
                    created.append(candidate.memory_id)
                if persisted.status == "candidate" and self._eligible_for_auto_activation(
                    connection, persisted, conversation_id, assessed_at_utc
                ):
                    old_id = self._activate_batch_candidate(
                        connection,
                        persisted,
                        conversation_id,
                        session_job_id,
                        assessed_at_utc,
                    )
                    activated.append(persisted.memory_id)
                    if old_id is not None:
                        superseded.append(old_id)

            for review in reviews:
                outcome, old_id = self._apply_review(
                    connection,
                    review,
                    conversation_id,
                    session_job_id,
                    assessed_at_utc,
                )
                reviewed.append(review.memory_id)
                if outcome == "active":
                    activated.append(review.memory_id)
                elif outcome == "rejected":
                    rejected.append(review.memory_id)
                elif outcome == "expired":
                    expired.append(review.memory_id)
                if old_id is not None:
                    superseded.append(old_id)
            result = MemoryBatchResult(
                tuple(created),
                tuple(reviewed),
                tuple(dict.fromkeys(activated)),
                tuple(dict.fromkeys(superseded)),
                tuple(rejected),
                tuple(expired),
                tuple(restated),
            )
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            return result
        except BaseException:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise

    def try_record_confirmation_presentation(
        self,
        *,
        conversation_id: str,
        memory_id: str,
        fragment_key: str,
        trigger_event_id: str,
        context_version: int,
        presented_at_utc: datetime,
    ) -> bool:
        conversation_id = _nonempty_text(conversation_id, "conversation_id")
        memory_id = _nonempty_text(memory_id, "memory_id")
        fragment_key = _nonempty_text(fragment_key, "fragment_key")
        trigger_event_id = _nonempty_text(trigger_event_id, "trigger_event_id")
        if type(context_version) is not int or context_version < 0:
            raise ValueError("context_version must be a non-negative integer")
        presented_at_utc = _utc(presented_at_utc, "presented_at_utc")
        with self.database.transaction() as connection:
            try:
                record = self._get(connection, memory_id)
            except KeyError:
                return False
            if (
                self._validate_evidence(connection, record.memory_evidence) != conversation_id
                or record.recall_scope != "confirmation"
                or record.valid_from_utc > presented_at_utc
                or (
                    record.valid_until_utc is not None
                    and record.valid_until_utc < presented_at_utc
                )
            ):
                return False
            trigger = connection.execute(
                "SELECT conversation_id, direction, actor, kind, status, occurred_at_utc, sequence "
                "FROM conversation_events WHERE event_id = ?",
                (trigger_event_id,),
            ).fetchone()
            if (
                trigger is None
                or trigger["conversation_id"] != conversation_id
                or trigger["direction"] != "inbound"
                or trigger["actor"] != "mumo"
                or trigger["kind"] != "text"
                or trigger["status"] != "received"
            ):
                return False
            latest_evidence = max(item.occurred_at_utc for item in record.memory_evidence)
            trigger_time = _stored_utc(trigger["occurred_at_utc"], "trigger occurred_at_utc")
            if trigger_time < latest_evidence or presented_at_utc < trigger_time:
                return False
            newest = connection.execute(
                "SELECT MAX(sequence) AS sequence FROM conversation_events WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if newest is None or trigger["sequence"] != newest["sequence"]:
                return False
            cursor = connection.execute(
                "SELECT context_version FROM conversation_cursors WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if cursor is None or cursor["context_version"] != context_version:
                return False
            if connection.execute(
                "SELECT 1 FROM memory_confirmation_presentations "
                "WHERE conversation_id = ? AND fragment_key = ?",
                (conversation_id, fragment_key),
            ).fetchone() is not None:
                return False
            latest = connection.execute(
                "SELECT presented_at_utc FROM memory_confirmation_presentations "
                "WHERE memory_id = ? ORDER BY presented_at_utc DESC, presentation_id DESC LIMIT 1",
                (memory_id,),
            ).fetchone()
            if latest is not None:
                last_at = _stored_utc(latest["presented_at_utc"], "presented_at_utc")
                if presented_at_utc < last_at + _CONFIRMATION_COOLDOWN:
                    return False
            presentation_id = str(
                uuid5(
                    NAMESPACE_URL,
                    "qichi:memory:confirmation-presentation:"
                    f"{conversation_id}:{fragment_key}:{memory_id}:{trigger_event_id}:"
                    f"{context_version}",
                )
            )
            try:
                connection.execute(
                    "INSERT INTO memory_confirmation_presentations "
                    "(presentation_id, conversation_id, memory_id, fragment_key, "
                    "trigger_event_id, context_version, presented_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        presentation_id,
                        conversation_id,
                        memory_id,
                        fragment_key,
                        trigger_event_id,
                        context_version,
                        presented_at_utc.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            return True

    def _list_by_statuses(
        self,
        conversation_id: str,
        at_utc: datetime,
        statuses: tuple[str, ...],
        *,
        require_current_validity: bool,
    ) -> tuple[MemoryRecord, ...]:
        conversation_id = _nonempty_text(conversation_id, "conversation_id")
        at_utc = _utc(at_utc, "at_utc")
        placeholders = ",".join("?" for _ in statuses)
        validity_clause = (
            "AND m.valid_from_utc <= ? "
            "AND (m.valid_until_utc IS NULL OR m.valid_until_utc >= ?) "
            if require_current_validity
            else ""
        )
        parameters: tuple[object, ...] = (conversation_id, *statuses)
        if require_current_validity:
            parameters += (at_utc.isoformat(), at_utc.isoformat())
        rows = self.database.connection.execute(
            "SELECT DISTINCT m.* FROM memory_records AS m "
            "JOIN memory_evidence AS me ON me.memory_id = m.memory_id "
            "JOIN conversation_events AS e ON e.event_id = me.event_id "
            f"WHERE e.conversation_id = ? AND m.status IN ({placeholders}) "
            f"{validity_clause}"
            "ORDER BY m.created_at_utc, m.memory_id",
            parameters,
        ).fetchall()
        records = tuple(self._record_from_row(self.database.connection, row) for row in rows)
        for record in records:
            if self._validate_evidence(self.database.connection, record.memory_evidence) != conversation_id:
                raise ValueError("memory evidence must belong to the queried conversation")
            if record.status == "active":
                self._validate_active_evidence(record)
        return records

    def _prepare_batch_candidate(
        self,
        connection: sqlite3.Connection,
        record: MemoryRecord,
        conversation_id: str,
        assessed_at_utc: datetime,
    ) -> MemoryRecord:
        if record.status != "candidate":
            raise ValueError("V2 candidate status must be candidate")
        if record.type not in _BATCH_TYPES:
            raise ValueError("V2 candidate type is invalid")
        if record.certainty == "unassessed" or record.assessment_reason_code is None:
            raise ValueError("V2 candidate must be assessed")
        if len(record.memory_evidence) > 4:
            raise ValueError("V2 candidate may contain at most 4 evidence items")
        self._validate_candidate_assessment(record)
        if self._validate_evidence(connection, record.memory_evidence) != conversation_id:
            raise ValueError("candidate evidence does not belong to conversation_id")
        self._validate_correction_shape(record)
        self._validate_candidate_ownership(record)
        if record.supersedes_id is not None:
            target = self._get(connection, record.supersedes_id)
            if self._validate_evidence(connection, target.memory_evidence) != conversation_id:
                raise ValueError(
                    "correction and superseded memory must belong to the same conversation"
                )
            if target.status != "active":
                raise ValueError(f"correction target must be active, not {target.status}")
        return replace(record, assessed_at_utc=assessed_at_utc)

    @staticmethod
    def _validate_candidate_assessment(record: MemoryRecord) -> None:
        if not MemoryRepository._assessment_is_consistent(
            record.certainty, record.assessment_reason_code
        ):
            raise ValueError("candidate assessment fields are inconsistent")

    @staticmethod
    def _assessment_is_consistent(certainty: str, reason: str | None) -> bool:
        allowed = {
            "unsupported": {"unsupported_or_transient"},
            "ambiguous": {"ambiguous_scope"},
            "explicit": {
                "explicit_user_statement",
                "user_correction",
                "historical_event",
            },
            "confirmed": {
                "later_user_confirmation",
                "bilateral_agreement",
                "user_correction",
            },
        }
        return reason in allowed.get(certainty, set())

    @staticmethod
    def _validate_candidate_ownership(record: MemoryRecord) -> None:
        if record.type == "self_expression":
            if not any(item.actor == "qichi" for item in record.memory_evidence):
                raise ValueError("self_expression requires qichi evidence")
        elif not any(item.actor == "mumo" for item in record.memory_evidence):
            raise ValueError(f"{record.type} requires mumo evidence")
        if record.type == "correction" and not any(
            item.actor == "mumo" and item.evidence_role == "correction"
            for item in record.memory_evidence
        ):
            raise ValueError("correction requires mumo correction evidence")
        expected_roles = _CANDIDATE_ROLE_SETS[record.type]
        actual_roles = {item.evidence_role for item in record.memory_evidence}
        if record.type == "agreement":
            if not actual_roles or not actual_roles <= expected_roles:
                raise ValueError("candidate evidence roles are inconsistent")
        elif actual_roles != expected_roles:
            raise ValueError("candidate evidence roles are inconsistent")
        for item in record.memory_evidence:
            if record.type == "agreement":
                expected_actor = "mumo" if item.evidence_role == "proposal" else "qichi"
            else:
                expected_actor = "qichi" if record.type == "self_expression" else "mumo"
            if item.actor != expected_actor:
                raise ValueError("candidate evidence role actor is inconsistent")

    def _create_batch_candidate(
        self,
        connection: sqlite3.Connection,
        record: MemoryRecord,
        session_job_id: str,
        assessed_at_utc: datetime,
    ) -> tuple[MemoryRecord, bool]:
        row = connection.execute(
            "SELECT * FROM memory_records WHERE memory_id = ?", (record.memory_id,)
        ).fetchone()
        if row is not None:
            existing = self._record_from_row(connection, row)
            if not self._same_identity(existing, record):
                raise ValueError("memory_id immutable identity conflict")
            persisted = {self._evidence_key(item): item for item in existing.memory_evidence}
            for item in record.memory_evidence:
                old = persisted.get(self._evidence_key(item))
                if old is None:
                    raise ValueError("existing memories must receive new evidence through reviews")
                if old.evidence_role != item.evidence_role:
                    raise ValueError("existing evidence role conflict")
            return existing, False
        self._insert_record(connection, record)
        persisted = self._get(connection, record.memory_id)
        self._write_audit(
            connection,
            record.memory_id,
            session_job_id,
            "create",
            None,
            persisted,
            record.assessment_reason_code,
            assessed_at_utc,
        )
        return persisted, True

    def _validate_review(
        self,
        connection: sqlite3.Connection,
        review: MemoryReview,
        conversation_id: str,
    ) -> None:
        if len(review.evidence) > 4:
            raise ValueError("review may contain at most 4 evidence items")
        target = self._get(connection, review.memory_id)
        if target.status not in {"candidate", "active"}:
            raise ValueError(f"review target cannot be {target.status}")
        if target.status == "active" and review.action == "support":
            raise ValueError("active memory cannot receive support")
        if (
            self._validate_evidence(connection, target.memory_evidence) != conversation_id
            or self._validate_evidence(connection, review.evidence) != conversation_id
        ):
            raise ValueError("review evidence and target must belong to conversation_id")
        if review.temporal_scope == "bounded" and target.valid_until_utc is None:
            raise ValueError("review cannot create a bounded scope without a valid_until_utc")
        roles = {item.evidence_role for item in review.evidence}
        actors = {item.actor for item in review.evidence}
        if review.action == "confirm":
            if (
                review.certainty != "confirmed"
                or review.assessment_reason_code != "later_user_confirmation"
                or roles != {"confirmation"}
                or actors != {"mumo"}
            ):
                raise ValueError("confirm review fields are inconsistent")
            latest_old = max(item.occurred_at_utc for item in target.memory_evidence)
            if not any(item.occurred_at_utc > latest_old for item in review.evidence):
                raise ValueError("confirm review requires later user evidence")
        elif review.action == "reject":
            if (
                review.certainty != "unsupported"
                or review.importance != 0
                or review.temporal_scope != "unclassified"
                or review.assessment_reason_code != "contradicted_by_user"
                or roles != {"counterevidence"}
                or actors != {"mumo"}
            ):
                raise ValueError("reject review fields are inconsistent")
            latest_old = max(item.occurred_at_utc for item in target.memory_evidence)
            if not any(item.occurred_at_utc > latest_old for item in review.evidence):
                raise ValueError("reject review requires later counterevidence")
        elif review.action == "expire":
            if (
                review.certainty != "unsupported"
                or
                review.importance != 0
                or review.assessment_reason_code != "expired_or_completed"
                or roles != {"counterevidence"}
                or actors != {"mumo"}
            ):
                raise ValueError("expire review fields are inconsistent")
            latest_old = max(item.occurred_at_utc for item in target.memory_evidence)
            if not any(item.occurred_at_utc > latest_old for item in review.evidence):
                raise ValueError("expire review requires later counterevidence")
        elif review.action == "support":
            allowed_roles = {
                "explicit_user_statement": {"source"},
                "bilateral_agreement": {"proposal", "acceptance"},
                "user_correction": {"correction"},
                "ambiguous_scope": {"source", "proposal", "acceptance"},
                "historical_event": {"source"},
            }
            if (
                review.assessment_reason_code not in _SUPPORT_REASONS
                or not self._assessment_is_consistent(
                    review.certainty, review.assessment_reason_code
                )
                or not roles
                or not roles <= allowed_roles[review.assessment_reason_code]
                or "platform" in actors
            ):
                raise ValueError("support review fields are inconsistent")
            expected_roles = {
                "explicit_user_statement": {"source"},
                "bilateral_agreement": {"proposal", "acceptance"},
                "user_correction": {"correction"},
                "ambiguous_scope": {"source", "proposal", "acceptance"},
                "historical_event": {"source"},
            }[review.assessment_reason_code]
            for item in review.evidence:
                if review.assessment_reason_code == "bilateral_agreement":
                    expected_actor = "mumo" if item.evidence_role == "proposal" else "qichi"
                else:
                    expected_actor = "mumo"
                if item.actor != expected_actor or item.evidence_role not in expected_roles:
                    raise ValueError("support evidence role actor is inconsistent")
        else:
            raise ValueError("review action is invalid")

    def _validate_public_active_matrix(
        self, connection: sqlite3.Connection, record: MemoryRecord, conversation_id: str
    ) -> None:
        if record.type == "self_expression":
            self._validate_active_evidence(record)
            return
        if not self._eligible_for_auto_activation(
            connection, replace(record, status="candidate"), conversation_id,
            record.assessed_at_utc or record.created_at_utc,
        ):
            raise ValueError("memory does not satisfy active matrix")

    def _activate_with_audit(
        self, connection: sqlite3.Connection, memory_id: str, conversation_id: str,
        session_job_id: str, assessed_at_utc: datetime, *, allow_manual_self_expression: bool = False,
    ) -> None:
        before = self._get(connection, memory_id)
        if before.status != "candidate":
            self._raise_transition_error(connection, memory_id, "activate")
        cursor = connection.execute(
            "UPDATE memory_records SET status = 'active' WHERE memory_id = ? AND status = 'candidate'",
            (memory_id,),
        )
        if cursor.rowcount != 1:
            self._raise_transition_error(connection, memory_id, "activate")
        after = self._get(connection, memory_id)
        self._validate_active_evidence(after)
        self._write_audit(
            connection, memory_id, session_job_id, "activate", before, after,
            "legacy_manual_review" if session_job_id == _LEGACY_API_SESSION_JOB
            else after.assessment_reason_code,
            assessed_at_utc,
        )
        if after.supersedes_id is not None:
            old = self._get(connection, after.supersedes_id)
            if old.status != "active":
                raise ValueError("correction target is no longer active")
            connection.execute(
                "UPDATE memory_records SET status = 'superseded' WHERE memory_id = ? AND status = 'active'",
                (old.memory_id,),
            )
            self._write_audit(connection, old.memory_id, session_job_id, "supersede", old,
                               self._get(connection, old.memory_id), "user_correction", assessed_at_utc)
    def _apply_review(
        self,
        connection: sqlite3.Connection,
        review: MemoryReview,
        conversation_id: str,
        session_job_id: str,
        assessed_at_utc: datetime,
    ) -> tuple[str | None, str | None]:
        before = self._get(connection, review.memory_id)
        if before.status not in {"candidate", "active"}:
            raise ValueError(f"review target cannot be {before.status}")
        self._add_review_evidence(connection, review.evidence)
        status = (
            "rejected"
            if review.action == "reject"
            else "expired"
            if review.action == "expire"
            else before.status
        )
        cursor = connection.execute(
            "UPDATE memory_records SET status = ?, certainty = ?, importance = ?, "
            "temporal_scope = ?, assessment_reason_code = ?, assessed_at_utc = ?, "
            "privacy_class = ?, recall_policy = ? "
            "WHERE memory_id = ? AND status = ?",
            (
                status,
                review.certainty,
                review.importance,
                review.temporal_scope,
                review.assessment_reason_code,
                assessed_at_utc.isoformat(),
                (
                    before.privacy_class
                    if before.privacy_class != "ordinary"
                    and review.privacy_class == "ordinary"
                    and review.recall_policy == "daily_safe"
                    else review.privacy_class
                ),
                (
                    before.recall_policy
                    if before.privacy_class != "ordinary"
                    and review.privacy_class == "ordinary"
                    and review.recall_policy == "daily_safe"
                    else review.recall_policy
                ),
                review.memory_id,
                before.status,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("review target changed concurrently")
        after = self._get(connection, review.memory_id)
        if after.status == "active":
            self._validate_active_evidence(after)
        self._write_audit(
            connection,
            review.memory_id,
            session_job_id,
            review.action,
            before,
            after,
            review.assessment_reason_code,
            assessed_at_utc,
        )
        if after.status == "candidate" and self._eligible_for_auto_activation(
            connection, after, conversation_id, assessed_at_utc
        ):
            old_id = self._activate_batch_candidate(
                connection, after, conversation_id, session_job_id, assessed_at_utc
            )
            return "active", old_id
        if review.action == "reject":
            return "rejected", None
        if review.action == "expire":
            return "expired", None
        return None, None

    def _add_review_evidence(
        self, connection: sqlite3.Connection, evidence_items: tuple[MemoryEvidence, ...]
    ) -> None:
        for evidence in evidence_items:
            row = connection.execute(
                "SELECT actor, occurred_at_utc, evidence_role FROM memory_evidence "
                "WHERE memory_id = ? AND event_id = ? AND exact_quote = ?",
                (evidence.memory_id, evidence.event_id, evidence.exact_quote),
            ).fetchone()
            if row is not None:
                if (
                    row["actor"] != evidence.actor
                    or _stored_utc(row["occurred_at_utc"], "evidence occurred_at_utc")
                    != evidence.occurred_at_utc
                    or row["evidence_role"] != evidence.evidence_role
                ):
                    raise ValueError("existing evidence identity conflict")
                continue
            connection.execute(
                "INSERT INTO memory_evidence "
                "(memory_id, event_id, actor, exact_quote, occurred_at_utc, evidence_role) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evidence.memory_id,
                    evidence.event_id,
                    evidence.actor,
                    evidence.exact_quote,
                    evidence.occurred_at_utc.isoformat(),
                    evidence.evidence_role,
                ),
            )

    def _eligible_for_auto_activation(
        self,
        connection: sqlite3.Connection,
        record: MemoryRecord,
        conversation_id: str,
        assessed_at_utc: datetime,
    ) -> bool:
        if (
            record.status != "candidate"
            or record.modality != "explicit_statement"
            or record.certainty not in {"explicit", "confirmed"}
            or record.importance < 1
            or record.temporal_scope == "unclassified"
            or (
                record.valid_until_utc is not None
                and record.valid_until_utc < assessed_at_utc
            )
            or any(item.evidence_role == "counterevidence" for item in record.memory_evidence)
            or self._validate_evidence(connection, record.memory_evidence) != conversation_id
        ):
            return False
        if record.type == "preference":
            return record.temporal_scope in {"ongoing", "bounded"} and any(
                item.actor == "mumo" for item in record.memory_evidence
            )
        if record.type == "episode":
            return record.temporal_scope == "historical" and any(
                item.actor == "mumo" for item in record.memory_evidence
            )
        if record.type == "agreement":
            return record.certainty == "confirmed" and self._has_bilateral_agreement(
                record.memory_evidence
            )
        if record.type == "correction":
            if not any(
                item.actor == "mumo" and item.evidence_role == "correction"
                for item in record.memory_evidence
            ) or record.supersedes_id is None:
                return False
            try:
                target = self._get(connection, record.supersedes_id)
            except KeyError:
                return False
            return target.status == "active"
        return False

    @staticmethod
    def _has_bilateral_agreement(evidence: tuple[MemoryEvidence, ...]) -> bool:
        proposals = [item for item in evidence if item.evidence_role == "proposal"]
        acceptances = [item for item in evidence if item.evidence_role == "acceptance"]
        if any(item.actor == "platform" for item in (*proposals, *acceptances)):
            return False
        return any(
            proposal.event_id != acceptance.event_id
            and proposal.actor != acceptance.actor
            and {proposal.actor, acceptance.actor} == {"mumo", "qichi"}
            for proposal in proposals
            for acceptance in acceptances
        )

    def _activate_batch_candidate(
        self,
        connection: sqlite3.Connection,
        record: MemoryRecord,
        conversation_id: str,
        session_job_id: str,
        assessed_at_utc: datetime,
    ) -> str | None:
        before = self._get(connection, record.memory_id)
        cursor = connection.execute(
            "UPDATE memory_records SET status = 'active' "
            "WHERE memory_id = ? AND status = 'candidate'",
            (record.memory_id,),
        )
        if cursor.rowcount != 1:
            self._raise_transition_error(connection, record.memory_id, "activate")
        after = self._get(connection, record.memory_id)
        self._validate_active_evidence(after)
        self._write_audit(
            connection,
            record.memory_id,
            session_job_id,
            "activate",
            before,
            after,
            after.assessment_reason_code,
            assessed_at_utc,
        )
        if after.supersedes_id is None:
            return None
        old = self._get(connection, after.supersedes_id)
        if (
            self._validate_evidence(connection, old.memory_evidence) != conversation_id
            or old.status != "active"
        ):
            raise ValueError("correction target is no longer active in this conversation")
        cursor = connection.execute(
            "UPDATE memory_records SET status = 'superseded' "
            "WHERE memory_id = ? AND status = 'active'",
            (old.memory_id,),
        )
        if cursor.rowcount != 1:
            raise ValueError("superseded memory changed concurrently")
        superseded = self._get(connection, old.memory_id)
        self._write_audit(
            connection,
            old.memory_id,
            session_job_id,
            "supersede",
            old,
            superseded,
            "user_correction",
            assessed_at_utc,
        )
        return old.memory_id

    def _write_audit(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
        session_job_id: str,
        action: str,
        before: MemoryRecord | None,
        after: MemoryRecord,
        reason: str | None,
        occurred_at_utc: datetime,
    ) -> None:
        if reason is None:
            raise ValueError("audited memory action requires an assessment reason")
        audit_id = str(
            uuid5(
                NAMESPACE_URL,
                "qichi:memory:audit:"
                f"{session_job_id}:{memory_id}:{action}:"
                f"{before.status if before else ''}:{after.status}:"
                f"{after.certainty}:{after.importance}:{after.temporal_scope}:"
                f"{occurred_at_utc.isoformat()}",
            )
        )
        values = (
            audit_id,
            memory_id,
            session_job_id,
            action,
            before.status if before is not None else None,
            after.status,
            before.certainty if before is not None else None,
            after.certainty,
            before.importance if before is not None else None,
            after.importance,
            before.temporal_scope if before is not None else None,
            after.temporal_scope,
            reason,
            occurred_at_utc.isoformat(),
        )
        existing = connection.execute(
            "SELECT audit_event_id, memory_id, session_job_id, action, before_status, "
            "after_status, before_certainty, after_certainty, before_importance, "
            "after_importance, before_temporal_scope, after_temporal_scope, "
            "assessment_reason_code, occurred_at_utc FROM memory_audit_events "
            "WHERE audit_event_id = ?",
            (audit_id,),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != values:
                raise ValueError("memory audit identity conflict")
            return
        connection.execute(
            "INSERT INTO memory_audit_events "
            "(audit_event_id, memory_id, session_job_id, action, before_status, "
            "after_status, before_certainty, after_certainty, before_importance, "
            "after_importance, before_temporal_scope, after_temporal_scope, "
            "assessment_reason_code, occurred_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

    def _insert_record(self, connection: sqlite3.Connection, record: MemoryRecord) -> None:
        connection.execute(
            "INSERT INTO memory_records "
            "(memory_id, type, normalized_fact, modality, status, valid_from_utc, "
            "valid_until_utc, supersedes_id, created_at_utc, certainty, importance, "
            "temporal_scope, assessment_reason_code, assessed_at_utc, privacy_class, "
            "recall_policy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.memory_id,
                record.type,
                record.normalized_fact,
                record.modality,
                record.status,
                record.valid_from_utc.isoformat(),
                record.valid_until_utc.isoformat() if record.valid_until_utc else None,
                record.supersedes_id,
                record.created_at_utc.isoformat(),
                record.certainty,
                record.importance,
                record.temporal_scope,
                record.assessment_reason_code,
                record.assessed_at_utc.isoformat() if record.assessed_at_utc else None,
                record.privacy_class,
                record.recall_policy,
            ),
        )
        for evidence in record.memory_evidence:
            connection.execute(
                "INSERT INTO memory_evidence "
                "(memory_id, event_id, actor, exact_quote, occurred_at_utc, evidence_role) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evidence.memory_id,
                    evidence.event_id,
                    evidence.actor,
                    evidence.exact_quote,
                    evidence.occurred_at_utc.isoformat(),
                    evidence.evidence_role,
                ),
            )

    def _transition(
        self,
        memory_id: str,
        source: str,
        target: str,
        action: str,
        *,
        reason: str = "legacy_manual_review",
    ) -> MemoryRecord:
        memory_id = _nonempty_text(memory_id, "memory_id")
        with self.database.transaction() as connection:
            record = self._get(connection, memory_id)
            if record.status != source:
                raise ValueError(f"cannot {action} from {record.status}")
            cursor = connection.execute(
                "UPDATE memory_records SET status = ? WHERE memory_id = ? AND status = ?",
                (target, memory_id, source),
            )
            if cursor.rowcount != 1:
                self._raise_transition_error(connection, memory_id, action)
            after = self._get(connection, memory_id)
            self._write_audit(
                connection,
                memory_id,
                _LEGACY_API_SESSION_JOB,
                action,
                record,
                after,
                reason,
                record.assessed_at_utc or record.created_at_utc,
            )
            return after

    def _activate_correction(
        self, connection: sqlite3.Connection, memory_id: str, conversation_id: str
    ) -> None:
        correction = self._get(connection, memory_id)
        if correction.supersedes_id is None:
            raise ValueError("correction requires supersedes_id")
        target = self._get(connection, correction.supersedes_id)
        if self._validate_evidence(connection, target.memory_evidence) != conversation_id:
            raise ValueError(
                "correction and superseded memory must belong to the same conversation"
            )
        if target.status != "active":
            raise ValueError(f"cannot supersede memory from {target.status}")
        updated_new = connection.execute(
            "UPDATE memory_records SET status = 'active' "
            "WHERE memory_id = ? AND status = 'candidate'",
            (memory_id,),
        )
        if updated_new.rowcount != 1:
            self._raise_transition_error(connection, memory_id, "activate")
        updated_old = connection.execute(
            "UPDATE memory_records SET status = 'superseded' "
            "WHERE memory_id = ? AND status = 'active'",
            (target.memory_id,),
        )
        if updated_old.rowcount != 1:
            raise ValueError("superseded memory changed concurrently")

    @staticmethod
    def _validate_correction_shape(record: MemoryRecord) -> None:
        if record.supersedes_id is not None and record.type != "correction":
            raise ValueError("only correction memory may set supersedes_id")
        if record.type == "correction" and record.supersedes_id is None:
            raise ValueError("correction requires supersedes_id")
        if record.supersedes_id == record.memory_id:
            raise ValueError("memory cannot supersede itself")

    @staticmethod
    def _validate_active_evidence(record: MemoryRecord) -> None:
        required_actor = "qichi" if record.type == "self_expression" else "mumo"
        if not any(item.actor == required_actor for item in record.memory_evidence):
            raise ValueError(f"active {record.type} memory requires {required_actor} evidence")

    def _validate_evidence(
        self, connection: sqlite3.Connection, evidence_items: tuple[MemoryEvidence, ...]
    ) -> str:
        conversations: set[str] = set()
        identities: dict[tuple[str, str], str] = {}
        for evidence in evidence_items:
            if not evidence.exact_quote.strip():
                raise ValueError("evidence exact quote must contain a non-whitespace character")
            identity = (evidence.event_id, evidence.exact_quote)
            if identity in identities:
                if identities[identity] != evidence.evidence_role:
                    raise ValueError("duplicate evidence has conflicting roles")
                raise ValueError("duplicate evidence")
            identities[identity] = evidence.evidence_role
            row = connection.execute(
                "SELECT conversation_id, actor, direction, kind, status, text, occurred_at_utc, metadata_json "
                "FROM conversation_events WHERE event_id = ?",
                (evidence.event_id,),
            ).fetchone()
            if row is None:
                raise ValueError("evidence event does not exist")
            reliable = (
                row["actor"] == "mumo" and row["direction"] == "inbound"
                and row["kind"] == "text" and row["status"] in {"received", "failed"}
            ) or (
                row["actor"] == "qichi" and row["direction"] == "outbound"
                and row["kind"] == "text" and row["status"] == "sent"
                and _outbound_source(row["metadata_json"])
            )
            if not reliable:
                raise ValueError("reliable evidence is required")
            if row["actor"] != evidence.actor:
                raise ValueError("evidence actor does not match source event")
            if _stored_utc(row["occurred_at_utc"], "event occurred_at_utc") != evidence.occurred_at_utc:
                raise ValueError("evidence time does not match source event")
            text = row["text"]
            if not isinstance(text, str) or evidence.exact_quote not in text:
                raise ValueError("evidence exact quote is absent from source event")
            conversations.add(_nonempty_text(row["conversation_id"], "event conversation_id"))
        if len(conversations) != 1:
            raise ValueError("memory evidence must belong to the same conversation")
        return conversations.pop()

    def _get(self, connection: sqlite3.Connection, memory_id: str) -> MemoryRecord:
        row = connection.execute(
            "SELECT * FROM memory_records WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._record_from_row(connection, row)

    def _record_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> MemoryRecord:
        memory_id = _nonempty_text(row["memory_id"], "persisted memory_id")
        status = row["status"]
        if status not in _STATUSES:
            raise ValueError("persisted memory status is invalid")
        evidence_rows = connection.execute(
            "SELECT me.memory_id, me.event_id, me.actor, me.exact_quote, "
            "me.occurred_at_utc, me.evidence_role FROM memory_evidence AS me "
            "JOIN conversation_events AS e ON e.event_id = me.event_id "
            "WHERE me.memory_id = ? "
            "ORDER BY me.occurred_at_utc, e.sequence, e.event_id, me.exact_quote",
            (memory_id,),
        ).fetchall()
        if not evidence_rows:
            raise ValueError("persisted memory has no evidence")
        evidence = tuple(
            MemoryEvidence(
                _nonempty_text(item["memory_id"], "persisted evidence memory_id"),
                _nonempty_text(item["event_id"], "persisted evidence event_id"),
                item["actor"],
                _nonempty_text(item["exact_quote"], "persisted evidence exact_quote"),
                _stored_utc(item["occurred_at_utc"], "evidence occurred_at_utc"),
                item["evidence_role"],
            )
            for item in evidence_rows
        )
        valid_until = row["valid_until_utc"]
        assessed_at = row["assessed_at_utc"]
        record = MemoryRecord(
            memory_id,
            _nonempty_text(row["type"], "persisted memory type"),
            _nonempty_text(row["normalized_fact"], "persisted normalized_fact"),
            _nonempty_text(row["modality"], "persisted modality"),
            status,
            _stored_utc(row["valid_from_utc"], "valid_from_utc"),
            _stored_utc(valid_until, "valid_until_utc") if valid_until is not None else None,
            row["supersedes_id"],
            _stored_utc(row["created_at_utc"], "created_at_utc"),
            evidence,
            row["certainty"],
            row["importance"],
            row["temporal_scope"],
            row["assessment_reason_code"],
            _stored_utc(assessed_at, "assessed_at_utc") if assessed_at is not None else None,
            row["privacy_class"] if "privacy_class" in row.keys() else "ordinary",
            row["recall_policy"] if "recall_policy" in row.keys() else "daily_safe",
        )
        self._validate_correction_shape(record)
        self._validate_evidence(connection, evidence)
        if record.status == "active":
            self._validate_active_evidence(record)
        return record

    @classmethod
    def _same_identity(cls, existing: MemoryRecord, requested: MemoryRecord) -> bool:
        left = (
            existing.memory_id,
            existing.type,
            existing.normalized_fact,
            existing.modality,
            existing.valid_from_utc,
            existing.valid_until_utc,
            existing.supersedes_id,
        )
        right = (
            requested.memory_id,
            requested.type,
            requested.normalized_fact,
            requested.modality,
            requested.valid_from_utc,
            requested.valid_until_utc,
            requested.supersedes_id,
        )
        return left == right and cls._identity_evidence(
            existing.memory_evidence
        ) == cls._identity_evidence(requested.memory_evidence)

    @staticmethod
    def _identity_evidence(
        evidence_items: tuple[MemoryEvidence, ...]
    ) -> tuple[str, str, str]:
        anchors = [
            item
            for item in evidence_items
            if item.evidence_role in {"source", "proposal", "correction"}
        ] or list(evidence_items)
        anchor = min(
            anchors,
            key=lambda item: (item.occurred_at_utc, item.event_id, item.exact_quote),
        )
        return anchor.event_id, anchor.actor, anchor.exact_quote

    @staticmethod
    def _evidence_key(evidence: MemoryEvidence) -> tuple[str, str]:
        return evidence.event_id, evidence.exact_quote

    @staticmethod
    def _raise_transition_error(
        connection: sqlite3.Connection, memory_id: str, action: str
    ) -> None:
        row = connection.execute(
            "SELECT status FROM memory_records WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        raise ValueError(f"cannot {action} from {row['status']}")
