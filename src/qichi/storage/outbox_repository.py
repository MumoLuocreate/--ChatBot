from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Mapping

from .database import Database


# 2026-09-14：record = QQ 语音段。与其它动作一样只做幂等与未知态收口，本层不判断语义。
ACTION_KINDS = frozenset({"text", "face", "reaction", "poke", "record"})
TERMINAL_STATUSES = frozenset({"sent", "unknown", "failed"})
OUTBOX_STATUSES = frozenset({"pending", "dispatched", "sent", "unknown", "failed"})


def _require_nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _as_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _stored_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be stored as text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be stored in UTC")
    return parsed


def _freeze_json(value: object, field: str = "JSON value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain finite floats")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} keys must be strings")
            frozen[key] = _freeze_json(child, field)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child, field) for child in value)
    raise TypeError(f"{field} must be JSON-safe")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True)
class OutboxRecord:
    operation_key: str
    event_id: str
    payload: Mapping[str, object]
    status: str
    attempt_count: int
    response: object | None
    created_at_utc: datetime
    updated_at_utc: datetime


class OutboxRepository:
    def __init__(self, database: Database):
        self.database = database

    def create_intent(
        self,
        operation_key: str,
        event_id: str,
        payload: Mapping[str, object],
        created_at_utc: datetime,
    ) -> OutboxRecord:
        operation_key = _require_nonempty_text(operation_key, "operation_key")
        event_id = _require_nonempty_text(event_id, "event_id")
        created_at_utc = _as_utc(created_at_utc, "created_at_utc")
        frozen_payload = _freeze_json(payload, "payload")
        if not isinstance(frozen_payload, Mapping):
            raise TypeError("payload must be a mapping")
        action_kind = frozen_payload.get("action_kind")
        if not isinstance(action_kind, str) or action_kind not in ACTION_KINDS:
            raise ValueError("payload action_kind is invalid")
        if action_kind == "reaction" and frozen_payload.get("set") is not True:
            raise ValueError("reaction payload requires set=true")
        payload_json = json.dumps(_thaw_json(frozen_payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM outbox WHERE operation_key = ?", (operation_key,)
            ).fetchone()
            if existing is not None:
                if existing["event_id"] != event_id or existing["payload_json"] != payload_json:
                    raise ValueError("operation_key immutable intent identity conflict")
                return self._record_from_row(existing)
            connection.execute(
                "INSERT INTO outbox (operation_key, event_id, payload_json, status, attempt_count, "
                "response_json, created_at_utc, updated_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    operation_key, event_id, payload_json, "pending", 0, None,
                    created_at_utc.isoformat(), created_at_utc.isoformat(),
                ),
            )
            return self.get(operation_key)

    def get(self, operation_key: str) -> OutboxRecord:
        operation_key = _require_nonempty_text(operation_key, "operation_key")
        row = self.database.connection.execute(
            "SELECT * FROM outbox WHERE operation_key = ?", (operation_key,)
        ).fetchone()
        if row is None:
            raise KeyError(operation_key)
        return self._record_from_row(row)

    def begin_dispatch(self, operation_key: str, dispatched_at_utc: datetime) -> OutboxRecord:
        operation_key = _require_nonempty_text(operation_key, "operation_key")
        dispatched_at_utc = _as_utc(dispatched_at_utc, "dispatched_at_utc")
        with self.database.transaction() as connection:
            current = self._require_current(connection, operation_key)
            self._require_not_before(dispatched_at_utc, current.updated_at_utc)
            cursor = connection.execute(
                "UPDATE outbox SET status = ?, attempt_count = attempt_count + 1, updated_at_utc = ? "
                "WHERE operation_key = ? AND status = ?",
                ("dispatched", dispatched_at_utc.isoformat(), operation_key, "pending"),
            )
            if cursor.rowcount != 1:
                self._raise_transition_error(connection, operation_key, "dispatch")
            return self.get(operation_key)

    def complete_dispatch(
        self,
        operation_key: str,
        status: str,
        response: object | None,
        completed_at_utc: datetime,
    ) -> OutboxRecord:
        operation_key = _require_nonempty_text(operation_key, "operation_key")
        if status not in TERMINAL_STATUSES:
            raise ValueError("status must be sent, unknown, or failed")
        completed_at_utc = _as_utc(completed_at_utc, "completed_at_utc")
        frozen_response = _freeze_json(response, "response")
        response_json = json.dumps(_thaw_json(frozen_response), ensure_ascii=False, sort_keys=True, separators=(",", ":")) if response is not None else None
        with self.database.transaction() as connection:
            current = self._require_current(connection, operation_key)
            self._require_not_before(completed_at_utc, current.updated_at_utc)
            cursor = connection.execute(
                "UPDATE outbox SET status = ?, response_json = ?, updated_at_utc = ? "
                "WHERE operation_key = ? AND status = ?",
                (status, response_json, completed_at_utc.isoformat(), operation_key, "dispatched"),
            )
            if cursor.rowcount != 1:
                self._raise_transition_error(connection, operation_key, f"complete as {status}")
            return self.get(operation_key)

    def recover_idempotent_reaction(
        self, operation_key: str, recovered_at_utc: datetime
    ) -> OutboxRecord:
        operation_key = _require_nonempty_text(operation_key, "operation_key")
        recovered_at_utc = _as_utc(recovered_at_utc, "recovered_at_utc")
        with self.database.transaction() as connection:
            current = self._require_current(connection, operation_key)
            self._require_not_before(recovered_at_utc, current.updated_at_utc)
            if current.status != "unknown" or current.payload.get("action_kind") != "reaction" or current.payload.get("set") is not True:
                raise ValueError("only unknown reaction set=true can recover")
            connection.execute(
                "UPDATE outbox SET status = ?, updated_at_utc = ? WHERE operation_key = ?",
                ("pending", recovered_at_utc.isoformat(), operation_key),
            )
            return self.get(operation_key)

    def recoverable_pending(self) -> tuple[OutboxRecord, ...]:
        rows = self.database.connection.execute(
            "SELECT * FROM outbox ORDER BY created_at_utc, operation_key"
        ).fetchall()
        records = tuple(self._record_from_row(row) for row in rows)
        return tuple(record for record in records if record.status == "pending")

    def outstanding(self) -> tuple[OutboxRecord, ...]:
        rows = self.database.connection.execute(
            "SELECT * FROM outbox ORDER BY created_at_utc, operation_key"
        ).fetchall()
        records = tuple(self._record_from_row(row) for row in rows)
        return tuple(record for record in records if record.status in {"pending", "dispatched", "unknown"})

    def _require_current(self, connection: sqlite3.Connection, operation_key: str) -> OutboxRecord:
        row = connection.execute("SELECT * FROM outbox WHERE operation_key = ?", (operation_key,)).fetchone()
        if row is None:
            raise KeyError(operation_key)
        return self._record_from_row(row)

    def _require_not_before(self, value: datetime, previous: datetime) -> None:
        if value < previous:
            raise ValueError("timestamp must not move backward")

    def _raise_transition_error(self, connection: sqlite3.Connection, operation_key: str, action: str) -> None:
        row = connection.execute(
            "SELECT status FROM outbox WHERE operation_key = ?", (operation_key,)
        ).fetchone()
        if row is None:
            raise KeyError(operation_key)
        raise ValueError(f"cannot {action} from {row['status']}")

    def _record_from_row(self, row: sqlite3.Row) -> OutboxRecord:
        operation_key = _require_nonempty_text(row["operation_key"], "persisted operation_key")
        event_id = _require_nonempty_text(row["event_id"], "persisted event_id")
        payload = _freeze_json(json.loads(row["payload_json"]), "payload")
        response = _freeze_json(json.loads(row["response_json"]), "response") if row["response_json"] is not None else None
        if not isinstance(payload, Mapping):
            raise ValueError("persisted payload must be a mapping")
        action_kind = payload.get("action_kind")
        if not isinstance(action_kind, str) or action_kind not in ACTION_KINDS:
            raise ValueError("persisted payload action_kind is invalid")
        if action_kind == "reaction" and payload.get("set") is not True:
            raise ValueError("persisted reaction payload requires set=true")
        if row["status"] not in OUTBOX_STATUSES:
            raise ValueError("persisted outbox status is invalid")
        if type(row["attempt_count"]) is not int or row["attempt_count"] < 0:
            raise ValueError("persisted attempt_count is invalid")
        if row["status"] != "pending" and row["attempt_count"] < 1:
            raise ValueError("persisted terminal or dispatched status requires an attempt")
        if (
            row["status"] == "pending"
            and row["attempt_count"] > 0
            and not (action_kind == "reaction" and payload.get("set") is True)
        ):
            raise ValueError("persisted pending action with prior attempt is invalid")
        created_at_utc = _stored_utc(row["created_at_utc"], "created_at_utc")
        updated_at_utc = _stored_utc(row["updated_at_utc"], "updated_at_utc")
        if updated_at_utc < created_at_utc:
            raise ValueError("persisted timestamps move backward")
        return OutboxRecord(
            operation_key=operation_key, event_id=event_id, payload=payload,
            status=row["status"], attempt_count=row["attempt_count"], response=response,
            created_at_utc=created_at_utc, updated_at_utc=updated_at_utc,
        )
