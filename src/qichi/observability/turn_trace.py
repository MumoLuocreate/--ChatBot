"""Append-only turn traces containing evidence identities, never hidden reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from qichi.domain.events import JSONValue, _freeze_json, _thaw_json
from qichi.storage.database import Database


_SOURCES = frozenset({"dialogue", "interaction", "initiative"})
_PHASES = frozenset({"received", "context", "generation", "delivery", "failure", "cancelled"})
_FORBIDDEN_DETAIL_KEYS = frozenset({
    "api_key", "access_token", "authorization", "prompt", "messages",
    "completion", "reasoning", "chain_of_thought", "raw_payload", "text",
})


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _validate_detail_keys(value: object, path: str = "details") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key.casefold() in _FORBIDDEN_DETAIL_KEYS:
                raise ValueError("turn trace details contain a sensitive field")
            _validate_detail_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_detail_keys(child, f"{path}[]")


@dataclass(frozen=True, slots=True)
class TurnTraceEvent:
    trace_event_id: str
    trace_id: str
    conversation_id: str
    trigger_event_id: str
    source: str
    phase: str
    occurred_at_utc: datetime
    details: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        for field in ("trace_event_id", "trace_id", "conversation_id", "trigger_event_id"):
            _text(getattr(self, field), field)
        if self.source not in _SOURCES:
            raise ValueError("trace source is invalid")
        if self.phase not in _PHASES:
            raise ValueError("trace phase is invalid")
        object.__setattr__(self, "occurred_at_utc", _utc(self.occurred_at_utc, "occurred_at_utc"))
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        _validate_detail_keys(self.details)
        object.__setattr__(self, "details", _freeze_json(self.details, "details"))


class TurnTraceRepository:
    def __init__(self, database: Database) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self.database = database

    @staticmethod
    def trace_id_for(trigger_event_id: str) -> str:
        _text(trigger_event_id, "trigger_event_id")
        return str(uuid5(NAMESPACE_URL, f"qichi:turn:{trigger_event_id}"))

    @staticmethod
    def trace_event_id_for(trace_id: str, phase: str) -> str:
        _text(trace_id, "trace_id")
        if phase not in _PHASES:
            raise ValueError("trace phase is invalid")
        return str(uuid5(NAMESPACE_URL, f"qichi:turn-event:{trace_id}:{phase}"))

    def append_once(
        self,
        *,
        trigger_event_id: str,
        conversation_id: str,
        source: str,
        phase: str,
        occurred_at_utc: datetime,
        details: Mapping[str, JSONValue] | None = None,
    ) -> TurnTraceEvent:
        trace_id = self.trace_id_for(trigger_event_id)
        candidate = TurnTraceEvent(
            trace_event_id=self.trace_event_id_for(trace_id, phase),
            trace_id=trace_id,
            conversation_id=conversation_id,
            trigger_event_id=trigger_event_id,
            source=source,
            phase=phase,
            occurred_at_utc=occurred_at_utc,
            details={} if details is None else details,
        )
        serialized = json.dumps(
            _thaw_json(candidate.details), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM turn_trace_events WHERE trace_id=? AND phase=?",
                (trace_id, phase),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO turn_trace_events(trace_event_id,trace_id,conversation_id,"
                    "trigger_event_id,source,phase,occurred_at_utc,details_json) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        candidate.trace_event_id, candidate.trace_id, candidate.conversation_id,
                        candidate.trigger_event_id, candidate.source, candidate.phase,
                        candidate.occurred_at_utc.isoformat(), serialized,
                    ),
                )
                return candidate
        restored = self.get(trace_id, phase)
        if restored != candidate:
            raise ValueError("turn trace phase identity conflict")
        return restored

    def get(self, trace_id: str, phase: str) -> TurnTraceEvent:
        row = self.database.connection.execute(
            "SELECT * FROM turn_trace_events WHERE trace_id=? AND phase=?", (trace_id, phase)
        ).fetchone()
        if row is None:
            raise KeyError((trace_id, phase))
        return TurnTraceEvent(
            trace_event_id=row["trace_event_id"], trace_id=row["trace_id"],
            conversation_id=row["conversation_id"], trigger_event_id=row["trigger_event_id"],
            source=row["source"], phase=row["phase"],
            occurred_at_utc=datetime.fromisoformat(row["occurred_at_utc"]),
            details=json.loads(row["details_json"]),
        )

    def list_for_trace(self, trace_id: str) -> tuple[TurnTraceEvent, ...]:
        rows = self.database.connection.execute(
            "SELECT phase FROM turn_trace_events WHERE trace_id=? "
            "ORDER BY occurred_at_utc,trace_event_id", (trace_id,)
        ).fetchall()
        return tuple(self.get(trace_id, row["phase"]) for row in rows)
