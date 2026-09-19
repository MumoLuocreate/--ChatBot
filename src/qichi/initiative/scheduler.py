from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from qichi.domain.dialogue import DialogueSkip
from qichi.storage.database import Database
from qichi.transport.sender import Sender

from .policy import InitiativePolicy


_STATUSES = frozenset({"claimed", "sent", "skipped", "failed", "unknown", "cancelled"})
_STATE_FIELDS = frozenset(
    {
        "slot",
        "status",
        "owner",
        "claimed_at",
        "activity",
        "context_version",
        "claim_id",
        "outbound_event_id",
        "unanswered_attempts",
        "completed_at",
        "next_due_at",
        "failure_category",
    }
)
_CLAIM_FIELDS = frozenset(
    {
        "slot",
        "status",
        "owner",
        "claimed_at",
        "activity",
        "context_version",
        "claim_id",
        "outbound_event_id",
        "unanswered_attempts",
    }
)

_FAILURE_CATEGORIES = frozenset({"recovered_orphan", "generation_failed", "delivery_failed", "delivery_unknown"})
_CONTROL_FIELDS = frozenset({"paused", "deferred_until"})


def _utc(value: datetime, field: str = "clock") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be an aware datetime")
    return value.astimezone(timezone.utc)


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"initiative state {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"initiative state {field} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"initiative state {field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _latest_due_slot(activity: datetime, now: datetime, interval: timedelta) -> datetime | None:
    elapsed = now - activity
    if elapsed < interval:
        return None
    return activity + (elapsed // interval) * interval


def decode_initiative_state(raw: object) -> dict[str, Any]:
    """Decode the exact durable initiative-state contract without side effects."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("initiative state JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("initiative state must be an object")
    fields = set(value)
    if fields - _STATE_FIELDS:
        raise ValueError("initiative state has unknown fields")
    status = value.get("status")
    if status not in _STATUSES:
        raise ValueError("initiative state status is invalid")
    expected_fields = _CLAIM_FIELDS
    if status == "unknown":
        expected_fields |= {"completed_at"}
    elif status != "claimed":
        expected_fields |= {"completed_at", "next_due_at"}
    if fields != expected_fields and fields != expected_fields | {"failure_category"}:
        raise ValueError("initiative state fields do not match status")

    activity = _parse_utc(value.get("activity"), "activity")
    slot = _parse_utc(value.get("slot"), "slot")
    claimed_at = _parse_utc(value.get("claimed_at"), "claimed_at")
    if not activity <= slot <= claimed_at:
        raise ValueError("initiative state claim times are out of order")
    completed = (
        None
        if value.get("completed_at") is None
        else _parse_utc(value["completed_at"], "completed_at")
    )
    next_due = (
        None
        if value.get("next_due_at") is None
        else _parse_utc(value["next_due_at"], "next_due_at")
    )
    if type(value.get("context_version")) is not int or value["context_version"] < 0:
        raise ValueError("initiative state context_version is invalid")
    if (
        type(value.get("unanswered_attempts")) is not int
        or value["unanswered_attempts"] < 0
    ):
        raise ValueError("initiative state unanswered_attempts is invalid")
    failure_category = value.get("failure_category")
    if failure_category is not None and failure_category not in _FAILURE_CATEGORIES:
        raise ValueError("initiative state failure category is invalid")
    for field in ("owner", "claim_id"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"initiative state {field} is invalid")
    outbound = value.get("outbound_event_id")
    if not isinstance(outbound, str) or not outbound:
        raise ValueError("initiative state outbound_event_id is invalid")
    if status == "claimed":
        if completed is not None or next_due is not None:
            raise ValueError("claimed initiative state must not be completed")
    elif status == "unknown":
        if completed is None or next_due is not None:
            raise ValueError("unknown initiative state must await reconciliation")
    else:
        if completed is None or next_due is None or next_due <= completed:
            raise ValueError("terminal initiative state timing is invalid")
    if completed is not None and completed < claimed_at:
        raise ValueError("initiative completion precedes its claim")
    return value


class InitiativeScheduler:
    """Persist and execute at most one initiative attempt for the current idle slot."""

    def __init__(self, database: Database, application: Any, policy: InitiativePolicy, *, clock=None):
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        if not callable(getattr(application, "generate_initiative", None)):
            raise TypeError("application must provide generate_initiative")
        if not isinstance(policy, InitiativePolicy):
            raise TypeError("policy must be an InitiativePolicy")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self.database = database
        self.application = application
        self.policy = policy
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._claim_owner = uuid4().hex

    @staticmethod
    def _state_key(conversation_id: str) -> str:
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty text")
        return f"initiative:{conversation_id}"

    @staticmethod
    def _control_key(conversation_id: str) -> str:
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty text")
        return f"initiative-control:{conversation_id}"

    def control(self, conversation_id: str) -> dict[str, Any]:
        row = self.database.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = ?", (self._control_key(conversation_id),)
        ).fetchone()
        if row is None:
            return {"paused": False, "deferred_until": None}
        try:
            value = json.loads(row["value_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("initiative control JSON is invalid") from error
        if not isinstance(value, dict) or set(value) != _CONTROL_FIELDS:
            raise ValueError("initiative control fields are invalid")
        if type(value["paused"]) is not bool:
            raise ValueError("initiative control paused is invalid")
        if value["deferred_until"] is not None:
            _parse_utc(value["deferred_until"], "deferred_until")
        return value

    def _write_control(self, conversation_id: str, value: dict[str, Any], now: datetime) -> None:
        if set(value) != _CONTROL_FIELDS:
            raise ValueError("initiative control fields are invalid")
        now = _utc(now, "initiative control updated_at")
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with self.database.transaction() as connection:
            previous = connection.execute(
                "SELECT updated_at_utc FROM runtime_meta WHERE key = ?", (self._control_key(conversation_id),)
            ).fetchone()
            if previous is not None and now < _parse_utc(previous["updated_at_utc"], "updated_at_utc"):
                raise ValueError("initiative control updated_at must not move backward")
            connection.execute(
                "INSERT INTO runtime_meta(key, value_json, updated_at_utc) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc",
                (self._control_key(conversation_id), serialized, now.isoformat()),
            )

    def pause(self, conversation_id: str) -> None:
        value = self.control(conversation_id)
        value["paused"] = True
        self._write_control(conversation_id, value, _utc(self.clock()))

    def resume(self, conversation_id: str) -> None:
        value = self.control(conversation_id)
        value["paused"] = False
        self._write_control(conversation_id, value, _utc(self.clock()))

    def defer_once(self, conversation_id: str, deferred_until: datetime) -> None:
        deferred_until = _utc(deferred_until, "deferred_until")
        now = _utc(self.clock())
        if deferred_until <= now:
            raise ValueError("deferred_until must be in the future")
        value = self.control(conversation_id)
        value["deferred_until"] = deferred_until.isoformat()
        self._write_control(conversation_id, value, now)

    def timeline(self, conversation_id: str) -> tuple[dict[str, Any], ...]:
        state = self._state(conversation_id)
        if not state:
            return ()
        items: list[dict[str, Any]] = [{"category": "due", "at_utc": state["slot"], "claim_id": state["claim_id"]}]
        items.append({"category": "claim", "at_utc": state["claimed_at"], "claim_id": state["claim_id"]})
        if state["status"] != "claimed":
            item = {"category": state["status"], "at_utc": state["completed_at"], "claim_id": state["claim_id"], "failure_category": state.get("failure_category")}
            items.append(item)
        return tuple(items)

    def _decode_state(self, raw: object) -> dict[str, Any]:
        return decode_initiative_state(raw)

    def _state(self, conversation_id: str) -> dict[str, Any]:
        row = self.database.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (self._state_key(conversation_id),),
        ).fetchone()
        if row is None:
            return {}
        state = self._decode_state(row["value_json"])
        updated_at = _parse_utc(row["updated_at_utc"], "updated_at_utc")
        latest_state_time = _parse_utc(state["claimed_at"], "claimed_at")
        if state["status"] != "claimed":
            latest_state_time = max(
                latest_state_time,
                _parse_utc(state["completed_at"], "completed_at"),
            )
        if updated_at < latest_state_time:
            raise ValueError("initiative state updated_at precedes state data")
        return state

    def _completion_floor(
        self, connection: Any, conversation_id: str, state: dict[str, Any]
    ) -> datetime:
        floor = _parse_utc(state["claimed_at"], "claimed_at")
        state_row = connection.execute(
            "SELECT updated_at_utc FROM runtime_meta WHERE key = ?",
            (self._state_key(conversation_id),),
        ).fetchone()
        if state_row is not None:
            floor = max(
                floor,
                _parse_utc(state_row["updated_at_utc"], "updated_at_utc"),
            )
        outbox_row = connection.execute(
            "SELECT updated_at_utc FROM outbox WHERE operation_key = ?",
            (Sender.text_operation_key(state["outbound_event_id"]),),
        ).fetchone()
        if outbox_row is not None:
            floor = max(
                floor,
                _parse_utc(outbox_row["updated_at_utc"], "outbox.updated_at_utc"),
            )
        return floor

    def _write_state(self, connection: Any, conversation_id: str, state: dict[str, Any], now: datetime) -> None:
        now = _utc(now, "initiative state updated_at")
        serialized = json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self._decode_state(serialized)
        previous = connection.execute(
            "SELECT updated_at_utc FROM runtime_meta WHERE key = ?",
            (self._state_key(conversation_id),),
        ).fetchone()
        if previous is not None and now < _parse_utc(
            previous["updated_at_utc"], "updated_at_utc"
        ):
            raise ValueError("initiative state updated_at must not move backward")
        connection.execute(
            "INSERT INTO runtime_meta(key, value_json, updated_at_utc) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
            "updated_at_utc=excluded.updated_at_utc",
            (
                self._state_key(conversation_id),
                serialized,
                now.isoformat(),
            ),
        )

    def _terminal_state(
        self, claimed: dict[str, Any], status: str, now: datetime
    ) -> dict[str, Any]:
        terminal = dict(claimed)
        terminal["status"] = status
        terminal["completed_at"] = now.isoformat()
        # Only an actually sent message can remain unanswered. Skips, failed
        # sends and unknown delivery do not consume the user's attention
        # budget and therefore preserve the previous count.
        if status == "sent":
            terminal["unanswered_attempts"] = claimed["unanswered_attempts"] + 1
        terminal["failure_category"] = self._failure_category(status, recovered=False)
        if status == "unknown":
            terminal.pop("next_due_at", None)
        elif status == "skipped":
            next_due = _parse_utc(claimed["slot"], "slot") + self.policy.idle_delta
            terminal["next_due_at"] = (
                next_due if next_due > now else now + self.policy.idle_delta
            ).isoformat()
        else:
            terminal["next_due_at"] = (now + self.policy.idle_delta).isoformat()
        return terminal

    def _complete_state(
        self, conversation_id: str, claimed: dict[str, Any], status: str, now: datetime
    ) -> bool:
        if status not in _STATUSES - {"claimed"}:
            raise ValueError("initiative completion status is invalid")
        with self.database.transaction() as connection:
            state_row = connection.execute(
                "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
                (self._state_key(conversation_id),),
            ).fetchone()
            if state_row is None:
                return False
            current = self._decode_state(state_row["value_json"])
            identity_fields = (
                "slot",
                "activity",
                "context_version",
                "claim_id",
                "owner",
                "claimed_at",
                "outbound_event_id",
            )
            if current.get("status") != "claimed" or any(
                current.get(field) != claimed.get(field) for field in identity_fields
            ):
                return False
            effective_now = max(
                _utc(now), self._completion_floor(connection, conversation_id, current)
            )
            failure_category = self._failure_category(status, recovered=False)
            self._write_state(
                connection,
                conversation_id,
                self._terminal_state(current, status, effective_now),
                effective_now,
            )
            self._record_trigger_observation(
                connection, current, status, failure_category, effective_now
            )
            return True

    @staticmethod
    def _failure_category(status: str, *, recovered: bool) -> str | None:
        if status == "unknown":
            return "delivery_unknown"
        if status == "failed":
            return "recovered_orphan" if recovered else "generation_failed"
        return None

    def _record_trigger_observation(
        self,
        connection: Any,
        state: dict[str, Any],
        status: str,
        failure_category: str | None,
        completed_at: datetime,
    ) -> None:
        """Converge the durable internal trigger status with the claim state."""
        if failure_category is not None and failure_category not in _FAILURE_CATEGORIES:
            raise ValueError("initiative failure category is invalid")
        row = connection.execute(
            "SELECT event_id FROM conversation_events WHERE event_id = ?",
            (state["claim_id"],),
        ).fetchone()
        if row is None:
            return
        connection.execute(
            "UPDATE conversation_events SET status = ? WHERE event_id = ?",
            (status, state["claim_id"]),
        )

    def _durable_result(self, state: dict[str, Any]) -> str | None:
        trigger = self.database.connection.execute(
            "SELECT status FROM conversation_events WHERE event_id = ?", (state["claim_id"],)
        ).fetchone()
        outbound_id = state.get("outbound_event_id")
        outbound = None
        outbox = None
        if outbound_id is not None:
            outbound = self.database.connection.execute(
                "SELECT status FROM conversation_events WHERE event_id = ?", (outbound_id,)
            ).fetchone()
            outbox = self.database.connection.execute(
                "SELECT status FROM outbox WHERE operation_key = ?",
                (Sender.text_operation_key(outbound_id),),
            ).fetchone()
        if outbox is not None:
            outbox_status = outbox["status"]
            outbound_status = None if outbound is None else outbound["status"]
            if outbox_status == outbound_status == "sent":
                return "sent"
            if outbox_status == outbound_status == "failed":
                return "failed"
            return "unknown"
        if outbound is not None:
            return "unknown"
        if trigger is None:
            return None
        if trigger["status"] == "skipped":
            return "skipped"
        if trigger["status"] == "cancelled":
            return "cancelled"
        return "failed"

    def _claim(self, conversation_id: str, now: datetime) -> dict[str, Any] | None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "SELECT context_version, last_user_activity_utc FROM conversation_cursors "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if cursor is None or cursor["last_user_activity_utc"] is None:
                return None
            control = self.control(conversation_id)
            if control["paused"]:
                return None
            deferred_until = control["deferred_until"]
            if deferred_until is not None and now < _parse_utc(deferred_until, "deferred_until"):
                return None
            if deferred_until is not None:
                control["deferred_until"] = None
                connection.execute(
                    "UPDATE runtime_meta SET value_json = ?, updated_at_utc = ? WHERE key = ?",
                    (json.dumps(control, sort_keys=True, separators=(",", ":")), now.isoformat(), self._control_key(conversation_id)),
                )
            activity_raw = cursor["last_user_activity_utc"]
            activity = _parse_utc(activity_raw, "last_user_activity_utc")
            if type(cursor["context_version"]) is not int or cursor["context_version"] < 0:
                raise ValueError("conversation context_version is invalid")
            context_version = cursor["context_version"]
            slot = _latest_due_slot(activity, now, self.policy.idle_delta)
            if slot is None or self.policy.is_quiet_at(slot):
                return None

            state = self._state(conversation_id)
            same_anchor = bool(state) and (
                state["activity"] == activity_raw
                and state["context_version"] == context_version
            )
            if (
                same_anchor
                and state["unanswered_attempts"] >= self.policy.max_unanswered_attempts
            ):
                return None
            if same_anchor and state["status"] == "unknown":
                return None
            if same_anchor and state["status"] == "claimed":
                claimed_at = _parse_utc(state["claimed_at"], "claimed_at")
                if now < claimed_at + self.policy.idle_delta:
                    return None
                durable = self._durable_result(state)
                if durable is not None:
                    recovered_at = max(
                        now, self._completion_floor(connection, conversation_id, state)
                    )
                    terminal = self._terminal_state(state, durable, recovered_at)
                    terminal["failure_category"] = self._failure_category(
                        durable, recovered=durable == "failed"
                    )
                    self._write_state(
                        connection, conversation_id, terminal, recovered_at
                    )
                    self._record_trigger_observation(
                        connection,
                        state,
                        durable,
                        self._failure_category(durable, recovered=durable == "failed"),
                        recovered_at,
                    )
                    return None
            elif same_anchor:
                next_due_raw = state.get("next_due_at")
                next_due = (
                    _parse_utc(next_due_raw, "next_due_at")
                    if next_due_raw is not None
                    else _parse_utc(state["slot"], "slot") + self.policy.idle_delta
                )
                if now < next_due:
                    return None

            slot_text = slot.isoformat()
            claim_id = hashlib.sha256(
                f"{conversation_id}:{activity_raw}:{context_version}:{slot_text}".encode("utf-8")
            ).hexdigest()
            outbound_event_id = str(
                uuid5(NAMESPACE_URL, f"qichi:initiative:outbound:{claim_id}")
            )
            claimed = {
                "slot": slot_text,
                "status": "claimed",
                "owner": self._claim_owner,
                "claim_id": claim_id,
                "claimed_at": now.isoformat(),
                "activity": activity_raw,
                "context_version": context_version,
                "outbound_event_id": outbound_event_id,
                "unanswered_attempts": state.get("unanswered_attempts", 0) if same_anchor else 0,
            }
            self._write_state(connection, conversation_id, claimed, now)
            return claimed

    async def tick(self, conversation_id: str) -> Any | None:
        if not self.policy.enabled:
            return None
        now = _utc(self.clock())
        if self.policy.is_quiet_at(now):
            return None
        claimed = self._claim(conversation_id, now)
        if claimed is None:
            return None
        try:
            result = await self.application.generate_initiative(
                conversation_id,
                claimed["context_version"],
                _parse_utc(claimed["activity"], "activity"),
                now,
                claimed["claim_id"],
                claimed["outbound_event_id"],
                json.dumps(
                    claimed,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                self.policy.allows_initiative_at,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            durable = self._durable_result(claimed)
            self._complete_state(
                conversation_id,
                claimed,
                "failed" if durable is None else durable,
                _utc(self.clock()),
            )
            return None

        if result is None:
            status = "cancelled"
        elif isinstance(result, DialogueSkip):
            status = "skipped"
        elif getattr(result, "status", None) == "unknown":
            status = "unknown"
        elif getattr(result, "status", None) == "sent":
            status = "sent"
        else:
            status = "failed"
        self._complete_state(conversation_id, claimed, status, _utc(self.clock()))
        return result
