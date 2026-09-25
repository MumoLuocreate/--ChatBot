from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from qichi.domain.events import ConversationEvent, MessageSegment
from .database import Database


class EventRepository:
    def __init__(self, database: Database):
        self.database = database

    def insert_inbound(self, event: ConversationEvent, raw_payload: Any = None) -> ConversationEvent:
        return self.insert(event, raw_payload=raw_payload)

    def insert(self, event: ConversationEvent, raw_payload: Any = None) -> ConversationEvent:
        with self.database.transaction() as connection:
            persisted, _created = self.insert_in_transaction(
                connection, event, raw_payload=raw_payload
            )
            return persisted

    def insert_in_transaction(
        self, connection: Any, event: ConversationEvent, raw_payload: Any = None
    ) -> tuple[ConversationEvent, bool]:
        """Insert using the caller's active transaction and report deduplication."""
        if connection is not self.database.connection or not connection.in_transaction:
            raise ValueError("connection must be this repository's active transaction")
        existing = self._existing_platform_event(connection, event)
        if existing is not None:
            return self.get(existing), False
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM conversation_events "
            "WHERE conversation_id = ?",
            (event.conversation_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO conversation_events (event_id, platform_event_id, platform_message_id, "
            "conversation_id, sequence, direction, actor, kind, text, message_segments_json, "
            "reply_to_event_id, reply_to_platform_message_id, occurred_at_utc, received_at_utc, "
            "status, metadata_json, raw_payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.platform_event_id,
                event.platform_message_id,
                event.conversation_id,
                sequence,
                event.direction,
                event.actor,
                event.kind,
                event.text,
                json.dumps(
                    [segment.to_dict() for segment in event.message_segments],
                    ensure_ascii=False,
                ),
                event.reply_to_event_id,
                event.reply_to_platform_message_id,
                event.occurred_at_utc.isoformat(),
                event.received_at_utc.isoformat(),
                event.status,
                json.dumps(event.metadata, ensure_ascii=False),
                json.dumps(raw_payload, ensure_ascii=False)
                if raw_payload is not None
                else None,
            ),
        )
        if event.platform_message_id is not None:
            connection.execute(
                "INSERT INTO platform_message_map (platform_message_id, event_id, source, created_at_utc) "
                "VALUES (?, ?, ?, ?)",
                (
                    event.platform_message_id,
                    event.event_id,
                    "event_repository",
                    event.received_at_utc.isoformat(),
                ),
            )
        return self.get(event.event_id), True

    def _existing_platform_event(self, connection: Any, event: ConversationEvent) -> str | None:
        found: set[str] = set()
        for column, value in (("platform_event_id", event.platform_event_id), ("platform_message_id", event.platform_message_id)):
            if value is not None:
                row = connection.execute(f"SELECT event_id FROM conversation_events WHERE {column} = ?", (value,)).fetchone()
                if row is not None:
                    found.add(row[0])
        if len(found) > 1:
            raise ValueError("platform identity conflict")
        if not found:
            return None
        existing = self.get(found.pop())
        if (event.platform_event_id is not None and event.platform_event_id != existing.platform_event_id) or (event.platform_message_id is not None and event.platform_message_id != existing.platform_message_id):
            raise ValueError("platform identity conflict")
        return existing.event_id

    def get(self, event_id: str) -> ConversationEvent:
        # Event rows remain immutable; successful sends publish their platform
        # id in the authoritative mapping table. Hydrate that projection here
        # so callers never need to bypass the repository.
        row = self.database.connection.execute(
            "SELECT e.*, m.platform_message_id AS mapped_platform_message_id "
            "FROM conversation_events AS e "
            "LEFT JOIN platform_message_map AS m ON m.event_id = e.event_id "
            "WHERE e.event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(event_id)
        platform_message_id = row["platform_message_id"]
        if platform_message_id is None:
            platform_message_id = row["mapped_platform_message_id"]
        return ConversationEvent(event_id=row["event_id"], platform_event_id=row["platform_event_id"], platform_message_id=platform_message_id, conversation_id=row["conversation_id"], sequence=row["sequence"], direction=row["direction"], actor=row["actor"], kind=row["kind"], text=row["text"], message_segments=tuple(MessageSegment.from_dict(item) for item in json.loads(row["message_segments_json"])), reply_to_event_id=row["reply_to_event_id"], reply_to_platform_message_id=row["reply_to_platform_message_id"], occurred_at_utc=datetime.fromisoformat(row["occurred_at_utc"]), received_at_utc=datetime.fromisoformat(row["received_at_utc"]), status=row["status"], metadata=json.loads(row["metadata_json"]))

    def derive_handle(self, conversation_id: str, event_id: str) -> str | None:
        event = self.get(event_id)
        if event.conversation_id != conversation_id:
            raise KeyError(event_id)
        return event.visible_handle

    def resolve_handle(self, conversation_id: str, handle: str) -> ConversationEvent:
        match = re.fullmatch(r"([MQ])(\d+)", handle)
        if match is None:
            raise ValueError("invalid handle")
        sequence = int(match.group(2))
        if handle != f"{match.group(1)}{sequence}":
            raise ValueError("invalid handle")
        actor = "mumo" if match.group(1) == "M" else "qichi"
        row = self.database.connection.execute("SELECT event_id FROM conversation_events WHERE conversation_id = ? AND sequence = ? AND actor = ? AND direction != 'internal'", (conversation_id, sequence, actor)).fetchone()
        if row is None:
            raise KeyError(handle)
        return self.get(row[0])

    def update_status(self, event_id: str, status: str) -> None:
        if not isinstance(status, str):
            raise TypeError("status must be a string")
        if not status:
            raise ValueError("status must not be empty")
        with self.database.transaction() as connection:
            cursor = connection.execute("UPDATE conversation_events SET status = ? WHERE event_id = ?", (status, event_id))
            if cursor.rowcount != 1:
                raise KeyError(event_id)
