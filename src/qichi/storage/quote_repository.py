from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qichi.domain.events import ConversationEvent

from .database import Database
from .event_repository import EventRepository


@dataclass(frozen=True)
class ReplyResolution:
    status: str
    target_event_id: str | None


class QuoteRepository:
    def __init__(self, database: Database):
        self.database = database
        self.events = EventRepository(database)

    def find_platform_message(self, platform_message_id: str) -> ConversationEvent | None:
        row = self.database.connection.execute(
            "SELECT event_id FROM platform_message_map WHERE platform_message_id = ?", (platform_message_id,)
        ).fetchone()
        return self.events.get(row["event_id"]) if row is not None else None

    def persist_imported(self, event: ConversationEvent) -> ConversationEvent:
        inserted = self.events.insert(event)
        if inserted.platform_message_id is None:
            raise ValueError("imported quote requires a platform message id")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE platform_message_map SET source = ? WHERE platform_message_id = ? AND event_id = ?",
                ("imported", inserted.platform_message_id, inserted.event_id),
            )
        return inserted

    def append_reply_link(
        self,
        source_event_id: str,
        target_event_id: str | None,
        target_platform_message_id: str,
    ) -> None:
        identity = target_event_id or f"unresolved:{target_platform_message_id}"
        link_id = str(uuid5(NAMESPACE_URL, f"qichi:reply:{source_event_id}:{identity}"))
        status = "resolved" if target_event_id is not None else "unresolved"
        with self.database.transaction() as connection:
            existing_reply_links = connection.execute(
                "SELECT link_id, source_event_id, target_event_id, relation, status "
                "FROM message_links WHERE source_event_id = ? AND relation = ?",
                (source_event_id, "reply"),
            ).fetchall()
            if len(existing_reply_links) > 1:
                raise ValueError("message link collision")
            if existing_reply_links:
                existing = existing_reply_links[0]
                existing_status = existing["status"]
                if existing_status not in {"resolved", "unresolved"}:
                    raise ValueError("message link collision")
                if existing_status == "resolved":
                    # A durable resolution is authoritative and must never be
                    # downgraded by a later failed lookup.
                    if target_event_id is None or existing["target_event_id"] == target_event_id:
                        return
                    raise ValueError("message link collision")
                if target_event_id is None:
                    if existing["link_id"] != link_id:
                        raise ValueError("message link collision")
                    return
                # A prior unresolved observation is not proof that the target
                # does not exist. Promote it in place so a later successful
                # get_msg can become the single authoritative reply edge.
                conflict = connection.execute(
                    "SELECT source_event_id FROM message_links WHERE link_id = ?",
                    (link_id,),
                ).fetchone()
                if conflict is not None and conflict["source_event_id"] != source_event_id:
                    raise ValueError("message link collision")
                connection.execute(
                    "UPDATE message_links SET link_id = ?, target_event_id = ?, status = 'resolved' "
                    "WHERE source_event_id = ? AND relation = 'reply'",
                    (link_id, target_event_id, source_event_id),
                )
                return
            connection.execute(
                "INSERT INTO message_links "
                "(link_id, source_event_id, target_event_id, relation, status) VALUES (?, ?, ?, ?, ?)",
                (link_id, source_event_id, target_event_id, "reply", status),
            )

    def has_unresolved_reply(self, source_event_id: str, target_platform_message_id: str) -> bool:
        resolution = self.reply_resolution(source_event_id, target_platform_message_id)
        return resolution is not None and resolution.status == "unresolved"

    def reply_resolution(
        self, source_event_id: str, target_platform_message_id: str
    ) -> ReplyResolution | None:
        reply_links = self.database.connection.execute(
            "SELECT source_event_id, target_event_id, relation, status "
            "FROM message_links WHERE source_event_id = ? AND relation = ?",
            (source_event_id, "reply"),
        ).fetchall()
        if len(reply_links) > 1:
            raise ValueError("message link collision")
        for reply_link in reply_links:
            self._validate_reply_link(reply_link, source_event_id)
        link_id = str(uuid5(NAMESPACE_URL, f"qichi:reply:{source_event_id}:unresolved:{target_platform_message_id}"))
        row = self.database.connection.execute(
            "SELECT source_event_id, target_event_id, relation, status "
            "FROM message_links WHERE link_id = ?",
            (link_id,),
        ).fetchone()
        unresolved = self._validate_reply_link(row, source_event_id) if row is not None else None
        resolved_rows = self.database.connection.execute(
            "SELECT l.link_id, l.source_event_id, l.target_event_id, l.relation, l.status "
            "FROM message_links AS l "
            "JOIN conversation_events AS e ON e.event_id = l.target_event_id "
            "JOIN platform_message_map AS m ON m.event_id = e.event_id "
            "WHERE l.source_event_id = ? AND m.platform_message_id = ?",
            (source_event_id, target_platform_message_id),
        ).fetchall()
        resolved_candidates = []
        for resolved_row in resolved_rows:
            expected_link_id = str(
                uuid5(NAMESPACE_URL, f"qichi:reply:{source_event_id}:{resolved_row['target_event_id']}")
            )
            if resolved_row["relation"] == "reply" or resolved_row["link_id"] == expected_link_id:
                resolved_candidates.append(self._validate_reply_link(resolved_row, source_event_id))
        if len(resolved_candidates) > 1:
            raise ValueError("ambiguous reply resolution")
        resolved = resolved_candidates[0] if resolved_candidates else None
        if unresolved is not None and resolved is not None:
            raise ValueError("ambiguous reply resolution")
        return unresolved or resolved

    @staticmethod
    def _validate_reply_link(row: Any, source_event_id: str) -> ReplyResolution:
        source_id = row["source_event_id"]
        target_event_id = row["target_event_id"]
        relation = row["relation"]
        status = row["status"]
        if source_id != source_event_id or relation != "reply":
            raise ValueError("invalid reply link")
        if status == "resolved" and target_event_id is not None:
            return ReplyResolution(status=status, target_event_id=target_event_id)
        if status == "unresolved" and target_event_id is None:
            return ReplyResolution(status=status, target_event_id=None)
        raise ValueError("invalid reply link")
