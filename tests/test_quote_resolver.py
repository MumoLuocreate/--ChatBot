from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.quote_repository import QuoteRepository
from qichi.transport.quote_resolver import QuoteResolutionError, QuoteResolver


NOW = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
_DEFAULT_PLATFORM_MESSAGE_ID = object()


def event(event_id: str, *, conversation_id: str = "conversation", actor: str = "mumo", direction: str = "inbound", platform_message_id: str | None | object = _DEFAULT_PLATFORM_MESSAGE_ID, reply_to_platform_message_id: str | None = None) -> ConversationEvent:
    return ConversationEvent(
        event_id=event_id,
        platform_event_id=f"event-{event_id}",
        platform_message_id=(
            f"message-{event_id}"
            if platform_message_id is _DEFAULT_PLATFORM_MESSAGE_ID
            else platform_message_id
        ),
        conversation_id=conversation_id,
        sequence=0,
        direction=direction,
        actor=actor,
        kind="text",
        text=f"text-{event_id}",
        message_segments=(MessageSegment("text", {"text": f"text-{event_id}"}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=reply_to_platform_message_id,
        occurred_at_utc=NOW,
        received_at_utc=NOW,
        status="received" if direction == "inbound" else "sent",
        metadata={},
    )


@pytest.mark.parametrize(
    "source_actor, source_direction, target_actor, target_direction",
    [
        ("mumo", "inbound", "mumo", "inbound"),
        ("mumo", "inbound", "qichi", "outbound"),
        ("qichi", "outbound", "mumo", "inbound"),
        ("qichi", "outbound", "qichi", "outbound"),
    ],
)
def test_local_mapping_resolves_all_four_quote_directions_without_fetch(tmp_path, source_actor, source_direction, target_actor, target_direction):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        target = events.insert(event("target", actor=target_actor, direction=target_direction, platform_message_id="quoted-platform"))
        source = events.insert(event("source", actor=source_actor, direction=source_direction, reply_to_platform_message_id="quoted-platform"))
        calls: list[str] = []
        quoted = QuoteResolver(database).resolve(source.event_id, lambda platform_id: calls.append(platform_id))
        assert quoted is not None
        assert quoted.event_id == target.event_id
        assert quoted.actor == target_actor
        assert quoted.handle == target.visible_handle
        assert calls == []
        link = database.connection.execute("SELECT target_event_id, relation, status FROM message_links").fetchone()
        assert tuple(link) == (target.event_id, "reply", "resolved")
    finally:
        database.close()


def test_missing_mapping_fetches_once_persists_imported_snapshot_and_reopens(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    events = EventRepository(database)
    source = events.insert(event("source", reply_to_platform_message_id="missing-platform"))
    fetched = ConversationEvent(
        event_id="imported",
        platform_event_id="event-imported",
        platform_message_id="missing-platform",
        conversation_id="conversation",
        sequence=0,
        direction="outbound",
        actor="qichi",
        kind="text",
        text="imported text",
        message_segments=(
            MessageSegment("face", {"id": "14"}),
            MessageSegment("text", {"text": "imported text"}),
        ),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=datetime(2026, 8, 27, 10, 15, tzinfo=timezone.utc),
        received_at_utc=datetime(2026, 8, 27, 10, 17, tzinfo=timezone.utc),
        status="sent",
        metadata={},
    )
    calls: list[str] = []
    quoted = QuoteResolver(database).resolve(source.event_id, lambda platform_id: calls.append(platform_id) or fetched)
    assert quoted is not None
    assert quoted.event_id == "imported"
    assert calls == ["missing-platform"]
    assert database.connection.execute("SELECT source FROM platform_message_map WHERE platform_message_id = ?", ("missing-platform",)).fetchone()[0] == "imported"
    database.close()
    reopened = Database(path)
    try:
        quoted = QuoteResolver(reopened).resolve(source.event_id, lambda _: pytest.fail("local map must avoid fetch"))
        assert quoted is not None and quoted.event_id == "imported"
        restored = EventRepository(reopened).get("imported")
        assert restored.text == "imported text"
        assert restored.message_segments == fetched.message_segments
        assert restored.actor == "qichi"
        assert restored.occurred_at_utc == datetime(2026, 8, 27, 10, 15, tzinfo=timezone.utc)
        assert restored.received_at_utc == datetime(2026, 8, 27, 10, 17, tzinfo=timezone.utc)
        assert restored.platform_message_id == "missing-platform"
    finally:
        reopened.close()


def test_quote_to_sent_outbound_uses_hydrated_platform_message_id_after_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    try:
        events = EventRepository(database)
        target = events.insert(
            event(
                "outbound",
                actor="qichi",
                direction="outbound",
                platform_message_id=None,
            )
        )
        database.connection.execute(
            "INSERT INTO platform_message_map "
            "(platform_message_id, event_id, source, created_at_utc) VALUES (?, ?, ?, ?)",
            ("702", target.event_id, "sender", NOW.isoformat()),
        )
        source = events.insert(event("source", reply_to_platform_message_id="702"))
        quoted = QuoteResolver(database).resolve(
            source.event_id, lambda _: pytest.fail("local mapping must resolve")
        )
        assert quoted is not None
        assert quoted.event_id == target.event_id
        assert quoted.platform_message_id == "702"
    finally:
        database.close()

    reopened = Database(path)
    try:
        quoted = QuoteResolver(reopened).resolve(
            source.event_id, lambda _: pytest.fail("restart must retain local mapping")
        )
        assert quoted is not None
        assert quoted.event_id == target.event_id
        assert quoted.platform_message_id == "702"
    finally:
        reopened.close()


def test_unavailable_target_can_be_retried_and_promoted_without_mutating_source(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    database = Database(path)
    try:
        events = EventRepository(database)
        source = events.insert(event("source", reply_to_platform_message_id="unavailable-platform"))
        resolver = QuoteResolver(database)
        assert resolver.resolve(source.event_id, lambda _: None) is None
        fetched = event("imported", actor="qichi", direction="outbound", platform_message_id="unavailable-platform")
        quoted = resolver.resolve(source.event_id, lambda _: fetched)
        assert quoted is not None and quoted.event_id == "imported"
        restored = events.get(source.event_id)
        assert restored.reply_to_event_id is None
        assert restored.reply_to_platform_message_id == "unavailable-platform"
        link = database.connection.execute("SELECT target_event_id, relation, status FROM message_links").fetchone()
        assert tuple(link) == ("imported", "reply", "resolved")
        resolution = resolver.quotes.reply_resolution(source.event_id, "unavailable-platform")
        assert resolution is not None
        assert (resolution.status, resolution.target_event_id) == ("resolved", "imported")
    finally:
        database.close()
    reopened = Database(path)
    try:
        assert QuoteResolver(reopened).resolve(source.event_id, lambda _: pytest.fail("resolved restart must not fetch")) is not None
    finally:
        reopened.close()


def test_cross_conversation_local_target_and_imported_snapshot_are_rejected(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        events.insert(event("target", conversation_id="other", platform_message_id="quoted-platform"))
        source = events.insert(event("source", reply_to_platform_message_id="quoted-platform"))
        resolver = QuoteResolver(database)
        with pytest.raises(QuoteResolutionError, match="conversation"):
            resolver.resolve(source.event_id, lambda _: pytest.fail("cross-conversation local target must not fetch"))
        source = events.insert(event("source-fetch", reply_to_platform_message_id="fetched-platform"))
        fetched = event("fetched", conversation_id="other", platform_message_id="fetched-platform")
        with pytest.raises(QuoteResolutionError, match="conversation"):
            resolver.resolve(source.event_id, lambda _: fetched)
    finally:
        database.close()


def test_invalid_source_identity_mapping_collision_and_missing_reply_fail_closed(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        no_reply = events.insert(event("no-reply"))
        with pytest.raises(QuoteResolutionError, match="reply"):
            QuoteResolver(database).resolve(no_reply.event_id, lambda _: None)
        internal = events.insert(event("internal", actor="platform", direction="internal"))
        with pytest.raises(QuoteResolutionError, match="identity"):
            QuoteResolver(database).resolve(internal.event_id, lambda _: None)
        source = events.insert(event("source", reply_to_platform_message_id="collision"))
        existing = events.insert(event("existing", platform_message_id="collision"))
        colliding = event("colliding", platform_message_id="collision")
        with pytest.raises(ValueError, match="identity conflict"):
            QuoteRepository(database).persist_imported(colliding)
        assert events.get(existing.event_id).event_id == "existing"
    finally:
        database.close()


@pytest.mark.parametrize(
    "actor, direction",
    [("mumo", "outbound"), ("qichi", "inbound")],
)
def test_mismatched_owner_private_actor_direction_is_rejected(tmp_path, actor, direction):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        source = events.insert(event("source", actor=actor, direction=direction, reply_to_platform_message_id="quoted"))
        events.insert(event("target", platform_message_id="quoted"))
        with pytest.raises(QuoteResolutionError, match="identity"):
            QuoteResolver(database).resolve(source.event_id, lambda _: None)
    finally:
        database.close()


def test_existing_reply_link_collision_fails_closed(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        target = events.insert(event("target", platform_message_id="quoted"))
        source = events.insert(event("source", reply_to_platform_message_id="quoted"))
        resolver = QuoteResolver(database)
        assert resolver.resolve(source.event_id, lambda _: None) is not None
        database.connection.execute("UPDATE message_links SET status = ?", ("corrupt",))
        with pytest.raises(ValueError, match="collision"):
            resolver.resolve(source.event_id, lambda _: None)
        assert target.event_id == "target"
    finally:
        database.close()


def test_reply_resolution_matches_exact_requested_platform_message_id(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        target = events.insert(event("target", platform_message_id="quoted"))
        source = events.insert(event("source", reply_to_platform_message_id="quoted"))
        resolver = QuoteResolver(database)
        assert resolver.resolve(source.event_id, lambda _: None) is not None
        assert resolver.quotes.reply_resolution(source.event_id, "other-platform") is None
        resolution = resolver.quotes.reply_resolution(source.event_id, "quoted")
        assert resolution is not None
        assert resolution.target_event_id == target.event_id
    finally:
        database.close()


def test_unrelated_relation_coexists_with_reply_resolution(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        target = events.insert(event("target", platform_message_id="quoted"))
        source = events.insert(event("source", reply_to_platform_message_id="quoted"))
        database.connection.execute(
            "INSERT INTO message_links "
            "(link_id, source_event_id, target_event_id, relation, status) VALUES (?, ?, ?, ?, ?)",
            ("unrelated-link", source.event_id, target.event_id, "quote", "resolved"),
        )
        quoted = QuoteResolver(database).resolve(source.event_id, lambda _: pytest.fail("local target must resolve"))
        assert quoted is not None and quoted.event_id == target.event_id
        assert QuoteRepository(database).reply_resolution(source.event_id, "quoted") is not None
    finally:
        database.close()


def test_second_reply_target_for_source_is_a_collision(tmp_path):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        first = events.insert(event("first", platform_message_id="quoted"))
        second = events.insert(event("second", platform_message_id="other"))
        source = events.insert(event("source", reply_to_platform_message_id="quoted"))
        quotes = QuoteRepository(database)
        quotes.append_reply_link(source.event_id, first.event_id, "quoted")
        with pytest.raises(ValueError, match="collision"):
            quotes.append_reply_link(source.event_id, second.event_id, "other")
    finally:
        database.close()


@pytest.mark.parametrize(
    "target_event_id, status",
    [(None, "resolved"), ("target", "unresolved")],
)
def test_reply_resolution_rejects_corrupt_link_status_target_pairs(tmp_path, target_event_id, status):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        events = EventRepository(database)
        target = events.insert(event("target", platform_message_id="quoted"))
        source = events.insert(event("source", reply_to_platform_message_id="quoted"))
        quotes = QuoteRepository(database)
        quotes.append_reply_link(source.event_id, target.event_id, "quoted")
        database.connection.execute(
            "UPDATE message_links SET target_event_id = ?, status = ?",
            (target_event_id, status),
        )
        with pytest.raises(ValueError, match="invalid reply link"):
            quotes.reply_resolution(source.event_id, "quoted")
    finally:
        database.close()
