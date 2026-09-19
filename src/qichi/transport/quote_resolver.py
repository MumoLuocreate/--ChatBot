from __future__ import annotations

from collections.abc import Awaitable, Callable

from qichi.domain.dialogue import QuotedTarget
from qichi.domain.events import ConversationEvent
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.quote_repository import QuoteRepository


class QuoteResolutionError(ValueError):
    """A quote cannot safely be associated with the current owner conversation."""


class QuoteResolver:
    def __init__(self, database: Database):
        self.events = EventRepository(database)
        self.quotes = QuoteRepository(database)

    def resolve(
        self,
        source_event_id: str,
        get_msg: Callable[[str], ConversationEvent | None],
    ) -> QuotedTarget | None:
        source = self.events.get(source_event_id)
        self._validate_event(source, source.conversation_id, "source")
        platform_message_id = source.reply_to_platform_message_id
        if platform_message_id is None:
            raise QuoteResolutionError("source event has no reply target")
        target = self.quotes.find_platform_message(platform_message_id)
        if target is None:
            # An unresolved link records only that a previous lookup had no
            # usable snapshot. It is deliberately retryable: a temporary
            # NapCat failure or eventual message-map visibility must not make
            # the quote invisible for the rest of the conversation.
            fetched = get_msg(platform_message_id)
            if fetched is None:
                self.quotes.append_reply_link(source.event_id, None, platform_message_id)
                return None
            self._validate_event(fetched, source.conversation_id, "fetched target")
            if fetched.platform_message_id != platform_message_id:
                raise QuoteResolutionError("fetched target platform message id does not match reply")
            target = self.quotes.persist_imported(fetched)
        self._validate_event(target, source.conversation_id, "target")
        self.quotes.append_reply_link(source.event_id, target.event_id, platform_message_id)
        handle = target.visible_handle
        if handle is None:
            raise QuoteResolutionError("target identity cannot be quoted")
        return QuotedTarget(
            conversation_id=target.conversation_id,
            event_id=target.event_id,
            platform_message_id=target.platform_message_id,
            handle=handle,
            actor=target.actor,
            text=target.text,
            message_segments=target.message_segments,
            occurred_at_utc=target.occurred_at_utc,
        )

    async def resolve_async(
        self,
        source_event_id: str,
        get_msg: Callable[[str], Awaitable[ConversationEvent | None]],
    ) -> QuotedTarget | None:
        """Resolve a quote with an asynchronous platform ``get_msg`` fallback."""
        source = self.events.get(source_event_id)
        self._validate_event(source, source.conversation_id, "source")
        platform_message_id = source.reply_to_platform_message_id
        if platform_message_id is None:
            raise QuoteResolutionError("source event has no reply target")
        target = self.quotes.find_platform_message(platform_message_id)
        if target is None:
            # An unresolved link records only that a previous lookup had no
            # usable snapshot. It is deliberately retryable: a temporary
            # NapCat failure or eventual message-map visibility must not make
            # the quote invisible for the rest of the conversation.
            fetched = await get_msg(platform_message_id)
            if fetched is None:
                self.quotes.append_reply_link(source.event_id, None, platform_message_id)
                return None
            self._validate_event(fetched, source.conversation_id, "fetched target")
            if fetched.platform_message_id != platform_message_id:
                raise QuoteResolutionError("fetched target platform message id does not match reply")
            target = self.quotes.persist_imported(fetched)
        self._validate_event(target, source.conversation_id, "target")
        self.quotes.append_reply_link(source.event_id, target.event_id, platform_message_id)
        handle = target.visible_handle
        if handle is None:
            raise QuoteResolutionError("target identity cannot be quoted")
        return QuotedTarget(
            conversation_id=target.conversation_id,
            event_id=target.event_id,
            platform_message_id=target.platform_message_id,
            handle=handle,
            actor=target.actor,
            text=target.text,
            message_segments=target.message_segments,
            occurred_at_utc=target.occurred_at_utc,
        )

    def _validate_event(self, event: ConversationEvent, conversation_id: str, label: str) -> None:
        if event.conversation_id != conversation_id:
            raise QuoteResolutionError(f"{label} is from another conversation")
        if (event.actor, event.direction) not in {("mumo", "inbound"), ("qichi", "outbound")}:
            raise QuoteResolutionError(f"{label} has an invalid quote identity")
