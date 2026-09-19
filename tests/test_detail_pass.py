from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from qichi.dialogue.llm_client import LLMGeneration
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.memory.detail_pass import MemoryDetailPass, MemoryDetailPassError

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def event(sequence: int, actor: str = "mumo") -> ConversationEvent:
    text = f"第{sequence}条"
    return ConversationEvent(
        event_id=f"e{sequence}",
        platform_event_id=None,
        platform_message_id=f"pm-{sequence}",
        conversation_id="10001",
        sequence=sequence,
        direction="inbound" if actor == "mumo" else "outbound",
        actor=actor,
        kind="text",
        text=text,
        message_segments=(MessageSegment("text", {"text": text}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=NOW + timedelta(minutes=sequence),
        received_at_utc=NOW + timedelta(minutes=sequence),
        status="received" if actor == "mumo" else "sent",
        metadata={},
    )


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def generate(self, messages, *, thinking=None):
        self.calls.append({"messages": messages, "thinking": thinking})
        return LLMGeneration(self.response, "primary", "m", 10, 20, 1.0)


def detail(ordinal: int, event_id: str, *, kind: str = "message") -> dict:
    return {
        "ordinal": ordinal,
        "detail_kind": kind,
        "actor": "mumo",
        "reality_scope": "conversation",
        "normalized_detail": f"细节{ordinal}",
        "exact_quote": event_id.replace("e", "第") + "条",
        "source_event_id": event_id,
        "certainty": "explicit",
        "temporal_scope": "historical",
        "status": "active",
        "privacy_class": "ordinary",
        "recall_policy": "daily_safe",
        "evidence": [[event_id, "source"]],
    }


@pytest.mark.asyncio
async def test_the_pass_disables_thinking_and_returns_the_timeline():
    events = (event(1), event(2))
    client = FakeClient(json.dumps({"details": [detail(0, "e1"), detail(1, "e2")]}, ensure_ascii=False))
    pass_ = MemoryDetailPass(client)

    drafts = await pass_.generate(events)

    assert [d.ordinal for d in drafts] == [0, 1]
    assert client.calls[0]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_the_model_order_is_replaced_by_verified_event_order():
    events = (event(1), event(2), event(3))
    # The model lists them backwards and numbers them wrongly, as it did on two
    # of the four replay fragments; the code owns the order.
    payload = {"details": [detail(2, "e3"), detail(5, "e1"), detail(9, "e2")]}
    pass_ = MemoryDetailPass(FakeClient(json.dumps(payload, ensure_ascii=False)))

    drafts = await pass_.generate(events)

    assert [d.source_event_id for d in drafts] == ["e1", "e2", "e3"]
    assert [d.ordinal for d in drafts] == [0, 1, 2]


@pytest.mark.asyncio
async def test_an_invalid_entry_is_dropped_and_counted_without_breaking_the_order():
    events = (event(1), event(2), event(3))
    broken = detail(1, "e2")
    broken["detail_kind"] = "not_a_kind"
    payload = {"details": [detail(0, "e1"), broken, detail(2, "e3")]}
    pass_ = MemoryDetailPass(FakeClient(json.dumps(payload, ensure_ascii=False)))

    drafts = await pass_.generate(events)

    assert [d.source_event_id for d in drafts] == ["e1", "e3"]
    assert [d.ordinal for d in drafts] == [0, 1]
    assert pass_.dropped == 1


@pytest.mark.asyncio
async def test_malformed_output_fails_closed():
    events = (event(1),)
    for response in ("", "not json", "[]", json.dumps({"details": "no"}, ensure_ascii=False),
                     json.dumps({"other": []}, ensure_ascii=False)):
        with pytest.raises(MemoryDetailPassError):
            await MemoryDetailPass(FakeClient(response)).generate(events)


@pytest.mark.asyncio
async def test_an_empty_details_array_is_an_empty_timeline_not_an_error():
    events = (event(1),)
    pass_ = MemoryDetailPass(FakeClient(json.dumps({"details": []})))

    assert await pass_.generate(events) == ()
