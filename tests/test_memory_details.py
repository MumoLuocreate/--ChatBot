from datetime import datetime, timedelta, timezone
import json

import pytest

from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory_details import MemoryDetailDraft
from qichi.memory.extractor import MemoryExtractionResult, MemoryExtractor, MemoryOutcome
from qichi.memory.worker import MemoryWorker
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import MemoryDetailRepository
from qichi.storage.memory_repository import MemoryRepository


UTC = timezone.utc
BASE = datetime(2026, 9, 9, 4, 54, tzinfo=UTC)


def event(
    event_id: str,
    sequence: int,
    text: str,
    *,
    actor: str = "mumo",
    at: datetime | None = None,
) -> ConversationEvent:
    direction = "inbound" if actor == "mumo" else "outbound"
    status = "received" if actor == "mumo" else "sent"
    metadata = {} if actor == "mumo" else {"generation_metadata": {"source": "dialogue"}}
    occurred = at or BASE + timedelta(minutes=sequence)
    return ConversationEvent(
        event_id=event_id,
        platform_event_id=None,
        platform_message_id=f"pm-{event_id}",
        conversation_id="conversation-a",
        sequence=sequence,
        direction=direction,
        actor=actor,
        kind="text",
        text=text,
        message_segments=(MessageSegment("text", {"text": text}),),
        reply_to_event_id=None,
        reply_to_platform_message_id=None,
        occurred_at_utc=occurred,
        received_at_utc=occurred,
        status=status,
        metadata=metadata,
    )


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    async def generate(self, events):
        return self.response


def detail_payload(events: tuple[ConversationEvent, ...]) -> str:
    return json.dumps(
        {
            "outcome": {"kind": "memory_found", "reason_code": "historical_episode"},
            "fragment": {
                "fragment_type": "adult",
                "reality_scope": "shared_imagination",
                "summary": "双方在连续片段中共同推进并收束了一段想象情节",
                "privacy_class": "adult",
                "recall_policy": "explicit_request_only",
                "closed": True,
            },
            "details": [
                {
                    "ordinal": index,
                    "detail_kind": "message",
                    "actor": item.actor,
                    "reality_scope": "shared_imagination",
                    "normalized_detail": f"{item.actor}在片段中的原话",
                    "exact_quote": item.text,
                    "source_event_id": item.event_id,
                    "certainty": "explicit",
                    "temporal_scope": "historical",
                    "status": "candidate",
                    "privacy_class": "adult",
                    "recall_policy": "explicit_request_only",
                    # 与抽取提示词声明的唯一一种 evidence 形状一致：模型就是这么写的。
                    # 旧解析器只认 {event_id, role}，于是这条 payload 会让每个窗口都失败
                    # （真机 seq 7306-7320 被隔离，见 doc/诊断-20260916-记忆抽取证据契约冲突.md）。
                    "evidence": [
                        {
                            "event_id": item.event_id,
                            "actor": item.actor,
                            "exact_quote": item.text,
                            "role": "source",
                        }
                    ],
                }
                for index, item in enumerate(events)
            ],
            "candidates": [],
            "reviews": [],
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_extractor_parses_ordered_details_and_preserves_actor_ownership():
    events = (
        event("mumo-1", 1, "用户提出想象内容"),
        event("qichi-2", 2, "角色接住并调整", actor="qichi"),
    )
    result = await MemoryExtractor(FakeLLM(detail_payload(events))).extract(events)

    assert result.ok
    assert result.fragment is not None
    assert result.fragment.reality_scope == "shared_imagination"
    assert [item.ordinal for item in result.details] == [0, 1]
    assert [item.actor for item in result.details] == ["mumo", "qichi"]
    assert [item.source_event_id for item in result.details] == ["mumo-1", "qichi-2"]
    assert all(item.privacy_class == "adult" for item in result.details)
    assert all(item.recall_policy == "explicit_request_only" for item in result.details)


def _single_detail_payload(events: tuple[ConversationEvent, ...], evidence) -> str:
    """One detail whose evidence list is whatever the caller wants to test."""

    payload = json.loads(detail_payload(events[:1]))
    payload["details"][0]["evidence"] = evidence
    return json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_detail_evidence_accepts_both_the_general_and_the_minimal_shape():
    """两种形状都必须能过：提示词声明的那一种，和只用 event_id/role 的最小那一种。"""

    events = (event("mumo-1", 1, "用户提出想象内容"),)
    general = await MemoryExtractor(
        FakeLLM(
            _single_detail_payload(
                events,
                [
                    {
                        "event_id": "mumo-1",
                        "actor": "mumo",
                        "exact_quote": "用户提出想象内容",
                        "role": "source",
                    }
                ],
            )
        )
    ).extract(events)
    minimal = await MemoryExtractor(
        FakeLLM(_single_detail_payload(events, [{"event_id": "mumo-1", "role": "source"}]))
    ).extract(events)

    assert general.ok and len(general.details) == 1
    assert minimal.ok and len(minimal.details) == 1
    assert general.details[0].evidence == minimal.details[0].evidence == (("mumo-1", "source"),)


@pytest.mark.parametrize(
    "evidence, reason",
    [
        # 只放宽到「同一份 evidence 契约」，不是「什么都能塞」：多一个不在契约里的字段照样拒。
        (
            [
                {
                    "event_id": "mumo-1",
                    "actor": "mumo",
                    "exact_quote": "用户提出想象内容",
                    "role": "source",
                    "confidence": 0.9,
                }
            ],
            "unknown field in detail evidence",
        ),
        ([{"event_id": "mumo-1", "actor": "mumo", "exact_quote": "用户提出想象内容"}], "missing field: role"),
        ([{"actor": "mumo", "exact_quote": "用户提出想象内容", "role": "source"}], "missing field: event_id"),
        ([{"event_id": "mumo-1", "role": "narrator"}], "detail evidence role is invalid"),
        ([{"event_id": "ghost-9", "role": "source"}], "detail evidence event does not exist"),
        ([{"event_id": "mumo-1", "role": "source"}, {"event_id": "mumo-1", "role": "source"}], "duplicate detail evidence identity"),
        (["mumo-1"], "detail evidence must be an object"),
    ],
)
@pytest.mark.asyncio
async def test_detail_evidence_stays_fail_closed(evidence, reason):
    events = (event("mumo-1", 1, "用户提出想象内容"),)

    result = await MemoryExtractor(FakeLLM(_single_detail_payload(events, evidence))).extract(events)

    assert not result.ok
    assert result.failure.kind == "parse_error"
    assert reason in result.failure.reason


@pytest.mark.asyncio
async def test_temporary_scene_fallback_keeps_every_reliable_text_event_not_four_quotes():
    events = tuple(
        item
        for index in range(6)
        for item in (
            event(f"mumo-{index}", index * 2, f"用户原话{index}"),
            event(f"qichi-{index}", index * 2 + 1, f"角色回应{index}", actor="qichi"),
        )
    )
    response = json.dumps(
        {
            "outcome": {
                "kind": "no_persistent_memory",
                "reason_code": "temporary_scene_or_roleplay",
            },
            "candidates": [],
            "reviews": [],
        },
        ensure_ascii=False,
    )
    result = await MemoryExtractor(FakeLLM(response)).extract(events)

    assert result.ok
    assert result.fragment is not None
    assert len(result.details) == len(events)
    assert [item.source_event_id for item in result.details] == [item.event_id for item in events]
    assert {item.actor for item in result.details} == {"mumo", "qichi"}
    assert all(item.reality_scope == "shared_imagination" for item in result.details)
    assert all(item.temporal_scope == "historical" for item in result.details)


@pytest.mark.asyncio
async def test_worker_persists_full_fragment_and_details_atomically(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    try:
        events = (
            event("mumo-1", 1, "第一句"),
            event("qichi-2", 2, "回应", actor="qichi"),
            event("mumo-3", 3, "收束"),
        )
        event_repo = EventRepository(db)
        for item in events:
            event_repo.insert(item)

        details = tuple(
            MemoryDetailDraft(
                ordinal=index,
                detail_kind="message",
                actor=item.actor,
                reality_scope="shared_imagination",
                normalized_detail=f"{item.actor}说了什么",
                exact_quote=item.text or "",
                source_event_id=item.event_id,
                certainty="explicit",
                temporal_scope="historical",
                status="candidate",
                privacy_class="adult",
                recall_policy="explicit_request_only",
                evidence=((item.event_id, "source"),),
            )
            for index, item in enumerate(events)
        )

        class DetailExtractor(MemoryExtractor):
            async def extract(self, frozen_events):
                return MemoryExtractionResult(
                    (),
                    None,
                    (),
                    {},
                    MemoryOutcome("no_persistent_memory", "temporary_scene_or_roleplay"),
                    None,
                    details,
                )

        memory = MemoryRepository(db)
        worker = MemoryWorker(
            DetailExtractor(FakeLLM("{}")),
            memory,
            database=db,
            conversation_id="conversation-a",
        )
        worker.notify_reliable_activity("conversation-a")
        worker.open_semantic_gate()
        runs = await worker.run_due(BASE + timedelta(minutes=35))

        assert runs and not runs[0].failed
        assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 1
        assert db.connection.execute("SELECT COUNT(*) FROM memory_fragment_events").fetchone()[0] == 3
        assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 3
        assert db.connection.execute(
            "SELECT GROUP_CONCAT(event_id, ',') FROM memory_fragment_events ORDER BY ordinal"
        ).fetchone()[0] == "mumo-1,qichi-2,mumo-3"
        assert db.connection.execute(
            "SELECT COUNT(*) FROM memory_detail_records WHERE privacy_class='adult' AND recall_policy='explicit_request_only'"
        ).fetchone()[0] == 3
        detail_repository = MemoryDetailRepository(db)
        assert detail_repository.list_details("conversation-a") == ()
        recalled = detail_repository.list_details(
            "conversation-a", query="回应", explicit_request=True
        )
        assert [item.source_event_id for item in recalled] == ["qichi-2"]
        full_recalled = detail_repository.list_details("conversation-a", explicit_request=True)
        assert [item.source_event_id for item in full_recalled] == [
            "mumo-1", "qichi-2", "mumo-3"
        ]
        assert db.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key='memory_worker:conversation-a:processed_sequence'"
        ).fetchone()[0] == "2"
    finally:
        db.close()
