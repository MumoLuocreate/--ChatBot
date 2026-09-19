from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import json

import pytest

from qichi.domain.dialogue import (
    CapabilityManifest,
    DialogueInput,
    DialogueResult,
    DialogueSkip,
    ExpressionIntent,
    ModelMessage,
    QuotedTarget,
)
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import (
    Agreement,
    Correction,
    MemoryEvidence,
    MemoryRecord,
    MemoryReview,
)


UTC = timezone.utc
OCCURRED = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
RECEIVED = datetime(2026, 8, 27, 10, 0, 1, tzinfo=UTC)


def make_event(**overrides) -> ConversationEvent:
    fields = {
        "event_id": "event-1",
        "platform_event_id": "platform-event-1",
        "platform_message_id": "message-1",
        "conversation_id": "owner-1",
        "sequence": 12,
        "direction": "inbound",
        "actor": "mumo",
        "kind": "text",
        "text": "今天在实验室",
        "message_segments": (MessageSegment("text", {"text": "今天在实验室"}),),
        "reply_to_event_id": None,
        "reply_to_platform_message_id": None,
        "occurred_at_utc": OCCURRED,
        "received_at_utc": RECEIVED,
        "status": "received",
        "metadata": {"source": "test", "nested": {"count": 1}},
    }
    fields.update(overrides)
    return ConversationEvent(**fields)


def make_evidence(**overrides) -> MemoryEvidence:
    fields = {
        "memory_id": "memory-1",
        "event_id": "event-1",
        "actor": "mumo",
        "exact_quote": "今天在实验室",
        "occurred_at_utc": OCCURRED,
    }
    fields.update(overrides)
    return MemoryEvidence(**fields)


def test_event_rejects_timezone_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        make_event(occurred_at_utc=datetime(2026, 8, 27, 10, 0))


def test_dialogue_and_memory_reject_timezone_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        make_dialogue_input(current_time=datetime(2026, 8, 27, 10, 0))

    with pytest.raises(ValueError, match="timezone-aware"):
        make_evidence(occurred_at_utc=datetime(2026, 8, 27, 10, 0))


def make_dialogue_input(**overrides) -> DialogueInput:
    fields = {
        "conversation_id": "owner-1",
        "trigger_event_ids": ("event-1",),
        "current_event_handle": "M12",
        "quoted_target": None,
        "context_version": 1,
        "role_messages": (ModelMessage("user", "你好"),),
        "current_time": RECEIVED,
        "platform_capabilities": CapabilityManifest({}),
        "source": "dialogue",
    }
    fields.update(overrides)
    return DialogueInput(**fields)


def test_event_rejects_invalid_direction_and_actor():
    with pytest.raises(ValueError, match="direction"):
        make_event(direction="sideways")
    with pytest.raises(ValueError, match="actor"):
        make_event(actor="unknown")


def test_domain_objects_are_frozen_and_nested_json_is_immutable():
    event = make_event()

    with pytest.raises(FrozenInstanceError):
        event.text = "changed"
    with pytest.raises(TypeError):
        event.metadata["new"] = "value"
    with pytest.raises(TypeError):
        event.metadata["nested"]["count"] = 2
    with pytest.raises(TypeError):
        event.message_segments[0].data["text"] = "changed"


def test_event_handle_is_derived_from_actor_and_sequence():
    assert make_event(actor="mumo", sequence=12).visible_handle == "M12"
    assert make_event(direction="outbound", actor="qichi", sequence=12).visible_handle == "Q12"


@pytest.mark.parametrize("actor", ["mumo", "qichi"])
def test_internal_event_never_gets_a_visible_handle(actor: str):
    assert make_event(direction="internal", actor=actor, sequence=12).visible_handle is None


def test_platform_event_has_no_visible_handle():
    assert make_event(actor="platform", sequence=12).visible_handle is None
    assert make_event(actor="platform", sequence=12).handle is None


def test_event_json_round_trip_preserves_all_fields():
    event = make_event(reply_to_event_id="quoted-event", reply_to_platform_message_id="quoted-message")

    serialized = event.to_json()
    assert json.loads(serialized)["text"] == "今天在实验室"
    assert ConversationEvent.from_json(serialized) == event
    assert ConversationEvent.from_dict(event.to_dict()) == event


def test_dialogue_objects_json_round_trip():
    segment = MessageSegment("reply", {"id": "message-0"})
    quoted = QuotedTarget(
        conversation_id="owner-1",
        event_id="event-0",
        platform_message_id="message-0",
        handle="Q11",
        actor="qichi",
        text="你自己说的",
        message_segments=(segment,),
        occurred_at_utc=OCCURRED,
    )
    input_value = DialogueInput(
        conversation_id="owner-1",
        trigger_event_ids=("event-1",),
        current_event_handle="M12",
        quoted_target=quoted,
        context_version=3,
        role_messages=(ModelMessage("user", "今天在实验室"),),
        current_time=RECEIVED,
        platform_capabilities=CapabilityManifest({"qq_face": True}),
        source="dialogue",
    )
    result = DialogueResult("那你先忙", "M12", ExpressionIntent("reaction", "heart", "M12"), "primary", 3)

    assert DialogueInput.from_json(input_value.to_json()) == input_value
    assert DialogueResult.from_json(result.to_json()) == result
    assert DialogueSkip.from_json(DialogueSkip("primary", 3).to_json()) == DialogueSkip("primary", 3)


def test_expression_intent_enforces_face_target_rule():
    assert ExpressionIntent("face", "shy", None).target_event_handle is None
    with pytest.raises(ValueError, match="face"):
        ExpressionIntent("face", "shy", "M12")


def test_dialogue_outcomes_reject_removed_fallback_route():
    with pytest.raises(ValueError, match="primary"):
        DialogueResult("text", None, None, "fallback", 1)
    with pytest.raises(ValueError, match="primary"):
        DialogueSkip("fallback", 1)
    with pytest.raises(ValueError, match="kind"):
        ExpressionIntent("sticker", "shy", None)

def test_dialogue_result_round_trips_the_voice_part_index():
    result = DialogueResult(
        "第一条\n\n第二条", None, None, "primary", 3,
        message_parts=("第一条", "第二条"), voice_part_index=2,
    )

    assert DialogueResult.from_json(result.to_json()) == result
    assert result.to_dict()["voice_part_index"] == 2
    assert DialogueResult.from_dict(result.to_dict()) == result


def test_dialogue_result_defaults_to_no_voice_part():
    assert DialogueResult("就一句", None, None, "primary", 1).voice_part_index is None
    assert DialogueResult.from_dict(
        {"text": "就一句", "reply_target": None, "expression_intent": None,
         "model_route": "primary", "context_version": 1, "message_parts": ["就一句"]}
    ).voice_part_index is None


@pytest.mark.parametrize("index", [0, -1, 3, 99])
def test_voice_part_index_must_point_at_an_existing_part(index):
    with pytest.raises(ValueError, match="voice_part_index"):
        DialogueResult("第一条\n\n第二条", None, None, "primary", 3,
                       message_parts=("第一条", "第二条"), voice_part_index=index)


def test_voice_part_index_must_be_an_int():
    with pytest.raises(TypeError, match="voice_part_index"):
        DialogueResult("就一句", None, None, "primary", 1, voice_part_index="1")
    with pytest.raises(TypeError, match="voice_part_index"):
        DialogueResult("就一句", None, None, "primary", 1, voice_part_index=True)

def test_split_voice_part_removes_only_the_spoken_part():
    from qichi.domain.dialogue import split_voice_part

    original = DialogueResult(
        "第一条\n\n第二条", "M9", ExpressionIntent("reaction", "heart", "M9"), "primary", 3,
        message_parts=("第一条", "第二条"), voice_part_index=2,
    )

    remaining, spoken = split_voice_part(original)

    assert spoken == "第二条"
    assert remaining is not None
    assert remaining.message_parts == ("第一条",)
    assert remaining.text == "第一条"
    assert remaining.voice_part_index is None
    assert (remaining.reply_target, remaining.expression_intent) == ("M9", original.expression_intent)


def test_split_voice_part_keeps_the_other_parts_in_order():
    from qichi.domain.dialogue import split_voice_part

    original = DialogueResult(
        "一\n\n二\n\n三", None, None, "primary", 1,
        message_parts=("一", "二", "三"), voice_part_index=2,
    )

    remaining, spoken = split_voice_part(original)

    assert spoken == "二"
    assert remaining is not None and remaining.message_parts == ("一", "三")


def test_split_voice_part_is_a_noop_without_a_voice_part():
    from qichi.domain.dialogue import split_voice_part

    original = DialogueResult("就一句", None, None, "primary", 1)

    remaining, spoken = split_voice_part(original)

    assert remaining is original and spoken == ""


def test_split_voice_part_reports_a_voice_only_turn():
    from qichi.domain.dialogue import split_voice_part

    original = DialogueResult(
        "就这一句", None, None, "primary", 1, message_parts=("就这一句",), voice_part_index=1,
    )

    remaining, spoken = split_voice_part(original)

    assert remaining is None
    assert spoken == "就这一句"


def test_split_voice_part_rejects_non_results():
    from qichi.domain.dialogue import split_voice_part

    with pytest.raises(TypeError):
        split_voice_part("nope")  # type: ignore[arg-type]




def test_memory_requires_evidence_and_round_trips():
    memory = MemoryRecord(
        memory_id="memory-1",
        type="preference",
        normalized_fact="用户喜欢在实验室点外卖",
        modality="explicit_statement",
        status="active",
        valid_from_utc=OCCURRED,
        valid_until_utc=None,
        supersedes_id=None,
        created_at_utc=RECEIVED,
        memory_evidence=(make_evidence(),),
    )

    assert MemoryRecord.from_json(memory.to_json()) == memory
    assert memory.evidence == memory.memory_evidence
    with pytest.raises(FrozenInstanceError):
        memory.status = "rejected"


def test_memory_v2_fields_round_trip_and_preserve_legacy_constructors():
    legacy_evidence = MemoryEvidence(
        "memory-1", "event-1", "mumo", "今天在实验室", OCCURRED
    )
    legacy = MemoryRecord(
        "memory-1",
        "preference",
        "用户喜欢在实验室点外卖",
        "explicit_statement",
        "active",
        OCCURRED,
        None,
        None,
        RECEIVED,
        (legacy_evidence,),
    )
    assert legacy_evidence.evidence_role == "source"
    assert legacy.certainty == "unassessed"
    assert legacy.importance == 0
    assert legacy.temporal_scope == "unclassified"
    assert legacy.assessment_reason_code is None
    assert legacy.assessed_at_utc is None
    assert legacy.recall_scope == "none"

    assessed = MemoryRecord(
        memory_id="memory-1",
        type="preference",
        normalized_fact="用户喜欢在实验室点外卖",
        modality="explicit_statement",
        status="active",
        valid_from_utc=OCCURRED,
        valid_until_utc=None,
        supersedes_id=None,
        created_at_utc=RECEIVED,
        memory_evidence=(make_evidence(evidence_role="confirmation"),),
        certainty="confirmed",
        importance=3,
        temporal_scope="ongoing",
        assessment_reason_code="later_user_confirmation",
        assessed_at_utc=RECEIVED,
    )
    assert assessed.recall_scope == "always"
    assert MemoryRecord.from_json(assessed.to_json()) == assessed


def test_memory_v2_reads_legacy_json_with_safe_defaults():
    legacy = {
        "memory_id": "memory-1",
        "type": "preference",
        "normalized_fact": "用户喜欢在实验室点外卖",
        "modality": "explicit_statement",
        "status": "active",
        "valid_from_utc": OCCURRED.isoformat(),
        "valid_until_utc": None,
        "supersedes_id": None,
        "created_at_utc": RECEIVED.isoformat(),
        "memory_evidence": [
            {
                "memory_id": "memory-1",
                "event_id": "event-1",
                "actor": "mumo",
                "exact_quote": "今天在实验室",
                "occurred_at_utc": OCCURRED.isoformat(),
            }
        ],
    }
    restored = MemoryRecord.from_dict(legacy)
    assert restored.certainty == "unassessed"
    assert restored.importance == 0
    assert restored.temporal_scope == "unclassified"
    assert restored.assessment_reason_code is None
    assert restored.assessed_at_utc is None
    assert restored.memory_evidence[0].evidence_role == "source"
    assert restored.recall_scope == "none"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("certainty", "certain"),
        ("importance", -1),
        ("importance", 4),
        ("importance", True),
        ("temporal_scope", "forever"),
        ("assessment_reason_code", "model_feels_so"),
    ],
)
def test_memory_v2_rejects_invalid_grades(field, value):
    fields = {
        "memory_id": "memory-1",
        "type": "preference",
        "normalized_fact": "事实",
        "modality": "explicit_statement",
        "status": "candidate",
        "valid_from_utc": OCCURRED,
        "valid_until_utc": None,
        "supersedes_id": None,
        "created_at_utc": RECEIVED,
        "memory_evidence": (make_evidence(),),
        "certainty": "ambiguous",
        "importance": 2,
        "temporal_scope": "unclassified",
        "assessment_reason_code": "ambiguous_scope",
        "assessed_at_utc": RECEIVED,
    }
    fields[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        MemoryRecord(**fields)


def test_memory_v2_rejects_invalid_cross_field_combinations():
    base = {
        "memory_id": "memory-1",
        "type": "preference",
        "normalized_fact": "事实",
        "modality": "explicit_statement",
        "status": "candidate",
        "valid_from_utc": OCCURRED,
        "valid_until_utc": None,
        "supersedes_id": None,
        "created_at_utc": RECEIVED,
        "memory_evidence": (make_evidence(),),
    }
    with pytest.raises(ValueError, match="bounded"):
        MemoryRecord(
            **base,
            certainty="ambiguous",
            importance=2,
            temporal_scope="bounded",
            assessment_reason_code="ambiguous_scope",
            assessed_at_utc=RECEIVED,
        )
    with pytest.raises(ValueError, match="assessment_reason_code"):
        MemoryRecord(
            **base,
            certainty="explicit",
            importance=2,
            temporal_scope="ongoing",
            assessed_at_utc=RECEIVED,
        )
    with pytest.raises(ValueError, match="assessed_at_utc"):
        MemoryRecord(
            **base,
            certainty="explicit",
            importance=2,
            temporal_scope="ongoing",
            assessment_reason_code="explicit_user_statement",
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        MemoryRecord(
            **base,
            certainty="explicit",
            importance=2,
            temporal_scope="ongoing",
            assessment_reason_code="explicit_user_statement",
            assessed_at_utc=datetime(2026, 8, 27, 10, 0),
        )
    with pytest.raises(ValueError, match="unassessed"):
        MemoryRecord(**base, importance=1)


@pytest.mark.parametrize(
    ("status", "certainty", "importance", "temporal_scope", "expected"),
    [
        ("active", "explicit", 3, "ongoing", "always"),
        ("active", "confirmed", 2, "ongoing", "topic"),
        ("active", "explicit", 3, "historical", "topic"),
        ("active", "ambiguous", 3, "ongoing", "none"),
        ("active", "explicit", 0, "ongoing", "none"),
        ("candidate", "ambiguous", 2, "unclassified", "confirmation"),
        ("candidate", "ambiguous", 1, "unclassified", "none"),
        ("rejected", "explicit", 3, "ongoing", "none"),
    ],
)
def test_memory_v2_derives_recall_scope(
    status, certainty, importance, temporal_scope, expected
):
    record = MemoryRecord(
        memory_id="memory-1",
        type="preference",
        normalized_fact="事实",
        modality="explicit_statement",
        status=status,
        valid_from_utc=OCCURRED,
        valid_until_utc=None,
        supersedes_id=None,
        created_at_utc=RECEIVED,
        memory_evidence=(make_evidence(),),
        certainty=certainty,
        importance=importance,
        temporal_scope=temporal_scope,
        assessment_reason_code=(
            "ambiguous_scope" if certainty == "ambiguous" else "explicit_user_statement"
        ),
        assessed_at_utc=RECEIVED,
    )
    assert record.recall_scope == expected


def test_memory_review_round_trips_without_free_text_reasoning():
    review = MemoryReview(
        memory_id="memory-1",
        action="confirm",
        certainty="confirmed",
        importance=3,
        temporal_scope="ongoing",
        assessment_reason_code="later_user_confirmation",
        evidence=(make_evidence(evidence_role="confirmation"),),
    )
    restored = MemoryReview.from_json(review.to_json())
    assert restored == review
    assert set(review.to_dict()) == {
        "memory_id", "action", "certainty", "importance", "temporal_scope",
        "assessment_reason_code", "evidence",
    }


def test_adult_memory_requires_non_daily_recall_and_never_becomes_always_by_accident():
    record = MemoryRecord(
        memory_id="adult-history",
        type="episode",
        normalized_fact="过去发生过一段成人共同想象",
        modality="explicit_statement",
        status="active",
        valid_from_utc=OCCURRED,
        valid_until_utc=None,
        supersedes_id=None,
        created_at_utc=RECEIVED,
        memory_evidence=(make_evidence(memory_id="adult-history"),),
        certainty="explicit",
        importance=3,
        temporal_scope="historical",
        assessment_reason_code="historical_event",
        assessed_at_utc=RECEIVED,
        privacy_class="adult",
        recall_policy="explicit_request_only",
    )
    assert record.recall_scope == "topic"
    with pytest.raises(ValueError, match="adult memory"):
        replace(record, recall_policy="daily_safe")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "activate"),
        ("certainty", "unassessed"),
        ("importance", True),
        ("importance", 4),
        ("temporal_scope", "forever"),
        ("assessment_reason_code", "because_model_said_so"),
        ("evidence", ()),
    ],
)
def test_memory_review_rejects_invalid_contract_fields(field, value):
    fields = {
        "memory_id": "memory-1",
        "action": "support",
        "certainty": "explicit",
        "importance": 2,
        "temporal_scope": "ongoing",
        "assessment_reason_code": "explicit_user_statement",
        "evidence": (make_evidence(),),
    }
    fields[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        MemoryReview(**fields)


def test_memory_review_rejects_mismatched_or_invalid_evidence():
    with pytest.raises(ValueError, match="evidence_role"):
        make_evidence(evidence_role="witness")
    with pytest.raises(ValueError, match="memory_id"):
        MemoryReview(
            memory_id="other-memory",
            action="reject",
            certainty="unsupported",
            importance=0,
            temporal_scope="unclassified",
            assessment_reason_code="contradicted_by_user",
            evidence=(make_evidence(),),
        )


def test_memory_rejects_empty_evidence_and_invalid_status():
    fields = {
        "memory_id": "memory-1",
        "type": "preference",
        "normalized_fact": "事实",
        "modality": "explicit_statement",
        "status": "active",
        "valid_from_utc": OCCURRED,
        "valid_until_utc": None,
        "supersedes_id": None,
        "created_at_utc": RECEIVED,
        "memory_evidence": (),
    }
    with pytest.raises(ValueError, match="evidence"):
        MemoryRecord(**fields)
    with pytest.raises(ValueError, match="status"):
        MemoryRecord(**{**fields, "status": "guessed", "memory_evidence": (make_evidence(),)})


def test_agreement_and_correction_json_round_trip():
    evidence = make_evidence()
    agreement = Agreement(
        agreement_id="agreement-1",
        normalized_agreement="明天继续聊",
        status="pending",
        valid_from_utc=OCCURRED,
        valid_until_utc=None,
        created_at_utc=RECEIVED,
        evidence=(evidence,),
    )
    correction = Correction(
        correction_id="correction-1",
        supersedes_memory_id="memory-old",
        normalized_fact="用户今天在实验室",
        created_at_utc=RECEIVED,
        evidence=(evidence,),
    )

    assert Agreement.from_json(agreement.to_json()) == agreement
    assert Correction.from_json(correction.to_json()) == correction


def test_json_fields_are_serializable_without_custom_encoder():
    event = make_event()
    json.dumps(event.to_dict(), ensure_ascii=False)
    json.dumps(CapabilityManifest({"face": ["shy", "heart"]}).to_dict(), ensure_ascii=False)
