from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.transport.normalizer import NormalizationError, normalize_event


NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures" / "onebot"


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def normalize(raw: object, **overrides: object):
    values = {"bot_qq": "20001", "owner_qq": "10001", "received_at_utc": NOW, "event_id_factory": lambda: "local-id"}
    values.update(overrides)
    return normalize_event(raw, **values)


def test_owner_private_message_preserves_segments_reply_and_visible_text(tmp_path):
    raw = fixture("owner_private_message.json")
    original = copy.deepcopy(raw)
    event = normalize(raw)
    assert event is not None
    assert event.event_id == "local-id"
    assert event.platform_event_id == "message:20001:30001"
    assert event.platform_message_id == "30001"
    assert event.conversation_id == "10001"
    assert (event.direction, event.actor, event.kind, event.status) == ("inbound", "mumo", "text", "received")
    assert event.text == "你好，角色"
    assert [segment.type for segment in event.message_segments] == ["reply", "text", "face", "image", "text"]
    assert event.reply_to_event_id is None
    assert event.reply_to_platform_message_id == "29999"
    assert event.occurred_at_utc == datetime.fromtimestamp(1787803200, timezone.utc)
    assert raw == original
    with pytest.raises(Exception):
        event.metadata["changed"] = True
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        assert EventRepository(database).insert(event).platform_event_id == event.platform_event_id
    finally:
        database.close()


def test_repeated_delivery_has_stable_platform_identity_and_new_local_ids():
    raw = fixture("owner_private_message.json")
    ids = iter(("first", "second"))
    first = normalize(raw, event_id_factory=lambda: next(ids))
    second = normalize(raw, event_id_factory=lambda: next(ids))
    assert first is not None and second is not None
    assert first.event_id != second.event_id
    assert first.platform_event_id == second.platform_event_id
    assert first.platform_message_id == second.platform_message_id


def test_identical_same_second_pokes_do_not_claim_a_fake_platform_identity():
    raw = fixture("owner_poke.json")
    ids = iter(("first", "second"))
    first = normalize(raw, event_id_factory=lambda: next(ids))
    second = normalize(raw, event_id_factory=lambda: next(ids))
    assert first is not None and second is not None
    assert first.event_id != second.event_id
    assert first.platform_event_id is None
    assert first.platform_message_id is None
    assert second.platform_event_id is None


def test_outbound_private_message_is_bound_to_bot_not_text():
    event = normalize(fixture("outbound_private_message.json"))
    assert event is not None
    assert (event.direction, event.actor, event.kind, event.status) == ("outbound", "qichi", "text", "sent")
    assert event.text == "收到啦"
    assert event.conversation_id == "10001"


def test_text_segments_only_form_text_and_image_only_does_not_describe_image():
    raw = fixture("owner_private_message.json")
    raw["message"] = [{"type": "image", "data": {"file": "only-image"}}, {"type": "face", "data": {"id": "1"}}]
    event = normalize(raw)
    assert event is not None
    assert event.text is None
    assert [segment.type for segment in event.message_segments] == ["image", "face"]


def test_empty_and_unicode_text_are_preserved_without_platform_ids_in_text():
    raw = fixture("owner_private_message.json")
    raw["message"] = [{"type": "text", "data": {"text": ""}}, {"type": "text", "data": {"text": "雪豹🙂"}}]
    event = normalize(raw)
    assert event is not None
    assert event.text == "雪豹🙂"
    assert "30001" not in event.text


def test_owner_poke_is_structured_platform_input_not_user_text():
    event = normalize(fixture("owner_poke.json"))
    assert event is not None
    assert (event.direction, event.actor, event.kind, event.text) == ("inbound", "mumo", "poke", None)
    assert event.metadata == {"notice_type": "notify", "sub_type": "poke", "sender_id": "10001", "target_id": "20001"}


def test_bot_to_owner_poke_is_derived_from_explicit_sender_id():
    poke = fixture("owner_poke.json")
    poke["sender_id"] = "20001"
    poke["target_id"] = "10001"
    poke_event = normalize(poke)
    assert poke_event is not None
    assert (poke_event.direction, poke_event.actor, poke_event.kind) == ("outbound", "qichi", "poke")


def test_poke_uses_sender_id_and_group_reaction_is_ignored():
    poke = fixture("owner_poke.json")
    poke["target_id"] = "10001"
    with pytest.raises(NormalizationError):
        normalize(poke)
    group_reaction = {"post_type": "notice", "notice_type": "group_msg_emoji_like", "self_id": 20001, "group_id": 1, "user_id": 10001, "message_id": 30001, "likes": [], "is_add": True, "time": 1787803203}
    assert normalize(group_reaction) is None
    assert normalize({"post_type": "notice", "notice_type": "notify", "sub_type": "other"}) is None


@pytest.mark.parametrize(
    "raw",
    [
        {"post_type": "message", "message_type": "group", "self_id": 20001, "user_id": 10001},
        {"post_type": "message", "message_type": "private", "self_id": 20001, "user_id": 99999, "target_id": 99999, "message_id": 1, "time": 1, "message": []},
        {"post_type": "meta_event", "self_id": 20001},
        {"post_type": "notice", "notice_type": "notify", "sub_type": "poke", "self_id": 20001, "user_id": 99999, "sender_id": 99999, "target_id": 20001, "time": 1},
    ],
)
def test_valid_out_of_scope_events_are_ignored(raw):
    assert normalize(raw) is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("self_id", 99999), ("message_id", None), ("time", True), ("time", -1),
        ("message", [{"type": "text", "data": {"text": {"not": "text"}}}]),
        ("message", [{"type": "text", "data": {"bad": {1}}}]),
    ],
)
def test_in_scope_message_malformed_fields_raise_classifiable_error(field, value):
    raw = fixture("owner_private_message.json")
    raw[field] = value
    with pytest.raises(NormalizationError):
        normalize(raw)


def test_malformed_raw_identity_and_reply_targets_fail_closed():
    with pytest.raises(NormalizationError):
        normalize([])
    with pytest.raises(NormalizationError):
        normalize(fixture("owner_private_message.json"), bot_qq=True)
    with pytest.raises(NormalizationError):
        normalize(fixture("owner_private_message.json"), received_at_utc=NOW.replace(tzinfo=None))
    raw = fixture("owner_private_message.json")
    raw["message"] = [{"type": "reply", "data": {"id": "1"}}, {"type": "reply", "data": {"id": "2"}}]
    with pytest.raises(NormalizationError):
        normalize(raw)
    raw = fixture("owner_private_message.json")
    raw["message"] = [{"type": "reply", "data": {"id": True}}]
    with pytest.raises(NormalizationError):
        normalize(raw)


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.pop("sender"),
    lambda raw: raw["sender"].update({"user_id": "99999"}),
])
def test_in_scope_message_requires_sender_identity_consistency(mutate):
    raw = fixture("owner_private_message.json")
    mutate(raw)
    with pytest.raises(NormalizationError):
        normalize(raw)


@pytest.mark.parametrize("message_type", [None, "unknown"])
def test_known_message_post_types_require_private_message_type(message_type):
    raw = fixture("owner_private_message.json")
    if message_type is None:
        raw.pop("message_type")
    else:
        raw["message_type"] = message_type
    with pytest.raises(NormalizationError):
        normalize(raw)


def test_private_identity_contradictions_raise_but_nonowner_events_are_ignored():
    inbound = fixture("owner_private_message.json")
    inbound["target_id"] = "99999"
    with pytest.raises(NormalizationError):
        normalize(inbound)
    outbound = fixture("outbound_private_message.json")
    outbound["user_id"] = "99999"
    outbound["sender"]["user_id"] = "99999"
    with pytest.raises(NormalizationError):
        normalize(outbound)
    outbound = fixture("outbound_private_message.json")
    outbound["target_id"] = "99999"
    assert normalize(outbound) is None
    inbound = fixture("owner_private_message.json")
    inbound["user_id"] = "99999"
    inbound["sender"]["user_id"] = "99999"
    inbound["target_id"] = "99999"
    assert normalize(inbound) is None


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.pop("sender_id"),
    lambda raw: raw.pop("time"),
    lambda raw: raw.update({"sender_id": True}),
    lambda raw: raw.update({"self_id": 99999}),
    lambda raw: raw.update({"sender_id": "10001", "target_id": "10001"}),
    lambda raw: raw.update({"sender_id": "20001", "target_id": "20001"}),
])
def test_in_scope_poke_malformed_or_impossible_identity_raises(mutate):
    raw = fixture("owner_poke.json")
    mutate(raw)
    with pytest.raises(NormalizationError):
        normalize(raw)


def test_event_id_factory_must_be_callable_and_return_nonempty_text():
    raw = fixture("owner_private_message.json")
    with pytest.raises(NormalizationError):
        normalize(raw, event_id_factory="not-callable")
    with pytest.raises(NormalizationError):
        normalize(raw, event_id_factory=lambda: "")


def test_unknown_json_safe_segment_is_preserved_and_non_json_segment_is_rejected():
    raw = fixture("owner_private_message.json")
    raw["message"] = [{"type": "custom", "data": {"x": [None, "safe"]}}]
    event = normalize(raw)
    assert event is not None
    assert event.message_segments[0].to_dict() == {"type": "custom", "data": {"x": [None, "safe"]}}
    raw["message"] = [{"type": "custom", "data": {"x": float("nan")}}]
    with pytest.raises(NormalizationError):
        normalize(raw)
