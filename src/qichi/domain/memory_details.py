from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, TypeAlias


FragmentType: TypeAlias = Literal["daily", "intimate", "adult", "mixed", "unknown"]
RealityScope: TypeAlias = Literal[
    "conversation", "shared_imagination", "hypothetical", "claimed_real", "mixed", "unknown"
]
DetailKind: TypeAlias = Literal[
    "message", "statement", "proposal", "acceptance", "boundary", "choice",
    "agreement", "plan", "uncertainty", "correction", "closure"
]
DetailActor: TypeAlias = Literal["mumo", "qichi", "joint", "unknown"]
DetailStatus: TypeAlias = Literal["candidate", "active", "superseded", "rejected"]
DetailCertainty: TypeAlias = Literal["explicit", "confirmed", "ambiguous", "unsupported"]
DetailTemporalScope: TypeAlias = Literal["historical", "ongoing", "future_plan", "unclassified"]
DetailPrivacyClass: TypeAlias = Literal["ordinary", "intimate", "adult"]
DetailRecallPolicy: TypeAlias = Literal[
    "daily_safe", "topic_only", "explicit_request_only"
]
DetailEvidenceRole: TypeAlias = Literal[
    "source", "proposal", "acceptance", "correction", "context"
]


_FRAGMENT_TYPES = frozenset({"daily", "intimate", "adult", "mixed", "unknown"})
_REALITY_SCOPES = frozenset(
    {"conversation", "shared_imagination", "hypothetical", "claimed_real", "mixed", "unknown"}
)
_DETAIL_KINDS = frozenset(
    {
        "message", "statement", "proposal", "acceptance", "boundary", "choice",
        "agreement", "plan", "uncertainty", "correction", "closure",
    }
)
_DETAIL_ACTORS = frozenset({"mumo", "qichi", "joint", "unknown"})
_DETAIL_STATUSES = frozenset({"candidate", "active", "superseded", "rejected"})
_CERTAINTIES = frozenset({"explicit", "confirmed", "ambiguous", "unsupported"})
_TEMPORAL_SCOPES = frozenset({"historical", "ongoing", "future_plan", "unclassified"})
_PRIVACY_CLASSES = frozenset({"ordinary", "intimate", "adult"})
_RECALL_POLICIES = frozenset({"daily_safe", "topic_only", "explicit_request_only"})
_EVIDENCE_ROLES = frozenset({"source", "proposal", "acceptance", "correction", "context"})


def _text(value: Any, field: str, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{field} exceeds the hard length limit")
    return value


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error


@dataclass(frozen=True, slots=True)
class MemoryFragmentSpec:
    """The model-provided, non-authoritative classification of a frozen fragment."""

    fragment_type: FragmentType
    reality_scope: RealityScope
    summary: str
    privacy_class: DetailPrivacyClass
    recall_policy: DetailRecallPolicy
    closed: bool

    def __post_init__(self) -> None:
        if self.fragment_type not in _FRAGMENT_TYPES:
            raise ValueError("fragment_type is invalid")
        if self.reality_scope not in _REALITY_SCOPES:
            raise ValueError("reality_scope is invalid")
        _text(self.summary, "summary", 1024)
        if self.privacy_class not in _PRIVACY_CLASSES:
            raise ValueError("privacy_class is invalid")
        if self.recall_policy not in _RECALL_POLICIES:
            raise ValueError("recall_policy is invalid")
        if self.privacy_class == "adult" and self.recall_policy == "daily_safe":
            raise ValueError("adult fragment cannot use daily_safe recall policy")
        if type(self.closed) is not bool:
            raise TypeError("closed must be a bool")


@dataclass(frozen=True, slots=True)
class MemoryDetailEvidence:
    event_id: str
    evidence_role: DetailEvidenceRole = "source"

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id", 256)
        if self.evidence_role not in _EVIDENCE_ROLES:
            raise ValueError("evidence_role is invalid")


@dataclass(frozen=True, slots=True)
class MemoryDetailDraft:
    """A detail before the worker binds it to a stable fragment/detail ID."""

    ordinal: int
    detail_kind: DetailKind
    actor: DetailActor
    reality_scope: RealityScope
    normalized_detail: str
    exact_quote: str
    source_event_id: str
    certainty: DetailCertainty
    temporal_scope: DetailTemporalScope
    status: DetailStatus
    privacy_class: DetailPrivacyClass
    recall_policy: DetailRecallPolicy
    evidence: tuple[tuple[str, DetailEvidenceRole], ...]

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if self.detail_kind not in _DETAIL_KINDS:
            raise ValueError("detail_kind is invalid")
        if self.actor not in _DETAIL_ACTORS:
            raise ValueError("actor is invalid")
        if self.reality_scope not in _REALITY_SCOPES:
            raise ValueError("reality_scope is invalid")
        _text(self.normalized_detail, "normalized_detail", 1024)
        _text(self.exact_quote, "exact_quote", 2048)
        _text(self.source_event_id, "source_event_id", 256)
        if self.certainty not in _CERTAINTIES:
            raise ValueError("certainty is invalid")
        if self.temporal_scope not in _TEMPORAL_SCOPES:
            raise ValueError("temporal_scope is invalid")
        if self.status not in _DETAIL_STATUSES:
            raise ValueError("status is invalid")
        if self.privacy_class not in _PRIVACY_CLASSES:
            raise ValueError("privacy_class is invalid")
        if self.recall_policy not in _RECALL_POLICIES:
            raise ValueError("recall_policy is invalid")
        if self.privacy_class == "adult" and self.recall_policy == "daily_safe":
            raise ValueError("adult detail cannot use daily_safe recall policy")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("evidence must contain at least one item")
        if not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and item[0]
            and item[1] in _EVIDENCE_ROLES
            for item in self.evidence
        ):
            raise TypeError("evidence must contain (event_id, role) tuples")
        if len({item[0] for item in self.evidence}) != len(self.evidence):
            raise ValueError("detail evidence event IDs must be unique")


@dataclass(frozen=True, slots=True)
class MemoryFragment:
    fragment_id: str
    conversation_id: str
    fragment_key: str
    start_sequence: int
    end_sequence: int
    start_event_id: str
    end_event_id: str
    started_at_utc: datetime
    ended_at_utc: datetime
    fragment_type: FragmentType
    reality_scope: RealityScope
    summary: str
    privacy_class: DetailPrivacyClass
    recall_policy: DetailRecallPolicy
    status: DetailStatus
    closed_at_utc: datetime | None
    created_at_utc: datetime

    def __post_init__(self) -> None:
        for field in ("fragment_id", "conversation_id", "fragment_key", "start_event_id", "end_event_id"):
            _text(getattr(self, field), field, 256)
        if type(self.start_sequence) is not int or self.start_sequence < 0:
            raise ValueError("start_sequence must be a non-negative integer")
        if type(self.end_sequence) is not int or self.end_sequence < self.start_sequence:
            raise ValueError("end_sequence must not precede start_sequence")
        object.__setattr__(self, "started_at_utc", _utc(self.started_at_utc, "started_at_utc"))
        object.__setattr__(self, "ended_at_utc", _utc(self.ended_at_utc, "ended_at_utc"))
        if self.ended_at_utc < self.started_at_utc:
            raise ValueError("ended_at_utc must not precede started_at_utc")
        if self.fragment_type not in _FRAGMENT_TYPES:
            raise ValueError("fragment_type is invalid")
        if self.reality_scope not in _REALITY_SCOPES:
            raise ValueError("reality_scope is invalid")
        _text(self.summary, "summary", 1024)
        if self.privacy_class not in _PRIVACY_CLASSES:
            raise ValueError("privacy_class is invalid")
        if self.recall_policy not in _RECALL_POLICIES:
            raise ValueError("recall_policy is invalid")
        if self.privacy_class == "adult" and self.recall_policy == "daily_safe":
            raise ValueError("adult fragment cannot use daily_safe recall policy")
        if self.status not in _DETAIL_STATUSES:
            raise ValueError("status is invalid")
        if self.closed_at_utc is not None:
            object.__setattr__(self, "closed_at_utc", _utc(self.closed_at_utc, "closed_at_utc"))
        object.__setattr__(self, "created_at_utc", _utc(self.created_at_utc, "created_at_utc"))


@dataclass(frozen=True, slots=True)
class MemoryDetailRecord:
    detail_id: str
    fragment_id: str
    ordinal: int
    detail_kind: DetailKind
    actor: DetailActor
    reality_scope: RealityScope
    normalized_detail: str
    exact_quote: str
    source_event_id: str
    occurred_at_utc: datetime
    certainty: DetailCertainty
    temporal_scope: DetailTemporalScope
    status: DetailStatus
    privacy_class: DetailPrivacyClass
    recall_policy: DetailRecallPolicy
    evidence: tuple[MemoryDetailEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.detail_id, "detail_id", 256)
        _text(self.fragment_id, "fragment_id", 256)
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if self.detail_kind not in _DETAIL_KINDS:
            raise ValueError("detail_kind is invalid")
        if self.actor not in _DETAIL_ACTORS:
            raise ValueError("actor is invalid")
        if self.reality_scope not in _REALITY_SCOPES:
            raise ValueError("reality_scope is invalid")
        _text(self.normalized_detail, "normalized_detail", 1024)
        _text(self.exact_quote, "exact_quote", 2048)
        _text(self.source_event_id, "source_event_id", 256)
        object.__setattr__(self, "occurred_at_utc", _utc(self.occurred_at_utc, "occurred_at_utc"))
        if self.certainty not in _CERTAINTIES:
            raise ValueError("certainty is invalid")
        if self.temporal_scope not in _TEMPORAL_SCOPES:
            raise ValueError("temporal_scope is invalid")
        if self.status not in _DETAIL_STATUSES:
            raise ValueError("status is invalid")
        if self.privacy_class not in _PRIVACY_CLASSES:
            raise ValueError("privacy_class is invalid")
        if self.recall_policy not in _RECALL_POLICIES:
            raise ValueError("recall_policy is invalid")
        if self.privacy_class == "adult" and self.recall_policy == "daily_safe":
            raise ValueError("adult detail cannot use daily_safe recall policy")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("evidence must contain at least one item")
        if not all(isinstance(item, MemoryDetailEvidence) for item in self.evidence):
            raise TypeError("evidence must contain MemoryDetailEvidence")
        if self.source_event_id not in {item.event_id for item in self.evidence}:
            raise ValueError("source_event_id must be included in detail evidence")

