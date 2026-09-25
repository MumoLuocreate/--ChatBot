from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Literal, Mapping, TypeAlias


MemoryStatus: TypeAlias = Literal["candidate", "active", "superseded", "rejected", "expired"]
MemoryActor: TypeAlias = Literal["mumo", "qichi", "platform"]
AgreementStatus: TypeAlias = Literal["pending", "completed", "cancelled", "expired"]
MemoryCertainty: TypeAlias = Literal[
    "unsupported", "ambiguous", "explicit", "confirmed", "unassessed"
]
MemoryTemporalScope: TypeAlias = Literal[
    "ongoing", "bounded", "historical", "unclassified"
]
MemoryEvidenceRole: TypeAlias = Literal[
    "source", "proposal", "acceptance", "confirmation", "correction", "counterevidence"
]
MemoryAssessmentReason: TypeAlias = Literal[
    "explicit_user_statement",
    "bilateral_agreement",
    "later_user_confirmation",
    "user_correction",
    "ambiguous_scope",
    "historical_event",
    "contradicted_by_user",
    "expired_or_completed",
    "legacy_manual_review",
    "unsupported_or_transient",
]
MemoryRecallScope: TypeAlias = Literal["always", "topic", "confirmation", "none"]
MemoryReviewAction: TypeAlias = Literal["support", "confirm", "reject", "expire"]

# 可作为长期记忆证据的出站来源（2026-09-21 卡⑤：加入 initiative）。
# 用户裁定「主动消息进记忆，但一般主动消息很少有有效信息，除非她找我本身也是一件
# 值得记录的事情」——准入放开后，边界由取证角色矩阵保证（关于用户的记录必须以用户
# 的话为证据；她自己的记录走 self_expression）。
# 有意例外：extractor._structural_sensitive_episode 使用更窄的 {dialogue, interaction}
# ——那里只需要「双边交换够长」的门槛，放进 initiative 会削弱场景结构守卫。
# 这个谓词曾在三处各写一份；现在统一从这里取，避免再次漂移。
MEMORY_RELIABLE_SOURCES: frozenset[str] = frozenset(
    {"dialogue", "interaction", "initiative"}
)
MemoryPrivacyClass: TypeAlias = Literal["ordinary", "intimate", "adult"]
MemoryRecallPolicy: TypeAlias = Literal[
    "daily_safe", "topic_only", "explicit_request_only"
]


_CERTAINTIES = frozenset({"unsupported", "ambiguous", "explicit", "confirmed", "unassessed"})
_TEMPORAL_SCOPES = frozenset({"ongoing", "bounded", "historical", "unclassified"})
_EVIDENCE_ROLES = frozenset(
    {"source", "proposal", "acceptance", "confirmation", "correction", "counterevidence"}
)
_ASSESSMENT_REASONS = frozenset(
    {
        "explicit_user_statement",
        "bilateral_agreement",
        "later_user_confirmation",
        "user_correction",
        "ambiguous_scope",
        "historical_event",
        "contradicted_by_user",
        "expired_or_completed",
        "legacy_manual_review",
        "unsupported_or_transient",
    }
)
_REVIEW_ACTIONS = frozenset({"support", "confirm", "reject", "expire"})
_PRIVACY_CLASSES = frozenset({"ordinary", "intimate", "adult"})
_RECALL_POLICIES = frozenset({"daily_safe", "topic_only", "explicit_request_only"})


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{field_name} must be a non-empty string or None")
    return value


@dataclass(frozen=True)
class MemoryEvidence:
    memory_id: str
    event_id: str
    actor: MemoryActor
    exact_quote: str
    occurred_at_utc: datetime
    evidence_role: MemoryEvidenceRole = "source"

    def __post_init__(self) -> None:
        _text(self.memory_id, "memory_id")
        _text(self.event_id, "event_id")
        if self.actor not in {"mumo", "qichi", "platform"}:
            raise ValueError("actor must be mumo, qichi, or platform")
        _text(self.exact_quote, "exact_quote")
        object.__setattr__(self, "occurred_at_utc", _aware_utc(self.occurred_at_utc, "occurred_at_utc"))
        if self.evidence_role not in _EVIDENCE_ROLES:
            raise ValueError(
                "evidence_role must be source, proposal, acceptance, confirmation, "
                "correction, or counterevidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "event_id": self.event_id,
            "actor": self.actor,
            "exact_quote": self.exact_quote,
            "occurred_at_utc": self.occurred_at_utc.isoformat(),
            "evidence_role": self.evidence_role,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryEvidence":
        if not isinstance(value, Mapping):
            raise TypeError("memory evidence must be a mapping")
        return cls(
            memory_id=value.get("memory_id"),
            event_id=value.get("event_id"),
            actor=value.get("actor"),
            exact_quote=value.get("exact_quote"),
            occurred_at_utc=datetime.fromisoformat(value.get("occurred_at_utc")),
            evidence_role=value.get("evidence_role", "source"),
        )


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    type: str
    normalized_fact: str
    modality: str
    status: MemoryStatus
    valid_from_utc: datetime
    valid_until_utc: datetime | None
    supersedes_id: str | None
    created_at_utc: datetime
    memory_evidence: tuple[MemoryEvidence, ...]
    certainty: MemoryCertainty = "unassessed"
    importance: int = 0
    temporal_scope: MemoryTemporalScope = "unclassified"
    assessment_reason_code: MemoryAssessmentReason | None = None
    assessed_at_utc: datetime | None = None
    privacy_class: MemoryPrivacyClass = "ordinary"
    recall_policy: MemoryRecallPolicy = "daily_safe"

    def __post_init__(self) -> None:
        _text(self.memory_id, "memory_id")
        _text(self.type, "type")
        _text(self.normalized_fact, "normalized_fact")
        _text(self.modality, "modality")
        if self.status not in {"candidate", "active", "superseded", "rejected", "expired"}:
            raise ValueError("status must be candidate, active, superseded, rejected, or expired")
        object.__setattr__(self, "valid_from_utc", _aware_utc(self.valid_from_utc, "valid_from_utc"))
        if self.valid_until_utc is not None:
            object.__setattr__(self, "valid_until_utc", _aware_utc(self.valid_until_utc, "valid_until_utc"))
            if self.valid_until_utc < self.valid_from_utc:
                raise ValueError("valid_until_utc must not precede valid_from_utc")
        _optional_text(self.supersedes_id, "supersedes_id")
        object.__setattr__(self, "created_at_utc", _aware_utc(self.created_at_utc, "created_at_utc"))
        if not isinstance(self.memory_evidence, tuple) or not self.memory_evidence:
            raise ValueError("memory_evidence must contain at least one item")
        if not all(isinstance(item, MemoryEvidence) for item in self.memory_evidence):
            raise TypeError("memory_evidence must be a tuple of MemoryEvidence")
        if any(item.memory_id != self.memory_id for item in self.memory_evidence):
            raise ValueError("memory evidence memory_id must match the record")
        if self.certainty not in _CERTAINTIES:
            raise ValueError(
                "certainty must be unsupported, ambiguous, explicit, confirmed, or unassessed"
            )
        if type(self.importance) is not int:
            raise TypeError("importance must be an integer")
        if not 0 <= self.importance <= 3:
            raise ValueError("importance must be between 0 and 3")
        if self.temporal_scope not in _TEMPORAL_SCOPES:
            raise ValueError(
                "temporal_scope must be ongoing, bounded, historical, or unclassified"
            )
        if self.temporal_scope == "bounded" and self.valid_until_utc is None:
            raise ValueError("bounded temporal_scope requires valid_until_utc")
        if (
            self.assessment_reason_code is not None
            and self.assessment_reason_code not in _ASSESSMENT_REASONS
        ):
            raise ValueError("assessment_reason_code is invalid")
        if self.certainty == "unassessed":
            if (
                self.importance != 0
                or self.temporal_scope != "unclassified"
                or self.assessment_reason_code is not None
                or self.assessed_at_utc is not None
            ):
                raise ValueError("unassessed memory must use safe grading defaults")
        else:
            if self.assessment_reason_code is None:
                raise ValueError("assessed memory requires assessment_reason_code")
            if self.assessed_at_utc is None:
                raise ValueError("assessed memory requires assessed_at_utc")
            object.__setattr__(
                self,
                "assessed_at_utc",
                _aware_utc(self.assessed_at_utc, "assessed_at_utc"),
            )
        if self.privacy_class not in _PRIVACY_CLASSES:
            raise ValueError("privacy_class must be ordinary, intimate, or adult")
        if self.recall_policy not in _RECALL_POLICIES:
            raise ValueError(
                "recall_policy must be daily_safe, topic_only, or explicit_request_only"
            )
        if self.privacy_class == "adult" and self.recall_policy == "daily_safe":
            raise ValueError("adult memory cannot use daily_safe recall policy")

    @property
    def evidence(self) -> tuple[MemoryEvidence, ...]:
        return self.memory_evidence

    @property
    def recall_scope(self) -> MemoryRecallScope:
        if (
            self.status == "candidate"
            and self.certainty == "ambiguous"
            and self.importance in {2, 3}
        ):
            return "confirmation"
        is_factual_active = (
            self.status == "active"
            and self.certainty in {"explicit", "confirmed"}
            and self.importance >= 1
            and self.temporal_scope != "unclassified"
        )
        if not is_factual_active:
            return "none"
        if (
            self.importance == 3
            and self.temporal_scope == "ongoing"
            and self.recall_policy == "daily_safe"
        ):
            return "always"
        return "topic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "type": self.type,
            "normalized_fact": self.normalized_fact,
            "modality": self.modality,
            "status": self.status,
            "valid_from_utc": self.valid_from_utc.isoformat(),
            "valid_until_utc": self.valid_until_utc.isoformat() if self.valid_until_utc is not None else None,
            "supersedes_id": self.supersedes_id,
            "created_at_utc": self.created_at_utc.isoformat(),
            "memory_evidence": [item.to_dict() for item in self.memory_evidence],
            "certainty": self.certainty,
            "importance": self.importance,
            "temporal_scope": self.temporal_scope,
            "assessment_reason_code": self.assessment_reason_code,
            "assessed_at_utc": (
                self.assessed_at_utc.isoformat() if self.assessed_at_utc is not None else None
            ),
            "privacy_class": self.privacy_class,
            "recall_policy": self.recall_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryRecord":
        if not isinstance(value, Mapping):
            raise TypeError("memory record must be a mapping")
        evidence = value.get("memory_evidence")
        if not isinstance(evidence, (list, tuple)):
            raise TypeError("memory_evidence must be a list")
        valid_until = value.get("valid_until_utc")
        assessed_at = value.get("assessed_at_utc")
        return cls(
            memory_id=value.get("memory_id"),
            type=value.get("type"),
            normalized_fact=value.get("normalized_fact"),
            modality=value.get("modality"),
            status=value.get("status"),
            valid_from_utc=datetime.fromisoformat(value.get("valid_from_utc")),
            valid_until_utc=datetime.fromisoformat(valid_until) if valid_until is not None else None,
            supersedes_id=value.get("supersedes_id"),
            created_at_utc=datetime.fromisoformat(value.get("created_at_utc")),
            memory_evidence=tuple(MemoryEvidence.from_dict(item) for item in evidence),
            certainty=value.get("certainty", "unassessed"),
            importance=value.get("importance", 0),
            temporal_scope=value.get("temporal_scope", "unclassified"),
            assessment_reason_code=value.get("assessment_reason_code"),
            assessed_at_utc=(
                datetime.fromisoformat(assessed_at) if assessed_at is not None else None
            ),
            privacy_class=value.get("privacy_class", "ordinary"),
            recall_policy=value.get("recall_policy", "daily_safe"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "MemoryRecord":
        return cls.from_dict(json.loads(value))


def working_set_resident(
    record: Any,
    at_utc: datetime,
    *,
    episode_max_age_days: int,
) -> bool:
    """这一条是否还该留在**常驻**工作集里（卡B④ 的年龄门）。

    只有 episode 会因年龄退场。偏好、约定、纠正是常驻关系状态（AGENTS.md 冻结
    不变式），再老也不退——2026-09-22 的审计里 107 条 active preference 全部常驻，
    那是有意为之，不是冗余。

    **退场不等于删除**：记录、证据、索引都还在库里，仍可被检索按相关度带回来；
    这里只决定它是否每轮都在场。episode_max_age_days=0 表示不按年龄退场。

    判据与实测见 doc/方案-20260922-写入侧作用范围与移出可逆.md 第 5 节。
    """
    if type(episode_max_age_days) is not int or episode_max_age_days < 0:
        raise ValueError("episode_max_age_days must be a non-negative integer")
    if episode_max_age_days == 0 or getattr(record, "type", None) != "episode":
        return True
    if not isinstance(at_utc, datetime):
        raise TypeError("at_utc must be a datetime")
    age = _aware_utc(at_utc, "at_utc") - record.valid_from_utc
    return age <= timedelta(days=episode_max_age_days)


@dataclass(frozen=True)
class MemoryReview:
    memory_id: str
    action: MemoryReviewAction
    certainty: MemoryCertainty
    importance: int
    temporal_scope: MemoryTemporalScope
    assessment_reason_code: MemoryAssessmentReason
    evidence: tuple[MemoryEvidence, ...]
    privacy_class: MemoryPrivacyClass = "ordinary"
    recall_policy: MemoryRecallPolicy = "daily_safe"

    def __post_init__(self) -> None:
        _text(self.memory_id, "memory_id")
        if self.action not in _REVIEW_ACTIONS:
            raise ValueError("action must be support, confirm, reject, or expire")
        if self.certainty not in _CERTAINTIES or self.certainty == "unassessed":
            raise ValueError(
                "certainty must be unsupported, ambiguous, explicit, or confirmed"
            )
        if type(self.importance) is not int:
            raise TypeError("importance must be an integer")
        if not 0 <= self.importance <= 3:
            raise ValueError("importance must be between 0 and 3")
        if self.temporal_scope not in _TEMPORAL_SCOPES:
            raise ValueError(
                "temporal_scope must be ongoing, bounded, historical, or unclassified"
            )
        if self.assessment_reason_code not in _ASSESSMENT_REASONS:
            raise ValueError("assessment_reason_code is invalid")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("evidence must contain at least one item")
        if not all(isinstance(item, MemoryEvidence) for item in self.evidence):
            raise TypeError("evidence must be a tuple of MemoryEvidence")
        if any(item.memory_id != self.memory_id for item in self.evidence):
            raise ValueError("evidence memory_id must match the review memory_id")
        if self.privacy_class not in _PRIVACY_CLASSES:
            raise ValueError("privacy_class must be ordinary, intimate, or adult")
        if self.recall_policy not in _RECALL_POLICIES:
            raise ValueError(
                "recall_policy must be daily_safe, topic_only, or explicit_request_only"
            )
        if self.privacy_class == "adult" and self.recall_policy == "daily_safe":
            raise ValueError("adult review cannot use daily_safe recall policy")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "memory_id": self.memory_id,
            "action": self.action,
            "certainty": self.certainty,
            "importance": self.importance,
            "temporal_scope": self.temporal_scope,
            "assessment_reason_code": self.assessment_reason_code,
            "evidence": [item.to_dict() for item in self.evidence],
        }
        if self.privacy_class != "ordinary" or self.recall_policy != "daily_safe":
            result["privacy_class"] = self.privacy_class
            result["recall_policy"] = self.recall_policy
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryReview":
        if not isinstance(value, Mapping):
            raise TypeError("memory review must be a mapping")
        evidence = value.get("evidence")
        if not isinstance(evidence, (list, tuple)):
            raise TypeError("evidence must be a list")
        return cls(
            memory_id=value.get("memory_id"),
            action=value.get("action"),
            certainty=value.get("certainty"),
            importance=value.get("importance"),
            temporal_scope=value.get("temporal_scope"),
            assessment_reason_code=value.get("assessment_reason_code"),
            evidence=tuple(MemoryEvidence.from_dict(item) for item in evidence),
            privacy_class=value.get("privacy_class", "ordinary"),
            recall_policy=value.get("recall_policy", "daily_safe"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "MemoryReview":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class Agreement:
    agreement_id: str
    normalized_agreement: str
    status: AgreementStatus
    valid_from_utc: datetime
    valid_until_utc: datetime | None
    created_at_utc: datetime
    evidence: tuple[MemoryEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.agreement_id, "agreement_id")
        _text(self.normalized_agreement, "normalized_agreement")
        if self.status not in {"pending", "completed", "cancelled", "expired"}:
            raise ValueError("status must be pending, completed, cancelled, or expired")
        object.__setattr__(self, "valid_from_utc", _aware_utc(self.valid_from_utc, "valid_from_utc"))
        if self.valid_until_utc is not None:
            object.__setattr__(self, "valid_until_utc", _aware_utc(self.valid_until_utc, "valid_until_utc"))
            if self.valid_until_utc < self.valid_from_utc:
                raise ValueError("valid_until_utc must not precede valid_from_utc")
        object.__setattr__(self, "created_at_utc", _aware_utc(self.created_at_utc, "created_at_utc"))
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("evidence must contain at least one item")
        if not all(isinstance(item, MemoryEvidence) for item in self.evidence):
            raise TypeError("evidence must be a tuple of MemoryEvidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement_id": self.agreement_id,
            "normalized_agreement": self.normalized_agreement,
            "status": self.status,
            "valid_from_utc": self.valid_from_utc.isoformat(),
            "valid_until_utc": self.valid_until_utc.isoformat() if self.valid_until_utc is not None else None,
            "created_at_utc": self.created_at_utc.isoformat(),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Agreement":
        if not isinstance(value, Mapping):
            raise TypeError("agreement must be a mapping")
        evidence = value.get("evidence")
        if not isinstance(evidence, (list, tuple)):
            raise TypeError("evidence must be a list")
        valid_until = value.get("valid_until_utc")
        return cls(
            agreement_id=value.get("agreement_id"),
            normalized_agreement=value.get("normalized_agreement"),
            status=value.get("status"),
            valid_from_utc=datetime.fromisoformat(value.get("valid_from_utc")),
            valid_until_utc=datetime.fromisoformat(valid_until) if valid_until is not None else None,
            created_at_utc=datetime.fromisoformat(value.get("created_at_utc")),
            evidence=tuple(MemoryEvidence.from_dict(item) for item in evidence),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "Agreement":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class Correction:
    correction_id: str
    supersedes_memory_id: str
    normalized_fact: str
    created_at_utc: datetime
    evidence: tuple[MemoryEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.correction_id, "correction_id")
        _text(self.supersedes_memory_id, "supersedes_memory_id")
        _text(self.normalized_fact, "normalized_fact")
        object.__setattr__(self, "created_at_utc", _aware_utc(self.created_at_utc, "created_at_utc"))
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("evidence must contain at least one item")
        if not all(isinstance(item, MemoryEvidence) for item in self.evidence):
            raise TypeError("evidence must be a tuple of MemoryEvidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "supersedes_memory_id": self.supersedes_memory_id,
            "normalized_fact": self.normalized_fact,
            "created_at_utc": self.created_at_utc.isoformat(),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Correction":
        if not isinstance(value, Mapping):
            raise TypeError("correction must be a mapping")
        evidence = value.get("evidence")
        if not isinstance(evidence, (list, tuple)):
            raise TypeError("evidence must be a list")
        return cls(
            correction_id=value.get("correction_id"),
            supersedes_memory_id=value.get("supersedes_memory_id"),
            normalized_fact=value.get("normalized_fact"),
            created_at_utc=datetime.fromisoformat(value.get("created_at_utc")),
            evidence=tuple(MemoryEvidence.from_dict(item) for item in evidence),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "Correction":
        return cls.from_dict(json.loads(value))
