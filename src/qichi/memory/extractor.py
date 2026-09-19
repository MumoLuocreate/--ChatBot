from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from qichi.domain.events import ConversationEvent
from qichi.domain.memory_details import MemoryDetailDraft, MemoryFragmentSpec
from qichi.dialogue.llm_client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
)
from qichi.domain.memory import MemoryEvidence, MemoryRecord, MemoryReview


_MAX_CANDIDATES = 12
_MAX_REVIEWS = 12
_MAX_EVIDENCE_PER_ITEM = 4
_MAX_DETAILS = 256
_MAX_DETAIL_EVIDENCE = 8
_MAX_ID_CHARS = 256
_MAX_FACT_CHARS = 1_024
_MAX_QUOTE_CHARS = 2_048
_MAX_LABEL_CHARS = 64
_MAX_DATE_CHARS = 64
_OUTCOME_KINDS = frozenset({"memory_found", "no_persistent_memory"})
_OUTCOME_REASON_CODES = frozenset(
    {
        "explicit_user_preference",
        "historical_episode",
        "bilateral_bounded_agreement",
        "existing_memory_review",
        "candidate_proposed",
        "review_proposed",
        "nothing_new",
        "temporary_scene_or_roleplay",
        "ambiguous_scope",
        "missing_bilateral_acceptance",
        "insufficient_user_evidence",
        "candidate_evidence_invalid",
    }
)
_NO_MEMORY_REASON_CODES = frozenset(
    {
        "nothing_new",
        "temporary_scene_or_roleplay",
        "ambiguous_scope",
        "missing_bilateral_acceptance",
        "insufficient_user_evidence",
        "candidate_evidence_invalid",
    }
)
_FOUND_REASON_CODES = frozenset(
    {
        "explicit_user_preference",
        "historical_episode",
        "bilateral_bounded_agreement",
        "existing_memory_review",
        "candidate_proposed",
        "review_proposed",
    }
)
_MEMORY_TYPES = frozenset(
    {"preference", "agreement", "correction", "episode", "self_expression"}
)
_CANDIDATE_FIELDS = frozenset(
    {
        "type",
        "normalized_fact",
        "modality",
        "certainty",
        "importance",
        "temporal_scope",
        "assessment_reason_code",
        "valid_from_utc",
        "valid_until_utc",
        "supersedes_id",
        "evidence",
        "privacy_class",
        "recall_policy",
    }
)
_REVIEW_FIELDS = frozenset(
    {
        "memory_id",
        "action",
        "certainty",
        "importance",
        "temporal_scope",
        "assessment_reason_code",
        "evidence",
        "privacy_class",
        "recall_policy",
    }
)
_CANDIDATE_REQUIRED_FIELDS = _CANDIDATE_FIELDS - {"privacy_class", "recall_policy"}
_REVIEW_REQUIRED_FIELDS = _REVIEW_FIELDS - {"privacy_class", "recall_policy"}
_EVIDENCE_FIELDS = frozenset({"event_id", "actor", "exact_quote", "role"})
_FRAGMENT_FIELDS = frozenset(
    {"fragment_type", "reality_scope", "summary", "privacy_class", "recall_policy", "closed"}
)
_DETAIL_FIELDS = frozenset(
    {
        "ordinal", "detail_kind", "actor", "reality_scope", "normalized_detail", "exact_quote",
        "source_event_id", "certainty", "temporal_scope", "status", "privacy_class",
        "recall_policy", "evidence",
    }
)
# detail 的 evidence 允许的字段与其它 evidence 相同（_EVIDENCE_FIELDS），但**只要求**
# event_id 与 role 在场：溯源锚点是 detail 自己的 source_event_id 与 exact_quote。
# 2026-09-16：旧版这里另立了一个 {event_id, role} 的**允许**集合，而抽取提示词只声明了
# 一条通用 evidence 规则，模型照提示词写四个字段 → 解析器判「未知字段」→ 整段窗口被隔离、
# 永久进不了记忆（真机 seq 7306-7320，见历史诊断）。
_DETAIL_EVIDENCE_REQUIRED_FIELDS = frozenset({"event_id", "role"})
_CANDIDATE_EVIDENCE_ROLES = {
    # A source plus later user confirmation is retained for compatibility
    # with the extractor's identity-stability contract; storage still
    # requires confirmation through a review action.
    "preference": frozenset({"source", "confirmation"}),
    "episode": frozenset({"source", "confirmation"}),
    "correction": frozenset({"correction"}),
    "self_expression": frozenset({"source"}),
    "agreement": frozenset({"proposal", "acceptance"}),
}
SAFE_PARSE_ERROR_CODES = frozenset(
    {
        "all_items_invalid",
        "candidate_duplicate",
        "candidate_evidence",
        "candidate_invalid",
        "candidate_schema",
        "candidate_time",
        "empty_response",
        "invalid_json",
        "item_limit",
        "response_schema",
        "review_duplicate",
        "review_evidence",
        "review_invalid",
        "review_schema",
        "review_target",
        "review_time",
    }
)


_SCHEMA_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]{1,40}(?:\.[A-Za-z0-9_]{1,40})?)\b")
# Only names from the published contract may be reported.  Error messages can
# quote model text, and a bare first word would carry it into the job ledger.
_SCHEMA_FIELDS = frozenset(
    set(_FRAGMENT_FIELDS)
    | set(_DETAIL_FIELDS)
    | set(_CANDIDATE_FIELDS)
    | set(_REVIEW_FIELDS)
    | set(_EVIDENCE_FIELDS)
    | {
        "outcome", "candidates", "reviews", "fragment", "details", "evidence",
        "valid_from_utc", "valid_until_utc", "supersedes_id", "normalized_fact",
        "detail", "candidate", "review", "status",
    }
)


def _schema_field(reason: str) -> str | None:
    """Name the contract field a schema error points at, or nothing."""

    if not isinstance(reason, str):
        return None
    match = _SCHEMA_FIELD_RE.match(reason.strip())
    if match is None:
        return None
    token = match.group(1)
    if token in _SCHEMA_FIELDS or token.split(".")[-1] in _SCHEMA_FIELDS:
        return token
    return None


class _AllItemsInvalid(ValueError):
    def __init__(self, reason: str, details: Mapping[str, str]) -> None:
        self.details = dict(details)
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ExtractionFailure:
    kind: str
    reason: str
    event_ids: tuple[str, ...]
    # Only a bounded, allow-listed diagnostic vocabulary may leave the
    # extractor.  The worker filters this again before durable persistence.
    details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryOutcome:
    """A short, public conclusion about one frozen memory fragment.

    This is deliberately not a rationale or hidden reasoning trace.  It lets
    operators distinguish a valid "nothing durable here" result from a
    malformed response or an infrastructure failure.
    """

    kind: str
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _OUTCOME_KINDS:
            raise ValueError("outcome kind is invalid")
        if not isinstance(self.reason_code, str) or self.reason_code not in _OUTCOME_REASON_CODES:
            raise ValueError("outcome reason_code is invalid")
        if self.kind == "no_persistent_memory" and self.reason_code not in _NO_MEMORY_REASON_CODES:
            raise ValueError("no_persistent_memory requires a no-memory reason_code")
        if self.kind == "memory_found" and self.reason_code not in _FOUND_REASON_CODES:
            raise ValueError("memory_found requires a memory-found reason_code")


@dataclass(frozen=True, slots=True)
class MemoryExtractionResult:
    candidates: tuple[MemoryRecord, ...]
    failure: ExtractionFailure | None = None
    reviews: tuple[MemoryReview, ...] = ()
    diagnostics: Mapping[str, str] = field(default_factory=dict)
    outcome: MemoryOutcome | None = None
    fragment: MemoryFragmentSpec | None = None
    details: tuple[MemoryDetailDraft, ...] = ()

    def __post_init__(self) -> None:
        if self.fragment is not None and not isinstance(self.fragment, MemoryFragmentSpec):
            raise TypeError("fragment must be a MemoryFragmentSpec or None")
        if not isinstance(self.details, tuple) or not all(
            isinstance(item, MemoryDetailDraft) for item in self.details
        ):
            raise TypeError("details must be a tuple of MemoryDetailDraft")
        if len(self.details) > _MAX_DETAILS:
            raise ValueError("details must contain at most 32 items")

    @property
    def ok(self) -> bool:
        return self.failure is None


class MemoryExtractor:
    def __init__(self, llm: Any):
        self.llm = llm

    async def extract(
        self,
        events: tuple[ConversationEvent, ...],
        *,
        evidence_event_ids: frozenset[str] | None = None,
    ) -> MemoryExtractionResult:
        if not isinstance(events, tuple) or not events or not all(
            isinstance(event, ConversationEvent) for event in events
        ):
            raise TypeError("events must be a non-empty tuple of ConversationEvent")
        event_ids = tuple(event.event_id for event in events)
        if evidence_event_ids is not None and (
            not isinstance(evidence_event_ids, frozenset)
            or not all(isinstance(item, str) for item in evidence_event_ids)
        ):
            raise TypeError("evidence_event_ids must be a frozenset of strings or None")
        if len(set(event_ids)) != len(event_ids):
            return self._failure("input_error", "duplicate event IDs in fragment", event_ids)
        if len({event.conversation_id for event in events}) != 1:
            return self._failure(
                "input_error", "fragment events must share one conversation", event_ids
            )
        allowed_evidence_ids = (
            frozenset(event_ids) if evidence_event_ids is None else evidence_event_ids
        )
        if not allowed_evidence_ids <= frozenset(event_ids):
            return self._failure(
                "input_error",
                "evidence_event_ids must be a subset of the fragment event IDs",
                event_ids,
            )
        try:
            generator = self.llm.generate if hasattr(self.llm, "generate") else self.llm
            response = (
                generator(events)
                if evidence_event_ids is None
                else generator(events, evidence_event_ids=evidence_event_ids)
            )
            if inspect.isawaitable(response):
                response = await response
        except Exception as error:
            kind, details = self._classify_provider_error(error)
            return self._failure(
                kind,
                f"{error.__class__.__name__}: provider generation failed",
                event_ids,
                details=details,
            )
        try:
            candidates, reviews, diagnostics, outcome, fragment, details = self._parse(
                response, events, allowed_evidence_ids
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            reason = str(error) or error.__class__.__name__
            if isinstance(error, json.JSONDecodeError):
                reason = f"invalid JSON: {reason}"
            details = dict(
                error.details
                if isinstance(error, _AllItemsInvalid)
                else {"parse_error_code": self._parse_failure_code(error)}
            )
            # A truncated response and a malformed one look identical in the
            # message alone; finish_reason separates them, and the offending
            # field name says which part of the contract the model missed.
            finish_reason = getattr(self.llm, "last_finish_reason", None)
            if isinstance(finish_reason, str) and finish_reason:
                details["finish_reason"] = finish_reason
            field = _schema_field(reason)
            if field is not None:
                details["schema_field"] = field
            return self._failure("parse_error", reason, event_ids, details=details)
        # A mutually engaged adult scene is itself a historical relationship
        # episode.  Providers may conservatively label the whole frozen
        # fragment "temporary_scene_or_roleplay" even after both sides have
        # continued and ended it.  Preserve that evidence-backed episode
        # without trying to interpret or rewrite the explicit details: the
        # original event ledger remains the source of truth and is only
        # exposed by the privacy/recall gate when the topic is relevant.
        if outcome.kind == "no_persistent_memory" and outcome.reason_code == "temporary_scene_or_roleplay":
            structural = self._structural_sensitive_episode(
                events, allowed_evidence_ids, max(event.received_at_utc for event in events)
            )
            if structural is not None:
                record, structural_fragment, structural_details = structural
                return MemoryExtractionResult(
                    (record,), None, (),
                    {**diagnostics, "structural_sensitive_episode": "1"},
                    MemoryOutcome("memory_found", "historical_episode"),
                    structural_fragment,
                    structural_details,
                )
        return MemoryExtractionResult(
            candidates, None, reviews, diagnostics, outcome, fragment, details
        )

    @staticmethod
    def _structural_sensitive_episode(
        events: tuple[ConversationEvent, ...],
        evidence_event_ids: frozenset[str],
        assessed_at_utc: datetime,
    ) -> tuple[MemoryRecord, MemoryFragmentSpec, tuple[MemoryDetailDraft, ...]] | None:
        """Preserve a completed bilateral scene when semantic extraction abstains.

        This is deliberately structural rather than a word/keyword classifier:
        the primary extractor has already identified the fragment as a
        temporary scene, and the fallback only requires a sufficiently long
        two-sided exchange with at least two user evidence events.  It stores a
        neutral fact and exact user quotes; detailed content stays in the
        immutable event ledger and is never injected into daily context.
        """
        mumo = [
            event
            for event in events
            if event.event_id in evidence_event_ids
            and event.actor == "mumo"
            and event.direction == "inbound"
            and event.kind == "text"
            and isinstance(event.text, str)
            and event.text
        ]
        qichi = [
            event
            for event in events
            if event.actor == "qichi"
            and event.direction == "outbound"
            and event.kind == "text"
            and event.status == "sent"
            and isinstance(event.text, str)
            and event.text
            and isinstance(event.metadata, Mapping)
            and isinstance(event.metadata.get("generation_metadata"), Mapping)
            and event.metadata["generation_metadata"].get("source") in {"dialogue", "interaction"}
        ]
        if len(mumo) < 2 or len(qichi) < 2:
            return None
        ordered = sorted(mumo, key=lambda item: (item.occurred_at_utc, item.sequence, item.event_id))
        selected = ordered if len(ordered) <= _MAX_EVIDENCE_PER_ITEM else (
            ordered[:2] + ordered[-2:]
        )
        selected_by_id: dict[str, ConversationEvent] = {}
        for item in selected:
            selected_by_id.setdefault(item.event_id, item)
        selected = list(selected_by_id.values())[:_MAX_EVIDENCE_PER_ITEM]
        source = selected[0]
        normalized_fact = (
            "双方在该冻结片段中连续接续并完成了一段成人共同想象经历；"
            "具体做法与收束方式只以所列原话和事件账本为准"
        )
        memory_id = str(
            uuid5(
                NAMESPACE_URL,
                "qichi:memory:structural-adult-episode:"
                f"{source.conversation_id}:{source.event_id}:{normalized_fact}",
            )
        )
        evidence = tuple(
            MemoryEvidence(
                memory_id=memory_id,
                event_id=item.event_id,
                actor="mumo",
                exact_quote=item.text or "",
                occurred_at_utc=item.occurred_at_utc,
                evidence_role="source",
            )
            for item in selected
        )
        record = MemoryRecord(
            memory_id=memory_id,
            type="episode",
            normalized_fact=normalized_fact,
            modality="explicit_statement",
            status="candidate",
            valid_from_utc=source.occurred_at_utc,
            valid_until_utc=None,
            supersedes_id=None,
            created_at_utc=source.received_at_utc,
            memory_evidence=evidence,
            # The structure proves a bilateral multi-turn exchange, but it
            # cannot prove from code alone that the scene was completed rather
            # than still hypothetical.  Keep it as a confirmation candidate;
            # later user confirmation or a semantic candidate can promote it.
            certainty="ambiguous",
            importance=2,
            temporal_scope="historical",
            assessment_reason_code="ambiguous_scope",
            assessed_at_utc=assessed_at_utc,
            privacy_class="adult",
            recall_policy="explicit_request_only",
        )
        detail_events = [
            event
            for event in sorted(events, key=lambda item: (item.sequence, item.event_id))
            if event.event_id in evidence_event_ids
            and event.kind == "text"
            and isinstance(event.text, str)
            and event.text
            and event.actor in {"mumo", "qichi"}
        ]
        details = tuple(
            MemoryDetailDraft(
                ordinal=index,
                detail_kind="message",
                actor=event.actor,
                reality_scope="shared_imagination",
                normalized_detail=(
                    "用户在该历史想象片段中说出了一句原话"
                    if event.actor == "mumo"
                    else "角色在该历史想象片段中作出了一次当时的回应"
                ),
                exact_quote=event.text or "",
                source_event_id=event.event_id,
                certainty="explicit",
                temporal_scope="historical",
                status="candidate",
                privacy_class="adult",
                recall_policy="explicit_request_only",
                evidence=((event.event_id, "source"),),
            )
            for index, event in enumerate(detail_events)
        )
        fragment = MemoryFragmentSpec(
            fragment_type="adult",
            reality_scope="shared_imagination",
            summary="双方在该连续片段中共同推进并收束了一段成人共同想象经历；具体原话以事件账本为准",
            privacy_class="adult",
            recall_policy="explicit_request_only",
            closed=True,
        )
        return record, fragment, details

    def _parse(
        self,
        response: Any,
        events: tuple[ConversationEvent, ...],
        evidence_event_ids: frozenset[str],
    ) -> tuple[
        tuple[MemoryRecord, ...],
        tuple[MemoryReview, ...],
        Mapping[str, str],
        MemoryOutcome,
        MemoryFragmentSpec | None,
        tuple[MemoryDetailDraft, ...],
    ]:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty JSON string")
        payload = json.loads(
            self._unwrap_json_fence(response), object_pairs_hook=self._object_pairs
        )
        if not isinstance(payload, Mapping):
            raise TypeError("response must be a JSON object")
        payload_keys = set(payload)
        allowed_payload_keys = {"outcome", "candidates", "reviews", "fragment", "details"}
        if set(payload) == {"candidates"}:
            raise ValueError("legacy empty result is not accepted; outcome is required")
        if set(payload) == {"candidates", "reviews"}:
            # Non-empty legacy V2 responses remain readable during rollout.
            # An empty response is intentionally not accepted here: without
            # an outcome it is impossible to distinguish no-memory from a
            # model omission.
            if payload["candidates"] == [] and payload["reviews"] == []:
                raise ValueError("empty result must include outcome")
            outcome = MemoryOutcome("memory_found", "candidate_proposed")
        elif not {"outcome", "candidates", "reviews"} <= payload_keys:
            raise ValueError("response must contain outcome, candidates, and reviews")
        else:
            if not payload_keys <= allowed_payload_keys:
                raise ValueError("response contains unknown top-level field")
            outcome = self._parse_outcome(payload["outcome"])

        fragment = self._parse_fragment(payload.get("fragment")) if "fragment" in payload else None

        raw_candidates = payload["candidates"]
        raw_reviews = payload["reviews"]
        if not isinstance(raw_candidates, list):
            raise TypeError("candidates must be a list")
        if not isinstance(raw_reviews, list):
            raise TypeError("reviews must be a list")
        if len(raw_candidates) > _MAX_CANDIDATES:
            raise ValueError("candidates must contain at most 12 items")
        if len(raw_reviews) > _MAX_REVIEWS:
            raise ValueError("reviews must contain at most 12 items")

        by_id = {event.event_id: event for event in events}
        assessed_at_utc = max(event.received_at_utc for event in events)
        candidates: list[MemoryRecord] = []
        candidate_ids: set[str] = set()
        candidate_errors: list[str] = []
        first_item_error: str | None = None
        for raw in raw_candidates:
            try:
                record = self._parse_candidate(
                    raw, by_id, evidence_event_ids, assessed_at_utc
                )
                if record.memory_id in candidate_ids:
                    raise ValueError("duplicate candidate identity")
            except (TypeError, ValueError) as error:
                candidate_errors.append(self._item_parse_error_code("candidate", error))
                if first_item_error is None:
                    first_item_error = str(error) or "candidate is invalid"
                continue
            candidate_ids.add(record.memory_id)
            candidates.append(record)

        reviews: list[MemoryReview] = []
        reviewed_ids: set[str] = set()
        review_errors: list[str] = []
        for raw in raw_reviews:
            try:
                review = self._parse_review(raw, by_id, evidence_event_ids)
                if review.memory_id in reviewed_ids:
                    raise ValueError("duplicate review target")
            except (TypeError, ValueError) as error:
                review_errors.append(self._item_parse_error_code("review", error))
                if first_item_error is None:
                    first_item_error = str(error) or "review is invalid"
                continue
            reviewed_ids.add(review.memory_id)
            reviews.append(review)
        details = self._parse_details(
            payload.get("details", []), by_id, evidence_event_ids
        )
        diagnostics = self._parse_diagnostics(candidate_errors, review_errors)
        if (raw_candidates or raw_reviews) and not candidates and not reviews:
            # 2026-09-17（用户裁定后收窄）：原来只要候选项/审核项全被丢弃就**整窗作废**，
            # 于是模型的一条候选项把 actor/role 写反，代价是同一段已经解析好的**明细（时间线）
            # 与片段一起丢掉**——而明细才是回忆入口，候选项只是提议（补录工具本来就不写
            # records 层）。真机上这样的窗口切到 10 条、重试 3 次都过不去（错误类别稳定：
            # candidate evidence role actor is inconsistent / episode requires mumo evidence）。
            # 现在只在**确实什么都没剩下**时才判失败。
            if not details and fragment is None:
                raise _AllItemsInvalid(
                    first_item_error or "all proposed memory items are invalid",
                    {"parse_error_code": "all_items_invalid", **diagnostics},
                )
        if outcome.kind == "no_persistent_memory" and (candidates or reviews):
            raise ValueError("no_persistent_memory cannot contain candidates or reviews")
        if outcome.kind == "memory_found" and not (candidates or reviews or details or fragment):
            raise ValueError("memory_found requires a candidate or review")
        if fragment is None:
            # The prompt treats fragment as optional, so an omission is compliant
            # behaviour -- failing here would throw away candidates and details
            # that already parsed (2026-09-11: five hours of unrecorded memory
            # lost to one missing metadata object).
            fragment = MemoryExtractor._derive_fragment(candidates, details)
        return tuple(candidates), tuple(reviews), diagnostics, outcome, fragment, details

    _DERIVED_FRAGMENT_SUMMARY = "冻结片段的完整原文索引；具体内容以事件账本为准"

    @staticmethod
    def _derive_fragment(
        candidates: Sequence[MemoryRecord],
        details: Sequence[MemoryDetailDraft],
    ) -> MemoryFragmentSpec | None:
        """Rebuild fragment metadata from the parsed items, never inventing any.

        Every value follows from what the model did emit: the strictest privacy
        class present decides the type and the recall policy, the detail scopes
        decide the reality scope, and the session is closed by definition (the
        job only runs after the quiet window elapsed).
        """

        privacies = {item.privacy_class for item in candidates} | {
            item.privacy_class for item in details
        }
        if not privacies:
            return None
        rank = {"ordinary": 0, "intimate": 1, "adult": 2}
        privacy = max(privacies, key=lambda item: rank[item])
        if privacies == {"adult"}:
            fragment_type, recall_policy = "adult", "explicit_request_only"
        elif privacies == {"intimate"}:
            fragment_type, recall_policy = "intimate", "topic_only"
        elif len(privacies) > 1:
            fragment_type = "mixed"
            recall_policy = "explicit_request_only" if "adult" in privacies else "topic_only"
        else:
            fragment_type, recall_policy = "daily", "daily_safe"
        scopes = {item.reality_scope for item in details}
        reality_scope = scopes.pop() if len(scopes) == 1 else ("mixed" if scopes else "conversation")
        return MemoryFragmentSpec(
            fragment_type=fragment_type,
            reality_scope=reality_scope,
            summary=MemoryExtractor._DERIVED_FRAGMENT_SUMMARY,
            privacy_class=privacy,
            recall_policy=recall_policy,
            closed=True,
        )

    @staticmethod
    def _parse_fragment(raw: Any) -> MemoryFragmentSpec:
        if not isinstance(raw, Mapping):
            raise TypeError("fragment must be an object")
        MemoryExtractor._require_exact_fields(raw, _FRAGMENT_FIELDS, "fragment")
        closed = raw["closed"]
        if type(closed) is not bool:
            raise TypeError("fragment.closed must be a bool")
        return MemoryFragmentSpec(
            fragment_type=raw["fragment_type"],
            reality_scope=raw["reality_scope"],
            summary=MemoryExtractor._text(raw["summary"], "fragment.summary", _MAX_FACT_CHARS),
            privacy_class=raw["privacy_class"],
            recall_policy=raw["recall_policy"],
            closed=closed,
        )

    def _parse_details(
        self,
        raw_items: Any,
        by_id: Mapping[str, ConversationEvent],
        evidence_event_ids: frozenset[str],
    ) -> tuple[MemoryDetailDraft, ...]:
        if not isinstance(raw_items, list):
            raise TypeError("details must be a list")
        if len(raw_items) > _MAX_DETAILS:
            raise ValueError("details must contain at most 32 items")
        output: list[MemoryDetailDraft] = []
        for expected_ordinal, raw in enumerate(raw_items):
            self._require_exact_fields(raw, _DETAIL_FIELDS, "detail")
            assert isinstance(raw, Mapping)
            ordinal = raw["ordinal"]
            if type(ordinal) is not int or ordinal != expected_ordinal:
                raise ValueError("detail ordinals must be contiguous")
            evidence_raw = raw["evidence"]
            if not isinstance(evidence_raw, list) or not evidence_raw:
                raise ValueError("detail evidence must be a non-empty list")
            if len(evidence_raw) > _MAX_DETAIL_EVIDENCE:
                raise ValueError("detail evidence must contain at most 8 items")
            evidence: list[tuple[str, str]] = []
            seen: set[str] = set()
            for evidence_item in evidence_raw:
                # 允许与其它 evidence 同形；actor/exact_quote 在这里是冗余信息，
                # 存在即接受、不改变来源（下面按 event_id 与 role 逐条校验）。
                self._require_exact_fields(
                    evidence_item,
                    _EVIDENCE_FIELDS,
                    "detail evidence",
                    required=_DETAIL_EVIDENCE_REQUIRED_FIELDS,
                )
                assert isinstance(evidence_item, Mapping)
                event_id = self._text(evidence_item["event_id"], "detail evidence event_id", _MAX_ID_CHARS)
                role = evidence_item["role"]
                if role not in {"source", "proposal", "acceptance", "correction", "context"}:
                    raise ValueError("detail evidence role is invalid")
                if event_id in seen:
                    raise ValueError("duplicate detail evidence identity")
                if event_id not in by_id:
                    raise ValueError("detail evidence event does not exist")
                if event_id not in evidence_event_ids:
                    raise ValueError("detail evidence event is context-only")
                seen.add(event_id)
                evidence.append((event_id, role))
            source_event_id = self._text(raw["source_event_id"], "source_event_id", _MAX_ID_CHARS)
            source = by_id.get(source_event_id)
            if source is None:
                raise ValueError("detail source event does not exist")
            if source_event_id not in evidence_event_ids:
                raise ValueError("detail source event is context-only")
            actor = raw["actor"]
            if actor not in {"mumo", "qichi", "joint", "unknown"}:
                raise ValueError("detail actor is invalid")
            if actor in {"mumo", "qichi"} and source.actor != actor:
                raise ValueError("detail actor does not match source event")
            quote = self._text(raw["exact_quote"], "exact_quote", _MAX_QUOTE_CHARS)
            if source.text is None or quote not in source.text:
                raise ValueError("detail exact quote is absent from source event")
            if source_event_id not in seen:
                raise ValueError("detail source event must be included in evidence")
            output.append(
                MemoryDetailDraft(
                    ordinal=ordinal,
                    detail_kind=raw["detail_kind"],
                    actor=actor,
                    reality_scope=raw["reality_scope"],
                    normalized_detail=self._text(
                        raw["normalized_detail"], "normalized_detail", _MAX_FACT_CHARS
                    ),
                    exact_quote=quote,
                    source_event_id=source_event_id,
                    certainty=raw["certainty"],
                    temporal_scope=raw["temporal_scope"],
                    status=raw["status"],
                    privacy_class=raw["privacy_class"],
                    recall_policy=raw["recall_policy"],
                    evidence=tuple(evidence),
                )
            )
        return tuple(output)

    @staticmethod
    def _parse_outcome(raw: Any) -> MemoryOutcome:
        if not isinstance(raw, Mapping):
            raise TypeError("outcome must be an object")
        if set(raw) != {"kind", "reason_code"}:
            raise ValueError("outcome must contain only kind and reason_code")
        kind = raw["kind"]
        reason_code = raw["reason_code"]
        if not isinstance(kind, str) or not isinstance(reason_code, str):
            raise TypeError("outcome kind and reason_code must be strings")
        return MemoryOutcome(kind, reason_code)

    @staticmethod
    def _item_parse_error_code(kind: str, error: Exception) -> str:
        reason = str(error)
        if "duplicate" in reason:
            suffix = "duplicate"
        elif any(
            marker in reason
            for marker in (
                "evidence",
                "actor",
                "exact quote",
                "cross-actor",
            )
        ):
            suffix = "evidence"
        elif "field" in reason or "must be an object" in reason:
            suffix = "schema"
        elif "valid_from" in reason or "valid_until" in reason or "timezone" in reason:
            suffix = "time"
        else:
            suffix = "invalid"
        code = f"{kind}_{suffix}"
        return code if code in SAFE_PARSE_ERROR_CODES else f"{kind}_invalid"

    @staticmethod
    def _parse_diagnostics(
        candidate_errors: list[str], review_errors: list[str]
    ) -> Mapping[str, str]:
        details: dict[str, str] = {}
        if candidate_errors:
            details["dropped_candidate_count"] = str(len(candidate_errors))
            details["candidate_error_codes"] = ",".join(sorted(set(candidate_errors)))
        if review_errors:
            details["dropped_review_count"] = str(len(review_errors))
            details["review_error_codes"] = ",".join(sorted(set(review_errors)))
        return details

    @staticmethod
    def _parse_failure_code(error: Exception) -> str:
        if isinstance(error, json.JSONDecodeError):
            return "invalid_json"
        reason = str(error)
        if "non-empty JSON string" in reason or "contain JSON inside" in reason:
            return "empty_response"
        if "at most 12" in reason:
            return "item_limit"
        return "response_schema"

    def _parse_candidate(
        self,
        raw: Any,
        by_id: Mapping[str, ConversationEvent],
        evidence_event_ids: frozenset[str],
        assessed_at_utc: datetime,
    ) -> MemoryRecord:
        self._require_exact_fields(
            raw, _CANDIDATE_FIELDS, "candidate", required=_CANDIDATE_REQUIRED_FIELDS
        )
        assert isinstance(raw, Mapping)

        memory_type = self._text(raw["type"], "type", _MAX_LABEL_CHARS)
        if memory_type not in _MEMORY_TYPES:
            raise ValueError("candidate type is invalid")
        normalized_fact = self._text(
            raw["normalized_fact"], "normalized_fact", _MAX_FACT_CHARS
        )
        modality = self._text(raw["modality"], "modality", _MAX_LABEL_CHARS)
        valid_from = self._date(raw["valid_from_utc"], "valid_from_utc", optional=False)
        valid_until = self._date(raw["valid_until_utc"], "valid_until_utc", optional=True)
        supersedes_id = self._optional_text(
            raw["supersedes_id"], "supersedes_id", _MAX_ID_CHARS
        )
        if supersedes_id is not None and memory_type != "correction":
            raise ValueError("only correction memory may set supersedes_id")
        if memory_type == "correction" and supersedes_id is None:
            raise ValueError("correction requires supersedes_id")
        privacy_class = raw.get("privacy_class", "ordinary")
        recall_policy = raw.get("recall_policy")
        if recall_policy is None:
            recall_policy = (
                "explicit_request_only"
                if privacy_class == "adult" and raw.get("temporal_scope") == "historical"
                else "topic_only"
                if privacy_class in {"adult", "intimate"}
                else "daily_safe"
            )

        required_actor = "qichi" if memory_type == "self_expression" else "mumo"
        raw_evidence = raw["evidence"]
        if not isinstance(raw_evidence, list) or not any(
            isinstance(item, Mapping) and item.get("actor") == required_actor
            for item in raw_evidence
        ):
            raise ValueError(f"{memory_type} requires {required_actor} evidence")
        provisional = self._parse_evidence(
            raw_evidence, "pending", by_id, evidence_event_ids
        )
        self._validate_candidate_evidence_shape(memory_type, provisional)
        if memory_type == "correction" and not any(
            item.actor == "mumo" and item.evidence_role == "correction"
            for item in provisional
        ):
            raise ValueError("correction requires mumo correction evidence")

        anchor = provisional[0]
        source = by_id[anchor.event_id]
        assert valid_from is not None
        memory_id = str(
            uuid5(
                NAMESPACE_URL,
                "qichi:memory:candidate:"
                f"{source.conversation_id}:{anchor.event_id}:{anchor.actor}:{memory_type}:"
                f"{normalized_fact}:{modality}:{anchor.exact_quote}:{valid_from.isoformat()}:"
                f"{valid_until.isoformat() if valid_until is not None else ''}:"
                f"{supersedes_id or ''}",
            )
        )
        evidence = tuple(
            MemoryEvidence(
                memory_id=memory_id,
                event_id=item.event_id,
                actor=item.actor,
                exact_quote=item.exact_quote,
                occurred_at_utc=item.occurred_at_utc,
                evidence_role=item.evidence_role,
            )
            for item in provisional
        )
        return MemoryRecord(
            memory_id=memory_id,
            type=memory_type,
            normalized_fact=normalized_fact,
            modality=modality,
            status="candidate",
            valid_from_utc=valid_from,
            valid_until_utc=valid_until,
            supersedes_id=supersedes_id,
            created_at_utc=source.received_at_utc,
            memory_evidence=evidence,
            certainty=raw["certainty"],
            importance=raw["importance"],
            temporal_scope=raw["temporal_scope"],
            assessment_reason_code=raw["assessment_reason_code"],
            assessed_at_utc=assessed_at_utc,
            privacy_class=privacy_class,
            recall_policy=recall_policy,
        )

    @staticmethod
    def _validate_candidate_evidence_shape(
        memory_type: str, evidence: tuple[MemoryEvidence, ...]
    ) -> None:
        """Enforce ownership roles before a candidate can reach storage."""
        expected_roles = _CANDIDATE_EVIDENCE_ROLES[memory_type]
        actual_roles = {item.evidence_role for item in evidence}
        if memory_type in {"preference", "episode"}:
            if not actual_roles or not actual_roles <= expected_roles or "source" not in actual_roles:
                raise ValueError("candidate evidence roles are inconsistent")
        elif actual_roles != expected_roles:
            raise ValueError("candidate evidence roles are inconsistent")
        if memory_type == "agreement":
            proposals = [item for item in evidence if item.evidence_role == "proposal"]
            acceptances = [item for item in evidence if item.evidence_role == "acceptance"]
            if not any(
                proposal.actor == "mumo"
                and acceptance.actor == "qichi"
                and proposal.event_id != acceptance.event_id
                for proposal in proposals
                for acceptance in acceptances
            ):
                raise ValueError("agreement requires cross-actor proposal and acceptance")
            return
        expected_actor = "qichi" if memory_type == "self_expression" else "mumo"
        if any(item.actor != expected_actor for item in evidence):
            raise ValueError("candidate evidence role actor is inconsistent")

    def _parse_review(
        self,
        raw: Any,
        by_id: Mapping[str, ConversationEvent],
        evidence_event_ids: frozenset[str],
    ) -> MemoryReview:
        self._require_exact_fields(
            raw, _REVIEW_FIELDS, "review", required=_REVIEW_REQUIRED_FIELDS
        )
        assert isinstance(raw, Mapping)
        memory_id = self._text(raw["memory_id"], "memory_id", _MAX_ID_CHARS)
        evidence = self._parse_evidence(
            raw["evidence"], memory_id, by_id, evidence_event_ids
        )
        review = MemoryReview(
            memory_id=memory_id,
            action=raw["action"],
            certainty=raw["certainty"],
            importance=raw["importance"],
            temporal_scope=raw["temporal_scope"],
            assessment_reason_code=raw["assessment_reason_code"],
            evidence=evidence,
            privacy_class=raw.get("privacy_class", "ordinary"),
            recall_policy=raw.get("recall_policy", "daily_safe"),
        )
        self._validate_review_shape(review)
        return review

    @staticmethod
    def _validate_review_shape(review: MemoryReview) -> None:
        """Reject item-local review contradictions before batch persistence."""
        roles = {item.evidence_role for item in review.evidence}
        actors = {item.actor for item in review.evidence}
        if review.action == "confirm":
            if (
                review.certainty != "confirmed"
                or review.assessment_reason_code != "later_user_confirmation"
                or roles != {"confirmation"}
                or actors != {"mumo"}
            ):
                raise ValueError("confirm review is internally inconsistent")
            return
        if review.action == "reject":
            if (
                review.certainty != "unsupported"
                or review.importance != 0
                or review.temporal_scope != "unclassified"
                or review.assessment_reason_code != "contradicted_by_user"
                or roles != {"counterevidence"}
                or actors != {"mumo"}
            ):
                raise ValueError("reject review is internally inconsistent")
            return
        if review.action == "expire":
            if (
                # The repository also requires unsupported here; leaving it out
                # let a review pass this check and fail the whole persistence
                # transaction instead of being dropped and counted.
                review.certainty != "unsupported"
                or review.importance != 0
                or review.assessment_reason_code != "expired_or_completed"
                or roles != {"counterevidence"}
                or actors != {"mumo"}
            ):
                raise ValueError("expire review is internally inconsistent")
            return

        allowed_roles = {
            "explicit_user_statement": {"source"},
            "bilateral_agreement": {"proposal", "acceptance"},
            "user_correction": {"correction"},
            "ambiguous_scope": {"source", "proposal", "acceptance"},
            "historical_event": {"source"},
        }
        expected_certainty = {
            "explicit_user_statement": "explicit",
            "bilateral_agreement": "confirmed",
            "user_correction": review.certainty,
            "ambiguous_scope": "ambiguous",
            "historical_event": "explicit",
        }
        allowed = allowed_roles.get(review.assessment_reason_code)
        if (
            review.action != "support"
            or allowed is None
            or review.certainty != expected_certainty[review.assessment_reason_code]
            or not roles
            or not roles <= allowed
        ):
            raise ValueError("support review is internally inconsistent")
        for item in review.evidence:
            expected_actor = (
                "mumo"
                if item.evidence_role != "acceptance"
                else "qichi"
            )
            if item.actor != expected_actor:
                raise ValueError("support review is internally inconsistent")

    def _parse_evidence(
        self,
        raw_items: Any,
        memory_id: str,
        by_id: Mapping[str, ConversationEvent],
        evidence_event_ids: frozenset[str],
    ) -> tuple[MemoryEvidence, ...]:
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("evidence must be a non-empty list")
        if len(raw_items) > _MAX_EVIDENCE_PER_ITEM:
            raise ValueError("evidence must contain at most 4 items")
        output: list[MemoryEvidence] = []
        identities: set[tuple[str, str]] = set()
        for raw in raw_items:
            self._require_exact_fields(raw, _EVIDENCE_FIELDS, "evidence")
            assert isinstance(raw, Mapping)
            event_id = self._text(raw["event_id"], "event_id", _MAX_ID_CHARS)
            source = by_id.get(event_id)
            if source is None:
                raise ValueError("evidence event does not exist")
            if event_id not in evidence_event_ids:
                raise ValueError("evidence event is context-only and not eligible as evidence")
            actor = raw["actor"]
            if actor not in {"mumo", "qichi", "platform"}:
                raise ValueError("actor is invalid")
            if source.actor != actor:
                raise ValueError("evidence actor does not match source event")
            quote = self._text(raw["exact_quote"], "exact_quote", _MAX_QUOTE_CHARS)
            if source.text is None or quote not in source.text:
                raise ValueError("exact quote is absent from source event")
            identity = (event_id, quote)
            if identity in identities:
                raise ValueError("duplicate evidence identity")
            identities.add(identity)
            output.append(
                MemoryEvidence(
                    memory_id=memory_id,
                    event_id=event_id,
                    actor=actor,
                    exact_quote=quote,
                    occurred_at_utc=source.occurred_at_utc,
                    evidence_role=raw["role"],
                )
            )
        output.sort(
            key=lambda item: (
                item.occurred_at_utc,
                by_id[item.event_id].sequence,
                item.event_id,
                item.exact_quote,
            )
        )
        return tuple(output)

    @staticmethod
    def _require_exact_fields(
        raw: Any,
        expected: frozenset[str],
        name: str,
        *,
        required: frozenset[str] | None = None,
    ) -> None:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{name} must be an object")
        unknown = set(raw) - expected
        missing = (expected if required is None else required) - set(raw)
        if unknown:
            raise ValueError(f"unknown field in {name}")
        if missing:
            raise ValueError(f"missing field: {sorted(missing)[0]}")

    @staticmethod
    def _unwrap_json_fence(response: str) -> str:
        """Accept one exact Markdown JSON wrapper without accepting narration."""
        text = response.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0].strip().casefold() not in {"```", "```json"}
            or lines[-1].strip() != "```"
        ):
            return text
        inner = "\n".join(lines[1:-1]).strip()
        if not inner:
            raise ValueError("response must contain JSON inside the fence")
        return inner

    @staticmethod
    def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    @staticmethod
    def _text(value: Any, field: str, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        if len(value) > maximum:
            raise ValueError(f"{field} exceeds the hard length limit")
        return value

    @classmethod
    def _optional_text(cls, value: Any, field: str, maximum: int) -> str | None:
        if value is None:
            return None
        return cls._text(value, field, maximum)

    @classmethod
    def _date(cls, value: Any, field: str, *, optional: bool) -> datetime | None:
        if value is None:
            if optional:
                return None
            raise ValueError(f"{field} must be an ISO string")
        if not isinstance(value, str):
            raise TypeError(f"{field} must be an ISO string")
        if len(value) > _MAX_DATE_CHARS:
            raise ValueError(f"{field} exceeds the hard length limit")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{field} is invalid") from error
        try:
            offset = parsed.utcoffset()
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{field} is invalid") from error
        if parsed.tzinfo is None or offset is None:
            raise ValueError(f"{field} must be timezone-aware")
        try:
            return parsed.astimezone(timezone.utc)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{field} is invalid") from error

    @staticmethod
    def _classify_provider_error(error: Exception) -> tuple[str, Mapping[str, str]]:
        """Map provider failures to safe operational categories.

        The exception text is intentionally not persisted.  Authentication,
        model and generic request failures retain the historical ``llm_error``
        retry contract while exposing a non-sensitive reason code for support.
        """
        if isinstance(error, LLMTimeoutError):
            return "timeout", {"provider_error": "timeout"}
        if isinstance(error, LLMConnectionError):
            return "network_error", {"provider_error": "connection"}
        if isinstance(error, LLMRateLimitError):
            return "rate_limit", {"provider_error": "rate_limit"}
        if isinstance(error, LLMServerError):
            return "server_error", {"provider_error": "server_error"}
        if isinstance(error, LLMProtocolError):
            return "llm_error", {"provider_error": "protocol"}
        if isinstance(error, LLMAuthenticationError):
            return "llm_error", {"provider_error": "authentication"}
        if isinstance(error, LLMModelNotFoundError):
            return "llm_error", {"provider_error": "model_not_found"}
        if isinstance(error, LLMRequestError):
            return "llm_error", {"provider_error": "request"}
        return "llm_error", {"provider_error": "unknown"}

    @staticmethod
    def _failure(
        kind: str,
        reason: str,
        event_ids: tuple[str, ...],
        *,
        details: Mapping[str, str] | None = None,
    ) -> MemoryExtractionResult:
        return MemoryExtractionResult(
            (), ExtractionFailure(kind, reason, event_ids, dict(details or {})), ()
        )
