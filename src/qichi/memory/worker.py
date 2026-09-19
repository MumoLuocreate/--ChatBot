from __future__ import annotations

"""Durable, post-send Memory V2 session consolidation."""

import asyncio
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from qichi.domain.events import ConversationEvent
from qichi.memory.detail_pass import MemoryDetailPass, MemoryDetailPassError
from qichi.memory.extractor import (
    SAFE_PARSE_ERROR_CODES,
    ExtractionFailure,
    MemoryExtractionResult,
    MemoryExtractor,
    MemoryOutcome,
)
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_detail_repository import (
    MAX_DETAILS_PER_FRAGMENT,
    MemoryDetailRepository,
)
from qichi.storage.memory_repository import MemoryBatchResult, MemoryRepository


_MAX_DIAGNOSTIC_ITEMS = 12
_OUTCOME_REASON_KINDS = {
    "memory_found": frozenset(
        {
            "explicit_user_preference",
            "historical_episode",
            "bilateral_bounded_agreement",
            "existing_memory_review",
            "candidate_proposed",
            "review_proposed",
        }
    ),
    "no_persistent_memory": frozenset(
        {
            "nothing_new",
            "temporary_scene_or_roleplay",
            "ambiguous_scope",
            "missing_bilateral_acceptance",
            "insufficient_user_evidence",
            "candidate_evidence_invalid",
        }
    ),
}

# Diagnostic vocabulary for the job ledger: fixed codes only, never free text.
_SAFE_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "other"})
# 2026-09-12：时间线补跑失败过去是静默的（片段落地、时间线为空、什么都没记），
# 于是「这一夜为什么是空的」查不出来。这几个码只说明失败的种类。
_SAFE_DETAIL_PASS_CODES = frozenset(
    {"detail_pass_invalid", "detail_pass_timeout", "detail_pass_error"}
)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{1,40}$")

# 2026-09-17：明细覆盖自检的末尾余量。末尾几条可能是收尾、被合并进上一条、或本来就没什么
# 可记的，允许它们没有自己的明细；再往前就没盖住，说明时间线提前停了。
_COVERAGE_TAIL_MARGIN = 3


def _drafts_cover_the_window(events: tuple, drafts: tuple) -> bool:
    """这一批明细有没有盖到窗口末尾（2026-09-17）。

    起因（真机）：seq 7551-7726（172 条事件）的片段，明细只有 29 条、只盖到第 57 条；
    凌晨那句「什么时候会想要我」落在覆盖之外，当晚他问起时她只能说「我这儿翻不着」——
    不是幻觉，也不怪她，是那段尾巴根本没进记忆层（她复述出来的那几件，全部落在覆盖之内）。

    原来那道保险丝为什么烧不断：提示词叫模型自己「最多 32 条、取最重要的 32 条」，
    于是 len(drafts) > MAX_DETAILS_PER_FRAGMENT 永远不成立，长片段必然截断。
    覆盖自检把触发条件换成「明细有没有盖到末尾」——这才是有信息量的那一半。

    没有明细不算截断（那是另一条路径的事，这里不为它触发切分）。
    """

    if not events or not drafts:
        return True
    order = {event.event_id: event.sequence for event in events}
    covered = [
        order[draft.source_event_id]
        for draft in drafts
        if getattr(draft, "source_event_id", None) in order
    ]
    if not covered:
        return False
    return max(covered) >= max(event.sequence for event in events) - _COVERAGE_TAIL_MARGIN


def _detail_pass_failure_code(error: BaseException) -> str:
    """Name what went wrong with the timeline call, in the fixed vocabulary."""

    if isinstance(error, MemoryDetailPassError):
        return "detail_pass_invalid"
    if "timeout" in type(error).__name__.casefold():
        return "detail_pass_timeout"
    return "detail_pass_error"


def safe_domain_error(message: Any) -> str | None:
    """Keep a code-owned validation message, never model or driver text.

    Domain checks speak in short ASCII sentences about field names; model output
    is free text (and, in this project, almost always Chinese), so shape is a
    reliable and cheap filter.
    """

    if not isinstance(message, str) or not 5 <= len(message) <= 120:
        return None
    if message[0].islower() is False:
        return None
    if any(character in message for character in "\"'\\/"):
        return None
    if not all(0x20 <= ord(character) < 0x7F for character in message):
        return None
    return message


@dataclass(frozen=True, slots=True)
class WorkerRun:
    fragment_key: str
    written_memory_ids: tuple[str, ...] = ()
    activated_memory_ids: tuple[str, ...] = ()
    cancelled: bool = False
    failed: bool = False
    failure: ExtractionFailure | None = None


@dataclass(frozen=True, slots=True)
class _Snapshot:
    conversation_id: str
    events: tuple[ConversationEvent, ...]
    start_sequence: int
    end_sequence: int
    anchor_event_id: str
    anchor_sequence: int
    anchor_received_at_utc: datetime
    deadline_utc: datetime
    context_version: int
    fragment_key: str


@dataclass(frozen=True, slots=True)
class _Claim:
    job: Mapping[str, Any]
    snapshot: _Snapshot
    canonical_snapshot: _Snapshot
    token: str
    owner: str
    lease_until_utc: datetime


# 2026-09-15：明细上限是**每片段** 32 条，而 detail pass 偶尔一条事件给一条明细
# （真机 35 条事件 → 35 条明细）。旧行为是 build_details 直接抛错 → 整段原文的证据与明细
# 全都落不了库。**超上限时切分，不丢整段**：对半切、各自重抽，必要时递归到这个深度。
#
# 2026-09-17：3 → 4。深度 3 最多 8 段，而对半切到最底层时单元仍有 40+ 条事件，明细又会
# 撞上每片段 32 条的上限——实测残留 ddb8912d 覆盖 56 条事件却只盖到第 32 条。补录量化：
# 19 个待补目标里 330/313/261 条那三个需要约 11 段，深度 4（16 段）即可全部覆盖。
# 这一处**生产同样受影响**（同一函数在 worker 的正常路径上），所以将来的长夜也会被截断。
# 回退：改回 3（代价是超大窗口的尾巴落不了明细）。
MEMORY_SPLIT_MAX_DEPTH = 4


class MemoryWorker:
    """Run one durable 30-minute consolidation job per conversation.

    Notification only records a canonical snapshot.  The extractor is called
    after a short claim transaction has committed and released SQLite.
    """

    SESSION_IDLE_MINUTES = 30
    # A fragment has to stay small enough that the extractor can return a complete
    # detail timeline for it inside its own output budget (max_output_tokens 4096).
    # At 131_072 a single fragment swallowed a whole evening of 182 events and the
    # model then timed out and returned invalid JSON, so nothing was recorded.
    DEFAULT_MAX_FRAGMENT_TOKENS = 4_096
    DEFAULT_LEASE_SECONDS = 600
    NETWORK_RETRY_DELAYS_SECONDS = (30, 120, 600)
    PARSE_RETRY_DELAY_SECONDS = 1
    TERMINAL_REOPEN_COOLDOWN = timedelta(minutes=30)
    _TRANSIENT_FAILURE_CATEGORIES = frozenset(
        {
            "llm_error",
            "network_error",
            "timeout",
            "rate_limit",
            "server_error",
            "repository_error",
        }
    )
    # A deterministic/content-bound failure cannot be retried safely from a
    # partial response.  Its event range is quarantined (never marked
    # processed) so later sessions can proceed while the original evidence
    # remains available for a future explicit repair.
    _QUARANTINABLE_FAILURE_CATEGORIES = frozenset(
        {"oversized_event", "parse_error", "schema_error", "evidence_error", "worker_error"}
    )
    # 2026-09-15：内容层失败不再「一次定生死」，先给这么多次**冷却重试**再隔离。
    # 起因：DeepSeek 官方崩溃那一晚，后台模型连续超时并返回残缺 JSON，一次 parse_error
    # 就让 seq 6566–6600（35 条，正是用户纠正她的那一段）在两次尝试后被永久隔离，
    # 那一段再没进记忆层（历史修复计划）。
    # 关键判断：**上游崩坏时返回的残缺 JSON，和「这段内容真的抽不出来」长得一模一样**，
    # 所以只能靠「等一会儿再试一次」来区分，不能靠一次失败下结论。
    # 代价有界：每段最多多 CONTENT_FAILURE_REOPEN_ROUNDS 次后台调用，间隔 ≥ 冷却时间。
    CONTENT_FAILURE_REOPEN_ROUNDS = 2
    _OWNER_RE = re.compile(r"^memory-(\d+)-")

    def __init__(
        self,
        extractor: MemoryExtractor,
        repository: MemoryRepository,
        *,
        quiet_minutes: int = SESSION_IDLE_MINUTES,
        auto_commit: str = "explicit_only",
        max_fragment_tokens: int = DEFAULT_MAX_FRAGMENT_TOKENS,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        retry_delays_seconds: tuple[int, int, int] = NETWORK_RETRY_DELAYS_SECONDS,
        parse_retry_delay_seconds: int = PARSE_RETRY_DELAY_SECONDS,
        database: Database | None = None,
        conversation_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
        owner: str | None = None,
        owner_alive: Callable[[str], bool] | None = None,
        detail_pass: MemoryDetailPass | None = None,
    ) -> None:
        if not isinstance(extractor, MemoryExtractor):
            raise TypeError("extractor must be a MemoryExtractor")
        if not isinstance(repository, MemoryRepository):
            raise TypeError("repository must be a MemoryRepository")
        if type(quiet_minutes) is not int or quiet_minutes != self.SESSION_IDLE_MINUTES:
            raise ValueError("quiet_minutes is fixed at 30 for Memory V2")
        if auto_commit not in {"explicit_only", "verified_explicit"}:
            raise ValueError("auto_commit must be explicit_only or verified_explicit")
        if type(max_fragment_tokens) is not int or max_fragment_tokens < 1:
            raise ValueError("max_fragment_tokens must be positive")
        if type(lease_seconds) is not int or lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        if (
            not isinstance(retry_delays_seconds, tuple)
            or len(retry_delays_seconds) != 3
            or any(type(item) is not int or item < 1 for item in retry_delays_seconds)
        ):
            raise ValueError("retry_delays_seconds must contain three positive integers")
        if type(parse_retry_delay_seconds) is not int or parse_retry_delay_seconds < 1:
            raise ValueError("parse_retry_delay_seconds must be positive")
        if database is not None and not isinstance(database, Database):
            raise TypeError("database must be a Database")
        if database is not None and (not isinstance(conversation_id, str) or not conversation_id):
            raise ValueError("conversation_id is required with database")
        if conversation_id is not None and (not isinstance(conversation_id, str) or not conversation_id):
            raise ValueError("conversation_id must be non-empty text")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if owner_alive is not None and not callable(owner_alive):
            raise TypeError("owner_alive must be callable")
        if owner is not None and (not isinstance(owner, str) or not owner):
            raise ValueError("owner must be non-empty text")

        self.extractor = extractor
        self.repository = repository
        self.detail_pass = detail_pass
        self.database = database
        self.conversation_id = conversation_id
        self.detail_repository = (
            MemoryDetailRepository(database) if database is not None else None
        )
        self.auto_commit = auto_commit
        self.max_fragment_tokens = max_fragment_tokens
        self.lease_seconds = lease_seconds
        self.retry_delays_seconds = retry_delays_seconds
        self.parse_retry_delay_seconds = parse_retry_delay_seconds
        self._clock_explicit = clock is not None
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.owner = owner or f"memory-{os.getpid()}-{uuid.uuid4().hex}"
        self._owner_alive_probe = owner_alive
        self._semantic_gate = False
        self._last_failures: dict[str, ExtractionFailure] = {}

    @property
    def semantic_gate_open(self) -> bool:
        return self._semantic_gate

    def open_semantic_gate(self) -> None:
        self._semantic_gate = True

    @staticmethod
    def _cursor_key(conversation_id: str) -> str:
        return f"memory_worker:{conversation_id}:processed_sequence"

    @staticmethod
    def _baseline_key(conversation_id: str) -> str:
        return f"memory_worker:{conversation_id}:baseline_sequence"

    @staticmethod
    def _quarantine_key(conversation_id: str) -> str:
        return f"memory_worker:{conversation_id}:quarantined_sequence"

    def _conversation(self, conversation_id: str | None = None) -> str:
        value = conversation_id or self.conversation_id
        if not isinstance(value, str) or not value:
            raise ValueError("conversation_id must be non-empty text")
        if self.conversation_id is not None and value != self.conversation_id:
            raise ValueError("conversation_id does not match configured worker")
        return value

    def _now(self) -> datetime:
        return self._utc(self.clock(), "clock result")

    def recover_pending(self, conversation_id: str | None = None) -> bool:
        """Recover local job state without invoking the extractor."""
        if self.database is None:
            return False
        conversation = self._conversation(conversation_id)
        now = self._now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM memory_session_jobs WHERE conversation_id = ?",
                (conversation,),
            ).fetchone()
            if row is not None:
                if row["status"] == "claimed" and self._claim_reclaimable(row, now):
                    connection.execute(
                        "UPDATE memory_session_jobs SET status='pending', claim_token=NULL, "
                        "claim_owner=NULL, claim_lease_until_utc=NULL, updated_at_utc=? "
                        "WHERE job_id=? AND status='claimed'",
                        (now.isoformat(), row["job_id"]),
                    )
                elif row["status"] == "failed":
                    if self._failure_is_retryable_tx(connection, row):
                        if self._terminal_failure_old_enough(row, now):
                            snapshot = self._snapshot_tx(connection, conversation)
                            if snapshot is not None:
                                self._reopen_failed_tx(connection, row, snapshot, now)
                    else:
                        # A content-bound failure must not pin the cursor.  Keep
                        # its range unresolved, but move the scan cursor past it
                        # and create a later job when later reliable activity is
                        # already present.
                        self._quarantine_range_tx(connection, row, now)
                        snapshot = self._snapshot_tx(connection, conversation)
                        if snapshot is not None:
                            self._upsert_snapshot_tx(
                                connection,
                                snapshot,
                                now,
                                revision=int(row["revision"]) + 1,
                                created_at=row["created_at_utc"],
                            )
                            reopened = connection.execute(
                                "SELECT * FROM memory_session_jobs WHERE job_id=?",
                                (row["job_id"],),
                            ).fetchone()
                            if reopened is not None:
                                self._append_job_event_tx(
                                    connection, reopened, "reopened", now=now
                                )
                return True
            snapshot = self._snapshot_tx(connection, conversation)
            if snapshot is None:
                return False
            self._upsert_snapshot_tx(connection, snapshot, now)
            return True

    def bootstrap_existing_events(self, conversation_id: str | None = None) -> int:
        """Atomically mark events present before worker deployment as a boundary."""
        if self.database is None:
            raise ValueError("durable database and conversation_id are required")
        conversation = self._conversation(conversation_id)
        key = self._baseline_key(conversation)
        now = self._now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key=?", (key,)
            ).fetchone()
            if row is not None:
                return self._decode_meta_int(row["value_json"], key)
            row = connection.execute(
                "SELECT MAX(sequence) AS sequence FROM conversation_events WHERE conversation_id=?",
                (conversation,),
            ).fetchone()
            baseline = int(row["sequence"]) if row and row["sequence"] is not None else -1
            connection.execute(
                "INSERT INTO runtime_meta(key,value_json,updated_at_utc) VALUES (?,?,?)",
                (key, json.dumps(baseline), now.isoformat()),
            )
            return baseline

    def notify_reliable_activity(self, conversation_id: str) -> bool:
        """Coalesce reliable activity into one durable, revisioned job."""
        if self.database is None:
            return False
        conversation = self._conversation(conversation_id)
        now = self._now()
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM memory_session_jobs WHERE conversation_id=?",
                (conversation,),
            ).fetchone()

            # A terminal provider failure is recoverable when a later reliable
            # activity gives us a new durable trigger.  Re-open the old frozen
            # range instead of silently skipping its evidence.
            # 2026-09-15：内容层失败在冷却重试额度内也走同一条路（判定在
            # _failure_is_retryable_tx 里，只有一处）。
            retryable_failure = bool(
                existing is not None
                and existing["status"] == "failed"
                and self._failure_is_retryable_tx(connection, existing)
            )
            if existing is not None and existing["status"] == "failed":
                if retryable_failure:
                    if not self._has_reliable_activity_after_tx(
                        connection, conversation, int(existing["end_sequence"])
                    ):
                        return False
                else:
                    self._quarantine_range_tx(connection, existing, now)

            snapshot = self._snapshot_tx(connection, conversation)
            if snapshot is None:
                return False
            if (
                existing is not None
                and self._job_matches_snapshot(existing, snapshot)
                and not retryable_failure
            ):
                return False

            if existing is not None and existing["status"] == "failed":
                if retryable_failure:
                    self._reopen_failed_tx(connection, existing, snapshot, now)
                else:
                    self._upsert_snapshot_tx(
                        connection,
                        snapshot,
                        now,
                        revision=int(existing["revision"]) + 1,
                        created_at=existing["created_at_utc"],
                    )
                    reopened = connection.execute(
                        "SELECT * FROM memory_session_jobs WHERE job_id=?",
                        (existing["job_id"],),
                    ).fetchone()
                    if reopened is not None:
                        self._append_job_event_tx(connection, reopened, "reopened", now=now)
            else:
                self._upsert_snapshot_tx(
                    connection,
                    snapshot,
                    now,
                    revision=(int(existing["revision"]) + 1 if existing is not None else 1),
                    created_at=(existing["created_at_utc"] if existing is not None else now.isoformat()),
                )
            return True

    async def run_due(self, now_utc: datetime) -> tuple[WorkerRun, ...]:
        """Claim one due job, call extractor outside SQLite, then CAS commit."""
        if self.database is None or not self._semantic_gate:
            return ()
        now = self._utc(now_utc, "now_utc")
        claim = self._claim_due(now)
        if claim is None:
            if self.conversation_id is not None:
                row = self.database.connection.execute(
                    "SELECT fragment_key, failure_category FROM memory_session_jobs "
                    "WHERE conversation_id=? AND status='failed' ORDER BY updated_at_utc DESC LIMIT 1",
                    (self.conversation_id,),
                ).fetchone()
                if row is not None and row["failure_category"] == "oversized_event":
                    failure = ExtractionFailure("oversized_event", "event exceeds consolidation token limit", ())
                    return (WorkerRun(row["fragment_key"], failed=True, failure=failure),)
            return ()
        try:
            result = await self.extractor.extract(claim.snapshot.events)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result = MemoryExtractionResult(
                (),
                ExtractionFailure(
                    "llm_error",
                    f"{error.__class__.__name__}: extractor failed",
                    tuple(item.event_id for item in claim.snapshot.events),
                ),
                (),
            )
        if not isinstance(result, MemoryExtractionResult):
            result = MemoryExtractionResult(
                (),
                ExtractionFailure(
                    "parse_error",
                    "extractor returned an invalid result",
                    tuple(item.event_id for item in claim.snapshot.events),
                ),
                (),
            )
        elif result.ok and (
            not isinstance(result.outcome, MemoryOutcome)
            or (
                result.outcome.kind == "no_persistent_memory"
                and (result.candidates or result.reviews)
            )
            or (
                result.outcome.kind == "memory_found"
                and not (result.candidates or result.reviews or result.details or result.fragment)
            )
        ):
            # A successful extraction without the public outcome contract is
            # indistinguishable from a silent omission.  A contradictory
            # outcome is equally unsafe. Fail closed before touching memories
            # or advancing the processed watermark.
            result = MemoryExtractionResult(
                (),
                ExtractionFailure(
                    "parse_error",
                    "extractor result missing public outcome",
                    tuple(item.event_id for item in claim.snapshot.events),
                    {"parse_error_code": "response_schema"},
                ),
                (),
            )
        if not result.ok:
            finish_now = self._now() if self._clock_explicit else now
            return (self._finish_failure(claim, result.failure, finish_now),)
        try:
            # The lease is measured against the clock after the extractor
            # returns; a slow provider must never commit an expired claim.
            finish_now = self._now() if self._clock_explicit else now
            extra_details: tuple = ()
            detail_failure: str | None = None
            detail_split = False
            if self.detail_pass is not None and not result.details:
                # The timeline is an enhancement, never a gate.  Its own call runs
                # outside the transaction, and if it fails the fragment still lands
                # with its event index and an empty timeline -- but the failure is
                # now named instead of silent (2026-09-12).
                extra_details, detail_failure, detail_split = await self._detail_timeline(
                    claim.snapshot.events
                )
            # 明细超上限 → 切成若干片段存（不丢整段原文），各段自己重抽。
            drafts = tuple(result.details or extra_details)
            stores = None
            # 超过上限要切分；**没盖到末尾**同样要切分（2026-09-17）——长片段里模型会
            # 自己截到 32 条就停，日志上看不出任何异常，而尾巴对她就等于不存在。
            if len(drafts) > MAX_DETAILS_PER_FRAGMENT or not _drafts_cover_the_window(
                claim.snapshot.events, drafts
            ):
                stores = await self._stores_within_cap(
                    claim.snapshot.events, result.fragment, drafts
                )
            return (
                self._finish_success(
                    claim,
                    result,
                    finish_now,
                    extra_details,
                    detail_failure=detail_failure,
                    detail_split=detail_split,
                    stores=stores,
                ),
            )
        except Exception as error:
            failure = ExtractionFailure(
                "repository_error",
                f"{error.__class__.__name__}: repository transaction failed",
                tuple(item.event_id for item in claim.snapshot.events),
            )
            finish_now = self._now() if self._clock_explicit else now
            return (self._finish_failure(claim, failure, finish_now),)

    def failures(self) -> tuple[ExtractionFailure, ...]:
        if self.database is None:
            return tuple(self._last_failures[key] for key in sorted(self._last_failures))
        if self.conversation_id is None:
            return ()
        rows = self.database.connection.execute(
            "SELECT * FROM memory_session_jobs WHERE conversation_id=? AND status='failed'",
            (self.conversation_id,),
        ).fetchall()
        failures: list[ExtractionFailure] = []
        for row in rows:
            cached = self._last_failures.get(row["fragment_key"])
            if cached is not None:
                failures.append(cached)
                continue
            failures.append(
                ExtractionFailure(
                    row["failure_category"] or "worker_error",
                    "durable memory consolidation failed",
                    self._event_ids_for_range(
                        self.conversation_id, row["start_sequence"], row["end_sequence"]
                    ),
                )
            )
        return tuple(failures)

    def _claim_due(self, now: datetime) -> _Claim | None:
        query = "SELECT * FROM memory_session_jobs WHERE status IN ('pending','retry','claimed')"
        params: list[object] = []
        if self.conversation_id is not None:
            query += " AND conversation_id=?"
            params.append(self.conversation_id)
        query += " ORDER BY deadline_utc, conversation_id"
        with self.database.transaction() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
            for raw_row in rows:
                row = raw_row
                status = row["status"]
                if status == "claimed":
                    if not self._claim_reclaimable(row, now):
                        continue
                    connection.execute(
                        "UPDATE memory_session_jobs SET status='pending', claim_token=NULL, "
                        "claim_owner=NULL, claim_lease_until_utc=NULL, updated_at_utc=? "
                        "WHERE job_id=? AND status='claimed'",
                        (now.isoformat(), row["job_id"]),
                    )
                    row = connection.execute(
                        "SELECT * FROM memory_session_jobs WHERE job_id=?", (row["job_id"],)
                    ).fetchone()
                    status = row["status"]
                if status == "retry":
                    retry_at = self._parse_optional_utc(row["next_retry_at_utc"], "next_retry_at_utc")
                    if retry_at is None or now < retry_at:
                        continue
                deadline = self._parse_utc(row["deadline_utc"], "deadline_utc")
                if now < deadline:
                    continue
                snapshot = self._snapshot_tx(connection, row["conversation_id"])
                if snapshot is None:
                    continue
                if not self._job_matches_snapshot(row, snapshot):
                    self._upsert_snapshot_tx(
                        connection,
                        snapshot,
                        now,
                        revision=int(row["revision"]) + 1,
                        created_at=row["created_at_utc"],
                    )
                    continue
                bounded = self._bound_snapshot(snapshot)
                canonical_snapshot = snapshot
                first_estimate = max(1, (len(snapshot.events[0].text or "") + 3) // 4)
                if first_estimate > self.max_fragment_tokens:
                    attempts = int(row["attempt_count"]) + 1
                    connection.execute(
                        "UPDATE memory_session_jobs SET status='failed', attempt_count=?, failure_category='oversized_event', "
                        "claim_token=NULL,claim_owner=NULL,claim_lease_until_utc=NULL,next_retry_at_utc=NULL,updated_at_utc=? "
                        "WHERE job_id=? AND revision=? AND status IN ('pending','retry')",
                        (attempts, now.isoformat(), row['job_id'], row['revision']),
                    )
                    failed_row = connection.execute(
                        "SELECT * FROM memory_session_jobs WHERE job_id=?", (row["job_id"],)
                    ).fetchone()
                    if failed_row is not None:
                        self._append_job_event_tx(
                            connection, failed_row, "failed", now=now,
                            failure_category="oversized_event",
                        )
                        self._quarantine_range_tx(connection, failed_row, now)
                    continue
                if bounded.fragment_key != snapshot.fragment_key:
                    self._upsert_snapshot_tx(
                        connection,
                        bounded,
                        now,
                        revision=int(row["revision"]) + 1,
                        created_at=row["created_at_utc"],
                    )
                    snapshot = bounded
                    row = connection.execute(
                        "SELECT * FROM memory_session_jobs WHERE job_id=?", (row["job_id"],)
                    ).fetchone()
                token = uuid.uuid4().hex
                lease = now + timedelta(seconds=self.lease_seconds)
                changed = connection.execute(
                    "UPDATE memory_session_jobs SET status='claimed',claim_token=?,claim_owner=?,"
                    "claim_lease_until_utc=?,next_retry_at_utc=NULL,failure_category=NULL,updated_at_utc=? WHERE job_id=? AND revision=? "
                    "AND status IN ('pending','retry')",
                    (
                        token,
                        self.owner,
                        lease.isoformat(),
                        now.isoformat(),
                        row["job_id"],
                        row["revision"],
                    ),
                )
                if changed.rowcount != 1:
                    continue
                return _Claim(dict(row), snapshot, canonical_snapshot, token, self.owner, lease)
        return None

    def _finish_failure(
        self,
        claim: _Claim,
        failure: ExtractionFailure | None,
        now: datetime,
    ) -> WorkerRun:
        if failure is None:
            failure = ExtractionFailure(
                "worker_error",
                "extractor failed without a category",
                tuple(item.event_id for item in claim.snapshot.events),
            )
        category = self._failure_category(failure)
        self._last_failures[claim.snapshot.fragment_key] = failure
        with self.database.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM memory_session_jobs WHERE job_id=?", (claim.job["job_id"],)
            ).fetchone()
            if not self._claim_matches(current, claim, now):
                self._refresh_after_stale_claim(connection, claim.snapshot.conversation_id, now)
                return WorkerRun(claim.snapshot.fragment_key, cancelled=True)
            attempts = int(current["attempt_count"]) + 1
            retry_delay = self._retry_delay(category, attempts)
            terminal = retry_delay is None
            status = "failed" if terminal else "retry"
            next_retry = None if terminal else (now + timedelta(seconds=retry_delay)).isoformat()
            connection.execute(
                "UPDATE memory_session_jobs SET status=?,attempt_count=?,failure_category=?,"
                "next_retry_at_utc=?,claim_token=NULL,claim_owner=NULL,claim_lease_until_utc=NULL,"
                "updated_at_utc=? WHERE job_id=? AND claim_token=?",
                (
                    status,
                    attempts,
                    category,
                    next_retry,
                    now.isoformat(),
                    claim.job["job_id"],
                    claim.token,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM memory_session_jobs WHERE job_id=?", (claim.job["job_id"],)
            ).fetchone()
            if updated is not None:
                self._append_job_event_tx(
                    connection,
                    updated,
                    "failed" if terminal else "retry_scheduled",
                    now=now,
                    failure_category=category,
                    details=self._safe_failure_details(failure),
                )
                # 额度没用完就让它留在 failed：恢复路径会在冷却之后重开同一段
                # （2026-09-15：上游崩坏导致的 parse_error 不该一次定生死）。
                if terminal and not self._failure_is_retryable_tx(connection, updated):
                    self._quarantine_range_tx(connection, updated, now)
        return WorkerRun(claim.snapshot.fragment_key, failed=True, failure=failure)

    async def _detail_timeline(
        self, events: tuple[ConversationEvent, ...]
    ) -> tuple[tuple, str | None, bool]:
        """The timeline for one fragment: whole window first, then each half.

        One 1.5-hour fragment (362 messages) came back empty on 2026-09-12 because
        the single call failed and nobody could tell why.  A whole-window call is
        still the first attempt; a failure is named, and the window is then split
        in half so one bad call costs half a timeline instead of all of it.
        """

        try:
            return await self.detail_pass.generate(events), None, False
        except Exception as error:
            failure = _detail_pass_failure_code(error)
        if len(events) < 2:
            return (), failure, False
        middle = len(events) // 2
        collected: list = []
        for half in (events[:middle], events[middle:]):
            try:
                collected.extend(await self.detail_pass.generate(half))
            except Exception:
                continue
        if not collected:
            return (), failure, True
        order = {event.event_id: index for index, event in enumerate(events)}
        collected.sort(key=lambda draft: order.get(draft.source_event_id, len(order)))
        merged = tuple(replace(draft, ordinal=index) for index, draft in enumerate(collected))
        return merged, failure, True

    async def _stores_within_cap(
        self,
        events: tuple[ConversationEvent, ...],
        fragment_spec: Any,
        drafts: tuple,
        depth: int = 0,
    ) -> tuple[tuple[tuple[ConversationEvent, ...], Any, tuple], ...]:
        """把一次抽取切成若干「明细不超上限」的存储单元（必要时各自重抽）。

        2026-09-15：build_details 在明细超过每片段上限时抛错，而 detail pass 偶尔
        一条事件给一条明细（真机 35 条事件 → 35 条明细），于是整段原文的证据与明细
        全都落不了库。**超上限就切分，不丢整段**：对半切、两半各自抽一次；还不够就递归。
        半段抽不出来时退回「整段 + 截到上限的明细」——片段与证据仍然留下。
        """

        drafts = tuple(drafts)
        if not drafts and self.detail_pass is not None:
            extra, _failure, _split = await self._detail_timeline(events)
            drafts = tuple(extra)
        if (
            (
                len(drafts) <= MAX_DETAILS_PER_FRAGMENT
                and _drafts_cover_the_window(events, drafts)
            )
            or depth >= MEMORY_SPLIT_MAX_DEPTH
            or len(events) < 2
        ):
            return ((events, fragment_spec, drafts[:MAX_DETAILS_PER_FRAGMENT]),)
        middle = len(events) // 2
        stores: list[tuple[tuple[ConversationEvent, ...], Any, tuple]] = []
        for half in (events[:middle], events[middle:]):
            half_result = await self.extractor.extract(half)
            if not half_result.ok:
                return ((events, fragment_spec, drafts[:MAX_DETAILS_PER_FRAGMENT]),)
            stores.extend(
                await self._stores_within_cap(
                    half, half_result.fragment, tuple(half_result.details or ()), depth + 1
                )
            )
        return tuple(stores)

    def _finish_success(
        self,
        claim: _Claim,
        result: MemoryExtractionResult,
        now: datetime,
        extra_details: tuple = (),
        detail_failure: str | None = None,
        detail_split: bool = False,
        stores: tuple | None = None,
    ) -> WorkerRun:
        try:
            with self.database.transaction() as connection:
                current = connection.execute(
                    "SELECT * FROM memory_session_jobs WHERE job_id=?", (claim.job["job_id"],)
                ).fetchone()
                if not self._claim_matches(current, claim, now):
                    self._refresh_after_stale_claim(connection, claim.snapshot.conversation_id, now)
                    return WorkerRun(claim.snapshot.fragment_key, cancelled=True)
                current_snapshot = self._snapshot_tx(connection, claim.snapshot.conversation_id)
                if current_snapshot is None:
                    self._refresh_after_stale_claim(connection, claim.snapshot.conversation_id, now)
                    return WorkerRun(claim.snapshot.fragment_key, cancelled=True)
                if current_snapshot.events != claim.canonical_snapshot.events:
                    self._refresh_after_stale_claim(connection, claim.snapshot.conversation_id, now)
                    return WorkerRun(claim.snapshot.fragment_key, cancelled=True)
                if self.detail_repository is None:
                    raise RuntimeError("detail repository is unavailable")
                # 一次抽取默认落成一个片段；超明细上限时由 _stores_within_cap 切分，
                # 每个单元各自的事件与明细落成自己的片段。
                units = stores if stores is not None else (
                    (claim.snapshot.events, result.fragment, tuple(result.details or extra_details)),
                )
                for unit_events, unit_spec, unit_drafts in units:
                    unit_key = (
                        claim.snapshot.fragment_key
                        if unit_events == claim.snapshot.events
                        else self._snapshot_key(
                            claim.snapshot.conversation_id, unit_events, claim.snapshot.context_version
                        )
                    )
                    fragment = self.detail_repository.build_fragment(
                        unit_events, unit_key, unit_spec, created_at_utc=now
                    )
                    details = self.detail_repository.build_details(
                        fragment, tuple(unit_drafts), unit_events
                    )
                    self.detail_repository.store_in_transaction(
                        connection, fragment=fragment, events=unit_events, details=details
                    )
                reviews, dropped_review_count = (
                    self.repository.filter_applicable_reviews_in_transaction(
                        connection,
                        conversation_id=claim.snapshot.conversation_id,
                        reviews=result.reviews,
                    )
                )
                batch: MemoryBatchResult = self.repository.apply_consolidation_in_transaction(
                    connection,
                    conversation_id=claim.snapshot.conversation_id,
                    session_job_id=claim.job["job_id"],
                    candidates=result.candidates,
                    reviews=reviews,
                    assessed_at_utc=now,
                )
                self._append_job_event_tx(
                    connection,
                    current,
                    "completed",
                    now=now,
                    details=self._merge_diagnostics(
                        result.diagnostics,
                        candidate_count=len(result.candidates),
                        review_count=len(result.reviews),
                        dropped_review_count=dropped_review_count,
                        outcome_kind=(result.outcome.kind if result.outcome is not None else None),
                        outcome_reason_code=(
                            result.outcome.reason_code if result.outcome is not None else None
                        ),
                        detail_pass_failure=detail_failure,
                        detail_pass_split=detail_split,
                        detail_pass_count=sum(len(unit[2]) for unit in units),
                        split_fragment_count=len(units) if len(units) > 1 else None,
                        restated_memory_count=len(batch.restated_memory_ids),
                    ),
                )
                self._persist_watermark_tx(
                    connection,
                    claim.snapshot.conversation_id,
                    claim.snapshot.end_sequence,
                    now,
                )
                connection.execute(
                    "DELETE FROM memory_session_jobs WHERE job_id=? AND claim_token=?",
                    (claim.job["job_id"], claim.token),
                )
                remainder = tuple(item for item in claim.canonical_snapshot.events if item.sequence > claim.snapshot.end_sequence)
                if remainder:
                    self._upsert_snapshot_tx(connection, self._snapshot_from_events(claim.snapshot.conversation_id, remainder, claim.snapshot.context_version), now)
                remainder = self._snapshot_tx(connection, claim.snapshot.conversation_id)
                if remainder is not None:
                    self._upsert_snapshot_tx(connection, remainder, now, revision=1)
        except Exception as error:
            # Domain validation messages are code-owned constants ("fragment_type
            # is invalid"), so they may be kept; anything else stays a class name
            # (a database or provider message can quote user content).
            details = {}
            message = safe_domain_error(str(error))
            if message is not None:
                details["domain_error"] = message
            failure = ExtractionFailure(
                "repository_error",
                f"{error.__class__.__name__}: consolidation transaction failed",
                tuple(item.event_id for item in claim.snapshot.events),
                details,
            )
            return self._finish_failure(claim, failure, now)
        self._last_failures.pop(claim.snapshot.fragment_key, None)
        return WorkerRun(
            claim.snapshot.fragment_key,
            tuple(batch.created_memory_ids),
            tuple(batch.activated_memory_ids),
        )

    def _refresh_after_stale_claim(self, connection: Any, conversation_id: str, now: datetime) -> None:
        snapshot = self._snapshot_tx(connection, conversation_id)
        if snapshot is None:
            return
        row = connection.execute(
            "SELECT revision,created_at_utc,status FROM memory_session_jobs WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        # A different worker may own a fresh claim. Never overwrite it while
        # attempting to refresh this worker's stale snapshot.
        if row is not None and row["status"] in {"claimed", "retry", "failed"}:
            return
        self._upsert_snapshot_tx(
            connection,
            snapshot,
            now,
            revision=(int(row["revision"]) + 1 if row else 1),
            created_at=(row["created_at_utc"] if row else now.isoformat()),
        )

    def _terminal_failure_old_enough(self, row: Mapping[str, Any], now: datetime) -> bool:
        """Keep startup recovery local and bounded by a durable cooldown.

        A failed provider call is not retried merely because the process
        restarted.  Once the persisted failure is old enough, recovery may
        reopen the same frozen range; the actual model call still waits for the
        normal semantic gate and deadline checks.
        """
        try:
            updated = self._parse_utc(row["updated_at_utc"], "updated_at_utc")
        except ValueError:
            return False
        return now >= updated + self.TERMINAL_REOPEN_COOLDOWN

    def _has_reliable_activity_after_tx(
        self, connection: Any, conversation_id: str, sequence: int
    ) -> bool:
        rows = connection.execute(
            "SELECT event_id FROM conversation_events "
            "WHERE conversation_id=? AND sequence>? ORDER BY sequence,event_id",
            (conversation_id, int(sequence)),
        ).fetchall()
        repository = EventRepository(self.database)
        return any(
            self._is_reliable(repository.get(row["event_id"])) for row in rows
        )

    def _append_job_event_tx(
        self,
        connection: Any,
        row: Mapping[str, Any],
        action: str,
        *,
        now: datetime,
        failure_category: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a public, non-content lifecycle fact for a memory job."""
        if action not in {
            "retry_scheduled",
            "failed",
            "reopened",
            "quarantined",
            "completed",
        }:
            raise ValueError("unknown memory job audit action")
        connection.execute(
            "INSERT INTO memory_job_events ("
            "job_event_id,job_id,conversation_id,revision,fragment_key,"
            "start_sequence,end_sequence,action,failure_category,attempt_count,"
            "next_retry_at_utc,occurred_at_utc,details_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex,
                row["job_id"],
                row["conversation_id"],
                int(row["revision"]),
                row["fragment_key"],
                int(row["start_sequence"]),
                int(row["end_sequence"]),
                action,
                failure_category,
                int(row["attempt_count"]),
                row["next_retry_at_utc"],
                now.isoformat(),
                json.dumps(dict(details or {}), ensure_ascii=False, sort_keys=True),
            ),
        )

    @staticmethod
    def _safe_failure_details(failure: ExtractionFailure | None) -> Mapping[str, str]:
        """Return only the public diagnostic fields allowed in the job ledger."""
        if failure is None or not isinstance(failure.details, Mapping):
            return {}
        details = dict(MemoryWorker._safe_parse_details(failure.details))
        reason_code = MemoryWorker._reason_code(failure.reason)
        if reason_code is not None:
            # Which failure it was, never what it said: 2026-09-11 lost five
            # hours to "repository_error" with an empty detail map.
            details["reason_code"] = reason_code
        allowed = {"timeout", "connection", "rate_limit", "server_error", "protocol", "authentication", "model_not_found", "request", "unknown"}
        value = failure.details.get("provider_error")
        if isinstance(value, str) and value in allowed:
            details["provider_error"] = value
        return details

    # Only exception-class names qualify: a bare leading word would leak model or
    # database text into the job ledger (2026-09-11: "sensitive model output").
    _REASON_CODE_RE = re.compile(
        r"^([A-Za-z][A-Za-z0-9_]{1,40}(?:Error|Exception|Timeout|Failure))(?::|\s|$)"
    )

    @staticmethod
    def _reason_code(reason: Any) -> str | None:
        """The exception class or leading identifier of a failure reason."""

        if not isinstance(reason, str):
            return None
        match = MemoryWorker._REASON_CODE_RE.match(reason.strip())
        if match is None:
            return None
        token = match.group(1)
        if "/" in token or "\\" in token:
            return None
        return token

    @staticmethod
    def _safe_parse_details(details: Mapping[str, Any]) -> Mapping[str, str]:
        """Bound partial-parse telemetry to fixed codes and small counters."""
        if not isinstance(details, Mapping):
            return {}
        output: dict[str, str] = {}
        outcome_kind = details.get("outcome_kind")
        outcome_reason_code = details.get("outcome_reason_code")
        if (
            isinstance(outcome_kind, str)
            and isinstance(outcome_reason_code, str)
            and outcome_reason_code in _OUTCOME_REASON_KINDS.get(outcome_kind, ())
        ):
            output["outcome_kind"] = outcome_kind
            output["outcome_reason_code"] = outcome_reason_code
        parse_error_code = details.get("parse_error_code")
        if isinstance(parse_error_code, str) and parse_error_code in SAFE_PARSE_ERROR_CODES:
            output["parse_error_code"] = parse_error_code
        finish_reason = details.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason in _SAFE_FINISH_REASONS:
            output["finish_reason"] = finish_reason
        schema_field = details.get("schema_field")
        if isinstance(schema_field, str) and _SAFE_IDENTIFIER_RE.match(schema_field):
            output["schema_field"] = schema_field
        domain_error = safe_domain_error(details.get("domain_error"))
        if domain_error is not None:
            output["domain_error"] = domain_error
        detail_failure = details.get("detail_pass_failure")
        if isinstance(detail_failure, str) and detail_failure in _SAFE_DETAIL_PASS_CODES:
            output["detail_pass_failure"] = detail_failure
        for key in (
            "candidate_count",
            "review_count",
            "dropped_candidate_count",
            "dropped_review_count",
            "detail_pass_count",
            "restated_memory_count",
        ):
            value = details.get(key)
            if isinstance(value, str) and value.isascii() and value.isdigit():
                count = int(value)
                if 0 <= count <= 12:
                    output[key] = str(count)
        empty_result = details.get("empty_result")
        if empty_result == "1":
            output["empty_result"] = "1"
        for key in ("candidate_error_codes", "review_error_codes"):
            value = details.get(key)
            if not isinstance(value, str):
                continue
            codes = value.split(",")
            if codes and len(codes) <= 12 and all(
                code in SAFE_PARSE_ERROR_CODES for code in codes
            ):
                output[key] = ",".join(sorted(set(codes)))
        return output

    @staticmethod
    def _merge_diagnostics(
        details: Mapping[str, Any],
        *,
        candidate_count: int,
        review_count: int,
        dropped_review_count: int,
        outcome_kind: str | None,
        outcome_reason_code: str | None,
        detail_pass_failure: str | None = None,
        detail_pass_split: bool = False,
        detail_pass_count: int = 0,
        split_fragment_count: int | None = None,
        restated_memory_count: int = 0,
    ) -> Mapping[str, str]:
        output = dict(MemoryWorker._safe_parse_details(details))
        if not 0 <= candidate_count <= _MAX_DIAGNOSTIC_ITEMS:
            raise ValueError("candidate_count is out of range")
        if not 0 <= review_count <= _MAX_DIAGNOSTIC_ITEMS:
            raise ValueError("review_count is out of range")
        if outcome_kind is not None or outcome_reason_code is not None:
            if outcome_kind not in _OUTCOME_REASON_KINDS:
                raise ValueError("outcome_kind is invalid")
            if (
                not isinstance(outcome_reason_code, str)
                or outcome_reason_code not in _OUTCOME_REASON_KINDS[outcome_kind]
            ):
                raise ValueError("outcome_reason_code is invalid")
            output["outcome_kind"] = outcome_kind
            output["outcome_reason_code"] = outcome_reason_code
        if detail_pass_failure is not None:
            if detail_pass_failure not in _SAFE_DETAIL_PASS_CODES:
                raise ValueError("detail_pass_failure is invalid")
            output["detail_pass_failure"] = detail_pass_failure
        if detail_pass_split:
            output["detail_pass_split"] = "1"
        if detail_pass_count:
            # 上限是「每片段 32 条」，而 2026-09-15 起超上限的窗口会被切成多个片段，
            # 一次作业里的明细总数因此可以更大（诊断值的上界放宽到切分深度允许的规模）。
            if not 0 <= detail_pass_count <= MAX_DETAILS_PER_FRAGMENT * (2 ** MEMORY_SPLIT_MAX_DEPTH):
                raise ValueError("detail_pass_count is out of range")
            output["detail_pass_count"] = str(detail_pass_count)
        if split_fragment_count is not None:
            # 2026-09-15：明细超上限时这一段被切成了几个片段（事实，不是异常）。
            # 2026-09-17：深度上限从 3 提到 4（最多 16 段），上界跟着放宽。
            if not 2 <= split_fragment_count <= 2 ** MEMORY_SPLIT_MAX_DEPTH:
                raise ValueError("split_fragment_count is out of range")
            output["split_fragment_count"] = str(split_fragment_count)
        output["candidate_count"] = str(candidate_count)
        output["review_count"] = str(review_count)
        if candidate_count == 0 and review_count == 0:
            output["empty_result"] = "1"
        else:
            output.pop("empty_result", None)
        if dropped_review_count <= 0:
            return output
        previous = int(output.get("dropped_review_count", "0"))
        output["dropped_review_count"] = str(min(12, previous + dropped_review_count))
        codes = set(filter(None, output.get("review_error_codes", "").split(",")))
        codes.add("review_target")
        output["review_error_codes"] = ",".join(sorted(codes))
        return output

    def _content_failure_rounds_tx(
        self, connection: Any, conversation_id: str, start_sequence: int
    ) -> int:
        """这一段（以范围头为准）已经内容层失败过几次。

        只数内容层类别：瞬时失败（超时/网络）不计入，否则一次网络抖动会把后面的
        冷却重试额度白白吃掉。按范围头计数，范围随新消息增长也不会重置额度。
        """

        categories = tuple(sorted(self._QUARANTINABLE_FAILURE_CATEGORIES))
        placeholders = ",".join("?" for _ in categories)
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM memory_job_events WHERE conversation_id=? "
            "AND action='failed' AND start_sequence=? AND failure_category IN (" + placeholders + ")",
            (conversation_id, int(start_sequence), *categories),
        ).fetchone()
        return int(row["count"] if row is not None else 0)

    def _failure_is_retryable_tx(self, connection: Any, row: Mapping[str, Any]) -> bool:
        """这条失败值不值得冷却后再试一次，而不是直接永久隔离。

        判定原则的唯一来源（recover_pending / notify_reliable_activity / _finish_failure
        都走这里）：瞬时类一直可重试；内容层失败给固定次数的冷却重试；额度用完才隔离。
        """

        category = row["failure_category"] or "worker_error"
        if category in self._TRANSIENT_FAILURE_CATEGORIES:
            return True
        if category not in self._QUARANTINABLE_FAILURE_CATEGORIES:
            return True
        rounds = self._content_failure_rounds_tx(
            connection, str(row["conversation_id"]), int(row["start_sequence"])
        )
        return rounds <= self.CONTENT_FAILURE_REOPEN_ROUNDS

    def _quarantine_range_tx(
        self, connection: Any, row: Mapping[str, Any], now: datetime
    ) -> None:
        """Skip a permanently unprocessable range without advancing processed.

        The quarantine cursor is deliberately separate from the processed
        watermark.  This preserves the distinction between "memory was
        written" and "this range was left unresolved" while allowing later
        sessions to make progress.
        """
        conversation = str(row["conversation_id"])
        key = self._quarantine_key(conversation)
        old_row = connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key=?", (key,)
        ).fetchone()
        old_value = (
            self._decode_meta_int(old_row["value_json"], key) if old_row is not None else -1
        )
        end_sequence = int(row["end_sequence"])
        value = max(old_value, end_sequence)
        connection.execute(
            "INSERT INTO runtime_meta(key,value_json,updated_at_utc) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,"
            "updated_at_utc=excluded.updated_at_utc",
            (key, json.dumps(value), now.isoformat()),
        )
        seen = connection.execute(
            "SELECT 1 FROM memory_job_events WHERE job_id=? AND revision=? "
            "AND action='quarantined' AND start_sequence=? AND end_sequence=? LIMIT 1",
            (
                row["job_id"],
                int(row["revision"]),
                int(row["start_sequence"]),
                end_sequence,
            ),
        ).fetchone()
        if seen is None:
            self._append_job_event_tx(
                connection,
                row,
                "quarantined",
                now=now,
                failure_category=row["failure_category"],
            )

    def _reopen_failed_tx(
        self,
        connection: Any,
        row: Mapping[str, Any],
        snapshot: _Snapshot,
        now: datetime,
    ) -> None:
        """Re-open a terminal transient failure with a fresh revision/token."""
        self._upsert_snapshot_tx(
            connection,
            snapshot,
            now,
            revision=int(row["revision"]) + 1,
            created_at=row["created_at_utc"],
        )
        reopened = connection.execute(
            "SELECT * FROM memory_session_jobs WHERE conversation_id=?",
            (snapshot.conversation_id,),
        ).fetchone()
        if reopened is not None:
            self._append_job_event_tx(connection, reopened, "reopened", now=now)

    def _read_quarantine_tx(self, connection: Any, conversation_id: str) -> int:
        key = self._quarantine_key(conversation_id)
        row = connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return -1
        return self._decode_meta_int(row["value_json"], key)

    def _snapshot_tx(self, connection: Any, conversation_id: str) -> _Snapshot | None:
        # Quarantine is a scan boundary only; it must never be reported as a
        # successfully processed memory watermark.
        watermark = max(
            self._read_watermark_tx(connection, conversation_id),
            self._read_quarantine_tx(connection, conversation_id),
        )
        rows = connection.execute(
            "SELECT event_id FROM conversation_events WHERE conversation_id=? AND sequence>? "
            "ORDER BY sequence,event_id",
            (conversation_id, watermark),
        ).fetchall()
        repository = EventRepository(self.database)
        loaded: list[ConversationEvent] = []
        for row in rows:
            event = repository.get(row["event_id"])
            if self._is_reliable(event):
                loaded.append(event)
        if not loaded:
            return None
        ordered = tuple(sorted(loaded, key=lambda item: (item.sequence, item.event_id)))
        # A durable job represents the earliest continuous session only. A
        # later session remains in the ledger and becomes a new job after the
        # first fragment advances its watermark.
        continuous: list[ConversationEvent] = [ordered[0]]
        previous = self._utc(ordered[0].received_at_utc, "received_at_utc")
        for event in ordered[1:]:
            received = self._utc(event.received_at_utc, "received_at_utc")
            if received - previous >= timedelta(minutes=self.SESSION_IDLE_MINUTES):
                break
            continuous.append(event)
            previous = received
        return self._snapshot_from_events(
            conversation_id, tuple(continuous), self._context_version_tx(connection, conversation_id)
        )

    def _bound_snapshot(self, snapshot: _Snapshot) -> _Snapshot:
        total = 0
        selected: list[ConversationEvent] = []
        previous_received: datetime | None = None
        for event in snapshot.events:
            received = self._utc(event.received_at_utc, "received_at_utc")
            if previous_received is not None and received - previous_received >= timedelta(minutes=self.SESSION_IDLE_MINUTES):
                break
            estimate = max(1, (len(event.text or "") + 3) // 4)
            if not selected and estimate > self.max_fragment_tokens:
                return snapshot
            if selected and total + estimate > self.max_fragment_tokens:
                break
            selected.append(event)
            total += estimate
            previous_received = received
        if len(selected) == len(snapshot.events):
            return snapshot
        return self._snapshot_from_events(
            snapshot.conversation_id, tuple(selected), snapshot.context_version
        )

    def _snapshot_from_events(
        self,
        conversation_id: str,
        events: tuple[ConversationEvent, ...],
        context_version: int,
    ) -> _Snapshot:
        ordered = tuple(sorted(events, key=lambda item: (item.sequence, item.event_id)))
        anchor = max(
            ordered,
            key=lambda item: (
                self._utc(item.received_at_utc, "received_at_utc"),
                item.sequence,
                item.event_id,
            ),
        )
        anchor_at = self._utc(anchor.received_at_utc, "anchor_received_at_utc")
        return _Snapshot(
            conversation_id,
            ordered,
            ordered[0].sequence,
            ordered[-1].sequence,
            anchor.event_id,
            anchor.sequence,
            anchor_at,
            anchor_at + timedelta(minutes=self.SESSION_IDLE_MINUTES),
            context_version,
            self._snapshot_key(conversation_id, ordered, context_version),
        )

    def _upsert_snapshot_tx(
        self,
        connection: Any,
        snapshot: _Snapshot,
        now: datetime,
        *,
        revision: int | None = None,
        created_at: str | None = None,
    ) -> None:
        existing = connection.execute(
            "SELECT revision,created_at_utc FROM memory_session_jobs WHERE conversation_id=?",
            (snapshot.conversation_id,),
        ).fetchone()
        if revision is None:
            revision = int(existing["revision"]) + 1 if existing else 1
        if created_at is None:
            created_at = existing["created_at_utc"] if existing else now.isoformat()
        connection.execute(
            """INSERT INTO memory_session_jobs (
                job_id,conversation_id,revision,fragment_key,start_sequence,end_sequence,
                anchor_event_id,anchor_sequence,anchor_received_at_utc,deadline_utc,
                context_version,status,claim_token,claim_owner,claim_lease_until_utc,
                next_retry_at_utc,attempt_count,failure_category,created_at_utc,updated_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                revision=excluded.revision,fragment_key=excluded.fragment_key,
                start_sequence=excluded.start_sequence,end_sequence=excluded.end_sequence,
                anchor_event_id=excluded.anchor_event_id,anchor_sequence=excluded.anchor_sequence,
                anchor_received_at_utc=excluded.anchor_received_at_utc,deadline_utc=excluded.deadline_utc,
                context_version=excluded.context_version,status='pending',claim_token=NULL,
                claim_owner=NULL,claim_lease_until_utc=NULL,next_retry_at_utc=NULL,
                attempt_count=0,failure_category=NULL,updated_at_utc=excluded.updated_at_utc""",
            (
                uuid.uuid4().hex,
                snapshot.conversation_id,
                revision,
                snapshot.fragment_key,
                snapshot.start_sequence,
                snapshot.end_sequence,
                snapshot.anchor_event_id,
                snapshot.anchor_sequence,
                snapshot.anchor_received_at_utc.isoformat(),
                snapshot.deadline_utc.isoformat(),
                snapshot.context_version,
                "pending",
                None,
                None,
                None,
                None,
                0,
                None,
                created_at,
                now.isoformat(),
            ),
        )

    def _job_matches_snapshot(self, row: Mapping[str, Any], snapshot: _Snapshot) -> bool:
        # A later, separate session advances the global context version without
        # changing this earliest frozen evidence range. Preserve its retry
        # budget; in-flight freshness compares the actual event tuple.
        return (
            row["conversation_id"] == snapshot.conversation_id
            and int(row["start_sequence"]) == snapshot.start_sequence
            and int(row["end_sequence"]) == snapshot.end_sequence
            and row["anchor_event_id"] == snapshot.anchor_event_id
            and int(row["anchor_sequence"]) == snapshot.anchor_sequence
            and self._parse_utc(row["anchor_received_at_utc"], "anchor_received_at_utc") == snapshot.anchor_received_at_utc
            and self._parse_utc(row["deadline_utc"], "deadline_utc") == snapshot.deadline_utc
        )

    def _claim_matches(self, row: Mapping[str, Any] | None, claim: _Claim, now: datetime) -> bool:
        if row is None or row["status"] != "claimed":
            return False
        lease = self._parse_optional_utc(row["claim_lease_until_utc"], "claim_lease_until_utc")
        return (
            row["claim_token"] == claim.token
            and row["claim_owner"] == claim.owner
            and int(row["revision"]) == int(claim.job["revision"])
            and lease is not None
            and now < lease
        )

    def _claim_reclaimable(self, row: Mapping[str, Any], now: datetime) -> bool:
        if row["status"] != "claimed":
            return False
        lease = self._parse_optional_utc(row["claim_lease_until_utc"], "claim_lease_until_utc")
        if lease is not None and now >= lease:
            return True
        owner = row["claim_owner"]
        if not isinstance(owner, str) or not owner or owner == self.owner:
            return False
        if self._owner_alive_probe is not None:
            try:
                return not bool(self._owner_alive_probe(owner))
            except Exception:
                return False
        match = self._OWNER_RE.match(owner)
        if match is None:
            return False
        try:
            from qichi.readiness import pid_alive

            return not pid_alive(int(match.group(1)))
        except Exception:
            return False

    def _read_watermark_tx(self, connection: Any, conversation_id: str) -> int:
        values: list[int] = []
        for key in (self._cursor_key(conversation_id), self._baseline_key(conversation_id)):
            row = connection.execute("SELECT value_json FROM runtime_meta WHERE key=?", (key,)).fetchone()
            if row is not None:
                values.append(self._decode_meta_int(row["value_json"], key))
        return max(values, default=-1)

    def _persist_watermark_tx(
        self, connection: Any, conversation_id: str, sequence: int, now: datetime
    ) -> None:
        value = max(self._read_watermark_tx(connection, conversation_id), int(sequence))
        connection.execute(
            "INSERT INTO runtime_meta(key,value_json,updated_at_utc) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc",
            (self._cursor_key(conversation_id), json.dumps(value), now.isoformat()),
        )

    @staticmethod
    def _context_version_tx(connection: Any, conversation_id: str) -> int:
        row = connection.execute(
            "SELECT context_version FROM conversation_cursors WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return 0
        value = row["context_version"]
        if type(value) is not int or value < 0:
            raise ValueError("conversation context_version is invalid")
        return value

    def _event_ids_for_range(self, conversation_id: str, start: int, end: int) -> tuple[str, ...]:
        rows = self.database.connection.execute(
            "SELECT event_id FROM conversation_events WHERE conversation_id=? AND sequence BETWEEN ? AND ? ORDER BY sequence,event_id",
            (conversation_id, start, end),
        ).fetchall()
        return tuple(row["event_id"] for row in rows)

    @staticmethod
    def _is_reliable(event: ConversationEvent) -> bool:
        if event.direction == "inbound" and event.actor == "mumo":
            return event.kind == "text" and event.status in {"received", "failed"}
        if (
            event.direction != "outbound"
            or event.actor != "qichi"
            or event.kind != "text"
            or event.status != "sent"
        ):
            return False
        metadata = event.metadata
        generation = metadata.get("generation_metadata") if isinstance(metadata, Mapping) else None
        return isinstance(generation, Mapping) and generation.get("source") in {"dialogue", "interaction"}

    @staticmethod
    def _snapshot_key(
        conversation_id: str, events: tuple[ConversationEvent, ...], context_version: int
    ) -> str:
        payload = {
            "conversation_id": conversation_id,
            "context_version": context_version,
            "events": [
                {
                    "event_id": event.event_id,
                    "sequence": event.sequence,
                    "direction": event.direction,
                    "actor": event.actor,
                    "kind": event.kind,
                    "text": event.text,
                    "segments": [segment.to_dict() for segment in event.message_segments],
                    "reply_to_event_id": event.reply_to_event_id,
                    "reply_to_platform_message_id": event.reply_to_platform_message_id,
                    "occurred_at_utc": event.occurred_at_utc.isoformat(),
                    "received_at_utc": event.received_at_utc.isoformat(),
                    "status": event.status,
                    "metadata": event.metadata,
                }
                for event in events
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _failure_category(failure: ExtractionFailure) -> str:
        return failure.kind if isinstance(failure.kind, str) and failure.kind else "worker_error"

    def _retry_delay(self, category: str, attempts: int) -> int | None:
        if category in {"parse_error", "schema_error", "evidence_error"}:
            return self.parse_retry_delay_seconds if attempts == 1 else None
        if category in {
            "llm_error",
            "network_error",
            "timeout",
            "rate_limit",
            "server_error",
            "repository_error",
        }:
            return (
                self.retry_delays_seconds[attempts - 1]
                if 1 <= attempts <= len(self.retry_delays_seconds)
                else None
            )
        return None

    @staticmethod
    def _decode_meta_int(value: object, key: str) -> int:
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"runtime metadata {key} is invalid") from error
        if type(parsed) is not int or parsed < -1:
            raise ValueError(f"runtime metadata {key} is invalid")
        return parsed

    @staticmethod
    def _parse_utc(value: object, field: str) -> datetime:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{field} is invalid") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _parse_optional_utc(cls, value: object, field: str) -> datetime | None:
        return None if value is None else cls._parse_utc(value, field)

    @staticmethod
    def _utc(value: datetime, field: str) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")
        return value.astimezone(timezone.utc)
