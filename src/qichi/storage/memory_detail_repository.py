from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qichi.domain.events import ConversationEvent
from qichi.domain.memory_details import (
    MemoryDetailDraft,
    MemoryDetailEvidence,
    MemoryDetailRecord,
    MemoryFragment,
    MemoryFragmentSpec,
)
from qichi.memory.lexical import (
    MIN_MATCH_FRAGMENTS,
    longest_shared_run,
    match_count,
    matches,
    query_fragments,
)
from qichi.memory.verbatim import VERBATIM_MIN_CHARS, shares_run

from .database import Database


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


# One response has to be able to carry a whole timeline: 32 details at roughly
# 100-150 output tokens each stays inside the extractor budget of 4096 tokens.
# The previous cap of 256 could never fit and only produced truncated JSON.
MAX_DETAILS_PER_FRAGMENT = 32


# 排序门槛：连 4 个字的连续共同串都没有时，只认「窗口命中 ≥2」这条粗信号。
_RANKING_MIN_RUN = 4


class MemoryDetailRepository:
    """Persist the full derived fragment index and ordered detail evidence."""

    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def fragment_id(conversation_id: str, fragment_key: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"qichi:memory-fragment:{conversation_id}:{fragment_key}"))

    @staticmethod
    def detail_id(fragment_id: str, draft: MemoryDetailDraft) -> str:
        evidence = ",".join(f"{event_id}:{role}" for event_id, role in draft.evidence)
        identity = "|".join(
            (
                fragment_id,
                str(draft.ordinal),
                draft.detail_kind,
                draft.actor,
                draft.reality_scope,
                draft.normalized_detail,
                draft.exact_quote,
                draft.source_event_id,
                draft.certainty,
                draft.temporal_scope,
                draft.status,
                draft.privacy_class,
                draft.recall_policy,
                evidence,
            )
        )
        return str(uuid5(NAMESPACE_URL, f"qichi:memory-detail:{identity}"))

    @classmethod
    def build_fragment(
        cls,
        events: tuple[ConversationEvent, ...],
        fragment_key: str,
        spec: MemoryFragmentSpec | None,
        *,
        created_at_utc: datetime,
    ) -> MemoryFragment:
        if not isinstance(events, tuple) or not events:
            raise ValueError("fragment events must be a non-empty tuple")
        if not all(isinstance(item, ConversationEvent) for item in events):
            raise TypeError("fragment events must contain ConversationEvent")
        ordered = tuple(sorted(events, key=lambda item: (item.sequence, item.event_id)))
        if len({item.event_id for item in ordered}) != len(ordered):
            raise ValueError("fragment events must have unique event IDs")
        if len({item.conversation_id for item in ordered}) != 1:
            raise ValueError("fragment events must share one conversation")
        if not isinstance(fragment_key, str) or not fragment_key:
            raise ValueError("fragment_key must be non-empty")
        created = _utc(created_at_utc, "created_at_utc")
        if spec is None:
            spec = MemoryFragmentSpec(
                fragment_type="daily",
                reality_scope="conversation",
                summary="冻结片段的完整原文索引；具体内容以事件账本为准",
                privacy_class="ordinary",
                recall_policy="daily_safe",
                closed=True,
            )
        return MemoryFragment(
            fragment_id=cls.fragment_id(ordered[0].conversation_id, fragment_key),
            conversation_id=ordered[0].conversation_id,
            fragment_key=fragment_key,
            start_sequence=ordered[0].sequence,
            end_sequence=ordered[-1].sequence,
            start_event_id=ordered[0].event_id,
            end_event_id=ordered[-1].event_id,
            started_at_utc=ordered[0].occurred_at_utc,
            ended_at_utc=ordered[-1].occurred_at_utc,
            fragment_type=spec.fragment_type,
            reality_scope=spec.reality_scope,
            summary=spec.summary,
            privacy_class=spec.privacy_class,
            recall_policy=spec.recall_policy,
            status="candidate",
            closed_at_utc=ordered[-1].occurred_at_utc if spec.closed else None,
            created_at_utc=created,
        )

    @classmethod
    def build_details(
        cls,
        fragment: MemoryFragment,
        drafts: tuple[MemoryDetailDraft, ...],
        events: tuple[ConversationEvent, ...],
    ) -> tuple[MemoryDetailRecord, ...]:
        if not isinstance(drafts, tuple) or not all(isinstance(item, MemoryDetailDraft) for item in drafts):
            raise TypeError("details must contain MemoryDetailDraft")
        if len(drafts) > MAX_DETAILS_PER_FRAGMENT:
            raise ValueError(f"details must contain at most {MAX_DETAILS_PER_FRAGMENT} items")
        ordered_events = {item.event_id: item for item in events}
        if len(ordered_events) != len(events):
            raise ValueError("fragment events must have unique event IDs")
        output: list[MemoryDetailRecord] = []
        for expected_ordinal, draft in enumerate(drafts):
            if draft.ordinal != expected_ordinal:
                raise ValueError("detail ordinals must be contiguous")
            source = ordered_events.get(draft.source_event_id)
            if source is None:
                raise ValueError("detail source event is outside the fragment")
            if source.text is None or draft.exact_quote not in source.text:
                raise ValueError("detail exact quote is absent from source event")
            if draft.actor in {"mumo", "qichi"} and source.actor != draft.actor:
                raise ValueError("detail actor does not match source event")
            evidence: list[MemoryDetailEvidence] = []
            seen: set[str] = set()
            for event_id, role in draft.evidence:
                evidence_source = ordered_events.get(event_id)
                if evidence_source is None:
                    raise ValueError("detail evidence event is outside the fragment")
                if event_id in seen:
                    raise ValueError("detail evidence event IDs must be unique")
                seen.add(event_id)
                evidence.append(MemoryDetailEvidence(event_id, role))
            if draft.source_event_id not in seen:
                raise ValueError("detail source event must be detail evidence")
            output.append(
                MemoryDetailRecord(
                    detail_id=cls.detail_id(fragment.fragment_id, draft),
                    fragment_id=fragment.fragment_id,
                    ordinal=draft.ordinal,
                    detail_kind=draft.detail_kind,
                    actor=draft.actor,
                    reality_scope=draft.reality_scope,
                    normalized_detail=draft.normalized_detail,
                    exact_quote=draft.exact_quote,
                    source_event_id=draft.source_event_id,
                    occurred_at_utc=source.occurred_at_utc,
                    certainty=draft.certainty,
                    temporal_scope=draft.temporal_scope,
                    status=draft.status,
                    privacy_class=draft.privacy_class,
                    recall_policy=draft.recall_policy,
                    evidence=tuple(evidence),
                )
            )
        return tuple(output)

    def store_in_transaction(
        self,
        connection: Any,
        *,
        fragment: MemoryFragment,
        events: tuple[ConversationEvent, ...],
        details: tuple[MemoryDetailRecord, ...],
    ) -> None:
        if connection is not self.database.connection or not connection.in_transaction:
            raise ValueError("connection must be this repository's active transaction")
        if not isinstance(fragment, MemoryFragment):
            raise TypeError("fragment must be a MemoryFragment")
        if not isinstance(events, tuple) or not events or not all(isinstance(item, ConversationEvent) for item in events):
            raise TypeError("events must be a non-empty tuple of ConversationEvent")
        if not isinstance(details, tuple) or not all(isinstance(item, MemoryDetailRecord) for item in details):
            raise TypeError("details must be a tuple of MemoryDetailRecord")
        ordered = tuple(sorted(events, key=lambda item: (item.sequence, item.event_id)))
        if any(item.conversation_id != fragment.conversation_id for item in ordered):
            raise ValueError("fragment event conversation does not match fragment")
        if ordered[0].event_id != fragment.start_event_id or ordered[-1].event_id != fragment.end_event_id:
            raise ValueError("fragment endpoints do not match events")
        if ordered[0].sequence != fragment.start_sequence or ordered[-1].sequence != fragment.end_sequence:
            raise ValueError("fragment sequences do not match events")
        if len(details) > 256:
            raise ValueError(f"details must contain at most {MAX_DETAILS_PER_FRAGMENT} items")
        if any(item.fragment_id != fragment.fragment_id for item in details):
            raise ValueError("detail fragment ID does not match fragment")

        connection.execute(
            "INSERT INTO memory_fragments (fragment_id,conversation_id,fragment_key,start_sequence,end_sequence,"
            "start_event_id,end_event_id,started_at_utc,ended_at_utc,fragment_type,reality_scope,summary,"
            "privacy_class,recall_policy,status,closed_at_utc,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(fragment_id) DO NOTHING",
            (
                fragment.fragment_id, fragment.conversation_id, fragment.fragment_key,
                fragment.start_sequence, fragment.end_sequence, fragment.start_event_id,
                fragment.end_event_id, fragment.started_at_utc.isoformat(), fragment.ended_at_utc.isoformat(),
                fragment.fragment_type, fragment.reality_scope, fragment.summary, fragment.privacy_class,
                fragment.recall_policy, fragment.status,
                fragment.closed_at_utc.isoformat() if fragment.closed_at_utc else None,
                fragment.created_at_utc.isoformat(),
            ),
        )
        existing = connection.execute(
            "SELECT conversation_id,fragment_key,start_sequence,end_sequence,start_event_id,end_event_id,"
            "fragment_type,reality_scope,summary,privacy_class,recall_policy,status,closed_at_utc "
            "FROM memory_fragments WHERE fragment_id=?",
            (fragment.fragment_id,),
        ).fetchone()
        # Identity is the conversation and the covered span.  Classification and
        # summary are derived and may legitimately differ when the same session is
        # replayed (2026-09-11: a whole session failed because a re-derived
        # fragment_type disagreed with the stored one); the first write wins, which
        # is what the ON CONFLICT DO NOTHING insert above already means.
        if existing is None or tuple(existing[:6]) != (
            fragment.conversation_id, fragment.fragment_key, fragment.start_sequence, fragment.end_sequence,
            fragment.start_event_id, fragment.end_event_id,
        ):
            raise ValueError("fragment identity conflict")

        for ordinal, item in enumerate(ordered):
            connection.execute(
                "INSERT INTO memory_fragment_events (fragment_id,event_id,ordinal,sequence,actor,occurred_at_utc) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(fragment_id,event_id) DO NOTHING",
                (fragment.fragment_id, item.event_id, ordinal, item.sequence, item.actor, item.occurred_at_utc.isoformat()),
            )
            row = connection.execute(
                "SELECT ordinal,sequence,actor,occurred_at_utc FROM memory_fragment_events WHERE fragment_id=? AND event_id=?",
                (fragment.fragment_id, item.event_id),
            ).fetchone()
            if row is None or tuple(row) != (ordinal, item.sequence, item.actor, item.occurred_at_utc.isoformat()):
                raise ValueError("fragment event identity conflict")
        stored_event_rows = connection.execute(
            "SELECT event_id FROM memory_fragment_events WHERE fragment_id=? ORDER BY ordinal",
            (fragment.fragment_id,),
        ).fetchall()
        if [row["event_id"] for row in stored_event_rows] != [item.event_id for item in ordered]:
            raise ValueError("fragment event set is incomplete or conflicting")

        for detail in details:
            connection.execute(
                "INSERT INTO memory_detail_records (detail_id,fragment_id,ordinal,detail_kind,actor,reality_scope,"
                "normalized_detail,exact_quote,source_event_id,occurred_at_utc,certainty,temporal_scope,status,"
                "privacy_class,recall_policy) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(detail_id) DO NOTHING",
                (
                    detail.detail_id, detail.fragment_id, detail.ordinal, detail.detail_kind, detail.actor,
                    detail.reality_scope, detail.normalized_detail, detail.exact_quote, detail.source_event_id,
                    detail.occurred_at_utc.isoformat(), detail.certainty, detail.temporal_scope, detail.status,
                    detail.privacy_class, detail.recall_policy,
                ),
            )
            for evidence_ordinal, evidence in enumerate(detail.evidence):
                connection.execute(
                    "INSERT INTO memory_detail_evidence (detail_id,event_id,evidence_role,ordinal) VALUES (?,?,?,?) "
                    "ON CONFLICT(detail_id,event_id) DO NOTHING",
                    (detail.detail_id, evidence.event_id, evidence.evidence_role, evidence_ordinal),
                )
            stored_detail = connection.execute(
                "SELECT fragment_id,ordinal,detail_kind,actor,reality_scope,normalized_detail,exact_quote,"
                "source_event_id,occurred_at_utc,certainty,temporal_scope,status,privacy_class,recall_policy "
                "FROM memory_detail_records WHERE detail_id=?",
                (detail.detail_id,),
            ).fetchone()
            expected_detail = (
                detail.fragment_id, detail.ordinal, detail.detail_kind, detail.actor, detail.reality_scope,
                detail.normalized_detail, detail.exact_quote, detail.source_event_id,
                detail.occurred_at_utc.isoformat(), detail.certainty, detail.temporal_scope, detail.status,
                detail.privacy_class, detail.recall_policy,
            )
            if stored_detail is None or tuple(stored_detail) != expected_detail:
                raise ValueError("detail identity conflict")
            stored_evidence = connection.execute(
                "SELECT event_id,evidence_role FROM memory_detail_evidence WHERE detail_id=? ORDER BY ordinal",
                (detail.detail_id,),
            ).fetchall()
            if [tuple(row) for row in stored_evidence] != [
                (item.event_id, item.evidence_role) for item in detail.evidence
            ]:
                raise ValueError("detail evidence set is incomplete or conflicting")

    def list_details(
        self,
        conversation_id: str,
        *,
        query: str = "",
        explicit_request: bool = False,
        event_ids: tuple[str, ...] | None = None,
        fragment_ids: tuple[str, ...] | None = None,
    ) -> tuple[MemoryDetailRecord, ...]:
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(explicit_request) is not bool:
            raise TypeError("explicit_request must be a bool")
        if event_ids is not None and (
            not isinstance(event_ids, tuple)
            or not all(isinstance(item, str) and item for item in event_ids)
        ):
            raise TypeError("event_ids must be a tuple of non-empty strings or None")
        if fragment_ids is not None and (
            not isinstance(fragment_ids, tuple)
            or not all(isinstance(item, str) and item for item in fragment_ids)
        ):
            raise TypeError("fragment_ids must be a tuple of non-empty strings or None")
        event_filter = set(event_ids or ())
        fragment_filter = set(fragment_ids or ())
        query_fragments_ = query_fragments(tuple(query.splitlines())) if query.strip() else ()
        rows = self.database.connection.execute(
            "SELECT d.*, f.summary AS fragment_summary FROM memory_detail_records d JOIN memory_fragments f ON f.fragment_id=d.fragment_id "
            "WHERE f.conversation_id=? AND d.status IN ('candidate','active') "
            "ORDER BY f.started_at_utc,d.ordinal,d.detail_id",
            (conversation_id,),
        ).fetchall()
        output: list[MemoryDetailRecord] = []
        for row in rows:
            if row["privacy_class"] == "adult" and not explicit_request:
                continue
            if row["privacy_class"] != "ordinary" and not explicit_request and not query.strip():
                continue
            if event_filter:
                linked = self.database.connection.execute(
                    "SELECT 1 FROM memory_fragment_events WHERE fragment_id=? AND event_id IN (%s) LIMIT 1"
                    % ",".join("?" for _ in event_filter),
                    (row["fragment_id"], *sorted(event_filter)),
                ).fetchone()
                if linked is None:
                    continue
            if fragment_filter and row["fragment_id"] not in fragment_filter:
                continue
            searchable = row["normalized_detail"] + row["exact_quote"] + row["fragment_summary"]
            if query_fragments_ and not matches(searchable, query_fragments_):
                continue
            evidence_rows = self.database.connection.execute(
                "SELECT event_id,evidence_role FROM memory_detail_evidence WHERE detail_id=? ORDER BY ordinal",
                (row["detail_id"],),
            ).fetchall()
            output.append(
                MemoryDetailRecord(
                    detail_id=row["detail_id"], fragment_id=row["fragment_id"], ordinal=row["ordinal"],
                    detail_kind=row["detail_kind"], actor=row["actor"], reality_scope=row["reality_scope"],
                    normalized_detail=row["normalized_detail"], exact_quote=row["exact_quote"],
                    source_event_id=row["source_event_id"], occurred_at_utc=datetime.fromisoformat(row["occurred_at_utc"]),
                    certainty=row["certainty"], temporal_scope=row["temporal_scope"], status=row["status"],
                    privacy_class=row["privacy_class"], recall_policy=row["recall_policy"],
                    evidence=tuple(MemoryDetailEvidence(item["event_id"], item["evidence_role"]) for item in evidence_rows),
                )
            )
        return tuple(output)

    def footprint_details(
        self,
        conversation_id: str,
        *,
        since: datetime,
        per_fragment: int = 4,
        limit: int = 12,
    ) -> tuple[MemoryDetailRecord, ...]:
        """最近若干天里每个片段的头几条逐字原话，按时间正序返回。

        2026-09-12 用户反馈「角色没法找到原文字段」：她的常驻上下文里只有改写层
        （工作集/索引），逐字原话只在钥匙开门时才注入。这个查询给「最近原文足迹」
        提供原料：最近几天、每个片段取头几条，够她认出「这事确实说过」即可。

        隐私门与既有规则一致：按**明细**判 privacy/recall（与明细块同一条规矩），
        所以 ordinary 明细可以来自 mixed/intimate 片段；成人或 explicit_request_only
        的明细一律不进日常上下文。
        """

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(since, datetime) or since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since must be timezone-aware")
        for value, field in ((per_fragment, "per_fragment"), (limit, "limit")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        rows = self.database.connection.execute(
            "SELECT d.*, f.started_at_utc AS fragment_started_at_utc "
            "FROM memory_detail_records d JOIN memory_fragments f ON f.fragment_id=d.fragment_id "
            "WHERE f.conversation_id=? AND d.privacy_class='ordinary' "
            "AND d.recall_policy='daily_safe' AND d.status IN ('candidate','active') "
            "AND f.started_at_utc >= ? "
            "ORDER BY f.started_at_utc DESC, d.ordinal ASC, d.detail_id ASC",
            (conversation_id, since.astimezone(timezone.utc).isoformat()),
        ).fetchall()
        per_fragment_count: dict[str, int] = {}
        chosen: list[Any] = []
        for row in rows:
            fragment = row["fragment_id"]
            used = per_fragment_count.get(fragment, 0)
            if used >= per_fragment:
                continue
            per_fragment_count[fragment] = used + 1
            chosen.append(row)
            if len(chosen) >= limit:
                break
        chosen.reverse()  # 时间正序读起来才是对话的样子
        output: list[MemoryDetailRecord] = []
        for row in chosen:
            evidence_rows = self.database.connection.execute(
                "SELECT event_id,evidence_role FROM memory_detail_evidence WHERE detail_id=? ORDER BY ordinal",
                (row["detail_id"],),
            ).fetchall()
            output.append(
                MemoryDetailRecord(
                    detail_id=row["detail_id"], fragment_id=row["fragment_id"], ordinal=row["ordinal"],
                    detail_kind=row["detail_kind"], actor=row["actor"], reality_scope=row["reality_scope"],
                    normalized_detail=row["normalized_detail"], exact_quote=row["exact_quote"],
                    source_event_id=row["source_event_id"], occurred_at_utc=datetime.fromisoformat(row["occurred_at_utc"]),
                    certainty=row["certainty"], temporal_scope=row["temporal_scope"], status=row["status"],
                    privacy_class=row["privacy_class"], recall_policy=row["recall_policy"],
                    evidence=tuple(MemoryDetailEvidence(item["event_id"], item["evidence_role"]) for item in evidence_rows),
                )
            )
        return tuple(output)

    def fragments_for_events(self, conversation_id: str, event_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Map quoted events to the episodes that contain them, oldest first."""

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(event_ids, tuple) or not all(
            isinstance(item, str) and item for item in event_ids
        ):
            raise TypeError("event_ids must be a tuple of non-empty strings")
        if not event_ids:
            return ()
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.database.connection.execute(
            "SELECT f.fragment_id FROM memory_fragments f "
            "JOIN memory_fragment_events fe ON fe.fragment_id=f.fragment_id "
            f"WHERE f.conversation_id=? AND fe.event_id IN ({placeholders}) "
            "GROUP BY f.fragment_id ORDER BY f.started_at_utc, f.fragment_id",
            (conversation_id, *event_ids),
        ).fetchall()
        return tuple(row[0] for row in rows)

    def fragments_on_dates(
        self,
        conversation_id: str,
        dates: tuple,
        *,
        local_zone,
        limit: int = 4,
        prefer_hours: tuple[tuple[int, int], ...] = (),
        anchors: tuple[int, ...] = (),
    ) -> tuple[str, ...]:
        """Return the episodes that happened on the named local days, newest first.

        A date is a fact about the calendar, so it decides on its own: a newer
        conversation that merely mentions the number must not outrank the episode
        that actually happened that day.

        2026-09-12（T6，用户裁定 A）：顺序从「最早优先」改成「最新优先」。名字里
        的一天往往有好几段（09-11 那天有 6 段），限额一截断，旧顺序留下的是那天
        一早的内容，而人嘴里说的「那天」通常是最近发生的那部分。
        """

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(dates, tuple) or not all(
            isinstance(item, date) for item in dates
        ):
            raise TypeError("dates must be a tuple of dates")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not dates:
            return ()
        wanted = set(dates)
        rows = self.database.connection.execute(
            "SELECT fragment_id, started_at_utc FROM memory_fragments "
            "WHERE conversation_id=? ORDER BY started_at_utc DESC, fragment_id DESC",
            (conversation_id,),
        ).fetchall()
        matched: list[tuple[str, int]] = []
        for row in rows:
            try:
                moment = datetime.fromisoformat(row[1]).astimezone(local_zone)
            except (TypeError, ValueError):
                continue
            if moment.date() in wanted:
                matched.append((row[0], moment.hour))
        # 用户点明了「凌晨/下午/晚上」时，先说中那一段的排前面；说了钟点（"零点过后"）
        # 时，离那个钟点最近的排最前。两者都没有就退回「最新优先」。排序是稳定的，
        # 所以同一档里仍然是新的在前（2026-09-12 T10）。
        def rank(item: tuple[str, int]) -> tuple[int, int]:
            hour = item[1]
            if anchors:
                distance = min(min(abs(hour - a), 24 - abs(hour - a)) for a in anchors)
            else:
                distance = 0
            band = 0 if any(start <= hour < end for start, end in prefer_hours) else 1
            return (distance, band)

        matched.sort(key=rank)
        return tuple(item[0] for item in matched[:limit])

    def fragments_matching(
        self, conversation_id: str, terms: tuple[str, ...], *, limit: int = 4
    ) -> tuple[str, ...]:
        """Find the episodes whose own stored words mention the query, newest first.

        A guess is only worth offering when the current turn (or the running
        conversation behind it) actually names something the episode contains.
        """

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(terms, tuple) or not all(isinstance(item, str) for item in terms):
            raise TypeError("terms must be a tuple of strings")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        fragments = query_fragments(terms)
        if not fragments:
            return ()
        rows = self.database.connection.execute(
            "SELECT d.fragment_id, d.normalized_detail || d.exact_quote || f.summary "
            "FROM memory_fragments f JOIN memory_detail_records d ON d.fragment_id=f.fragment_id "
            "WHERE f.conversation_id=? AND d.status IN ('candidate','active') "
            "ORDER BY f.started_at_utc DESC, f.fragment_id DESC, d.ordinal, d.detail_id",
            (conversation_id,),
        ).fetchall()
        scores: dict[str, int] = {}
        order: list[str] = []
        for row in rows:
            fragment_id = row[0]
            if fragment_id not in scores:
                scores[fragment_id] = 0
                order.append(fragment_id)
            score = match_count(row[1], fragments)
            if score > scores[fragment_id]:
                scores[fragment_id] = score
        needed = min(MIN_MATCH_FRAGMENTS, len(fragments))
        return tuple(item for item in order if scores[item] >= needed)[:limit]

    def fragments_ranked_by_text(
        self,
        conversation_id: str,
        terms: tuple[str, ...],
        *,
        within: tuple[str, ...] = (),
        limit: int = 4,
    ) -> tuple[tuple[str, int], ...]:
        """Rank the given episodes by how much of this turn's text they contain.

        2026-09-17: 点名日期/时段时，指针按「钟点距离 + 最新优先」取样；预算只装得下最前面那几段。
        真机上他要核对的那段排第三，于是永远进不来。这里只做**排序**——in 进的是已经被那把钥匙
        授权的片段，不新开任何片段，也不改授权集合。同分时新的在前。
        """

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(terms, tuple) or not all(isinstance(item, str) for item in terms):
            raise TypeError("terms must be a tuple of strings")
        if not isinstance(within, tuple) or not all(isinstance(item, str) for item in within):
            raise TypeError("within must be a tuple of strings")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        windows = query_fragments(terms)
        if not windows or not within:
            return ()
        allowed = set(within)
        rows = self.database.connection.execute(
            "SELECT d.fragment_id, d.normalized_detail || d.exact_quote || f.summary, "
            "f.started_at_utc FROM memory_fragments f "
            "JOIN memory_detail_records d ON d.fragment_id=f.fragment_id "
            "WHERE f.conversation_id=? AND d.status IN ('candidate','active') "
            "ORDER BY f.started_at_utc DESC, f.fragment_id DESC, d.ordinal, d.detail_id",
            (conversation_id,),
        ).fetchall()
        # 排序信号从粗到细：**最长公共连续串**优先（「他说的就是这句」），窗口命中数次之。
        # 2026-09-17 真机：只用窗口数时，通用词让无关的长片段排到了前面。
        scores: dict[str, tuple[int, int]] = {}
        recency: dict[str, str] = {}
        for row in rows:
            fragment_id = row[0]
            if fragment_id not in allowed:
                continue
            recency.setdefault(fragment_id, row[2])
            run = max(
                longest_shared_run(term, row[1]) for term in terms if isinstance(term, str)
            )
            hits = match_count(row[1], windows)
            previous = scores.get(fragment_id, (0, 0))
            if (run, hits) > previous:
                scores[fragment_id] = (run, hits)
        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1][0],
                -item[1][1],
                str(recency.get(item[0], "")),
                item[0],
            ),
        )
        return tuple(
            (fragment_id, run, hits)
            for fragment_id, (run, hits) in ranked
            if run >= _RANKING_MIN_RUN or hits >= 2
        )[:limit]


    def fragments_quoting(
        self, conversation_id: str, text: str, *, min_chars: int = VERBATIM_MIN_CHARS, limit: int = 4
    ) -> tuple[str, ...]:
        """Episodes whose stored verbatim quotes contain a long run of this text.

        Only exact quotes are compared: a paraphrase is the parser's wording, and
        matching against it is what let a two-character everyday word open an
        adult episode (2026-09-12 audit, P1).  Newest episode first, like every
        other locator.
        """

        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if len(text) < min_chars:
            return ()
        rows = self.database.connection.execute(
            "SELECT d.fragment_id, d.exact_quote "
            "FROM memory_fragments f JOIN memory_detail_records d ON d.fragment_id=f.fragment_id "
            "WHERE f.conversation_id=? AND d.status IN ('candidate','active') "
            "AND d.exact_quote IS NOT NULL AND d.exact_quote <> '' "
            "ORDER BY f.started_at_utc DESC, f.fragment_id DESC, d.ordinal, d.detail_id",
            (conversation_id,),
        ).fetchall()
        matched: list[str] = []
        for row in rows:
            fragment_id = row[0]
            if fragment_id in matched:
                continue
            if shares_run(text, row[1], min_chars=min_chars):
                matched.append(fragment_id)
                if len(matched) >= limit:
                    break
        return tuple(matched)


