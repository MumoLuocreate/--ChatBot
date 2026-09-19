from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, TypeAlias
from uuid import uuid4

from qichi.domain.events import ConversationEvent
from qichi.domain.memory import MemoryRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository


SearchMode: TypeAlias = Literal["fts5_trigram", "like_short_query", "like_degraded"]
_MAX_CANDIDATES = 24
_MAX_CONTEXT_CANDIDATES = 12
_MAX_TRIGRAMS = 64


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    candidates: tuple[MemoryRecord, ...]
    context_candidates: tuple[MemoryRecord, ...]
    evidence_events: Mapping[str, ConversationEvent]
    search_mode: SearchMode
    degraded_reason: str | None
    scores: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    reasons: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    confirmation_candidates: tuple[MemoryRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(item, MemoryRecord) for item in self.candidates
        ):
            raise TypeError("candidates must be a tuple of MemoryRecord")
        if len(self.candidates) > _MAX_CANDIDATES:
            raise ValueError("candidates exceeds the hard limit")
        if not isinstance(self.context_candidates, tuple) or not all(
            isinstance(item, MemoryRecord) for item in self.context_candidates
        ):
            raise TypeError("context_candidates must be a tuple of MemoryRecord")
        if len(self.context_candidates) > _MAX_CONTEXT_CANDIDATES:
            raise ValueError("context_candidates exceeds the hard limit")
        candidate_ids = {item.memory_id for item in self.candidates}
        if any(item.memory_id not in candidate_ids for item in self.context_candidates):
            raise ValueError("context candidates must be drawn from candidates")
        if len(self.confirmation_candidates) > 1 or any(
            item.status != "candidate" for item in self.confirmation_candidates
        ):
            raise ValueError("confirmation_candidates must contain at most one candidate")
        if not isinstance(self.evidence_events, Mapping) or not all(
            isinstance(key, str) and isinstance(value, ConversationEvent)
            for key, value in self.evidence_events.items()
        ):
            raise TypeError("evidence_events must map event IDs to ConversationEvent")
        object.__setattr__(self, "evidence_events", MappingProxyType(dict(self.evidence_events)))
        if self.search_mode not in {"fts5_trigram", "like_short_query", "like_degraded"}:
            raise ValueError("search_mode is invalid")
        if self.search_mode == "like_degraded":
            if not isinstance(self.degraded_reason, str) or not self.degraded_reason:
                raise ValueError("degraded retrieval requires a reason")
        elif self.degraded_reason is not None:
            raise ValueError("non-degraded retrieval cannot carry a degraded reason")
        for name, values in (("scores", self.scores), ("reasons", self.reasons)):
            if not isinstance(values, Mapping):
                raise TypeError(f"{name} must be a mapping")
            if set(values) != candidate_ids:
                raise ValueError(f"{name} must describe every candidate exactly once")
        if any(type(value) is not int or value < 0 for value in self.scores.values()):
            raise ValueError("scores must contain non-negative integers")
        if any(not isinstance(value, str) or not value for value in self.reasons.values()):
            raise ValueError("reasons must contain non-empty strings")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))

    @property
    def retrieved_memory_ids(self) -> tuple[str, ...]:
        return tuple(record.memory_id for record in self.context_candidates)


class MemoryRetriever:
    def __init__(
        self, database: Database, candidate_limit: int = 24, context_limit: int = 12
    ) -> None:
        if type(candidate_limit) is not int or not 1 <= candidate_limit <= _MAX_CANDIDATES:
            raise ValueError("candidate_limit must be between 1 and 24")
        if type(context_limit) is not int or not 1 <= context_limit <= _MAX_CONTEXT_CANDIDATES:
            raise ValueError("context_limit must be between 1 and 12")
        if context_limit > candidate_limit:
            raise ValueError("context_limit must not exceed candidate_limit")
        self.database = database
        self.candidate_limit = candidate_limit
        self.context_limit = context_limit
        self.memories = MemoryRepository(database)
        self.events = EventRepository(database)

    def retrieve(
        self,
        conversation_id: str,
        query: str,
        at_utc: datetime,
        query_terms: tuple[str, ...] | None = None,
        include_confirmation: bool = False,
        include_sensitive: bool = False,
    ) -> MemoryRetrievalResult:
        conversation_id = self._nonempty_text(conversation_id, "conversation_id")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(include_sensitive) is not bool:
            raise TypeError("include_sensitive must be a bool")
        at_utc = self._utc(at_utc, "at_utc")
        query = query.strip()
        if not query:
            return MemoryRetrievalResult((), (), {}, "like_short_query", None)
        terms = self._query_terms(query, query_terms)
        short_terms = tuple(term for term in terms if len(term) < 3)
        long_terms = tuple(term for term in terms if len(term) >= 3)

        if not long_terms:
            memory_ids = self._like_candidate_ids(conversation_id, short_terms, at_utc)
            search_mode: SearchMode = "like_short_query"
            degraded_reason = None
        elif self._supports_fts5_trigram():
            fts_ids = self._fts_candidate_ids(conversation_id, long_terms, at_utc)
            short_ids = self._like_candidate_ids(conversation_id, short_terms, at_utc)
            memory_ids = tuple(sorted(set(fts_ids) | set(short_ids)))
            search_mode = "fts5_trigram"
            degraded_reason = None
        else:
            memory_ids = self._like_candidate_ids(
                conversation_id,
                short_terms + self._query_trigrams(long_terms),
                at_utc,
            )
            search_mode = "like_degraded"
            degraded_reason = "SQLite FTS5 trigram tokenizer is unavailable"

        loaded = tuple(
            record
            for record in (self.memories.get(memory_id) for memory_id in memory_ids)
            if include_sensitive or record.privacy_class == "ordinary"
        )
        confirmation_loaded: tuple[MemoryRecord, ...] = ()
        if include_confirmation:
            confirmation_loaded = tuple(
                record
                for record in (
                    self.memories.get(i)
                    for i in self._confirmation_ids(conversation_id, at_utc, terms)
                )
                if include_sensitive or record.privacy_class == "ordinary"
            )
        ranked, all_scores, all_reasons = self._rank_candidates(loaded, terms, query)
        candidates = ranked[: self.candidate_limit]
        selected_ids = {record.memory_id for record in candidates}
        scores = {key: all_scores[key] for key in selected_ids}
        reasons = {key: all_reasons[key] for key in selected_ids}
        self._validate_candidates(candidates, conversation_id, at_utc)
        context_candidates = candidates[: self.context_limit]
        evidence_events = dict(self._evidence_events(context_candidates, conversation_id))
        confirmation_ranked, confirmation_scores, _ = self._rank_candidates(confirmation_loaded, terms, query)
        confirmation = tuple(item for item in confirmation_ranked if confirmation_scores.get(item.memory_id, 0) > 0)[:1]
        evidence_events.update(self._evidence_events(confirmation, conversation_id))
        return MemoryRetrievalResult(
            candidates=candidates,
            context_candidates=context_candidates,
            evidence_events=evidence_events,
            search_mode=search_mode,
            degraded_reason=degraded_reason,
            scores=scores,
            reasons=reasons,
            confirmation_candidates=confirmation,
        )

    def _confirmation_ids(self, conversation_id: str, at_utc: datetime, terms: tuple[str, ...]) -> tuple[str, ...]:
        rows = self.database.connection.execute(
            "SELECT DISTINCT m.memory_id FROM memory_records AS m "
            "JOIN memory_evidence AS me ON me.memory_id=m.memory_id "
            "JOIN conversation_events AS e ON e.event_id=me.event_id "
            "WHERE e.conversation_id=? AND m.status='candidate' "
            "AND certainty='ambiguous' AND importance IN (2,3) AND valid_from_utc<=? "
            "AND (valid_until_utc IS NULL OR valid_until_utc>=?) ORDER BY importance DESC",
            (conversation_id, at_utc.isoformat(), at_utc.isoformat()),
        ).fetchall()
        return tuple(row[0] for row in rows)

    @staticmethod
    def _query_terms(query: str, supplied: tuple[str, ...] | None) -> tuple[str, ...]:
        values: Iterable[str] = (query,) if supplied is None else supplied
        if isinstance(values, str):
            raise TypeError("query_terms must be a tuple of strings")
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise TypeError("query_terms must contain strings")
            value = value.strip()
            if value and value not in seen:
                seen.add(value)
                result.append(value)
                if len(result) == _MAX_TRIGRAMS:
                    break
        return tuple(result) or (query,)

    @classmethod
    def _rank_candidates(
        cls, records: tuple[MemoryRecord, ...], terms: tuple[str, ...], query: str
    ) -> tuple[tuple[MemoryRecord, ...], Mapping[str, int], Mapping[str, str]]:
        ranked: list[tuple[int, int, datetime, str, MemoryRecord, str]] = []
        scores: dict[str, int] = {}
        reasons: dict[str, str] = {}
        trigrams = cls._query_trigrams(terms)
        for record in records:
            fact = record.normalized_fact
            quotes = tuple(item.exact_quote for item in record.memory_evidence)
            fact_hits = sum(term in fact for term in terms)
            quote_hits = sum(any(term in quote for quote in quotes) for term in terms)
            trigram_fact_hits = sum(fragment in fact for fragment in trigrams)
            trigram_quote_hits = sum(
                any(fragment in quote for quote in quotes) for fragment in trigrams
            )
            exact_fact = bool(query and query in fact)
            exact_quote = any(query in quote for quote in quotes)
            score = (
                fact_hits * 2
                + quote_hits * 3
                + trigram_fact_hits
                + trigram_quote_hits * 2
                + exact_fact * 4
                + exact_quote * 5
            )
            matched = []
            if fact_hits:
                matched.append("normalized_fact")
            if quote_hits:
                matched.append("exact_quote")
            if exact_fact:
                matched.append("exact_query_in_fact")
            if exact_quote:
                matched.append("exact_query_in_quote")
            if trigram_fact_hits or trigram_quote_hits:
                matched.append(
                    f"trigram_overlap:fact={trigram_fact_hits};quote={trigram_quote_hits}"
                )
            reason = ",".join(matched) if matched else "no lexical hit"
            has_direct_hit = bool(fact_hits or quote_hits or exact_fact or exact_quote)
            if not has_direct_hit and trigram_fact_hits + trigram_quote_hits < 2:
                continue
            scores[record.memory_id] = score
            reasons[record.memory_id] = reason
            ranked.append(
                (
                    score,
                    record.importance,
                    record.created_at_utc,
                    record.memory_id,
                    record,
                    reason,
                )
            )
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        return tuple(item[4] for item in ranked), scores, reasons

    def _supports_fts5_trigram(self) -> bool:
        table_name = f"qichi_memory_fts_probe_{uuid4().hex}"
        try:
            self._create_fts_probe(table_name)
        except sqlite3.OperationalError as error:
            if self._is_fts_capability_error(error):
                return False
            raise
        finally:
            self.database.connection.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
        return True

    def _create_fts_probe(self, table_name: str) -> None:
        self.database.connection.execute(
            f"CREATE VIRTUAL TABLE temp.{table_name} "
            "USING fts5(searchable, tokenize='trigram')"
        )

    @staticmethod
    def _is_fts_capability_error(error: sqlite3.OperationalError) -> bool:
        message = str(error).lower()
        return (
            "no such module: fts5" in message
            or "no such tokenizer: trigram" in message
            or "unknown tokenizer: trigram" in message
        )

    def _fts_candidate_ids(
        self, conversation_id: str, terms: tuple[str, ...], at_utc: datetime
    ) -> tuple[str, ...]:
        table_name = f"qichi_memory_fts_{uuid4().hex}"
        try:
            self.database.connection.execute(
                f"CREATE VIRTUAL TABLE temp.{table_name} "
                "USING fts5(memory_id UNINDEXED, searchable, tokenize='trigram')"
            )
            self.database.connection.execute(
                f"INSERT INTO temp.{table_name} (memory_id, searchable) "
                "SELECT m.memory_id, m.normalized_fact || char(10) || "
                "group_concat(me.exact_quote, char(10)) "
                "FROM memory_records AS m "
                "JOIN memory_evidence AS me ON me.memory_id = m.memory_id "
                "JOIN conversation_events AS e ON e.event_id = me.event_id "
                "WHERE e.conversation_id = ? AND m.status = 'active' "
                "AND m.valid_from_utc <= ? "
                "AND (m.valid_until_utc IS NULL OR m.valid_until_utc >= ?) "
                "GROUP BY m.memory_id, m.normalized_fact",
                (conversation_id, at_utc.isoformat(), at_utc.isoformat()),
            )
            expression = " OR ".join(
                self._fts_phrase(item) for item in self._query_trigrams(terms)
            )
            rows = self.database.connection.execute(
                f"SELECT memory_id FROM temp.{table_name} WHERE {table_name} MATCH ? "
                "ORDER BY memory_id",
                (expression,),
            ).fetchall()
            return tuple(row["memory_id"] for row in rows)
        finally:
            self.database.connection.execute(f"DROP TABLE IF EXISTS temp.{table_name}")

    def _like_candidate_ids(
        self, conversation_id: str, terms: tuple[str, ...], at_utc: datetime
    ) -> tuple[str, ...]:
        patterns = tuple(f"%{self._escape_like(term)}%" for term in terms)
        if not patterns:
            return ()
        match_sql = " OR ".join(
            "m.normalized_fact LIKE ? ESCAPE '\\' OR me.exact_quote LIKE ? ESCAPE '\\'"
            for _ in patterns
        )
        parameters: list[object] = [conversation_id, at_utc.isoformat(), at_utc.isoformat()]
        for pattern in patterns:
            parameters.extend((pattern, pattern))
        rows = self.database.connection.execute(
            "SELECT DISTINCT m.memory_id FROM memory_records AS m "
            "JOIN memory_evidence AS me ON me.memory_id = m.memory_id "
            "JOIN conversation_events AS e ON e.event_id = me.event_id "
            "WHERE e.conversation_id = ? AND m.status = 'active' "
            "AND m.valid_from_utc <= ? "
            "AND (m.valid_until_utc IS NULL OR m.valid_until_utc >= ?) "
            f"AND ({match_sql}) ORDER BY m.memory_id",
            tuple(parameters),
        ).fetchall()
        return tuple(row["memory_id"] for row in rows)

    def _validate_candidates(
        self,
        candidates: tuple[MemoryRecord, ...],
        conversation_id: str,
        at_utc: datetime,
    ) -> None:
        seen: set[str] = set()
        for record in candidates:
            if record.memory_id in seen:
                raise ValueError("retrieval returned a duplicate memory")
            seen.add(record.memory_id)
            if record.status != "active":
                raise ValueError("retrieval returned an inactive memory")
            if record.valid_from_utc > at_utc or (
                record.valid_until_utc is not None and record.valid_until_utc < at_utc
            ):
                raise ValueError("retrieval returned a memory outside its validity interval")
            for evidence in record.memory_evidence:
                source = self.events.get(evidence.event_id)
                if source.conversation_id != conversation_id:
                    raise ValueError("retrieved memory belongs to another conversation")

    def _evidence_events(
        self, candidates: tuple[MemoryRecord, ...], conversation_id: str
    ) -> Mapping[str, ConversationEvent]:
        events: dict[str, ConversationEvent] = {}
        for record in candidates:
            for evidence in record.memory_evidence:
                source = self.events.get(evidence.event_id)
                if source.conversation_id != conversation_id:
                    raise ValueError("memory evidence belongs to another conversation")
                if source.actor != evidence.actor:
                    raise ValueError("memory evidence actor does not match source event")
                if source.occurred_at_utc != evidence.occurred_at_utc:
                    raise ValueError("memory evidence time does not match source event")
                if source.text is None or evidence.exact_quote not in source.text:
                    raise ValueError("memory evidence exact quote is absent from source event")
                events[source.event_id] = source
        return MappingProxyType(events)

    @staticmethod
    def _trigrams(query: str) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for index in range(len(query) - 2):
            fragment = query[index : index + 3]
            if fragment not in seen:
                seen.add(fragment)
                result.append(fragment)
                if len(result) == _MAX_TRIGRAMS:
                    break
        return tuple(result)

    @staticmethod
    def _query_trigrams(terms: tuple[str, ...]) -> tuple[str, ...]:
        """Share the bounded FTS budget across independent query sources."""
        positions = [0] * len(terms)
        seen: set[str] = set()
        result: list[str] = []
        while len(result) < _MAX_TRIGRAMS:
            progressed = False
            for term_index, term in enumerate(terms):
                while positions[term_index] <= len(term) - 3:
                    start = positions[term_index]
                    positions[term_index] += 1
                    fragment = term[start : start + 3]
                    if fragment in seen:
                        continue
                    seen.add(fragment)
                    result.append(fragment)
                    progressed = True
                    break
                if len(result) == _MAX_TRIGRAMS:
                    break
            if not progressed:
                break
        return tuple(result)

    @staticmethod
    def _fts_phrase(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _nonempty_text(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        if not value:
            raise ValueError(f"{field} must not be empty")
        return value

    @staticmethod
    def _utc(value: datetime, field: str) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field} must be timezone-aware")
        return value.astimezone(timezone.utc)
