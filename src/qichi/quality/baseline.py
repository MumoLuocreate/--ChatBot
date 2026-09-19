"""Strict, offline quality-baseline records.

The evaluator intentionally contains no semantic classifier. Humans rate
model samples after generation; code only validates provenance, score ranges
and aggregate completeness. This keeps evaluation outside the reply path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from qichi.domain.dialogue import ModelMessage


QUALITY_AXES = (
    "focus",
    "continuity",
    "evidence_fidelity",
    "time_accuracy",
    "agency_boundary",
    "natural_exchange",
    "non_repetition",
    "no_internal_leak",
)

_SCENARIO_FIELDS = frozenset({"scenario_id", "category", "messages", "evidence_event_ids"})
_RATING_FIELDS = frozenset({"scenario_id", "sample_index", "scores", "notes"})


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class QualityScenario:
    scenario_id: str
    category: str
    messages: tuple[ModelMessage, ...]
    evidence_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.scenario_id, "scenario_id")
        _nonempty(self.category, "category")
        if not self.messages or not all(isinstance(item, ModelMessage) for item in self.messages):
            raise TypeError("messages must be a non-empty tuple of ModelMessage")
        if not isinstance(self.evidence_event_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.evidence_event_ids
        ):
            raise TypeError("evidence_event_ids must be a tuple of non-empty strings")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("evidence_event_ids must be unique")


@dataclass(frozen=True, slots=True)
class QualityRating:
    scenario_id: str
    sample_index: int
    scores: Mapping[str, int]
    notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.scenario_id, "scenario_id")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        if not isinstance(self.scores, Mapping) or set(self.scores) != set(QUALITY_AXES):
            raise ValueError("scores must contain every quality axis exactly once")
        copied: dict[str, int] = {}
        for axis in QUALITY_AXES:
            score = self.scores[axis]
            if type(score) is not int or score not in {0, 1, 2}:
                raise ValueError("quality scores must be integers in range 0..2")
            copied[axis] = score
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("notes must be text or None")
        object.__setattr__(self, "scores", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class QualitySummary:
    sample_count: int
    scenario_count: int
    axis_means: Mapping[str, float]
    total_mean: float


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"line {line_number} must be a JSON object")
        yield line_number, value


def load_scenarios_jsonl(path: str | Path) -> tuple[QualityScenario, ...]:
    scenarios: list[QualityScenario] = []
    seen: set[str] = set()
    for line_number, value in _read_jsonl(path):
        if set(value) != _SCENARIO_FIELDS:
            raise ValueError(f"scenario line {line_number} has invalid fields")
        raw_messages = value["messages"]
        raw_evidence = value["evidence_event_ids"]
        if not isinstance(raw_messages, list) or not isinstance(raw_evidence, list):
            raise TypeError(f"scenario line {line_number} lists are invalid")
        scenario = QualityScenario(
            scenario_id=value["scenario_id"],
            category=value["category"],
            messages=tuple(ModelMessage.from_dict(item) for item in raw_messages),
            evidence_event_ids=tuple(raw_evidence),
        )
        if scenario.scenario_id in seen:
            raise ValueError("scenario IDs must be unique")
        seen.add(scenario.scenario_id)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError("quality baseline must contain at least one scenario")
    return tuple(scenarios)


def load_ratings_jsonl(path: str | Path) -> tuple[QualityRating, ...]:
    ratings: list[QualityRating] = []
    seen: set[tuple[str, int]] = set()
    for line_number, value in _read_jsonl(path):
        if set(value) != _RATING_FIELDS:
            raise ValueError(f"rating line {line_number} has invalid fields")
        rating = QualityRating(
            scenario_id=value["scenario_id"],
            sample_index=value["sample_index"],
            scores=value["scores"],
            notes=value["notes"],
        )
        identity = (rating.scenario_id, rating.sample_index)
        if identity in seen:
            raise ValueError("rating sample identities must be unique")
        seen.add(identity)
        ratings.append(rating)
    return tuple(ratings)


def summarize_ratings(
    scenarios: tuple[QualityScenario, ...], ratings: tuple[QualityRating, ...]
) -> QualitySummary:
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    scenario_ids = {item.scenario_id for item in scenarios}
    if any(item.scenario_id not in scenario_ids for item in ratings):
        raise ValueError("rating references an unknown scenario")
    if not ratings:
        return QualitySummary(0, len(scenarios), MappingProxyType({axis: 0.0 for axis in QUALITY_AXES}), 0.0)
    totals = {axis: 0 for axis in QUALITY_AXES}
    for rating in ratings:
        for axis in QUALITY_AXES:
            totals[axis] += rating.scores[axis]
    means = {axis: totals[axis] / len(ratings) for axis in QUALITY_AXES}
    return QualitySummary(
        sample_count=len(ratings),
        scenario_count=len(scenarios),
        axis_means=MappingProxyType(means),
        total_mean=sum(means.values()) / len(QUALITY_AXES),
    )
