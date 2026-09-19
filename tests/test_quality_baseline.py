from __future__ import annotations

import json
from pathlib import Path

import pytest

from qichi.domain.dialogue import ModelMessage
from qichi.quality import (
    QUALITY_AXES,
    QualityRating,
    load_ratings_jsonl,
    load_scenarios_jsonl,
    summarize_ratings,
)


FIXTURE = Path(__file__).parent / "fixtures" / "quality" / "baseline.jsonl"

ADULT_BOUNDARY_SCENARIOS = {
    "adult-enter-explicitly",
    "adult-direct-language",
    "adult-history-evidence",
    "adult-agency-adjustment",
    "adult-no-escalation",
    "adult-meta-discussion",
    "adult-old-consent-not-current",
    "adult-return-to-daily",
}


def test_baseline_fixture_covers_required_synthetic_categories():
    scenarios = load_scenarios_jsonl(FIXTURE)
    assert {item.category for item in scenarios} >= {
        "casual", "correction", "time", "quote", "memory", "relationship", "interaction", "initiative"
    }
    by_id = {item.scenario_id: item for item in scenarios}
    assert all(
        item.messages[-1].role == "user"
        for item in scenarios
        if item.category != "initiative"
    )
    assert by_id["poke-context"].messages[-1] == ModelMessage("user", "")
    assert by_id["initiative-skip"].messages[-1].role == "system"
    assert all(
        message.content not in {"[poke]", "[initiative]"}
        for scenario in scenarios
        for message in scenario.messages
    )


def test_baseline_fixture_keeps_adult_boundaries_paired_with_non_escalation_cases():
    scenarios = load_scenarios_jsonl(FIXTURE)
    by_id = {item.scenario_id: item for item in scenarios}

    assert set(by_id) >= ADULT_BOUNDARY_SCENARIOS
    assert "本轮没有进入成人共同想象" in by_id["adult-no-escalation"].messages[0].content
    assert "元讨论，不是场景邀请" in by_id["adult-meta-discussion"].messages[0].content
    assert "本轮尚未形成同意" in by_id["adult-old-consent-not-current"].messages[0].content
    assert "已经由双方明确结束" in by_id["adult-return-to-daily"].messages[0].content


def test_scenario_loader_rejects_unknown_fields_and_duplicate_ids(tmp_path):
    value = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    value["unexpected"] = True
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fields"):
        load_scenarios_jsonl(path)
    value.pop("unexpected")
    path.write_text("\n".join(json.dumps(value, ensure_ascii=False) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios_jsonl(path)


def test_ratings_require_every_axis_and_range(tmp_path):
    scores = {axis: 2 for axis in QUALITY_AXES}
    valid = {"scenario_id": "casual-short", "sample_index": 0, "scores": scores, "notes": None}
    path = tmp_path / "ratings.jsonl"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert load_ratings_jsonl(path)[0].scores["focus"] == 2
    with pytest.raises(ValueError, match="every quality axis"):
        QualityRating("casual-short", 1, {"focus": 2})
    with pytest.raises(ValueError, match="0..2"):
        QualityRating("casual-short", 1, {**scores, "focus": 3})


def test_summary_is_deterministic_and_rejects_unknown_scenario():
    scenarios = load_scenarios_jsonl(FIXTURE)
    high = QualityRating(scenarios[0].scenario_id, 0, {axis: 2 for axis in QUALITY_AXES})
    low = QualityRating(scenarios[0].scenario_id, 1, {axis: 0 for axis in QUALITY_AXES})
    summary = summarize_ratings(scenarios, (high, low))
    assert summary.sample_count == 2
    assert summary.total_mean == 1.0
    assert set(summary.axis_means) == set(QUALITY_AXES)
    unknown = QualityRating("missing", 0, {axis: 1 for axis in QUALITY_AXES})
    with pytest.raises(ValueError, match="unknown scenario"):
        summarize_ratings(scenarios, (unknown,))


def test_empty_ratings_are_an_explicit_unrated_baseline():
    scenarios = load_scenarios_jsonl(FIXTURE)
    summary = summarize_ratings(scenarios, ())
    assert summary.sample_count == 0
    assert summary.total_mean == 0.0
