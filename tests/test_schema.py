"""The data contract must survive a save/load round trip without losing information."""

from __future__ import annotations

import json
import math

import pytest

from minijev.schema import ROUTE_ORDER, Candidate, Decision, JevSample, Route, RouteScores, load_jsonl, save_jsonl


def test_roundtrip_preserves_core_fields(tmp_path):
    sample = JevSample(query="你好", confidence=0.9, route=Route.LOCAL, context=("上一句",))
    path = tmp_path / "rows.jsonl"
    save_jsonl([sample], path)
    (loaded,) = load_jsonl(path)
    assert loaded.query == sample.query
    assert loaded.confidence == pytest.approx(sample.confidence)
    assert loaded.route is sample.route
    assert loaded.context == sample.context


def test_roundtrip_preserves_unknown_fields():
    """Richer source schemas (MASSIVE intents, annotator votes) must not be dropped."""
    row = {"query": "查一下天气", "confidence": 1.0, "route": "rag", "intent": "weather_query", "scenario": "weather", "id": "massive_7"}
    sample = JevSample.from_dict(row)
    assert sample.metadata["intent"] == "weather_query"
    assert sample.metadata["scenario"] == "weather"
    assert sample.id == "massive_7"
    restored = sample.to_dict()
    assert restored["intent"] == "weather_query"
    assert restored["id"] == "massive_7"


def test_what_goes_through_json_survives(tmp_path):
    row = {"query": "查一下天气", "confidence": 1.0, "route": "rag", "intent": "weather_query", "id": "massive_7"}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (sample,) = load_jsonl(path)
    save_jsonl([sample], path)
    (again,) = load_jsonl(path)
    assert again.to_dict()["intent"] == "weather_query"


def test_confidence_must_be_a_probability():
    with pytest.raises(ValueError):
        JevSample(query="你好", confidence=1.5, route=Route.LOCAL)
    with pytest.raises(ValueError):
        JevSample(query="你好", confidence=-0.1, route=Route.LOCAL)


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        JevSample(query="   ", confidence=0.5, route=Route.LOCAL)


def test_candidates_roundtrip_and_expose_counterfactuals():
    sample = JevSample(
        query="设计一个分布式系统",
        confidence=0.5,
        route=Route.CLOUD,
        candidates=(
            Candidate(route=Route.LOCAL, reward=0.1, quality=0.4, cost=0.0, propensity=0.4),
            Candidate(route=Route.CLOUD, reward=0.8, quality=0.95, cost=1.0, propensity=0.1),
        ),
    )
    assert sample.has_counterfactuals
    restored = JevSample.from_dict(json.loads(json.dumps(sample.to_dict())))
    assert len(restored.candidates) == 2
    assert restored.candidates[1].route is Route.CLOUD
    assert restored.candidates[1].propensity == pytest.approx(0.1)
    assert restored.candidates[0].latency_ms is None


def test_route_scores_require_one_logit_per_route():
    with pytest.raises(ValueError):
        RouteScores(route_logits=(1.0, 2.0))
    assert len(RouteScores(route_logits=(1.0, 2.0, 3.0, 4.0)).route_logits) == len(ROUTE_ORDER)


def test_route_scores_carry_logits_and_nothing_else():
    """One source for confidence: derived by the policy, never supplied alongside."""
    fields = set(RouteScores.__dataclass_fields__)
    assert fields == {"route_logits"}
    assert "confidence" not in fields
    assert "agreement" not in fields


def test_decision_keeps_the_full_distribution_not_just_argmax():
    """A policy-gradient step must be able to sample and score any route."""
    decision = Decision(
        route=Route.RAG,
        confidence=0.8,
        route_logits=(1.0, 2.0, 0.0, 0.0),
        route_probs=(0.2, 0.7, 0.05, 0.05),
        policy="risk_ladder",
        confidence_measure="normalized_entropy",
    )
    assert decision.log_prob() == pytest.approx(math.log(0.7))
    assert decision.log_prob(Route.CLOUD) == pytest.approx(math.log(0.05))
    assert len(decision.route_probs) == len(ROUTE_ORDER)
    assert decision.top_route is Route.RAG, "the model preferred rag, whatever the policy did"
    assert decision.escalated is False


def test_decision_records_where_an_escalation_came_from():
    decision = Decision(
        route=Route.REVIEW,
        confidence=0.1,
        route_logits=(2.0, 1.0, 0.0, 0.0),
        route_probs=(0.6, 0.3, 0.05, 0.05),
        policy="risk_ladder",
        escalated_from=Route.LOCAL,
    )
    assert decision.escalated is True
    assert decision.escalated_from is Route.LOCAL
    assert decision.top_route is Route.LOCAL
