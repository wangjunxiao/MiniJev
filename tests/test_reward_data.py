"""Reward evidence must be complete before an optimiser is allowed to trust it."""

from __future__ import annotations

import json

import pytest

from minijev.convert import RouterBenchConvertor
from minijev.reward_data import (
    MisrouteJudgment,
    action_evidence_audit,
    build_misroute_queue,
    load_judgments,
    misroute_audit,
    save_judgments,
)
from minijev.schema import Candidate, JevSample, Route


def test_old_candidates_remain_loadable_but_are_not_mistaken_for_evidence():
    row = JevSample(
        id="old",
        query="q",
        route=Route.LOCAL,
        confidence=1.0,
        candidates=(Candidate(Route.LOCAL, 0.8, quality=0.8, cost=0.1), Candidate(Route.CLOUD, 0.9, quality=0.9, cost=1.0)),
    )
    audit = action_evidence_audit([row])
    assert audit["candidates"] == 2
    assert audit["all_fields_have_provenance"] is False
    assert audit["offline_counterfactual_ready"] is False
    assert audit["evidence_by_field"]["quality"]["missing"] == 2


def test_routerbench_conversion_marks_exported_fields_as_benchmark_not_production(tmp_path):
    source = tmp_path / "routerbench.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "q",
                "models": {
                    "small": {"score": 0.9, "cost": 0.01, "latency_ms": 20, "available": True},
                    "large": {"score": 0.95, "cost": 0.1, "latency_ms": 80, "available": True},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(RouterBenchConvertor().rows(source))
    audit = action_evidence_audit(rows)
    assert audit["all_fields_have_provenance"] is True
    assert audit["offline_counterfactual_ready"] is True
    assert audit["production_reward_ready"] is False
    assert audit["evidence_by_field"]["latency_ms"]["benchmark"] == 2


def test_missing_availability_and_latency_are_reported_not_imputed(tmp_path):
    source = tmp_path / "routerbench.jsonl"
    source.write_text(
        json.dumps({"prompt": "q", "models": {"small": {"score": 0.9, "cost": 0.01}, "large": {"score": 0.95, "cost": 0.1}}}) + "\n",
        encoding="utf-8",
    )
    audit = action_evidence_audit(RouterBenchConvertor().rows(source))
    assert audit["evidence_by_field"]["latency_ms"]["missing"] == 2
    assert audit["evidence_by_field"]["available"]["missing"] == 2
    assert audit["offline_counterfactual_ready"] is False


def test_review_priors_are_explicitly_declared_not_measured(tmp_path):
    source = tmp_path / "routerbench.jsonl"
    source.write_text(json.dumps({"prompt": "medical", "models": {"small": {"score": 0.2, "cost": 0.01}}}) + "\n", encoding="utf-8")
    row = next(RouterBenchConvertor().rows(source))
    evidence = row.metadata["candidate_evidence"]["human_review"]
    assert {entry["kind"] for entry in evidence.values()} == {"declared"}


def test_disagreement_queue_is_balanced_by_cell_and_prefers_high_confidence():
    report = {
        "disagreements": [
            {"sample_id": "a", "query": "a", "ground_truth": "local", "jev_prediction": "rag", "jev_confidence": 0.4, "frame_id": "f1"},
            {"sample_id": "b", "query": "b", "ground_truth": "local", "jev_prediction": "rag", "jev_confidence": 0.9, "frame_id": "f2"},
            {"sample_id": "c", "query": "c", "ground_truth": "review", "jev_prediction": "local", "jev_confidence": 0.8, "frame_id": "f3"},
        ]
    }
    queue = build_misroute_queue(report, per_cell=1)
    assert len(queue) == 2
    assert {row.sample_id for row in queue} == {"b", "c"}
    assert all(row.status == "pending" for row in queue)


def test_reviewed_judgment_requires_a_score_reviewer_and_rationale():
    with pytest.raises(ValueError):
        MisrouteJudgment("a", "q", Route.REVIEW, Route.LOCAL, status="reviewed", harm_score=4.0)
    with pytest.raises(ValueError):
        MisrouteJudgment("a", "q", Route.REVIEW, Route.LOCAL, status="reviewed", harm_score=5.0, reviewer_id="r", rationale="x")


def _review(sample_id: str, reviewer: str, score: float, *, kind: str = "human_elicited") -> MisrouteJudgment:
    return MisrouteJudgment(
        sample_id,
        "medical question",
        Route.REVIEW,
        Route.LOCAL,
        status="reviewed",
        harm_score=score,
        reviewer_id=reviewer,
        rationale="unsafe automation",
        evidence_kind=kind,
        direct_cost=100.0 if kind == "production" else None,
    )


def test_harm_matrix_requires_replicates_and_independent_reviewers():
    one = misroute_audit([_review("a", "reviewer-1", 4.0)])
    assert one["harm_matrix_ready"] is False
    repeated_same_person = misroute_audit([_review("a", "reviewer-1", 4.0), _review("b", "reviewer-1", 3.0)])
    assert repeated_same_person["harm_matrix_ready"] is False
    ready = misroute_audit([_review("a", "reviewer-1", 4.0), _review("b", "reviewer-2", 3.0)])
    assert ready["observed_cells_ready"] is True
    assert ready["harm_matrix_ready"] is False, "one ready cell is not a complete 4x4 off-diagonal matrix"
    assert ready["required_confusion_cells"] == 12
    assert ready["matrix"]["review>local"]["mean_harm_score"] == pytest.approx(3.5)
    assert ready["production_cost_ready"] is False


def test_observed_production_costs_still_do_not_complete_the_other_eleven_cells():
    audit = misroute_audit([_review("a", "r1", 4.0, kind="production"), _review("b", "r2", 3.0, kind="production")])
    assert audit["observed_cells_ready"] is True
    assert audit["production_cost_ready"] is False


def test_all_twelve_cells_with_observed_costs_pass_the_full_gate():
    rows = []
    for required in Route:
        for chosen in Route:
            if required is chosen:
                continue
            for reviewer in ("r1", "r2"):
                rows.append(
                    MisrouteJudgment(
                        f"{required.value}-{chosen.value}-{reviewer}",
                        "observed incident",
                        required,
                        chosen,
                        status="reviewed",
                        harm_score=2.0,
                        direct_cost=10.0,
                        reviewer_id=reviewer,
                        rationale="incident postmortem",
                        evidence_kind="production",
                    )
                )
    audit = misroute_audit(rows)
    assert audit["ready_cells"] == 12
    assert audit["harm_matrix_ready"] is True
    assert audit["production_cost_ready"] is True


def test_review_queue_round_trips_without_overwrite(tmp_path):
    path = tmp_path / "review.jsonl"
    pending = MisrouteJudgment("a", "q", Route.LOCAL, Route.RAG, source_model="jev", model_confidence=0.9)
    save_judgments([pending], path)
    restored = load_judgments(path)
    assert restored == [pending]
    with pytest.raises(FileExistsError):
        save_judgments([pending], path)
