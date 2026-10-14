"""The live Jev benchmark pipeline, exercised entirely offline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from minijev.jev_api import DecisionResponse, JevApiError
from minijev.jev_benchmark import (
    PROMPT_VERSION,
    RETRIEVAL_PROMPT_VERSION,
    annotate_one,
    api_state,
    benchmark_question,
    compare_prompt_records,
    evaluate_records,
    load_records,
    load_result_records,
    run_annotations,
    sample_key,
)
from minijev.schema import JevSample, Route


def _sample(identifier: str, query: str, route: Route, *, context: tuple[str, ...] = ()) -> JevSample:
    return JevSample(id=identifier, query=query, route=route, confidence=0.8, context=context, source="test")


def _response(route: Route, probabilities: dict[str, float], confidence: float = 0.7) -> DecisionResponse:
    raw = {
        "model": "typesafe/jev-1.13-test",
        "answers": {
            "route": {
                "type": "choice",
                "choice": route.value,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 4, "cost": 0.00002},
        "task_id": "task-test",
        "consumed": "0.00002",
    }
    return DecisionResponse(
        answers=raw["answers"],
        usage=raw["usage"],
        task_id=raw["task_id"],
        model=raw["model"],
        consumed=raw["consumed"],
        raw=raw,
    )


@dataclass
class FakeClient:
    responses: list[DecisionResponse | Exception]
    model: str = "typesafe/jev-1.13-test"
    calls: list[dict] = field(default_factory=list)

    def decide(self, state, questions, *, session_id=None, user=None):
        self.calls.append({"state": state, "questions": questions, "session_id": session_id, "user": user})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_the_versioned_question_encodes_the_route_precedence():
    question = benchmark_question()
    assert question.type == "choice"
    assert set(question.criteria) == {"local", "rag", "cloud", "review"}
    assert "precedence" in question.instructions.lower()
    assert "high-stakes" in question.instructions


def test_v2_makes_supplied_material_local_and_absent_material_rag():
    question = benchmark_question(RETRIEVAL_PROMPT_VERSION)
    assert "already present" in question.criteria["local"]
    assert "absent" in question.criteria["rag"]
    assert "supplied query and context" in question.instructions
    with pytest.raises(ValueError, match="unknown prompt version"):
        benchmark_question("future-version")


def test_only_query_and_context_cross_the_remote_boundary():
    sample = _sample("secret-label", "查合同", Route.REVIEW, context=("tenant=acme",))
    assert api_state(sample) == {"query": "查合同", "context": ["tenant=acme"]}
    assert "route" not in api_state(sample)
    assert "confidence" not in api_state(sample)


def test_a_successful_annotation_keeps_evidence_and_zero_one_agreement():
    sample = _sample("s1", "查合同", Route.RAG)
    client = FakeClient([_response(Route.RAG, {"local": 0.05, "rag": 0.8, "cloud": 0.1, "review": 0.05}, 0.73)])
    ticks = iter((1.0, 1.125))
    record = annotate_one(sample, client, clock=lambda: next(ticks))

    assert record["status"] == "ok"
    assert record["agreement_01"] == 1
    assert record["correct"] is True
    assert record["latency_ms"] == pytest.approx(125.0)
    assert record["cost"] == pytest.approx(0.00002)
    assert record["raw_response"]["task_id"] == "task-test"
    assert record["prompt_version"] == PROMPT_VERSION
    assert client.calls[0]["state"] == {"query": "查合同", "context": []}
    assert client.calls[0]["session_id"] == f"minijev:{PROMPT_VERSION}"


def test_retryable_failures_back_off_but_a_401_does_not_retry():
    sample = _sample("s1", "x", Route.LOCAL)
    response = _response(Route.LOCAL, {"local": 1.0, "rag": 0.0, "cloud": 0.0, "review": 0.0}, 1.0)
    retrying = FakeClient([JevApiError("busy", status=503), response])
    sleeps: list[float] = []
    record = annotate_one(sample, retrying, max_retries=3, sleep=sleeps.append)
    assert record["status"] == "ok"
    assert record["attempts"] == 2
    assert sleeps == [1.0]

    rejected = FakeClient([JevApiError("unauthorized", status=401, body='{"message":"bad key"}')])
    record = annotate_one(sample, rejected, max_retries=3, sleep=lambda _: pytest.fail("must not sleep"))
    assert record["status"] == "error"
    assert record["available"] is False
    assert record["attempts"] == 1
    assert record["error"]["http_status"] == 401
    assert "Authorization" not in json.dumps(record)


def test_batch_output_is_append_only_and_resume_skips_successes(tmp_path):
    samples = [_sample("a", "first", Route.LOCAL), _sample("b", "second", Route.RAG)]
    first = FakeClient([_response(Route.LOCAL, {"local": 1, "rag": 0, "cloud": 0, "review": 0}, 1.0)])
    output = tmp_path / "jev.jsonl"
    run_annotations(samples, first, output, limit=1, sleep=lambda _: None)
    assert len(load_records(output)) == 1

    second = FakeClient([_response(Route.RAG, {"local": 0, "rag": 1, "cloud": 0, "review": 0}, 1.0)])
    records = run_annotations(samples, second, output, resume=True, sleep=lambda _: None)
    assert len(records) == 2
    assert [call["state"]["query"] for call in second.calls] == ["second"]
    with pytest.raises(FileExistsError):
        run_annotations(samples, second, output)


def test_resume_does_not_reuse_a_response_from_another_prompt_version(tmp_path):
    sample = _sample("a", "first", Route.LOCAL)
    output = tmp_path / "ab.jsonl"
    v1 = FakeClient([_response(Route.RAG, {"local": 0, "rag": 1, "cloud": 0, "review": 0}, 1.0)])
    run_annotations([sample], v1, output, prompt_version=PROMPT_VERSION)
    v2 = FakeClient([_response(Route.LOCAL, {"local": 1, "rag": 0, "cloud": 0, "review": 0}, 1.0)])
    records = run_annotations([sample], v2, output, resume=True, prompt_version=RETRIEVAL_PROMPT_VERSION)
    assert len(records) == 2
    assert records[-1]["prompt_version"] == RETRIEVAL_PROMPT_VERSION
    assert v2.calls[0]["session_id"] == f"minijev:{RETRIEVAL_PROMPT_VERSION}"


def test_prompt_comparison_is_paired_and_counts_fixes_and_regressions():
    samples = [_sample("a", "has material", Route.LOCAL), _sample("b", "needs fetch", Route.RAG)]
    baseline = [
        annotate_one(samples[0], FakeClient([_response(Route.RAG, {"local": 0, "rag": 1, "cloud": 0, "review": 0}, 1.0)])),
        annotate_one(samples[1], FakeClient([_response(Route.RAG, {"local": 0, "rag": 1, "cloud": 0, "review": 0}, 1.0)])),
    ]
    candidate = [
        annotate_one(
            samples[0],
            FakeClient([_response(Route.LOCAL, {"local": 1, "rag": 0, "cloud": 0, "review": 0}, 1.0)]),
            prompt_version=RETRIEVAL_PROMPT_VERSION,
        ),
        annotate_one(
            samples[1],
            FakeClient([_response(Route.LOCAL, {"local": 1, "rag": 0, "cloud": 0, "review": 0}, 1.0)]),
            prompt_version=RETRIEVAL_PROMPT_VERSION,
        ),
    ]
    report = compare_prompt_records(baseline, candidate, samples)
    assert report["paired_rows"] == 2
    assert report["fixed_errors"] == 1
    assert report["regressions"] == 1
    assert report["net_fixes"] == 0
    assert report["prediction_transitions"] == {"rag>local": 2}


def test_presence_probe_is_six_matched_absent_supplied_pairs():
    from minijev.data import load_jsonl

    path = Path(__file__).resolve().parents[1] / "data" / "jev_presence_probe.jsonl"
    rows = load_jsonl(path)
    groups: dict[str, list[JevSample]] = {}
    for row in rows:
        groups.setdefault(str(row.metadata["semantic_group_id"]), []).append(row)
    assert len(groups) == 6
    for pair in groups.values():
        assert len(pair) == 2
        assert len({row.query for row in pair}) == 1
        assert {row.route for row in pair} == {Route.LOCAL, Route.RAG}
        assert {bool(row.context) for row in pair} == {False, True}


def test_report_contains_classification_calibration_cost_and_review_queue():
    samples = [
        _sample("a", "first", Route.LOCAL),
        _sample("b", "second", Route.RAG),
        _sample("c", "third", Route.CLOUD),
        _sample("d", "fourth", Route.REVIEW),
    ]
    predictions = [Route.LOCAL, Route.LOCAL, Route.CLOUD, Route.REVIEW]
    records = []
    for sample, prediction in zip(samples, predictions):
        probabilities = {route.value: 0.05 for route in Route}
        probabilities[prediction.value] = 0.85
        records.append(annotate_one(sample, FakeClient([_response(prediction, probabilities, 0.8)])))

    report = evaluate_records(records, samples)
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["coverage"]["fraction"] == pytest.approx(1.0)
    assert report["confusion_matrix"]["rag"]["local"] == 1
    assert report["annotator_api_measurements"]["total_cost"] == pytest.approx(0.00008)
    assert report["confidence"]["samples"] == 4
    assert report["probability_scores"]["multiclass_brier"] > 0
    assert len(report["disagreements"]) == 1
    assert report["disagreements"][0]["sample_id"] == "b"
    assert report["boundary"]["pairs"]["pairs"] == 0


def test_a_later_success_supersedes_an_earlier_failed_attempt_in_the_report():
    sample = _sample("a", "first", Route.LOCAL)
    failed = {
        "sample_id": sample_key(sample),
        "query": sample.query,
        "ground_truth": sample.route.value,
        "status": "error",
    }
    success = annotate_one(sample, FakeClient([_response(Route.LOCAL, {"local": 1, "rag": 0, "cloud": 0, "review": 0}, 1.0)]))
    report = evaluate_records([failed, success], [sample])
    assert report["coverage"]["successful_rows"] == 1
    assert report["accuracy"] == 1.0


def test_result_cannot_be_joined_to_a_changed_answer_key():
    original = _sample("a", "first", Route.LOCAL)
    record = annotate_one(original, FakeClient([_response(Route.LOCAL, {"local": 1, "rag": 0, "cloud": 0, "review": 0}, 1.0)]))
    changed = _sample("a", "first", Route.REVIEW)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_records([record], [changed])


def test_the_pre_runner_live_artifact_can_be_reused_without_another_api_call(tmp_path):
    sample = JevSample(
        id="frame_1_0",
        query="查合同",
        route=Route.RAG,
        confidence=0.8,
        metadata={"frame_id": "f1", "semantic_group_id": "frame:f1"},
    )
    legacy = {
        "records": [
            {
                "frame_id": "f1",
                "query": "查合同",
                "ours_route": "rag",
                "jev_route": "rag",
                "jev_route_probabilities": {"local": 0.0, "rag": 1.0, "cloud": 0.0, "review": 0.0},
                "jev_route_confidence": 1.0,
                "cost": 0.00004,
                "model": "typesafe/jev-1.13-test",
            }
        ]
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")
    records = load_result_records(path, [sample])
    assert records[0]["sample_id"] == "frame_1_0"
    assert records[0]["agreement_01"] == 1
    assert records[0]["legacy_record"]["frame_id"] == "f1"
    report = evaluate_records(records, [sample])
    assert report["accuracy"] == 1.0
    assert report["annotator_api_measurements"]["mean_latency_ms"] is None
