"""Batch Jev annotation and leakage-safe comparison against MiniJev benchmarks.

The live Jev response is evidence, not a replacement answer key.  This module
therefore writes an append-only sidecar JSONL and never mutates the source
benchmark.  A later report joins records by sample id, compares them with the
human/derived labels, and makes disagreements an explicit review queue.

Only ``query`` and ``context`` are sent to the remote API.  In particular, the
ground-truth route and frame metadata are attached *after* the response arrives,
so the label cannot leak into Jev's decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .boundary import evaluate_boundary
from .confidence import chance_corrected_top
from .data import load_jsonl
from .evaluation import classification_report, confidence_audit, expected_calibration_error, selective_audit
from .jev_api import DEFAULT_ENDPOINT, DEFAULT_MODEL, DecisionResponse, JevApiClient, JevApiError, Question
from .schema import ROUTE_ORDER, Decision, JevSample, Route

PROMPT_VERSION = "minijev-route-v1"
RETRIEVAL_PROMPT_VERSION = "minijev-route-v2-presence"
PROMPT_VERSIONS = (PROMPT_VERSION, RETRIEVAL_PROMPT_VERSION)
RESULT_SCHEMA_VERSION = "jev-benchmark-v1"

ROUTE_CRITERIA: Mapping[str, str] = {
    "local": "The request is answerable directly from the supplied query/context, without retrieval or a multi-step workflow.",
    "rag": "The request needs one external lookup or extraction from documents, user records, a database, or a service.",
    "cloud": "The request needs multi-step construction, synthesis, planning, or reasoning, but is sufficiently specified and not high-stakes.",
    "review": "A person must decide because the request is high-stakes, materially ambiguous, or missing information needed for safe routing.",
}

ROUTE_INSTRUCTIONS = (
    "Route the user's request to exactly one MiniJev handler. Apply precedence: choose review for high-stakes or materially "
    "underspecified requests; otherwise choose cloud for multi-step construction/reasoning; otherwise choose rag when external "
    "information must be retrieved; otherwise choose local. Judge only the supplied state."
)

PRESENCE_AWARE_ROUTE_CRITERIA: Mapping[str, str] = {
    "local": "All information needed to answer is already present in the supplied state, or the request is a self-contained calculation. Do not fetch anything.",
    "rag": "Information required to answer is absent from the supplied state and must be fetched from a document, user record, database, service, or the web.",
    "cloud": "The supplied state is sufficient, but the request needs multi-step construction, synthesis, planning, or reasoning beyond one lookup.",
    "review": "A person must decide because the request is high-stakes, materially ambiguous, or missing information that cannot be safely resolved by retrieval.",
}

PRESENCE_AWARE_INSTRUCTIONS = (
    "Route the request to exactly one MiniJev handler. Treat the supplied query and context as material already available to the handler: "
    "reading, extracting, counting, or summarising that supplied material is local, not rag. Choose rag only when required information is "
    "absent and must be fetched. Apply precedence after that distinction: review for high-stakes or irreducibly underspecified requests; "
    "otherwise cloud for multi-step construction/reasoning; otherwise rag for an external fetch; otherwise local."
)


class DecisionClient(Protocol):
    """The small client surface needed by the batch runner (and by offline tests)."""

    model: str

    def decide(
        self,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Mapping[str, Question],
        *,
        session_id: str | None = None,
        user: str | None = None,
    ) -> DecisionResponse: ...


def benchmark_question(prompt_version: str = PROMPT_VERSION) -> Question:
    """The versioned four-way question used for all comparable benchmark runs."""
    if prompt_version == PROMPT_VERSION:
        return Question(type="choice", instructions=ROUTE_INSTRUCTIONS, criteria=dict(ROUTE_CRITERIA))
    if prompt_version == RETRIEVAL_PROMPT_VERSION:
        return Question(type="choice", instructions=PRESENCE_AWARE_INSTRUCTIONS, criteria=dict(PRESENCE_AWARE_ROUTE_CRITERIA))
    raise ValueError(f"unknown prompt version {prompt_version!r}; expected one of {PROMPT_VERSIONS}")


def sample_key(sample: JevSample) -> str:
    """A stable join key, even for external rows that do not provide an id."""
    if sample.id:
        return sample.id
    payload = json.dumps(
        {"query": sample.query, "context": list(sample.context), "route": sample.route.value},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def api_state(sample: JevSample) -> dict[str, Any]:
    """The only sample fields allowed to cross the API boundary."""
    return {"query": sample.query, "context": list(sample.context)}


def _retryable(error: JevApiError) -> bool:
    return error.status is None or error.status == 429 or (error.status is not None and error.status >= 500)


def _safe_error(error: JevApiError) -> dict[str, Any]:
    """Keep diagnostics without ever serialising request headers or credentials."""
    detail = error.body[:500]
    try:
        parsed = json.loads(detail) if detail else None
    except json.JSONDecodeError:
        parsed = detail
    return {"type": type(error).__name__, "message": str(error), "http_status": error.status, "response": parsed}


def annotate_one(
    sample: JevSample,
    client: DecisionClient,
    *,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    """Call Jev once (with bounded retries) and build an auditable sidecar record."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    question = benchmark_question(prompt_version)
    key = sample_key(sample)
    started = clock()
    attempts = 0
    response: DecisionResponse | None = None
    last_error: JevApiError | None = None
    while attempts <= max_retries:
        attempts += 1
        try:
            response = client.decide(
                api_state(sample),
                {"route": question},
                session_id=f"minijev:{prompt_version}",
            )
            break
        except JevApiError as error:
            last_error = error
            if attempts > max_retries or not _retryable(error):
                break
            sleep(retry_backoff_s * (2 ** (attempts - 1)))

    latency_ms = (clock() - started) * 1_000.0
    base: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "sample_id": key,
        "query": sample.query,
        "context": list(sample.context),
        "ground_truth": sample.route.value,
        "source": sample.source,
        "frame_id": sample.metadata.get("frame_id"),
        "semantic_group_id": sample.metadata.get("semantic_group_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts,
        "latency_ms": latency_ms,
    }
    if response is None:
        assert last_error is not None
        return {**base, "status": "error", "available": False, "error": _safe_error(last_error)}

    prediction = response.choice("route")
    probabilities = response.probabilities("route")
    if prediction not in {route.value for route in ROUTE_ORDER}:
        error = JevApiError(f"Jev returned an unknown route {prediction!r}")
        return {**base, "status": "error", "available": True, "error": _safe_error(error), "raw_response": response.raw}
    if not probabilities:
        error = JevApiError("Jev returned a route without probabilities")
        return {**base, "status": "error", "available": True, "error": _safe_error(error), "raw_response": response.raw}

    ordered = [max(0.0, probabilities.get(route.value, 0.0)) for route in ROUTE_ORDER]
    total = sum(ordered)
    if not math.isfinite(total) or total <= 0:
        error = JevApiError("Jev returned an invalid probability distribution")
        return {**base, "status": "error", "available": True, "error": _safe_error(error), "raw_response": response.raw}
    normalized = {route.value: value / total for route, value in zip(ROUTE_ORDER, ordered)}
    reported_confidence = response.confidence("route")
    derived_confidence = chance_corrected_top(list(normalized.values()))
    return {
        **base,
        "status": "ok",
        "available": True,
        "jev_prediction": prediction,
        "correct": prediction == sample.route.value,
        "agreement_01": int(prediction == sample.route.value),
        "jev_probabilities": normalized,
        "jev_confidence": reported_confidence if reported_confidence is not None else derived_confidence,
        "derived_confidence": derived_confidence,
        "model": response.model or getattr(client, "model", DEFAULT_MODEL),
        "task_id": response.task_id,
        "usage": response.usage,
        "cost": response.cost,
        "consumed": response.consumed,
        "raw_response": response.raw,
    }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def load_result_records(path: str | Path, samples: Sequence[JevSample]) -> list[dict[str, Any]]:
    """Load native JSONL or the earlier frame-level validation JSON.

    The first live validation finished before this batch runner existed.  Its
    artifact is a pretty-printed object with a ``records`` array and legacy field
    names.  Importing it prevents hundreds of already-paid API calls from being
    repeated.  The converter is intentionally one-way and in-memory: the captured
    file remains immutable evidence.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return load_records(source)
    if isinstance(payload, list):
        return [dict(record) for record in payload]
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{source} is neither benchmark JSONL nor an object containing records")
    records = payload["records"]
    if not records or "ours_route" not in records[0]:
        return [dict(record) for record in records]

    by_frame: dict[str, list[JevSample]] = {}
    for sample in samples:
        frame_id = sample.metadata.get("frame_id")
        if frame_id is not None:
            by_frame.setdefault(str(frame_id), []).append(sample)
    converted: list[dict[str, Any]] = []
    for record in records:
        frame_id = str(record.get("frame_id", ""))
        candidates = by_frame.get(frame_id, [])
        sample = next((row for row in candidates if row.query == record.get("query")), None)
        if sample is None:
            raise ValueError(f"legacy result frame {frame_id!r} cannot be joined to the benchmark")
        probabilities = {str(key): float(value) for key, value in record["jev_route_probabilities"].items()}
        converted.append(
            {
                "schema_version": "jev-label-validation-legacy",
                "prompt_version": "legacy-unversioned",
                "sample_id": sample_key(sample),
                "query": sample.query,
                "context": list(sample.context),
                "ground_truth": sample.route.value,
                "frame_id": frame_id,
                "semantic_group_id": sample.metadata.get("semantic_group_id"),
                "status": "ok",
                "available": True,
                "jev_prediction": record["jev_route"],
                "correct": record["jev_route"] == sample.route.value,
                "agreement_01": int(record["jev_route"] == sample.route.value),
                "jev_probabilities": probabilities,
                "jev_confidence": float(record["jev_route_confidence"]),
                "derived_confidence": chance_corrected_top(list(probabilities.values())),
                "model": record.get("model"),
                "cost": record.get("cost"),
                "latency_ms": None,
                "legacy_record": record,
            }
        )
    return converted


def run_annotations(
    samples: Sequence[JevSample],
    client: DecisionClient,
    output: str | Path,
    *,
    resume: bool = False,
    limit: int | None = None,
    interval_ms: float = 0.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    prompt_version: str = PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """Append responses as they arrive, so interruption loses at most one call."""
    destination = Path(output)
    if destination.exists() and not resume:
        raise FileExistsError(f"{destination} already exists; pass --resume or choose a new artifact")
    existing = load_records(destination) if destination.exists() else []
    completed = {
        str(record.get("sample_id")) for record in existing if record.get("status") == "ok" and record.get("prompt_version") == prompt_version
    }
    keys = [sample_key(sample) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("benchmark sample ids are not unique")
    pending = [sample for sample in samples if sample_key(sample) not in completed]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        pending = pending[:limit]

    destination.parent.mkdir(parents=True, exist_ok=True)
    new_records: list[dict[str, Any]] = []
    with destination.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(pending):
            record = annotate_one(
                sample,
                client,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                sleep=sleep,
                prompt_version=prompt_version,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            new_records.append(record)
            if interval_ms > 0 and index + 1 < len(pending):
                sleep(interval_ms / 1_000.0)
    return existing + new_records


def _latest_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        key = str(record.get("sample_id", ""))
        if key:
            latest[key] = record
    return latest


class _RecordedRouter:
    """Adapt aligned sidecar probabilities to the existing boundary analyser."""

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.records = list(records)

    def decide(self, queries: Sequence[str]) -> list[Decision]:
        if len(queries) != len(self.records):
            raise ValueError("recorded decisions do not align with benchmark rows")
        decisions: list[Decision] = []
        for query, record in zip(queries, self.records):
            if query != record.get("query"):
                raise ValueError("recorded query does not align with benchmark row")
            probabilities = tuple(float(record["jev_probabilities"].get(route.value, 0.0)) for route in ROUTE_ORDER)
            route = Route(str(record["jev_prediction"]))
            decisions.append(
                Decision(
                    route=route,
                    confidence=float(record["jev_confidence"]),
                    route_logits=tuple(math.log(max(value, 1e-12)) for value in probabilities),
                    route_probs=probabilities,
                    policy="jev-api",
                    confidence_measure="chance_corrected_top",
                )
            )
        return decisions


def _probability_scores(records: Sequence[Mapping[str, Any]], samples: Sequence[JevSample]) -> dict[str, float]:
    brier = 0.0
    nll = 0.0
    for record, sample in zip(records, samples):
        probabilities = record["jev_probabilities"]
        for route in ROUTE_ORDER:
            target = 1.0 if route is sample.route else 0.0
            brier += (float(probabilities.get(route.value, 0.0)) - target) ** 2
        nll -= math.log(max(float(probabilities.get(sample.route.value, 0.0)), 1e-12))
    count = len(samples)
    return {"multiclass_brier": brier / count if count else 0.0, "negative_log_likelihood": nll / count if count else 0.0}


def evaluate_records(records: Sequence[Mapping[str, Any]], samples: Sequence[JevSample]) -> dict[str, Any]:
    """Join the sidecar to its benchmark and produce the full Jev audit."""
    latest = _latest_records(records)
    aligned_samples: list[JevSample] = []
    aligned_records: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for sample in samples:
        key = sample_key(sample)
        record = latest.get(key)
        if record is None or record.get("status") != "ok":
            missing.append(key)
            continue
        if record.get("query") != sample.query or record.get("ground_truth") != sample.route.value:
            raise ValueError(f"result {key!r} does not match the benchmark row")
        aligned_samples.append(sample)
        aligned_records.append(record)

    actual = [sample.route for sample in aligned_samples]
    predicted = [Route(str(record["jev_prediction"])) for record in aligned_records]
    confidences = [float(record["jev_confidence"]) for record in aligned_records]
    correct = [truth is guess for truth, guess in zip(actual, predicted)]
    report = classification_report(actual, predicted, name="jev")
    report["coverage"] = {
        "benchmark_rows": len(samples),
        "successful_rows": len(aligned_samples),
        "missing_or_failed_rows": len(missing),
        "fraction": len(aligned_samples) / len(samples) if samples else 0.0,
        "missing_sample_ids": missing,
    }
    report["confidence"] = confidence_audit(confidences, correct)
    report["selective"] = selective_audit(confidences, correct)
    report["probability_scores"] = _probability_scores(aligned_records, aligned_samples)
    report["reported_ece"] = expected_calibration_error(confidences, correct)
    formula_gaps = [abs(float(record["jev_confidence"]) - float(record["derived_confidence"])) for record in aligned_records]
    report["confidence_formula_mae"] = sum(formula_gaps) / len(formula_gaps) if formula_gaps else 0.0
    costs = [float(record["cost"]) for record in aligned_records if record.get("cost") is not None]
    latencies = [float(record["latency_ms"]) for record in aligned_records if record.get("latency_ms") is not None]
    report["annotator_api_measurements"] = {
        "calls": len(aligned_records),
        "total_cost": sum(costs),
        "mean_cost": sum(costs) / len(costs) if costs else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "model_versions": sorted({str(record.get("model", "")) for record in aligned_records if record.get("model")}),
        "note": "These are Jev annotation-call measurements, not the serving cost/latency of MiniJev route actions.",
    }
    report["prompt_versions"] = sorted({str(record.get("prompt_version", "unknown")) for record in aligned_records})
    report["disagreements"] = [
        {
            "sample_id": sample_key(sample),
            "query": sample.query,
            "ground_truth": sample.route.value,
            "jev_prediction": record["jev_prediction"],
            "jev_confidence": record["jev_confidence"],
            "jev_probabilities": record["jev_probabilities"],
            "frame_id": sample.metadata.get("frame_id"),
        }
        for sample, record in zip(aligned_samples, aligned_records)
        if record["jev_prediction"] != sample.route.value
    ]
    report["boundary"] = evaluate_boundary(_RecordedRouter(aligned_records), aligned_samples) if aligned_samples else None
    return report


def compare_prompt_records(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    samples: Sequence[JevSample],
) -> dict[str, Any]:
    """Paired prompt comparison; incomplete rows are excluded from both arms."""
    baseline = _latest_records(baseline_records)
    candidate = _latest_records(candidate_records)
    paired: list[tuple[JevSample, Mapping[str, Any], Mapping[str, Any]]] = []
    for sample in samples:
        key = sample_key(sample)
        left = baseline.get(key)
        right = candidate.get(key)
        if left is None or right is None or left.get("status") != "ok" or right.get("status") != "ok":
            continue
        for record in (left, right):
            if record.get("query") != sample.query or record.get("ground_truth") != sample.route.value:
                raise ValueError(f"result {key!r} does not match the benchmark row")
        paired.append((sample, left, right))

    baseline_predictions = [Route(str(left["jev_prediction"])) for _, left, _ in paired]
    candidate_predictions = [Route(str(right["jev_prediction"])) for _, _, right in paired]
    actual = [sample.route for sample, _, _ in paired]
    transitions: Counter[str] = Counter(f"{left.value}>{right.value}" for left, right in zip(baseline_predictions, candidate_predictions))
    fixes = sum(truth is not left and truth is right for truth, left, right in zip(actual, baseline_predictions, candidate_predictions))
    regressions = sum(truth is left and truth is not right for truth, left, right in zip(actual, baseline_predictions, candidate_predictions))
    disagreements = [
        {
            "sample_id": sample_key(sample),
            "query": sample.query,
            "ground_truth": sample.route.value,
            "baseline_prediction": left.value,
            "candidate_prediction": right.value,
            "baseline_confidence": left_record["jev_confidence"],
            "candidate_confidence": right_record["jev_confidence"],
        }
        for (sample, left_record, right_record), left, right in zip(paired, baseline_predictions, candidate_predictions)
        if left is not right
    ]
    return {
        "paired_rows": len(paired),
        "benchmark_rows": len(samples),
        "baseline_prompt_versions": sorted({str(left.get("prompt_version", "unknown")) for _, left, _ in paired}),
        "candidate_prompt_versions": sorted({str(right.get("prompt_version", "unknown")) for _, _, right in paired}),
        "baseline": classification_report(actual, baseline_predictions, name="baseline_prompt"),
        "candidate": classification_report(actual, candidate_predictions, name="candidate_prompt"),
        "fixed_errors": fixes,
        "regressions": regressions,
        "net_fixes": fixes - regressions,
        "prediction_transitions": dict(sorted(transitions.items())),
        "changed_predictions": disagreements,
    }


def compare_minijev(report: dict[str, Any], records: Sequence[Mapping[str, Any]], samples: Sequence[JevSample], checkpoint: Path) -> None:
    """Add an optional same-row comparison without changing the Jev sidecar."""
    from .inference import load_router

    latest = _latest_records(records)
    aligned = [(sample, latest.get(sample_key(sample))) for sample in samples]
    aligned = [(sample, record) for sample, record in aligned if record is not None and record.get("status") == "ok"]
    paired_samples = [sample for sample, _ in aligned]
    jev_predictions = [Route(str(record["jev_prediction"])) for _, record in aligned]
    router = load_router(checkpoint)
    decisions = router.decide([sample.query for sample in paired_samples])
    mini_predictions = [decision.route for decision in decisions]
    actual = [sample.route for sample in paired_samples]
    mini_report = classification_report(actual, mini_predictions, name="minijev")
    mini_report["boundary"] = evaluate_boundary(router, paired_samples)
    mini_report["head_to_head"] = {
        "rows": len(paired_samples),
        "both_correct": sum(t is j and t is m for t, j, m in zip(actual, jev_predictions, mini_predictions)),
        "only_jev_correct": sum(t is j and t is not m for t, j, m in zip(actual, jev_predictions, mini_predictions)),
        "only_minijev_correct": sum(t is not j and t is m for t, j, m in zip(actual, jev_predictions, mini_predictions)),
        "both_wrong": sum(t is not j and t is not m for t, j, m in zip(actual, jev_predictions, mini_predictions)),
        "same_prediction": sum(j is m for j, m in zip(jev_predictions, mini_predictions)),
    }
    mini_report["prediction_disagreements"] = [
        {
            "sample_id": sample_key(sample),
            "query": sample.query,
            "ground_truth": sample.route.value,
            "jev_prediction": jev.value,
            "minijev_prediction": mini.value,
        }
        for sample, jev, mini in zip(paired_samples, jev_predictions, mini_predictions)
        if jev is not mini
    ]
    report["minijev_comparison"] = mini_report


def _write_report(report: Mapping[str, Any], path: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if path is None:
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(
        f"Jev: rows={report['coverage']['successful_rows']} acc={report['accuracy']:.4f} "
        f"mF1={report['macro_f1']:.4f} ECE={report['reported_ece']:.4f}; wrote {path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.jev_benchmark", description="Run and audit Jev on a labelled MiniJev benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="call the live API and append an independent sidecar")
    run.add_argument("--data", type=Path, default=Path("data/hard_benchmark.jsonl"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--report-out", type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--interval-ms", type=float, default=0.0)
    run.add_argument("--max-retries", type=int, default=3)
    run.add_argument("--retry-backoff-s", type=float, default=1.0)
    run.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=RETRIEVAL_PROMPT_VERSION)

    audit = subparsers.add_parser("report", help="rebuild metrics from a saved sidecar without network access")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--results", type=Path, required=True)
    audit.add_argument("--report-out", type=Path)
    audit.add_argument("--checkpoint", type=Path)
    compare = subparsers.add_parser("compare-prompts", help="paired comparison of two saved prompt runs")
    compare.add_argument("--data", type=Path, required=True)
    compare.add_argument("--baseline-results", type=Path, required=True)
    compare.add_argument("--candidate-results", type=Path, required=True)
    compare.add_argument("--report-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    samples = load_jsonl(args.data)
    if args.command == "run":
        client = JevApiClient(endpoint=args.endpoint, model=args.model, timeout=args.timeout)
        records = run_annotations(
            samples,
            client,
            args.output,
            resume=args.resume,
            limit=args.limit,
            interval_ms=args.interval_ms,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
            prompt_version=args.prompt_version,
        )
    elif args.command == "report":
        records = load_result_records(args.results, samples)
    else:
        comparison = compare_prompt_records(
            load_result_records(args.baseline_results, samples),
            load_result_records(args.candidate_results, samples),
            samples,
        )
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"paired={comparison['paired_rows']} baseline_acc={comparison['baseline']['accuracy']:.4f} "
            f"candidate_acc={comparison['candidate']['accuracy']:.4f} net_fixes={comparison['net_fixes']}; wrote {args.report_out}"
        )
        return
    report = evaluate_records(records, samples)
    if args.checkpoint:
        compare_minijev(report, records, samples, args.checkpoint)
    _write_report(report, args.report_out)


if __name__ == "__main__":
    main()
