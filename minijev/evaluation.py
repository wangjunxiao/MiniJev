"""Metrics, calibration audits and reports.

Three deliberate choices, each of which exists because the first version of this
project reported a number that could not support the claim made from it:

* the majority-class baseline is always reported next to the model;
* a confidence audit is always reported, because a confidence head that collapsed
  to a constant looks identical to a well-calibrated one in an accuracy number;
* metrics can be restricted to the routes a source actually supports, so a dataset
  with no `cloud` samples cannot quietly contribute a zero to a macro average.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .confidence import CONFIDENCE_MEASURES, DEFAULT_MEASURE, distribution_shape
from .policies import FlatThresholdPolicy, Policy, RiskLadderPolicy, sweep_flat_threshold, sweep_ladder
from .routers import PriorRouter, Router
from .schema import ROUTE_ORDER, JevSample, Route, RouteScores

DEFAULT_ECE_BINS = 10


def confusion_matrix(actual: Sequence[Route], predicted: Sequence[Route]) -> dict[str, dict[str, int]]:
    matrix = {truth.value: {guess.value: 0 for guess in ROUTE_ORDER} for truth in ROUTE_ORDER}
    for truth, guess in zip(actual, predicted):
        matrix[truth.value][guess.value] += 1
    return matrix


def per_class_scores(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for route in ROUTE_ORDER:
        true_positive = matrix[route.value][route.value]
        predicted = sum(matrix[truth.value][route.value] for truth in ROUTE_ORDER)
        actual = sum(matrix[route.value].values())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[route.value] = {"support": actual, "precision": precision, "recall": recall, "f1": f1}
    return scores


def classification_report(
    actual: Sequence[Route],
    predicted: Sequence[Route],
    *,
    labels: Sequence[Route] = ROUTE_ORDER,
    name: str = "model",
    scope_to_labels: bool = False,
) -> dict[str, Any]:
    """Accuracy, macro-F1 and the confusion matrix over `labels`.

    `labels` alone decides which classes contribute to the macro average, so an
    absent class cannot drag it to zero. Set `scope_to_labels` as well to also drop
    rows whose true route is outside `labels` -- that answers "how good is this on
    the classes the source can actually evidence", which is a different and usually
    more useful question. Predictions that fall outside `labels` always count as
    errors, and their number is reported as `out_of_scope_predictions`.
    """
    in_scope = set(labels)
    pairs = list(zip(actual, predicted))
    if scope_to_labels:
        pairs = [(truth, guess) for truth, guess in pairs if truth in in_scope]
    scoped_actual = [truth for truth, _ in pairs]
    scoped_predicted = [guess for _, guess in pairs]

    matrix = confusion_matrix(scoped_actual, scoped_predicted)
    scores = per_class_scores(matrix)
    total = len(scoped_actual)
    correct = sum(matrix[route.value][route.value] for route in labels)
    f1_values = [scores[route.value]["f1"] for route in labels]
    supports = {route.value: scores[route.value]["support"] for route in labels}
    return {
        "name": name,
        "labels": [route.value for route in labels],
        "samples": total,
        "scoped_to_labels": scope_to_labels,
        "out_of_scope_predictions": sum(1 for guess in scoped_predicted if guess not in in_scope),
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": {route.value: scores[route.value] for route in labels},
        "supports": supports,
        "unrepresented_labels": [route.value for route in labels if supports[route.value] == 0],
        "confusion_matrix": {
            truth: {guess: count for guess, count in row.items() if guess in in_scope} for truth, row in matrix.items() if truth in in_scope
        },
    }


def majority_baseline(
    samples: Sequence[JevSample],
    *,
    labels: Sequence[Route] = ROUTE_ORDER,
    training_samples: Sequence[JevSample] | None = None,
    scope_to_labels: bool = False,
) -> dict[str, Any]:
    """The score a router gets by always answering with the most common route.

    Pass `training_samples` so the majority route is chosen from what the router
    would actually have seen. Leaving it out picks the majority of the evaluation
    set instead, which reads the answer key and makes the baseline look stronger
    than any real system could be. `scope_to_labels` must match whatever the model
    report used, or the two numbers are not comparable.
    """
    source = samples if training_samples is None else training_samples
    router = PriorRouter.from_routes([sample.route for sample in source])
    predicted = [router.route(sample.query).route for sample in samples]
    report = classification_report([sample.route for sample in samples], predicted, labels=labels, name="majority", scope_to_labels=scope_to_labels)
    report["majority_route"] = router.majority.value
    report["fitted_on"] = "evaluation set (optimistic)" if training_samples is None else "training set"
    report["training_distribution"] = router.distribution()
    return report


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_ECE_BINS) -> float:
    """Mean gap between stated confidence and observed accuracy."""
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [position for position, value in enumerate(confidences) if low <= value < high or (index == bins - 1 and value == 1.0)]
        if not members:
            continue
        accuracy = sum(1 for position in members if correct[position]) / len(members)
        mean_confidence = sum(confidences[position] for position in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def confidence_audit(confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    """Statistics that expose a collapsed confidence head.

    `spread` near zero means the head ignores its input. When that happens the
    accuracy, the F1 and the ECE all still look plausible, which is exactly why
    this audit is reported unconditionally.
    """
    if not confidences:
        return {"samples": 0, "collapsed": True}
    spread = max(confidences) - min(confidences)
    return {
        "samples": len(confidences),
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "spread": spread,
        "accuracy": sum(1 for value in correct if value) / len(correct),
        "ece": expected_calibration_error(confidences, correct, bins),
        "collapsed": spread < 0.05,
    }


def evaluate_router(
    router: Router,
    samples: Sequence[JevSample],
    *,
    labels: Sequence[Route] = ROUTE_ORDER,
    name: str | None = None,
) -> dict[str, Any]:
    """Full report for one router: classification, its baseline, and a calibration audit."""
    decisions = router.decide([sample.query for sample in samples])
    actual = [sample.route for sample in samples]
    predicted = [decision.route for decision in decisions]
    confidences = [decision.confidence for decision in decisions]
    correct = [guess is truth for truth, guess in zip(actual, predicted)]
    report = classification_report(actual, predicted, labels=labels, name=name or router.name)
    report["escalated_by_policy"] = sum(1 for decision in decisions if decision.escalated)
    report["confidence"] = confidence_audit(confidences, correct)
    return report


def evaluate_scores(
    scores: Sequence[RouteScores],
    samples: Sequence[JevSample],
    policy: Policy,
    *,
    labels: Sequence[Route] = ROUTE_ORDER,
    sweep: bool = False,
    name: str = "model",
    scope_to_labels: bool = False,
) -> dict[str, Any]:
    """Report a policy over precomputed scores, optionally sweeping its caution."""
    decisions = [policy.decide(value) for value in scores]
    actual = [sample.route for sample in samples]
    predicted = [decision.route for decision in decisions]
    report = classification_report(actual, predicted, labels=labels, name=name, scope_to_labels=scope_to_labels)
    report["escalated_by_policy"] = sum(1 for decision in decisions if decision.escalated)
    report["confidence"] = confidence_audit(
        [decision.confidence for decision in decisions], [guess is truth for truth, guess in zip(actual, predicted)]
    )
    measure = getattr(policy, "measure", DEFAULT_MEASURE)
    report["confidence"]["measure"] = measure
    report["confidence"]["escalation_rate"] = report["escalated_by_policy"] / (len(decisions) or 1)
    shapes = [distribution_shape(decision.route_probs, measure) for decision in decisions]
    report["distribution"] = {
        "measure": measure,
        "mean_effective_options": sum(shape["effective_options"] for shape in shapes) / (len(shapes) or 1),
        "mean_top_probability": sum(shape["top_probability"] for shape in shapes) / (len(shapes) or 1),
    }
    if sweep:
        report["sweep"] = _sweep(policy, scores, actual)
    return report


def _sweep(policy: Policy, scores: Sequence[RouteScores], actual: Sequence[Route]) -> dict[str, Any]:
    """How cautious should the system be? One knob, so it can actually be tuned."""
    if isinstance(policy, RiskLadderPolicy):
        best, points = sweep_ladder(policy, scores, actual)
        return {"kind": "ladder_caution", "best": best, "points": points}
    if isinstance(policy, FlatThresholdPolicy):
        best, points = sweep_flat_threshold(policy, scores, actual)
        return {"kind": "flat_threshold", "best": best, "points": points}
    return {"kind": "none", "best": None, "points": []}


def _parse_labels(value: str) -> tuple[Route, ...]:
    if value == "supported":
        return (Route.LOCAL, Route.RAG)
    routes = tuple(Route(part.strip()) for part in value.split(",") if part.strip())
    if not routes:
        raise argparse.ArgumentTypeError("at least one route label is required")
    return routes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.evaluation", description="Evaluate MiniJev routers")
    parser.add_argument("dataset", type=Path, nargs="+")
    parser.add_argument("--router", default="neural", help="router name from minijev.routers.ROUTERS")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--labels", type=_parse_labels, default=ROUTE_ORDER, help="comma-separated routes, or 'supported' for local,rag")
    parser.add_argument("--policy", default="risk_ladder", help="policy name from minijev.policies.POLICIES")
    parser.add_argument("--measure", default=DEFAULT_MEASURE, help="confidence measure: " + ", ".join(sorted(CONFIDENCE_MEASURES)))
    parser.add_argument("--caution", type=float, default=1.0, help="scale every route gate; >1 escalates more often")
    parser.add_argument("--sweep", action="store_true", help="sweep the caution level and report the trade-off curve")
    parser.add_argument("--baseline", action="store_true", help="also report the majority-class baseline")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .data import load_many
    from .routers import router_from_name

    args = build_parser().parse_args(argv)
    samples = load_many(args.dataset)

    kwargs: dict[str, object] = {}
    if args.policy == "risk_ladder":
        kwargs["policy"] = RiskLadderPolicy(measure=args.measure).scaled(args.caution)
    elif args.policy == "flat_threshold":
        kwargs["policy"] = FlatThresholdPolicy(measure=args.measure)
    if args.router == "neural":
        kwargs["checkpoint"] = None if args.checkpoint is None else str(args.checkpoint)
    elif args.router == "prior":
        kwargs["routes"] = [sample.route for sample in samples]
    router = router_from_name(args.router, **kwargs)

    scores = router.score([sample.query for sample in samples])
    report: dict[str, Any] = {
        "datasets": [str(path) for path in args.dataset],
        "router": router.name,
        "policy": router.policy.name,
        "labels": [route.value for route in args.labels],
        "all_labels": evaluate_scores(scores, samples, router.policy, name=router.name),
        "restricted": evaluate_scores(scores, samples, router.policy, labels=args.labels, sweep=args.sweep, name=router.name, scope_to_labels=True),
    }
    if args.baseline:
        # Reported once per scope so each baseline lines up with the model number
        # it is meant to be compared against. This CLI has no training split, so
        # the majority route comes from the evaluation set and the report says so.
        report["majority_baseline"] = majority_baseline(samples, labels=ROUTE_ORDER)
        if tuple(args.labels) != ROUTE_ORDER:
            report["majority_baseline_restricted"] = majority_baseline(samples, labels=args.labels, scope_to_labels=True)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
