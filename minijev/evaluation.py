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
import math
import random
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


def risk_coverage_curve(confidences: Sequence[float], correct: Sequence[bool]) -> list[tuple[float, float]]:
    """`(coverage, risk)` as the confidence gate is loosened from strictest to loosest.

    Sort by confidence descending and accept the top `k`. At coverage `k/n`, `risk` is the
    error rate among the accepted. A confidence that ranks errors below correct predictions
    produces a risk that climbs with coverage; a confidence that does not produces a flat line
    at the base error rate.

    Ties are broken by original position, `sorted` being stable, so a curve is reproducible
    for a given input order. That matters because a model emitting many identical confidences
    -- a collapsed head, say -- would otherwise give a different curve run to run and the
    difference would look like a result.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must align")
    if not confidences:
        return []
    order = sorted(range(len(confidences)), key=lambda index: -confidences[index])
    curve: list[tuple[float, float]] = []
    errors = 0
    for position, index in enumerate(order, start=1):
        if not correct[index]:
            errors += 1
        curve.append((position / len(order), errors / position))
    return curve


def area_under_risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean risk across every coverage level. Lower is better.

    The no-skill value is the base error rate: a confidence that ranks at random accepts errors
    in proportion to their frequency, so the curve stays flat at the overall error rate. A
    useful gate pushes the area *below* that. Reporting AURC without the no-skill value invites
    reading any small number as skill, so `selective_audit` reports both.
    """
    curve = risk_coverage_curve(confidences, correct)
    if not curve:
        return 0.0
    return sum(risk for _, risk in curve) / len(curve)


def random_ranking_aurc(correct: Sequence[bool], *, trials: int = 200, seed: int = 0) -> float:
    """The AURC of a confidence that carries no information about correctness.

    This is the correct no-skill reference, and it is *not* simply the base error rate. With all
    confidences tied, the curve depends entirely on how ties are broken, so a flat confidence can
    land on either side of the base rate by luck of input order -- a fact that made an earlier
    version of this module claim skill where there was none, and that two tests caught.

    Averaging the curve over random orderings gives the expected value a genuinely uninformative
    confidence would produce, which is what a skill ratio needs as its denominator.
    """
    if not correct:
        return 0.0
    rng = random.Random(seed)
    total = 0.0
    for _ in range(trials):
        # Fresh random values per row. Shuffling a list of indices and then using the index as the
        # confidence looks equivalent and is not: the curve sorts by confidence, so sorting by
        # index restores the original order and the "random" baseline silently measures the
        # identity ranking. That bug made a flat confidence appear to have 0.62 skill.
        confidences = [rng.random() for _ in range(len(correct))]
        total += area_under_risk_coverage(confidences, correct)
    return total / trials


def risk_at_coverage(confidences: Sequence[float], correct: Sequence[bool], coverage: float) -> float:
    """Error rate among the most confident `coverage` fraction.

    This is the number a deployment actually promises: "escalate 20%, and the rest is wrong
    this often".
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    curve = risk_coverage_curve(confidences, correct)
    if not curve:
        return 0.0
    index = max(0, min(len(curve) - 1, math.ceil(coverage * len(curve)) - 1))
    return curve[index][1]


def coverage_at_risk(confidences: Sequence[float], correct: Sequence[bool], target_risk: float) -> float:
    """How much can be accepted while keeping the error rate at or below `target_risk`.

    The decision-facing form of the same curve: "to hold errors under 10%, how much must be
    escalated?" Returns 0.0 when even the single most confident prediction is wrong.
    """
    curve = risk_coverage_curve(confidences, correct)
    best = 0.0
    for coverage, risk in curve:
        if risk <= target_risk:
            best = coverage
        else:
            break
    return best


def selective_audit(confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    """Calibration *and* discrimination, because they are different questions.

    `expected_calibration_error` asks whether confidence matches accuracy. `area_under_risk_
    coverage` asks whether confidence *ranks* errors below correct predictions. A model can
    satisfy the first and fail the second -- constant confidence is perfectly calibrated at the
    base rate and completely useless as a gate -- so a report that shows only ECE can declare
    victory while the escalation gate does nothing.

    The accepted-subset ECE is the paper's *selective calibration* (arXiv 2208.12084): among the
    predictions the gate lets through, is the stated confidence still honest? A gate that removes
    errors should leave a better-calibrated remainder; one that removes at random will not.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must align")
    if not confidences:
        return {"rows": 0}
    base_risk = 1.0 - sum(1 for value in correct if value) / len(correct)
    aurc = area_under_risk_coverage(confidences, correct)
    no_skill = random_ranking_aurc(correct)
    # Selective calibration: among the predictions a gate at this coverage would let through,
    # is the stated confidence still honest?
    accepted = sorted(range(len(confidences)), key=lambda index: -confidences[index])[: max(1, math.ceil(0.8 * len(confidences)))]
    accepted_confidence = [confidences[index] for index in accepted]
    accepted_correct = [correct[index] for index in accepted]
    return {
        "rows": len(confidences),
        "base_risk": base_risk,
        "aurc": aurc,
        "no_skill_aurc": no_skill,
        # Measured against a confidence that ranks at random, not against the base error rate:
        # a tie-broken flat confidence can fall either side of the base rate by input order.
        # 1.0 means no better than chance; negative means anti-correlated, which is a real
        # failure mode and worth seeing rather than clamping away.
        "skill_vs_random": (no_skill - aurc) / no_skill if no_skill > 0 else 0.0,
        "risk_at_80pct_coverage": risk_at_coverage(confidences, correct, 0.8),
        "risk_at_50pct_coverage": risk_at_coverage(confidences, correct, 0.5),
        "coverage_at_10pct_risk": coverage_at_risk(confidences, correct, 0.10),
        "coverage_at_20pct_risk": coverage_at_risk(confidences, correct, 0.20),
        "ece_all": expected_calibration_error(confidences, correct, bins),
        "ece_accepted_80pct": expected_calibration_error(accepted_confidence, accepted_correct, bins),
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_confidence_accepted": sum(accepted_confidence) / len(accepted_confidence),
    }


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
