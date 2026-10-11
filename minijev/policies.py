"""Turn model scores into a routing decision.

This is the only module that knows what a route means. The model produces a
distribution; a policy spends it. Keeping the two apart is what lets thresholds be
calibrated on validation data without retraining, and what lets a reward function
change the decision rule without touching the network.

Two policies are provided, and the contrast between them is the point:

* `RiskLadderPolicy` gates each route at its own confidence level, because the
  routes carry different consequences. Showing the wrong local answer is
  recoverable; sending a high-stakes question to a small local model is not.
* `FlatThresholdPolicy` uses one number for every route. It is kept so a
  single-threshold baseline stays measurable, not because it is a good idea.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol

from .confidence import DEFAULT_MEASURE, confidence_from, distribution_shape
from .schema import ROUTE_ORDER, Decision, Route, RouteScores

DEFAULT_REVIEW_THRESHOLD = 0.35


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    largest = max(logits)
    exponentials = [math.exp(value - largest) for value in logits]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _preferred_route(logits: Sequence[float]) -> Route:
    return ROUTE_ORDER[max(range(len(ROUTE_ORDER)), key=logits.__getitem__)]


class Policy(Protocol):
    """A decision rule: scores in, route out."""

    name: str

    def decide(self, scores: RouteScores) -> Decision: ...


@dataclass(frozen=True)
class RouteGate:
    """What one route demands before its action is taken without escalating.

    `accept` is a confidence floor, not a probability. `escalate_to` is where the
    decision goes when the floor is not met -- normally the next more capable or
    more careful route.
    """

    accept: float
    escalate_to: Route

    def __post_init__(self) -> None:
        if not 0.0 <= self.accept <= 1.0:
            raise ValueError("accept must be between 0 and 1")


# Gates are ordered by how bad it is to take the action wrongly, which is what the
# stakes imply -- not by position in the escalation chain:
#
#   local   a small local model answering badly is cheap and recoverable, so the
#           gate is low and most reads are simply accepted;
#   rag     grounding may miss, so it demands a little more;
#   cloud   expensive, and paying for a large model on the wrong question is the
#           worst outcome in this ladder, so it demands the most;
#   review  the safest place to be, so it accepts anything. A genuinely unknown
#           answer stays with a human rather than bouncing.
#
# These are reasoned starting points, not fitted values. `sweep_ladder` and
# `minijev.evaluation --sweep` give the accuracy/caution curve; where to sit on it
# depends on what a misroute costs, which is a product decision.
DEFAULT_LADDER: Mapping[Route, RouteGate] = MappingProxyType(
    {
        Route.LOCAL: RouteGate(accept=0.30, escalate_to=Route.RAG),
        Route.RAG: RouteGate(accept=0.35, escalate_to=Route.CLOUD),
        Route.CLOUD: RouteGate(accept=0.55, escalate_to=Route.REVIEW),
        Route.REVIEW: RouteGate(accept=0.00, escalate_to=Route.REVIEW),
    }
)


@dataclass(frozen=True)
class RiskLadderPolicy:
    """Per-route confidence gates: the threshold depends on what is at stake.

    A single number cannot express that a wrong `local` answer is cheap and a wrong
    `cloud` decision is expensive, so each route carries its own floor and its own
    escalation target. Raise a gate to make that route more cautious; the rest of
    the system is unaffected.
    """

    measure: str = DEFAULT_MEASURE
    ladder: Mapping[Route, RouteGate] = DEFAULT_LADDER
    max_escalations: int = 4
    name: str = "risk_ladder"

    def __post_init__(self) -> None:
        missing = [route.value for route in ROUTE_ORDER if route not in self.ladder]
        if missing:
            raise ValueError(f"ladder is missing gates for {missing}")

    def decide(self, scores: RouteScores) -> Decision:
        probabilities = softmax(scores.route_logits)
        confidence = confidence_from(probabilities, self.measure)
        preferred = _preferred_route(scores.route_logits)

        route = preferred
        for _ in range(self.max_escalations):
            gate = self.ladder[route]
            if confidence >= gate.accept:
                break
            route = gate.escalate_to

        return Decision(
            route=route,
            confidence=confidence,
            route_logits=scores.route_logits,
            route_probs=probabilities,
            policy=self.name,
            confidence_measure=self.measure,
            escalated_from=None if route is preferred else preferred,
        )

    def with_ladder(self, ladder: Mapping[Route, RouteGate]) -> "RiskLadderPolicy":
        return replace(self, ladder=ladder)

    def scaled(self, factor: float) -> "RiskLadderPolicy":
        """Every gate multiplied by `factor`, clamped to `[0, 1]`.

        Sweeping this one number answers "how cautious should the whole system be",
        which is the only sane way to tune four thresholds at once.
        """
        return replace(
            self,
            ladder={route: RouteGate(min(max(gate.accept * factor, 0.0), 1.0), gate.escalate_to) for route, gate in self.ladder.items()},
        )


@dataclass(frozen=True)
class FlatThresholdPolicy:
    """One confidence floor for every route, escalating straight to review.

    The pattern to measure against, not to ship: it cannot say that a wrong `local`
    answer is cheaper than a wrong `cloud` one.
    """

    review_threshold: float = DEFAULT_REVIEW_THRESHOLD
    measure: str = DEFAULT_MEASURE
    name: str = "flat_threshold"

    def decide(self, scores: RouteScores) -> Decision:
        probabilities = softmax(scores.route_logits)
        confidence = confidence_from(probabilities, self.measure)
        preferred = _preferred_route(scores.route_logits)
        route = Route.REVIEW if confidence < self.review_threshold else preferred
        return Decision(
            route=route,
            confidence=confidence,
            route_logits=scores.route_logits,
            route_probs=probabilities,
            policy=self.name,
            confidence_measure=self.measure,
            escalated_from=None if route is preferred else preferred,
        )

    def with_threshold(self, review_threshold: float) -> "FlatThresholdPolicy":
        return replace(self, review_threshold=review_threshold)


@dataclass(frozen=True)
class ArgmaxPolicy:
    """Ignore confidence entirely and always take the highest-scoring route."""

    measure: str = DEFAULT_MEASURE
    name: str = "argmax"

    def decide(self, scores: RouteScores) -> Decision:
        return Decision(
            route=_preferred_route(scores.route_logits),
            confidence=confidence_from(softmax(scores.route_logits), self.measure),
            route_logits=scores.route_logits,
            route_probs=softmax(scores.route_logits),
            policy=self.name,
            confidence_measure=self.measure,
        )


def _summarise(policy: Policy, scores: Sequence[RouteScores], expected: Sequence[Route], **extra: float) -> dict[str, float]:
    correct = 0
    escalated = 0
    for value, target in zip(scores, expected):
        decision = policy.decide(value)
        correct += int(decision.route is target)
        escalated += int(decision.escalated)
    total = len(scores) or 1
    return {"accuracy": correct / total, "escalation_rate": escalated / total, **extra}


def sweep_ladder(
    policy: RiskLadderPolicy,
    scores: Sequence[RouteScores],
    expected: Sequence[Route],
    *,
    factors: Sequence[float] | None = None,
    max_escalation_rate: float | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Scale every gate together and report the accuracy/caution trade-off.

    Returns the best factor and the whole sweep so the caller can plot it rather
    than quote one number. `max_escalation_rate` bounds how often the system may
    push work to a more expensive route.
    """
    grid = list(factors) if factors is not None else [round(0.25 * step, 2) for step in range(0, 17)]
    points = [_summarise(policy.scaled(factor), scores, expected, factor=factor) for factor in grid]
    return _pick(points, "factor", max_escalation_rate)


def sweep_flat_threshold(
    policy: FlatThresholdPolicy,
    scores: Sequence[RouteScores],
    expected: Sequence[Route],
    *,
    candidates: Sequence[float] | None = None,
    max_escalation_rate: float | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """The single-threshold equivalent, for comparison against the ladder."""
    grid = list(candidates) if candidates is not None else [round(0.05 * step, 2) for step in range(0, 20)]
    points = [_summarise(policy.with_threshold(value), scores, expected, threshold=value) for value in grid]
    return _pick(points, "threshold", max_escalation_rate)


def _pick(points: list[dict[str, float]], key: str, max_escalation_rate: float | None) -> tuple[float, list[dict[str, float]]]:
    eligible = points
    if max_escalation_rate is not None:
        within_budget = [point for point in points if point["escalation_rate"] <= max_escalation_rate]
        eligible = within_budget or points
    best = max(eligible, key=lambda point: (point["accuracy"], -point["escalation_rate"]))
    return best[key], points


POLICIES: dict[str, type] = {
    "risk_ladder": RiskLadderPolicy,
    "flat_threshold": FlatThresholdPolicy,
    "argmax": ArgmaxPolicy,
}


def policy_from_name(name: str, **kwargs: object) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; expected one of {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)  # type: ignore[call-arg]


__all__ = [
    "DEFAULT_LADDER",
    "DEFAULT_MEASURE",
    "DEFAULT_REVIEW_THRESHOLD",
    "POLICIES",
    "ArgmaxPolicy",
    "FlatThresholdPolicy",
    "Policy",
    "RiskLadderPolicy",
    "RouteGate",
    "distribution_shape",
    "policy_from_name",
    "softmax",
    "sweep_flat_threshold",
    "sweep_ladder",
]
