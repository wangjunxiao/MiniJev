"""Where reward comes from, kept separate from how a model is trained.

Two halves, deliberately decoupled:

* `compute_reward` is a pure function, the pluggable scoring rule;
* `RouterEnv` is the environment, the thing that knows what each route was worth.

The observation side (`RouterEnv.step`) is the natural hook for a future RL reward,
and it is separate from the decision side (a `Policy`), so instrumentation can be
added without touching the inference path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from .schema import ROUTE_ORDER, Candidate, JevSample, Route


@dataclass(frozen=True)
class RewardWeights:
    """Weights for `compute_reward`.

    Latency is off by default on purpose: its scale is not comparable to quality.
    A human review queue is measured in minutes and a local call in milliseconds,
    so any fixed latency weight either ignores latency entirely or lets it swamp
    quality. Enable it explicitly once there is a calibrated latency budget.
    """

    quality: float = 1.0
    cost: float = 0.05
    latency: float = 0.0
    cost_scale: float = 1.0
    latency_scale_ms: float = 1.0
    # Explicit product assumptions. An empty mapping preserves the original
    # quality-cost utility until real misroute outcomes calibrate these values.
    error_costs: Mapping[Route | str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost_scale <= 0.0 or self.latency_scale_ms <= 0.0:
            raise ValueError("cost_scale and latency_scale_ms must be positive")
        if not all(math.isfinite(float(value)) for value in (self.quality, self.cost, self.latency)):
            raise ValueError("reward weights must be finite")
        if any(float(value) < 0.0 for value in self.error_costs.values()):
            raise ValueError("error costs must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "quality": self.quality,
            "cost": self.cost,
            "latency": self.latency,
            "cost_scale": self.cost_scale,
            "latency_scale_ms": self.latency_scale_ms,
            "error_costs": {key.value if isinstance(key, Route) else str(key): value for key, value in self.error_costs.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RewardWeights":
        """Restore a serialised reward configuration from sample metadata."""

        raw_errors = payload.get("error_costs", {})
        if not isinstance(raw_errors, Mapping):
            raise ValueError("error_costs must be a mapping")
        return cls(
            quality=float(payload.get("quality", 1.0)),
            cost=float(payload.get("cost", 0.05)),
            latency=float(payload.get("latency", 0.0)),
            cost_scale=float(payload.get("cost_scale", 1.0)),
            latency_scale_ms=float(payload.get("latency_scale_ms", 1.0)),
            error_costs={str(key): float(value) for key, value in raw_errors.items()},
        )


DEFAULT_REWARD_WEIGHTS = RewardWeights()
REWARD_VERSION = "utility-v2"


def _route_error_cost(route: Route | str | None, weights: RewardWeights) -> float:
    if route is None:
        return 0.0
    route_value = route.value if isinstance(route, Route) else str(route)
    return float(weights.error_costs.get(route, weights.error_costs.get(route_value, 0.0)))


def compute_reward(
    route: Route | str | None = None,
    *,
    quality: float | None = None,
    cost: float | None = None,
    latency_ms: float | None = None,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
    eligible: bool = True,
    available: bool | None = None,
) -> float:
    """Quality-cost utility with an optional route-specific error penalty.

    The full form is ``w_q * quality - w_c * cost / cost_scale -
    w_l * latency_ms / latency_scale_ms - error_cost(route) * (1 - quality)``.
    Missing terms are treated as 0. An explicitly unavailable action is never
    rewarded; ``None`` means availability was not observed and is retained as an
    uncertainty in the audit rather than silently treated as a failure.

    Negative cost means "cheaper is better", matching the routing literature
    (`r(x, i) = a_i(x) - p * c_i`). An `eligible=False` action -- one that fails
    the quality bar it was recorded against -- scores -inf, because a slightly
    better score on a model that is not reliable enough is not a better action.
    """
    if not eligible or available is False:
        return float("-inf")
    total = 0.0
    if quality is not None:
        quality = float(quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be a finite number between 0 and 1")
        total += weights.quality * quality
        total -= _route_error_cost(route, weights) * (1.0 - quality)
    if cost is not None:
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("cost must be a finite non-negative number")
        total -= weights.cost * (cost / weights.cost_scale)
    if latency_ms is not None:
        latency_ms = float(latency_ms)
        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            raise ValueError("latency_ms must be a finite non-negative number")
        total -= weights.latency * (latency_ms / weights.latency_scale_ms)
    return total


def reward_for_candidate(candidate: Candidate, *, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> float:
    """Score one recorded counterfactual using the current utility contract."""

    return compute_reward(
        candidate.route,
        quality=candidate.quality,
        cost=candidate.cost,
        latency_ms=candidate.latency_ms,
        weights=weights,
        eligible=candidate.eligible,
        available=candidate.available,
    )


def reward_audit(samples: Iterable[JevSample], *, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> dict[str, Any]:
    """Summarise reward coverage before a bandit run is trusted.

    The audit distinguishes unavailable from unobserved availability and reports
    whether every route has at least one finite counterfactual. This catches the
    common failure where a reward table silently contains only one usable action.
    """

    rows = list(samples)
    route_counts = {route.value: 0 for route in ROUTE_ORDER}
    finite_counts = {route.value: 0 for route in ROUTE_ORDER}
    unavailable_counts = {route.value: 0 for route in ROUTE_ORDER}
    missing_counts = {route.value: 0 for route in ROUTE_ORDER}
    rewards: list[float] = []
    rows_with_no_finite_action = 0
    rows_with_single_finite_action = 0
    best_route_counts = {route.value: 0 for route in ROUTE_ORDER}
    for sample in rows:
        row_rewards: dict[Route, float] = {}
        for candidate in sample.candidates:
            route = candidate.route.value
            route_counts[route] += 1
            if candidate.available is False:
                unavailable_counts[route] += 1
            elif candidate.available is None:
                missing_counts[route] += 1
            value = reward_for_candidate(candidate, weights=weights)
            if math.isfinite(value):
                finite_counts[route] += 1
                rewards.append(value)
                row_rewards[candidate.route] = max(row_rewards.get(candidate.route, float("-inf")), value)
        if not row_rewards:
            rows_with_no_finite_action += 1
        elif len(row_rewards) == 1:
            rows_with_single_finite_action += 1
        if row_rewards:
            best_route = max(row_rewards, key=row_rewards.__getitem__)
            best_route_counts[best_route.value] += 1
    return {
        "rows": len(rows),
        "with_counterfactuals": sum(sample.has_counterfactuals for sample in rows),
        "route_candidate_counts": route_counts,
        "finite_reward_counts": finite_counts,
        "unavailable_counts": unavailable_counts,
        "availability_missing_counts": missing_counts,
        "all_routes_have_finite_candidate": all(finite_counts[route.value] > 0 for route in ROUTE_ORDER),
        "rows_with_no_finite_action": rows_with_no_finite_action,
        "rows_with_single_finite_action": rows_with_single_finite_action,
        "best_route_counts": best_route_counts,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "weights": weights.to_dict(),
        "reward_version": REWARD_VERSION,
    }


def _weights_from_samples(samples: Iterable[JevSample]) -> RewardWeights | None:
    """Read one consistent reward configuration from converted row metadata."""

    observed: list[RewardWeights] = []
    for sample in samples:
        payload = sample.metadata.get("reward_weights")
        if isinstance(payload, Mapping):
            observed.append(RewardWeights.from_dict(payload))
    if not observed:
        return None
    first = observed[0].to_dict()
    if any(weights.to_dict() != first for weights in observed[1:]):
        raise ValueError("samples contain multiple reward configurations; pass weights explicitly")
    return observed[0]


class RouterEnv(Protocol):
    """The environment a routing policy acts in."""

    name: str

    def step(self, query: str, route: Route) -> tuple[float, dict[str, Any]]: ...


class OfflineRouterEnv:
    """Replay rewards from a recorded counterfactual table.

    This is the honest environment: it can score every route because the data
    recorded what each route was worth. Use it whenever samples carry
    `candidates`. When every recorded candidate is ineligible, the only sensible
    counterfactual is the terminal `review` action, which the table reports with a
    neutral reward -- the alternative, letting an unreliable model win by default,
    is the failure a review route exists to prevent.
    """

    name = "offline"

    def __init__(self, samples: Iterable[JevSample], *, weights: RewardWeights | None = None):
        samples = list(samples)
        self._weights = weights or _weights_from_samples(samples) or DEFAULT_REWARD_WEIGHTS
        self._by_query: dict[str, tuple[Candidate, ...]] = {}
        for sample in samples:
            if sample.has_counterfactuals:
                self._by_query.setdefault(sample.query, sample.candidates)

    def __len__(self) -> int:
        return len(self._by_query)

    def reward_table(self, query: str) -> dict[Route, float]:
        candidates = self._by_query.get(query, ())
        table: dict[Route, float] = {}
        for candidate in candidates:
            value = reward_for_candidate(candidate, weights=self._weights)
            # Multiple models can map to one route tier. Keep the best observed
            # counterfactual instead of letting the last model overwrite it.
            table[candidate.route] = max(table.get(candidate.route, float("-inf")), value)
        if table and not any(math.isfinite(value) for value in table.values()):
            table[Route.REVIEW] = 0.0
        return table

    def step(self, query: str, route: Route) -> tuple[float, dict[str, Any]]:
        candidates = [candidate for candidate in self._by_query.get(query, ()) if candidate.route is route]
        if not candidates:
            # The terminal review action is a valid counterfactual when nothing
            # recorded was eligible, and must not crash the evaluation.
            all_candidates = self._by_query.get(query, ())
            if route is Route.REVIEW and all_candidates and not any(record.eligible and record.available is not False for record in all_candidates):
                return 0.0, {"source": "offline", "reward_version": REWARD_VERSION, "synthesized": True}
            raise KeyError(f"query has no recorded counterfactual for route {route.value!r}: {query!r}")
        candidate = max(candidates, key=lambda item: reward_for_candidate(item, weights=self._weights))
        reward = reward_for_candidate(candidate, weights=self._weights)
        return reward, {
            "propensity": candidate.propensity,
            "model": candidate.model,
            "available": candidate.available,
            "source": "offline",
            "reward_version": REWARD_VERSION,
        }


# A deliberately crude placeholder prior. It is query-independent by construction,
# because a mock has no way to know what any particular query deserves.
MOCK_ROUTE_PRIOR: Mapping[Route, tuple[float, float]] = {
    Route.LOCAL: (0.70, 0.00),
    Route.RAG: (0.85, 0.05),
    Route.CLOUD: (0.95, 1.00),
    Route.REVIEW: (0.90, 2.00),
}


class MockRouterEnv:
    """A stand-in environment for data that has no counterfactuals.

    Read the warning before using this for anything but plumbing tests.

    The reward it returns depends only on the route, never on the query. Any policy
    optimised against it therefore has exactly one optimum: always pick the route
    with the highest prior. That collapse is not a bug in this class, it is the
    concrete form of the "routing collapse" failure mode -- and it is the reason a
    policy-gradient step is pointless until real counterfactual rewards exist.

    Use it to exercise the RL plumbing. Do not use it to claim a result.
    """

    name = "mock"

    def __init__(self, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS):
        self._weights = weights
        self._rewards = {
            route: compute_reward(route, quality=quality, cost=cost, weights=weights) for route, (quality, cost) in MOCK_ROUTE_PRIOR.items()
        }

    def reward_table(self, query: str) -> dict[Route, float]:
        del query
        return dict(self._rewards)

    def step(self, query: str, route: Route) -> tuple[float, dict[str, Any]]:
        del query
        quality, cost = MOCK_ROUTE_PRIOR[route]
        return self._rewards[route], {"quality": quality, "cost": cost, "source": "mock", "reward_version": REWARD_VERSION}


def build_env(samples: Iterable[JevSample], *, weights: RewardWeights | None = None) -> RouterEnv:
    """Prefer recorded rewards; fall back to the mock only when there are none."""
    samples = list(samples)
    resolved = weights or _weights_from_samples(samples) or DEFAULT_REWARD_WEIGHTS
    offline = OfflineRouterEnv(samples, weights=resolved)
    if len(offline) > 0:
        return offline
    return MockRouterEnv(weights=resolved)


def constant_route_check(env: RouterEnv, queries: Iterable[str]) -> Route | None:
    """Return the route that maximises reward for every query, or None if none does.

    A non-None result means the environment cannot distinguish queries, so any
    policy trained on it will collapse. Call this before trusting an RL run.
    """
    queries = list(queries)
    if not queries:
        return None
    first = env.reward_table(queries[0])
    if not first:
        return None
    best = max(first, key=first.__getitem__)
    for query in queries[1:]:
        table = env.reward_table(query)
        if not table or max(table, key=table.__getitem__) is not best:
            return None
    return best


__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "MOCK_ROUTE_PRIOR",
    "REWARD_VERSION",
    "MockRouterEnv",
    "OfflineRouterEnv",
    "RewardWeights",
    "ROUTE_ORDER",
    "RouterEnv",
    "build_env",
    "compute_reward",
    "reward_audit",
    "reward_for_candidate",
    "constant_route_check",
]
