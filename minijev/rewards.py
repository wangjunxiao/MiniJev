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
    # Explicit product assumptions. An empty mapping preserves the original
    # quality-cost utility until real misroute outcomes calibrate these values.
    error_costs: Mapping[Route | str, float] = field(default_factory=dict)


DEFAULT_REWARD_WEIGHTS = RewardWeights()
REWARD_VERSION = "heuristic-v1"


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
) -> float:
    """Quality-cost utility with an optional route-specific error penalty.

    The full form is ``w_q * quality - w_c * cost - w_l * latency -
    error_cost(route) * (1 - quality)``. Missing terms are treated as 0.

    Negative cost means "cheaper is better", matching the routing literature
    (`r(x, i) = a_i(x) - p * c_i`). An `eligible=False` action -- one that fails
    the quality bar it was recorded against -- scores -inf, because a slightly
    better score on a model that is not reliable enough is not a better action.
    """
    if not eligible:
        return float("-inf")
    total = 0.0
    if quality is not None:
        total += weights.quality * quality
        total -= _route_error_cost(route, weights) * (1.0 - quality)
    if cost is not None:
        total -= weights.cost * cost
    if latency_ms is not None:
        total -= weights.latency * latency_ms
    return total


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

    def __init__(self, samples: Iterable[JevSample], *, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS):
        self._weights = weights
        self._by_query: dict[str, tuple[Candidate, ...]] = {}
        for sample in samples:
            if sample.has_counterfactuals:
                self._by_query.setdefault(sample.query, sample.candidates)

    def __len__(self) -> int:
        return len(self._by_query)

    def reward_table(self, query: str) -> dict[Route, float]:
        candidates = self._by_query.get(query, ())
        table = {
            candidate.route: compute_reward(
                candidate.route,
                quality=candidate.quality,
                cost=candidate.cost,
                latency_ms=candidate.latency_ms,
                weights=self._weights,
                eligible=candidate.eligible,
            )
            for candidate in candidates
        }
        if table and not any(math.isfinite(value) for value in table.values()):
            table[Route.REVIEW] = 0.0
        return table

    def step(self, query: str, route: Route) -> tuple[float, dict[str, Any]]:
        candidates = {candidate.route: candidate for candidate in self._by_query.get(query, ())}
        candidate = candidates.get(route)
        if candidate is None:
            # The terminal review action is a valid counterfactual when nothing
            # recorded was eligible, and must not crash the evaluation.
            if route is Route.REVIEW and candidates and not any(record.eligible for record in candidates.values()):
                return 0.0, {"source": "offline", "reward_version": REWARD_VERSION, "synthesized": True}
            raise KeyError(f"query has no recorded counterfactual for route {route.value!r}: {query!r}")
        reward = compute_reward(
            route,
            quality=candidate.quality,
            cost=candidate.cost,
            latency_ms=candidate.latency_ms,
            weights=self._weights,
            eligible=candidate.eligible,
        )
        return reward, {"propensity": candidate.propensity, "source": "offline", "reward_version": REWARD_VERSION}


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


def build_env(samples: Iterable[JevSample], *, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS) -> RouterEnv:
    """Prefer recorded rewards; fall back to the mock only when there are none."""
    offline = OfflineRouterEnv(samples, weights=weights)
    if len(offline) > 0:
        return offline
    return MockRouterEnv(weights=weights)


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
    "constant_route_check",
]
