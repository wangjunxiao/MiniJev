"""A declared world model, so the reward machinery has something to be tested against.

The gap this fills
------------------
Every reward-side claim in this project was untestable for one reason: there is no
counterfactual log with cost variation. RouterBench's real export is unreachable from this
machine, and the committed fixture is four rows. A preference-conditioned policy cannot be
shown to trade quality against cost on four rows, and `reward_baselines` has nothing to
compute a floor over.

What this is, and what it is not
--------------------------------
This is a **stated** world model, not measured data. Every number below is an assumption
written down so it can be argued with, and any conclusion drawn through it holds *only*
under that assumption. It exists to make the machinery falsifiable -- to answer "does one
policy actually move along the trade-off when the operating point moves?" -- and never to
support a claim about real traffic. `WORLD_NOTICE` is attached to every generated row so a
row cannot be mistaken for an observation.

The world
---------
Difficulty is a property of the query; quality falls with difficulty at different rates per
route, and cost is the price of that quality:

    route     quality            cost     intuition
    local     1 - 0.90*d         0.001    free, and wrong on anything hard
    rag       0.75 + 0.15*d      0.010    grounds an easy query, still limited
    cloud     0.85 + 0.13*d      0.100    100x local, and it holds up
    review    0.95               2.000    a human, 2000x local

The tension is real and is the point: `review` is the most accurate action and costs two
thousand times what `local` does. A router that ignores cost should always escalate; a
router that only counts cost should never escalate. The interesting behaviour is in between,
and it is only interesting if one set of weights can produce all of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rewards import Preference, RewardWeights, compute_reward
from .schema import ROUTE_ORDER, Candidate, JevSample, Route, save_jsonl

WORLD_NOTICE = "declared-world-v1: synthetic counterfactuals from a stated model, not observations"
"""Attached to every row's metadata. A generated row must never look like measured data."""


@dataclass(frozen=True)
class RouteWorld:
    """Per-route quality, cost and latency as declared functions of query difficulty."""

    name: str = "declared-v1"
    local_quality_intercept: float = 1.0
    local_quality_slope: float = -0.90
    rag_quality_intercept: float = 0.75
    rag_quality_slope: float = 0.15
    cloud_quality_intercept: float = 0.85
    cloud_quality_slope: float = 0.13
    review_quality: float = 0.95
    costs: tuple[tuple[Route, float], ...] = (
        (Route.LOCAL, 0.001),
        (Route.RAG, 0.010),
        (Route.CLOUD, 0.100),
        (Route.REVIEW, 2.000),
    )
    latencies_ms: tuple[tuple[Route, float], ...] = (
        (Route.LOCAL, 20.0),
        (Route.RAG, 120.0),
        (Route.CLOUD, 900.0),
        (Route.REVIEW, 3_600_000.0),
    )

    def __post_init__(self) -> None:
        routes = {route for route, _ in self.costs}
        if routes != set(ROUTE_ORDER):
            raise ValueError(f"costs must cover every route, got {sorted(r.value for r in routes)}")
        if any(cost < 0.0 for _, cost in self.costs):
            raise ValueError("costs must be non-negative")

    def quality(self, route: Route, difficulty: float) -> float:
        """Quality is clipped to [0, 1]; a declared model must not emit impossible values."""
        if route is Route.LOCAL:
            value = self.local_quality_intercept + self.local_quality_slope * difficulty
        elif route is Route.RAG:
            value = self.rag_quality_intercept + self.rag_quality_slope * difficulty
        elif route is Route.CLOUD:
            value = self.cloud_quality_intercept + self.cloud_quality_slope * difficulty
        else:
            value = self.review_quality
        return min(1.0, max(0.0, value))

    def cost(self, route: Route) -> float:
        return dict(self.costs)[route]

    def latency_ms(self, route: Route) -> float:
        return dict(self.latencies_ms)[route]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "costs": {route.value: cost for route, cost in self.costs},
            "latencies_ms": {route.value: value for route, value in self.latencies_ms},
            "quality_slopes": {
                "local": self.local_quality_slope,
                "rag": self.rag_quality_slope,
                "cloud": self.cloud_quality_slope,
                "review": 0.0,
            },
        }


DEFAULT_WORLD = RouteWorld()


def difficulty_of(query: str) -> float:
    """A deterministic difficulty in [0, 1] derived from the text.

    The first version hashed the text. That was deterministic and reproducible and completely
    useless: a hash has no relation to what the query says, so the reward-optimal route was
    unpredictable from the text *by construction* and the best any model could do was the
    majority class. It presented as a policy that trained to `val_macro_f1 = 0.16` and then
    decayed -- which looks like underfitting and was actually an impossible task.

    Difficulty is now a stated function of what the router can actually see: how long the
    request is and how many clauses it asks for. Both are surface facts a byte-level model can
    read, so the problem is hard rather than impossible. This remains a *declared* model -- it
    says nothing about real difficulty -- but it is at least a model of something observable.
    """
    length_term = min(1.0, len(query) / 90.0)
    clause_term = min(1.0, sum(query.count(marker) for marker in _CLAUSE_MARKERS) / 3.0)
    return min(1.0, max(0.0, 0.65 * length_term + 0.35 * clause_term))


_CLAUSE_MARKERS: tuple[str, ...] = ("，", "并", "然后", "最后", "再", "接着")
"""Multi-step and continuation cues. Their presence is a surface fact, not a claim about
meaning, which is exactly what makes the difficulty it implies learnable."""


def counterfactuals(query: str, world: RouteWorld = DEFAULT_WORLD, reference: Preference | None = None) -> tuple[Candidate, ...]:
    """Every route's quality, cost and latency for one query."""
    difficulty = difficulty_of(query)
    weights = RewardWeights().with_preference(reference or Preference(quality=1.0, cost=1.0))
    candidates = []
    for route in ROUTE_ORDER:
        quality = world.quality(route, difficulty)
        cost = world.cost(route)
        candidates.append(
            Candidate(
                route=route,
                reward=compute_reward(route, quality=quality, cost=cost, weights=weights),
                quality=quality,
                cost=cost,
                latency_ms=world.latency_ms(route),
                # A full counterfactual table: every action was observed for every query, so
                # the propensities of the logging policy are not needed and are left unset
                # rather than filled with a placeholder that would silently poison IPS.
                propensity=None,
                model=route.value,
                eligible=True,
            )
        )
    return tuple(candidates)


def generate_router_log(
    queries: Iterable[str],
    *,
    world: RouteWorld = DEFAULT_WORLD,
    reference: Preference | None = None,
    source: str = "declared-world",
) -> list[JevSample]:
    """Build a replayable counterfactual log: one row per query, one candidate per route.

    `reference` decides the supervised route label: a row is labelled with the route the
    reference operating point prefers. That keeps the label and the reward consistent -- a
    label that contradicted the reward table would teach the model to be wrong on purpose.
    """
    preference = reference or Preference(quality=1.0, cost=1.0)
    rows: list[JevSample] = []
    for index, query in enumerate(queries):
        candidates = counterfactuals(query, world, preference)
        finite = [candidate for candidate in candidates if candidate.reward is not None]
        best = max(finite, key=lambda candidate: float(candidate.reward))
        rows.append(
            JevSample(
                query=query,
                confidence=1.0,
                route=best.route,
                candidates=candidates,
                id=f"world_{index:05d}",
                source=source,
                metadata={
                    "semantic_group_id": f"world:{hashlib.sha256(query.encode()).hexdigest()[:12]}",
                    "world_notice": WORLD_NOTICE,
                    "world": world.to_dict(),
                    "difficulty": round(difficulty_of(query), 4),
                    "reference_cost_share": round(preference.cost_share, 4),
                    "route_rule": "labelled with the route the reference operating point prefers",
                },
            )
        )
    return rows


def world_audit(rows: Sequence[JevSample]) -> dict[str, Any]:
    """What the declared world actually contains, so its coverage can be checked.

    Reports where each route wins. A world in which one route wins everywhere cannot test a
    trade-off, and a world in which no route ever wins is broken.
    """
    winners: dict[str, int] = {route.value: 0 for route in ROUTE_ORDER}
    quality_ranges: dict[str, list[float]] = {route.value: [] for route in ROUTE_ORDER}
    for row in rows:
        finite = [candidate for candidate in row.candidates if candidate.reward is not None]
        if not finite:
            continue
        winners[max(finite, key=lambda c: float(c.reward)).route.value] += 1
        for candidate in row.candidates:
            if candidate.quality is not None:
                quality_ranges[candidate.route.value].append(candidate.quality)
    return {
        "rows": len(rows),
        "notice": WORLD_NOTICE,
        "winner_counts": winners,
        "routes_that_ever_win": sum(1 for count in winners.values() if count),
        "quality_range": {route: (round(min(values), 4), round(max(values), 4)) for route, values in quality_ranges.items() if values},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.routerlog", description="Generate a declared-world counterfactual log")
    parser.add_argument("--queries", type=Path, required=True, help="JSONL whose `query` field supplies the prompts")
    parser.add_argument("--output", type=Path, default=Path("data/router_world.jsonl"))
    parser.add_argument("--reference-cost-share", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .data import load_jsonl

    args = build_parser().parse_args(argv)
    queries = [row.query for row in load_jsonl(args.queries)]
    if args.limit:
        queries = queries[: args.limit]
    rows = generate_router_log(queries, reference=Preference.at_cost_share(args.reference_cost_share))
    save_jsonl(rows, args.output)
    print(json.dumps(world_audit(rows), ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
