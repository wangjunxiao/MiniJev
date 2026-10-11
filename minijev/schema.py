"""The MiniJev data contract: routes, samples, counterfactual candidates, decisions.

This module is the single source of truth for the label space. Everything that
produces or consumes a route imports `Route` from here, so the four-way routing
vocabulary is defined exactly once.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class Route(str, Enum):
    """The four handling paths a query can be routed to."""

    LOCAL = "local"
    RAG = "rag"
    CLOUD = "cloud"
    REVIEW = "review"


# Stable ordering: every tensor, confusion matrix and report indexes routes by this.
ROUTE_ORDER: tuple[Route, ...] = (Route.LOCAL, Route.RAG, Route.CLOUD, Route.REVIEW)
ROUTE_INDEX: dict[Route, int] = {route: index for index, route in enumerate(ROUTE_ORDER)}
INDEX_ROUTE: dict[int, Route] = {index: route for route, index in ROUTE_INDEX.items()}

CORE_FIELDS = frozenset({"id", "query", "confidence", "route", "context", "candidates"})


@dataclass(frozen=True)
class Candidate:
    """One counterfactual branch: what each route would have scored for this query.

    A single route label says which action was taken; a candidate table says what
    every action was worth. Off-policy evaluation and any future policy-gradient
    step need the table, not the label. Every numeric field except `reward` is
    optional so partially observed data still round-trips.

    `eligible` marks whether the action was *acceptable* at all. A model that fails
    the quality bar is not a candidate anyone should select, even if its raw score
    beats the alternatives -- without this flag a reward table would pick the
    biggest model for a medical query it gets right only 61% of the time, which is
    exactly the error a `review` route exists to prevent. Ineligible candidates
    score -inf in `minijev.rewards`, and when no candidate is eligible the correct
    counterfactual is the terminal `review` action.
    """

    route: Route
    reward: float
    quality: float | None = None
    cost: float | None = None
    latency_ms: float | None = None
    propensity: float | None = None
    eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"route": self.route.value, "reward": self.reward}
        for key in ("quality", "cost", "latency_ms", "propensity"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if not self.eligible:
            payload["eligible"] = False
        return payload

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Candidate":
        return cls(
            route=Route(str(row["route"])),
            reward=float(row["reward"]),
            quality=_optional_float(row.get("quality")),
            cost=_optional_float(row.get("cost")),
            latency_ms=_optional_float(row.get("latency_ms")),
            propensity=_optional_float(row.get("propensity")),
            eligible=row.get("eligible", True) is not False,
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class JevSample:
    """One labelled decision example.

    `route` and `confidence` are the supervised targets. `candidates` is the
    optional counterfactual table used by off-policy evaluation; it is empty for
    purely supervised data. Unknown keys survive in `metadata` so richer source
    schemas (MASSIVE intents, annotator votes, domain tags) are not destroyed by
    a load/save round trip.
    """

    query: str
    confidence: float
    route: Route
    context: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    id: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.query.strip():
            raise ValueError("query must not be empty")

    @property
    def has_counterfactuals(self) -> bool:
        return len(self.candidates) > 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": self.query, "confidence": self.confidence, "route": self.route.value, "context": list(self.context)}
        if self.candidates:
            payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        if self.id is not None:
            payload["id"] = self.id
        if self.source is not None:
            payload["source"] = self.source
        for key, value in self.metadata.items():
            payload.setdefault(key, value)
        return payload

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "JevSample":
        return cls(
            query=str(row["query"]),
            confidence=float(row["confidence"]),
            route=Route(str(row["route"])),
            context=tuple(str(item) for item in row.get("context", []) or ()),
            candidates=tuple(Candidate.from_dict(item) for item in row.get("candidates", []) or ()),
            id=None if row.get("id") is None else str(row["id"]),
            source=None if row.get("source") is None else str(row["source"]),
            metadata={key: value for key, value in row.items() if key not in CORE_FIELDS and key != "source"},
        )


@dataclass(frozen=True)
class RouteScores:
    """Raw model output: logits over the routes, and nothing else.

    There is no confidence field here on purpose. Confidence is a statistic of the
    route distribution, derived by `minijev.confidence`, so that there is exactly
    one source for it. An earlier version carried a separately-trained confidence
    here as well; the two disagreed, and the learned one was wrong.
    """

    route_logits: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.route_logits) != len(ROUTE_ORDER):
            raise ValueError(f"expected {len(ROUTE_ORDER)} route logits, got {len(self.route_logits)}")


@dataclass(frozen=True)
class Decision:
    """A routing decision, including what a policy step or RL update needs.

    `route_logits` and `route_probs` are kept as full distributions rather than an
    argmax index: a policy-gradient step must sample from them and compute a
    log-probability, which an argmax would have destroyed. `confidence` is a
    statistic of `route_probs`, and `confidence_measure` records which one, so two
    decisions are only comparable when they used the same measure.
    """

    route: Route
    confidence: float
    route_logits: tuple[float, ...]
    route_probs: tuple[float, ...]
    policy: str
    confidence_measure: str = "unknown"
    escalated_from: Route | None = None

    @property
    def escalated(self) -> bool:
        """Whether the policy raised the route above what the model preferred."""
        return self.escalated_from is not None

    @property
    def top_route(self) -> Route:
        """The route the model itself preferred, before any policy escalation."""
        return ROUTE_ORDER[max(range(len(ROUTE_ORDER)), key=self.route_logits.__getitem__)]

    def log_prob(self, route: Route | None = None) -> float:
        chosen = self.route if route is None else route
        probability = max(self.route_probs[ROUTE_INDEX[chosen]], 1e-12)
        return math.log(probability)


def save_jsonl(samples: Iterable[JevSample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[JevSample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [JevSample.from_dict(json.loads(line)) for line in handle if line.strip()]
