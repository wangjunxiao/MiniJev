"""Router implementations behind one interface.

Every router does the same two things: turn queries into scores, and turn scores
into a decision via a policy. Adding a router means adding one class and one
registry entry -- the evaluation harness, the CLIs and the policies all pick it up
through `ROUTERS` without changes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .config import JevConfig
from .model import TinyJev, load_checkpoint, load_checkpoint_config
from .policies import Policy, RiskLadderPolicy
from .schema import ROUTE_INDEX, ROUTE_ORDER, Decision, JevSample, Route, RouteScores
from .tokenizer import ByteTokenizer


class Router(ABC):
    """Base class: implement `score`, inherit the decision path.

    `contexts` is optional and positionally aligned with `queries`: entry *i* is the
    preceding turns for query *i*. It defaults to `None`, which means "no state", so every
    caller that predates context keeps working unchanged. A router whose config was not
    built for turn separators will simply concatenate the turns, which is why a
    single-turn checkpoint cannot use context -- see `contexts` module docs.
    """

    name: str = "router"

    def __init__(self, policy: Policy | None = None):
        self.policy: Policy = policy if policy is not None else RiskLadderPolicy()

    @abstractmethod
    def score(self, queries: Sequence[str], contexts: Sequence[Sequence[str]] | None = None) -> list[RouteScores]:
        """Return one `RouteScores` per query, in order."""

    def decide(self, queries: Sequence[str], contexts: Sequence[Sequence[str]] | None = None) -> list[Decision]:
        return [self.policy.decide(scores) for scores in self.score(queries, contexts)]

    def route(self, query: str, context: Sequence[str] | None = None) -> Decision:
        contexts = [context] if context is not None else None
        return self.decide([query], contexts)[0]

    def decide_samples(self, samples: Sequence[JevSample]) -> list[Decision]:
        """Decide for samples, using each sample's `context` when it has one."""
        queries = [sample.query for sample in samples]
        contexts = [tuple(sample.context) for sample in samples]
        if all(not turns for turns in contexts):
            return self.decide(queries)
        return self.decide(queries, contexts)

    @staticmethod
    def _turn_sequences(queries: Sequence[str], contexts: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]]:
        if contexts is None:
            return [(query,) for query in queries]
        if len(contexts) != len(queries):
            raise ValueError("contexts must align with queries")
        return [tuple(turns) + (query,) for query, turns in zip(queries, contexts)]


class NeuralRouter(Router):
    """The learned model: byte tokenizer, mean-pooled encoder, two heads."""

    name = "neural"

    def __init__(
        self,
        config: JevConfig | None = None,
        *,
        checkpoint: str | None = None,
        policy: Policy | None = None,
        batch_size: int = 128,
    ):
        super().__init__(policy)
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        # A checkpoint trained at a non-default width carries its shapes in a
        # sidecar; trusting the caller's default here would build a network the
        # weights cannot fit into.
        resolved = config
        if resolved is None and checkpoint is not None:
            resolved = load_checkpoint_config(checkpoint)
        self.config = resolved or JevConfig()
        self.tokenizer = ByteTokenizer(self.config)
        self.model = TinyJev(self.config)
        if checkpoint is not None:
            load_checkpoint(self.model, checkpoint)

    def score(self, queries: Sequence[str], contexts: Sequence[Sequence[str]] | None = None) -> list[RouteScores]:
        scores: list[RouteScores] = []
        turns = self._turn_sequences(queries, contexts)
        for start in range(0, len(turns), self.batch_size):
            batch = turns[start : start + self.batch_size]
            input_ids, attention_mask = self.tokenizer.batch_turns(batch)
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.route_logits.numpy().tolist()
            scores.extend(RouteScores(route_logits=tuple(float(x) for x in row)) for row in logits)
        return scores


# Keyword sets for the inspectable lower bound. Order matters: the first group that
# matches wins, so the risky and retrieval groups are checked before the easy ones.
RULE_KEYWORDS: tuple[tuple[Route, tuple[str, ...]], ...] = (
    (Route.REVIEW, ("不确定", "安全", "医疗", "法律", "风险", "不知道")),
    (Route.RAG, ("手册", "知识库", "文档", "政策", "查一下", "搜索", "查看")),
    (Route.CLOUD, ("设计", "比较", "分析", "算法", "并发", "策略")),
    (Route.LOCAL, ("你好", "加 2", "翻译", "星期")),
)

RULE_LOGIT_HIT = 2.0
RULE_LOGIT_MISS = -2.0


class RuleRouter(Router):
    """Small, inspectable keyword lower bound.

    Emits score vectors rather than routes so it flows through the same policy and
    evaluation code as the learned model. A keyword match is treated as a decisive
    read, so its derived confidence is high and the ladder escalates nothing --
    a rule baseline should not get the safety net the model is measured against.
    """

    name = "rule"

    def __init__(self, policy: Policy | None = None, *, default_route: Route = Route.REVIEW):
        super().__init__(policy)
        self.default_route = default_route

    def score(self, queries: Sequence[str], contexts: Sequence[Sequence[str]] | None = None) -> list[RouteScores]:
        turns = self._turn_sequences(queries, contexts)
        return [self._score_one(turn[-1]) for turn in turns]

    def _score_one(self, query: str) -> RouteScores:
        matched = self.default_route
        for route, words in RULE_KEYWORDS:
            if any(word in query for word in words):
                matched = route
                break
        logits = tuple(RULE_LOGIT_HIT if route is matched else RULE_LOGIT_MISS for route in ROUTE_ORDER)
        return RouteScores(route_logits=logits)


def _tally(routes: Iterable[Route]) -> tuple[int, ...]:
    counts = [0] * len(ROUTE_ORDER)
    for route in routes:
        counts[ROUTE_INDEX[route]] += 1
    return tuple(counts)


class PriorRouter(Router):
    """Always answers with the most frequent route seen in training.

    This is the baseline a real accuracy number has to beat. A model that scores
    below it is worse than knowing nothing about the query.
    """

    name = "prior"

    def __init__(self, routes: Iterable[Route] | None = None, policy: Policy | None = None):
        super().__init__(policy)
        self.counts: tuple[int, ...] = _tally(routes) if routes is not None else ()

    @classmethod
    def from_routes(cls, routes: Iterable[Route], policy: Policy | None = None) -> "PriorRouter":
        return cls(routes, policy)

    @property
    def majority(self) -> Route:
        if not self.counts or not any(self.counts):
            return Route.LOCAL
        return ROUTE_ORDER[max(range(len(ROUTE_ORDER)), key=self.counts.__getitem__)]

    def score(self, queries: Sequence[str], contexts: Sequence[Sequence[str]] | None = None) -> list[RouteScores]:
        logits = tuple(RULE_LOGIT_HIT if route is self.majority else RULE_LOGIT_MISS for route in ROUTE_ORDER)
        return [RouteScores(route_logits=logits) for _ in queries]

    def distribution(self) -> dict[str, float]:
        total = sum(self.counts) or 1
        return {route.value: self.counts[index] / total for index, route in enumerate(ROUTE_ORDER)}


@dataclass
class DecisionStats:
    """Sanity counters for a batch of decisions.

    Two failure modes this exists to catch, both of which leave accuracy looking
    fine: a derived confidence with no spread (the route head stopped
    discriminating) and an escalation rate at 0 or 1 (the ladder is not doing
    anything, or it is doing all of it).
    """

    count: int = 0
    confidence_min: float = math.inf
    confidence_max: float = -math.inf
    confidence_sum: float = 0.0
    escalated: int = 0
    route_counts: dict[str, int] = field(default_factory=dict)
    escalated_from_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, decision: Decision) -> None:
        self.count += 1
        self.confidence_min = min(self.confidence_min, decision.confidence)
        self.confidence_max = max(self.confidence_max, decision.confidence)
        self.confidence_sum += decision.confidence
        self.route_counts[decision.route.value] = self.route_counts.get(decision.route.value, 0) + 1
        if decision.escalated_from is not None:
            self.escalated += 1
            key = decision.escalated_from.value
            self.escalated_from_counts[key] = self.escalated_from_counts.get(key, 0) + 1

    @property
    def confidence_mean(self) -> float:
        return self.confidence_sum / self.count if self.count else 0.0

    @property
    def confidence_spread(self) -> float:
        return 0.0 if self.count == 0 else self.confidence_max - self.confidence_min

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "confidence_mean": self.confidence_mean,
            "confidence_min": None if self.count == 0 else self.confidence_min,
            "confidence_max": None if self.count == 0 else self.confidence_max,
            "confidence_spread": self.confidence_spread,
            "escalation_rate": self.escalation_rate,
            "escalated_from": dict(self.escalated_from_counts),
            "route_counts": dict(self.route_counts),
        }


ROUTERS: dict[str, type[Router]] = {
    "neural": NeuralRouter,
    "rule": RuleRouter,
    "prior": PriorRouter,
}


def router_from_name(name: str, **kwargs: object) -> Router:
    if name not in ROUTERS:
        raise KeyError(f"unknown router {name!r}; expected one of {sorted(ROUTERS)}")
    return ROUTERS[name](**kwargs)  # type: ignore[call-arg]


__all__ = [
    "ROUTERS",
    "RULE_KEYWORDS",
    "DecisionStats",
    "NeuralRouter",
    "PriorRouter",
    "Router",
    "RuleRouter",
    "router_from_name",
]
