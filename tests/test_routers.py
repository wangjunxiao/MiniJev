"""Routers produce scores; policies spend them. Neither may reach into the other."""

from __future__ import annotations

import pytest

from minijev.policies import ArgmaxPolicy, FlatThresholdPolicy, RiskLadderPolicy
from minijev.routers import ROUTERS, DecisionStats, NeuralRouter, PriorRouter, RuleRouter, router_from_name
from minijev.schema import ROUTE_ORDER, Route, RouteScores


def test_registry_exposes_the_three_routers():
    assert set(ROUTERS) == {"neural", "rule", "prior"}
    assert router_from_name("rule").name == "rule"
    with pytest.raises(KeyError):
        router_from_name("does-not-exist")


def test_every_router_returns_one_score_per_query_in_order():
    queries = ["你好", "查一下产品手册", "设计一个分布式系统"]
    for router in (RuleRouter(), PriorRouter([Route.LOCAL])):
        scores = router.score(queries)
        assert len(scores) == len(queries)
        assert all(len(value.route_logits) == len(ROUTE_ORDER) for value in scores)


def test_rule_router_routes_the_four_kinds():
    router = RuleRouter()
    assert router.route("你好").route is Route.LOCAL
    assert router.route("查一下产品手册").route is Route.RAG
    assert router.route("设计一个分布式系统").route is Route.CLOUD
    assert router.route("这个有法律风险吗").route is Route.REVIEW


def test_a_keyword_match_is_decisive_so_the_ladder_leaves_it_alone():
    """The baseline must not get the safety net the model is measured against."""
    decision = RuleRouter().route("你好")
    assert decision.confidence > RiskLadderPolicy().ladder[Route.LOCAL].accept
    assert decision.escalated is False


def test_prior_router_picks_the_majority_route():
    router = PriorRouter([Route.RAG] * 5 + [Route.LOCAL] * 2)
    assert router.majority is Route.RAG
    assert router.route("任意查询").route is Route.RAG
    assert router.distribution()["rag"] == pytest.approx(5 / 7)


def test_prior_router_with_no_history_is_not_a_crash():
    assert PriorRouter().majority is Route.LOCAL


def test_a_router_uses_whatever_policy_it_is_given():
    queries = ["设计一个分布式系统"]
    assert RuleRouter(ArgmaxPolicy()).route(queries[0]).route is Route.CLOUD
    assert RuleRouter(FlatThresholdPolicy(review_threshold=1.0)).route(queries[0]).route is Route.REVIEW


def test_neural_router_without_a_checkpoint_is_explicitly_untrained():
    """No silent fallback to random weights."""
    router = NeuralRouter()
    assert router.checkpoint is None
    scores = router.score(["你好"])
    assert len(scores) == 1


def test_decision_stats_expose_a_non_discriminating_route_head():
    """A flat distribution behind every query means the head learned nothing."""
    stats = DecisionStats()
    policy = RiskLadderPolicy()
    for _ in range(5):
        stats.observe(policy.decide(RouteScores(route_logits=(0.0, 0.0, 0.0, 0.0))))
    assert stats.confidence_spread < 1e-9
    assert stats.confidence_mean == pytest.approx(0.0)
    assert stats.route_counts == {"review": 5}, "a flat read escalates all the way to review"
    assert stats.escalation_rate == pytest.approx(1.0)


def test_decision_stats_separate_a_decisive_read_from_a_flat_one():
    stats = DecisionStats()
    policy = RiskLadderPolicy()
    stats.observe(policy.decide(RouteScores(route_logits=(6.0, 0.0, 0.0, 0.0))))
    stats.observe(policy.decide(RouteScores(route_logits=(0.3, 0.3, 0.3, 0.0))))
    assert stats.confidence_spread > 0.2
    assert stats.escalation_rate == pytest.approx(0.5), "only the flat read needs escalating"
    assert stats.route_counts == {"local": 1, "review": 1}
