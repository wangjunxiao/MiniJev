"""Reward is the missing ingredient for RL; these tests pin down what exists and what does not."""

from __future__ import annotations

import pytest

from minijev.rewards import (
    MOCK_ROUTE_PRIOR,
    MockRouterEnv,
    OfflineRouterEnv,
    RewardWeights,
    build_env,
    compute_reward,
    constant_route_check,
    reward_audit,
    reward_for_candidate,
)
from minijev.schema import Candidate, JevSample, Route


def test_reward_is_quality_minus_weighted_cost():
    weights = RewardWeights(quality=1.0, cost=0.5, latency=0.0)
    assert compute_reward(quality=0.8, cost=1.0, weights=weights) == pytest.approx(0.3)
    assert compute_reward(quality=0.8, weights=weights) == pytest.approx(0.8)


def test_missing_terms_are_ignored_not_treated_as_zero_quality():
    assert compute_reward(cost=2.0, weights=RewardWeights(quality=1.0, cost=0.1)) == pytest.approx(-0.2)


def test_latency_is_off_by_default_because_its_scale_is_incomparable():
    assert RewardWeights().latency == 0.0
    default = compute_reward(quality=1.0, latency_ms=100_000)
    enabled = compute_reward(quality=1.0, latency_ms=100_000, weights=RewardWeights(latency=0.001))
    assert default == pytest.approx(1.0)
    assert enabled < default


def test_latency_would_swamp_quality_if_it_were_on_by_default():
    """Documents why the default is off: a review queue is minutes, a local call is milliseconds."""
    review = compute_reward(quality=0.99, cost=2.0, latency_ms=3_600_000, weights=RewardWeights(latency=0.001))
    local = compute_reward(quality=0.70, cost=0.0, latency_ms=120, weights=RewardWeights(latency=0.001))
    assert review < local


def test_reward_normalizes_cost_and_latency_scales_explicitly():
    weights = RewardWeights(cost=0.5, latency=0.25, cost_scale=10.0, latency_scale_ms=1000.0)
    assert compute_reward(quality=0.8, cost=10.0, latency_ms=1000.0, weights=weights) == pytest.approx(0.05)


def test_offline_env_reuses_the_reward_configuration_recorded_with_rows():
    weights = RewardWeights(cost=0.5, cost_scale=10.0)
    sample = JevSample(
        query="configured",
        confidence=1.0,
        route=Route.LOCAL,
        candidates=(Candidate(route=Route.LOCAL, reward=0.3, quality=0.8, cost=10.0),),
        metadata={"reward_weights": weights.to_dict()},
    )
    env = OfflineRouterEnv([sample])
    reward, _ = env.step("configured", Route.LOCAL)
    assert reward == pytest.approx(0.3)


def test_unavailable_candidate_is_not_rewarded_but_unknown_availability_is_audited():
    unavailable = Candidate(route=Route.LOCAL, reward=0.0, quality=1.0, available=False)
    unknown = Candidate(route=Route.RAG, reward=0.0, quality=1.0, available=None)
    assert reward_for_candidate(unavailable) == float("-inf")
    assert reward_for_candidate(unknown) == pytest.approx(1.0)
    audit = reward_audit([JevSample(query="q", confidence=1.0, route=Route.RAG, candidates=(unavailable, unknown))])
    assert audit["unavailable_counts"]["local"] == 1
    assert audit["availability_missing_counts"]["rag"] == 1
    assert audit["best_route_counts"]["rag"] == 1


def _counterfactual_sample() -> JevSample:
    return JevSample(
        query="设计一个分布式系统",
        confidence=0.6,
        route=Route.CLOUD,
        candidates=(
            Candidate(route=Route.LOCAL, reward=0.2, quality=0.3, cost=0.0, propensity=0.5),
            Candidate(route=Route.CLOUD, reward=0.9, quality=0.95, cost=1.0, propensity=0.1),
        ),
    )


def test_offline_env_returns_a_recorded_reward_for_any_route():
    env = OfflineRouterEnv([_counterfactual_sample()])
    reward, info = env.step("设计一个分布式系统", Route.LOCAL)
    assert reward == pytest.approx(compute_reward(quality=0.3, cost=0.0))
    assert info["propensity"] == pytest.approx(0.5)
    assert len(env.reward_table("设计一个分布式系统")) == 2


def test_offline_env_refuses_to_invent_an_unrecorded_route():
    env = OfflineRouterEnv([_counterfactual_sample()])
    with pytest.raises(KeyError):
        env.step("设计一个分布式系统", Route.REVIEW)


def test_mock_env_reward_depends_only_on_the_route():
    """This is the routing-collapse trap, made explicit."""
    env = MockRouterEnv()
    first = env.step("你好", Route.LOCAL)[0]
    second = env.step("设计一个复杂系统", Route.LOCAL)[0]
    assert first == second
    assert set(MOCK_ROUTE_PRIOR) == set(Route)
    for route in Route:
        assert env.step("任意查询", route)[0] == pytest.approx(env.step("另一个查询", route)[0])


def test_constant_route_check_flags_the_mock_and_clears_the_offline_env():
    """Call this before trusting any RL run."""
    assert constant_route_check(MockRouterEnv(), ["a", "b", "c"]) is not None
    # Query-dependent rewards: the best route differs between the two queries, so
    # no single route dominates and a policy can actually learn something.
    offline = OfflineRouterEnv(
        [
            _counterfactual_sample(),
            JevSample(
                query="你好",
                confidence=1.0,
                route=Route.LOCAL,
                candidates=(
                    Candidate(route=Route.LOCAL, reward=0.99, quality=0.99, cost=0.0),
                    Candidate(route=Route.CLOUD, reward=0.45, quality=0.50, cost=1.0),
                ),
            ),
        ]
    )
    assert constant_route_check(offline, ["设计一个分布式系统", "你好"]) is None
    assert constant_route_check(offline, ["设计一个分布式系统"]) is Route.CLOUD
    assert constant_route_check(offline, ["你好"]) is Route.LOCAL


def test_build_env_prefers_recorded_rewards_over_the_mock():
    assert isinstance(build_env([_counterfactual_sample()]), OfflineRouterEnv)
    plain = JevSample(query="你好", confidence=1.0, route=Route.LOCAL)
    assert isinstance(build_env([plain]), MockRouterEnv)


def test_an_ineligible_action_scores_negative_infinity():
    """A model below the quality bar is not selectable, whatever its raw score."""
    assert compute_reward(quality=0.99, cost=0.0, eligible=False) == float("-inf")
    assert compute_reward(quality=0.61, cost=0.1, eligible=True) > float("-inf")


def _unreliable_sample() -> JevSample:
    """No model clears the bar: the correct counterfactual is review."""
    return JevSample(
        query="Give a definitive medical diagnosis from this short message.",
        confidence=0.4,
        route=Route.REVIEW,
        candidates=(
            Candidate(route=Route.LOCAL, reward=0.20, quality=0.20, cost=0.001, eligible=False),
            Candidate(route=Route.RAG, reward=0.42, quality=0.42, cost=0.01, eligible=False),
            Candidate(route=Route.CLOUD, reward=0.61, quality=0.61, cost=0.1, eligible=False),
        ),
    )


def test_reward_table_picks_review_when_nothing_is_eligible():
    """The failure the eligibility flag exists to prevent: the biggest model would otherwise win."""
    env = OfflineRouterEnv([_unreliable_sample()])
    table = env.reward_table("Give a definitive medical diagnosis from this short message.")
    finite = {route: value for route, value in table.items() if value != float("-inf")}
    assert finite == {Route.REVIEW: 0.0}


def test_step_synthesises_the_review_counterfactual_instead_of_crashing():
    env = OfflineRouterEnv([_unreliable_sample()])
    query = "Give a definitive medical diagnosis from this short message."
    reward, info = env.step(query, Route.REVIEW)
    assert reward == pytest.approx(0.0)
    assert info["synthesized"] is True
    # Recorded-but-ineligible actions are answerable: they score -inf, they do not crash.
    assert env.step(query, Route.LOCAL)[0] == float("-inf")
    assert env.step(query, Route.CLOUD)[0] == float("-inf")


def test_candidate_eligibility_roundtrips_through_json():
    import json

    sample = _unreliable_sample()
    payload = json.loads(json.dumps(sample.to_dict()))
    assert all(candidate["eligible"] is False for candidate in payload["candidates"])
    restored = JevSample.from_dict(payload)
    assert all(not candidate.eligible for candidate in restored.candidates)


# --- route-specific error costs ----------------------------------------------


def test_no_error_costs_means_the_plain_quality_cost_utility():
    """The default must not pretend to know how the routes differ in risk."""
    assert RewardWeights().error_costs == {}
    weights = RewardWeights(cost=0.05)
    assert compute_reward(Route.CLOUD, quality=0.6, cost=1.0, weights=weights) == pytest.approx(0.55)
    assert compute_reward(Route.LOCAL, quality=0.6, cost=1.0, weights=weights) == pytest.approx(0.55)


def test_an_error_cost_scales_with_the_shortfall_not_with_every_prediction():
    weights = RewardWeights(cost=0.0, error_costs={Route.CLOUD: 4.0})
    # Full quality -> no penalty, however large the cost.
    assert compute_reward(Route.CLOUD, quality=1.0, weights=weights) == pytest.approx(1.0)
    # Half quality -> half the penalty.
    assert compute_reward(Route.CLOUD, quality=0.5, weights=weights) == pytest.approx(0.5 - 2.0)
    assert compute_reward(Route.CLOUD, quality=0.0, weights=weights) == pytest.approx(-4.0)


def test_routes_without_an_entry_are_untouched():
    weights = RewardWeights(error_costs={Route.CLOUD: 5.0})
    assert compute_reward(Route.LOCAL, quality=0.6, cost=0.1, weights=weights) == pytest.approx(
        compute_reward(Route.LOCAL, quality=0.6, cost=0.1, weights=RewardWeights())
    )


def test_error_costs_accept_plain_strings():
    """Configs and JSON round trips hand back strings, not enum members."""
    weights = RewardWeights(error_costs={"rag": 2.0})
    assert compute_reward(Route.RAG, quality=0.5, weights=weights) == pytest.approx(0.5 - 1.0)
    assert compute_reward(Route.RAG.value, quality=0.5, weights=weights) == pytest.approx(0.5 - 1.0)


def test_an_error_cost_can_flip_which_route_a_reward_table_prefers():
    """The knob exists so a known-risky route can be made unattractive."""
    table = {
        Route.LOCAL: compute_reward(Route.LOCAL, quality=0.70, cost=0.0, weights=RewardWeights(cost=0.0)),
        Route.CLOUD: compute_reward(Route.CLOUD, quality=0.95, cost=0.0, weights=RewardWeights(cost=0.0)),
    }
    assert max(table, key=table.__getitem__) is Route.CLOUD

    risky = RewardWeights(cost=0.0, error_costs={Route.CLOUD: 20.0})
    weighted = {
        Route.LOCAL: compute_reward(Route.LOCAL, quality=0.70, cost=0.0, weights=risky),
        Route.CLOUD: compute_reward(Route.CLOUD, quality=0.95, cost=0.0, weights=risky),
    }
    assert max(weighted, key=weighted.__getitem__) is Route.LOCAL


def test_ineligibility_still_outranks_any_error_cost():
    weights = RewardWeights(error_costs={Route.LOCAL: 100.0})
    assert compute_reward(Route.LOCAL, quality=0.0, weights=weights, eligible=False) == float("-inf")
