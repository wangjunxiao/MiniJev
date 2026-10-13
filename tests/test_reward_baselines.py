"""Reward baselines must be scored over a common query set, or they lie.

The bug these tests exist for: a constant policy was credited only for the queries its route
could serve, so `always_rag` skipped the medical question its route cannot answer and its
total came out *above* the oracle's. A baseline that abstains from the hard rows is not a
baseline. Every policy now answers every query, paying the terminal review fallback when its
preferred route is ineligible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minijev.data import load_jsonl
from minijev.rewards import (
    PREFERENCE_SWEEP,
    Preference,
    RewardWeights,
    build_env,
    compute_reward,
    regret,
    reward_baselines,
)
from minijev.schema import Candidate, JevSample, Route

PROJECT = Path(__file__).resolve().parent.parent
FIXTURE = PROJECT / "data" / "external" / "routerbench" / "fixture_derived.jsonl"


@pytest.fixture(scope="module")
def fixture_rows():
    return load_jsonl(FIXTURE)


@pytest.fixture(scope="module")
def queries(fixture_rows):
    return [row.query for row in fixture_rows]


def _sample(query: str, *candidates: Candidate) -> JevSample:
    return JevSample(query=query, confidence=1.0, route=candidates[0].route, candidates=tuple(candidates))


# --- the invariant -----------------------------------------------------------


def test_the_oracle_is_never_below_the_best_constant_route(fixture_rows, queries):
    """Per-query maxima sum to at least any single route's sum. Violating this is a bug."""
    for preference in PREFERENCE_SWEEP:
        weights = RewardWeights().with_preference(preference)
        baselines = reward_baselines(build_env(fixture_rows, weights=weights), queries, weights=weights)
        assert baselines["oracle"] >= baselines["best_constant_total"] - 1e-9


def test_a_constant_route_pays_the_fallback_for_a_query_it_cannot_serve():
    """A route that cannot serve a query pays that query's terminal review, not nothing."""
    rows = [
        _sample(
            "开局问题",
            Candidate(Route.CLOUD, None, quality=0.9, cost=0.1),
            Candidate(Route.LOCAL, None, quality=0.9, cost=0.1, eligible=False),
        ),
        _sample(
            "高风险问题",
            Candidate(Route.REVIEW, None, quality=0.8, cost=0.2),
            Candidate(Route.CLOUD, None, quality=0.9, cost=0.1, eligible=False),
        ),
    ]
    env = build_env(rows)
    baselines = reward_baselines(env, [row.query for row in rows])
    assert baselines["queries"] == 2
    # cloud serves query 1 and pays query 2's review fallback.
    assert baselines["always_cloud"] == pytest.approx(
        compute_reward(Route.CLOUD, quality=0.9, cost=0.1) + compute_reward(Route.REVIEW, quality=0.8, cost=0.2)
    )
    # Abstaining would have scored cloud over one query only; that total must not be what
    # the floor reports.
    abstaining = compute_reward(Route.CLOUD, quality=0.9, cost=0.1)
    assert baselines["always_cloud"] != pytest.approx(abstaining)
    assert baselines["oracle"] >= baselines["always_cloud"] - 1e-9


def test_the_abstention_bug_is_visible_on_the_real_fixture(queries, fixture_rows):
    """The case that caught it: at cost share 0.5 the medical row is negative for everyone.

    A constant route scored only over the rows it can serve skipped that row and finished
    above the oracle. With the fallback it cannot.
    """
    weights = RewardWeights().with_preference(Preference.at_cost_share(0.5))
    baselines = reward_baselines(build_env(fixture_rows, weights=weights), queries, weights=weights)
    assert baselines["oracle"] >= baselines["best_constant_total"] - 1e-9
    for route in Route:
        assert baselines["oracle"] >= baselines[f"always_{route.value}"] - 1e-9


def test_a_query_nothing_is_eligible_for_still_gets_the_terminal_review():
    """The environment synthesises review, so no query is unserved and the guard stays quiet.

    Every route then ties at the fallback: the three ineligible ones pay it, and review *is*
    it. A tie is the honest result -- there is nothing to choose between them.
    """
    rows = [_sample("无解", Candidate(Route.CLOUD, None, quality=0.9, cost=0.1, eligible=False))]
    baselines = reward_baselines(build_env(rows), ["无解"])
    assert baselines["queries"] == 1
    assert baselines["fallback"] == Route.REVIEW.value
    assert baselines["oracle"] == pytest.approx(0.0)
    for route in Route:
        assert baselines[f"always_{route.value}"] == pytest.approx(0.0)


# --- headroom and regret -----------------------------------------------------


def test_the_best_constant_route_captures_no_headroom(queries, fixture_rows):
    baselines = reward_baselines(build_env(fixture_rows), queries)
    assert regret(baselines["best_constant_total"], baselines)["headroom_captured"] == pytest.approx(0.0)


def test_the_oracle_captures_all_of_the_headroom(fixture_rows, queries):
    baselines = reward_baselines(build_env(fixture_rows), queries)
    assert regret(baselines["oracle"], baselines)["headroom_captured"] == pytest.approx(1.0)


def test_a_policy_between_the_floor_and_the_ceiling_captures_a_fraction(fixture_rows, queries):
    baselines = reward_baselines(build_env(fixture_rows), queries)
    middle = (baselines["oracle"] + baselines["best_constant_total"]) / 2
    captured = regret(middle, baselines)["headroom_captured"]
    assert 0.0 < captured < 1.0


def test_a_policy_worse_than_always_taking_one_route_captures_negative_headroom(fixture_rows, queries):
    """A negative value is the signal not to ship, however good the raw total looks."""
    baselines = reward_baselines(build_env(fixture_rows), queries)
    worst = min(baselines[f"always_{route.value}"] for route in Route)
    assert regret(worst, baselines)["headroom_captured"] < 0.0


def test_headroom_shrinks_as_the_operator_weights_cost_more(queries, fixture_rows):
    """As cost dominates, every route converges, so the achievable gap must not grow."""
    headrooms = []
    for preference in PREFERENCE_SWEEP:
        weights = RewardWeights().with_preference(preference)
        baselines = reward_baselines(build_env(fixture_rows, weights=weights), queries, weights=weights)
        headrooms.append(baselines["oracle"] - baselines["best_constant_total"])
    assert all(later <= earlier + 1e-9 for earlier, later in zip(headrooms, headrooms[1:])), headrooms


# --- Preference --------------------------------------------------------------


def test_a_preference_sits_on_the_simplex():
    preference = Preference.at_cost_share(0.25)
    assert preference.normalized == pytest.approx((0.75, 0.25))
    assert preference.cost_share == pytest.approx(0.25)


def test_a_preference_with_no_weight_anywhere_is_refused():
    with pytest.raises(ValueError):
        Preference(quality=0.0, cost=0.0)
    with pytest.raises(ValueError):
        Preference(quality=-1.0, cost=1.0)


def test_an_out_of_range_cost_share_is_refused():
    with pytest.raises(ValueError):
        Preference.at_cost_share(1.5)
    with pytest.raises(ValueError):
        Preference.at_cost_share(-0.1)


def test_reweighting_for_a_preference_touches_nothing_else():
    base = RewardWeights(cost_scale=4.0, cost_cap=1.0, error_costs={Route.CLOUD: 3.0})
    reweighted = base.with_preference(Preference.at_cost_share(0.75))
    assert reweighted.quality == pytest.approx(0.25)
    assert reweighted.cost == pytest.approx(0.75)
    assert reweighted.cost_scale == base.cost_scale
    assert reweighted.cost_cap == base.cost_cap
    assert reweighted.error_costs == base.error_costs


# --- cost cap ----------------------------------------------------------------


def test_the_cost_cap_clamps_the_normalised_cost():
    uncapped = RewardWeights(cost=1.0, cost_scale=1.0)
    capped = RewardWeights(cost=1.0, cost_scale=1.0, cost_cap=1.0)
    assert compute_reward(Route.CLOUD, quality=0.0, cost=5.0, weights=uncapped) == pytest.approx(-5.0)
    assert compute_reward(Route.CLOUD, quality=0.0, cost=5.0, weights=capped) == pytest.approx(-1.0)
    # Below the cap nothing changes.
    assert compute_reward(Route.CLOUD, quality=0.0, cost=0.4, weights=capped) == pytest.approx(-0.4)


def test_the_default_is_still_uncapped_so_existing_numbers_do_not_move():
    """`utility-v2` totals must stay reproducible until a run opts into the cap."""
    assert RewardWeights().cost_cap is None
    assert compute_reward(Route.CLOUD, quality=0.6, cost=1.0) == pytest.approx(0.6 - 0.05)


def test_an_invalid_cost_cap_is_refused():
    with pytest.raises(ValueError):
        RewardWeights(cost_cap=0.0)
    with pytest.raises(ValueError):
        RewardWeights(cost_cap=-1.0)


def test_the_cap_survives_a_round_trip():
    original = RewardWeights(cost=0.2, cost_scale=3.0, cost_cap=1.5, error_costs={Route.CLOUD: 2.0})
    restored = RewardWeights.from_dict(original.to_dict())
    assert restored == original


def test_a_preference_and_a_cap_compose():
    weights = RewardWeights(cost_cap=1.0).with_preference(Preference.at_cost_share(0.5))
    # quality 0.5 * 0.9 - cost 0.5 * min(4.0, 1.0)
    assert compute_reward(Route.CLOUD, quality=0.9, cost=4.0, weights=weights) == pytest.approx(0.5 * 0.9 - 0.5)
