"""The declared world must be learnable, labelled as declared, and preference-consistent.

Two failure modes are guarded here, and both actually happened:

1. **An unlearnable world.** The first `difficulty_of` hashed the query text. Deterministic,
   reproducible, and impossible: a hash carries no information about what the query says, so
   the reward-optimal route could not be predicted from the text and the best a model could do
   was the majority class. It presented as underfitting.
2. **Contradictory supervision.** A fixed route label combined with a sampled operating point
   asks the same query for different answers with no way to tell which is wanted, and the
   cheapest solution is for the model to ignore the operating point. The target must be a
   function of the preference.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from minijev.rewards import Preference, RewardWeights, build_env
from minijev.routerlog import (
    DEFAULT_WORLD,
    WORLD_NOTICE,
    RouteWorld,
    counterfactuals,
    difficulty_of,
    generate_router_log,
    world_audit,
)
from minijev.schema import ROUTE_ORDER, Route
from minijev.training import _PREFERENCE_GRID, preference_targets, sample_preferences

QUERIES = [
    "查安装步骤",
    "查安装步骤，并给出下一步",
    "整理这份会议记录，然后核对关键依据，最后给出下一步",
]


# --- the world -----------------------------------------------------------------


def test_difficulty_depends_on_what_the_router_can_see():
    """Similar text must get similar difficulty; a hash would not do that."""
    near_a = difficulty_of("查安装步骤，并给出下一步")
    near_b = difficulty_of("查安装步骤，并给出下一步。")
    far = difficulty_of("查安装步骤")
    assert abs(near_a - near_b) < 0.05, "nearly identical queries must not get unrelated difficulty"
    assert far < near_a, "a longer, multi-clause request must be harder than a bare one"


def test_difficulty_is_bounded_and_deterministic():
    for query in QUERIES * 3:
        value = difficulty_of(query)
        assert 0.0 <= value <= 1.0
        assert value == difficulty_of(query)


def test_longer_requests_are_harder():
    short = difficulty_of("查")
    long = difficulty_of("查" * 200)
    assert long > short


# --- the counterfactual table ---------------------------------------------------


def test_every_route_gets_a_candidate():
    candidates = counterfactuals("查安装步骤")
    assert {candidate.route for candidate in candidates} == set(ROUTE_ORDER)
    assert all(candidate.eligible for candidate in candidates)
    assert all(candidate.quality is not None and 0.0 <= candidate.quality <= 1.0 for candidate in candidates)


def test_cost_grows_with_quality_so_there_is_a_trade_off():
    """A world where the best route is also the cheapest cannot test a preference."""
    candidates = {candidate.route: candidate for candidate in counterfactuals("查安装步骤，并给出下一步")}
    assert candidates[Route.LOCAL].cost < candidates[Route.CLOUD].cost < candidates[Route.REVIEW].cost
    assert candidates[Route.REVIEW].quality > candidates[Route.LOCAL].quality


def test_the_world_rejects_an_incomplete_cost_table():
    with pytest.raises(ValueError):
        RouteWorld(costs=((Route.LOCAL, 0.1),))
    with pytest.raises(ValueError):
        RouteWorld(costs=((Route.LOCAL, -1.0), (Route.RAG, 0.1), (Route.CLOUD, 0.1), (Route.REVIEW, 0.1)))


def test_quality_never_leaves_the_unit_interval():
    for difficulty in (0.0, 0.25, 0.5, 0.75, 1.0):
        for route in ROUTE_ORDER:
            assert 0.0 <= DEFAULT_WORLD.quality(route, difficulty) <= 1.0


# --- the generated log ----------------------------------------------------------


def test_every_row_says_it_is_declared_not_observed():
    """A generated row must never look like a measurement."""
    for row in generate_router_log(QUERIES):
        assert row.metadata["world_notice"] == WORLD_NOTICE
        assert row.source == "declared-world"
        assert "declared" in WORLD_NOTICE


def test_the_label_is_the_reference_preference_argmax():
    preference = Preference.at_cost_share(0.5)
    for row in generate_router_log(QUERIES, reference=preference):
        weights = RewardWeights().with_preference(preference)
        scored = [
            (
                (weights.quality * candidate.quality) - (weights.cost * candidate.cost),
                candidate.route,
            )
            for candidate in row.candidates
        ]
        assert row.route is max(scored, key=lambda item: item[0])[1]


def test_the_audit_reports_where_each_route_wins():
    audit = world_audit(generate_router_log(QUERIES))
    assert audit["rows"] == len(QUERIES)
    assert set(audit["winner_counts"]) == {route.value for route in ROUTE_ORDER}
    assert audit["routes_that_ever_win"] >= 1


def test_a_degenerate_world_is_visible_in_the_audit():
    """One route winning everything means the corpus cannot test a trade-off."""
    pinned = RouteWorld(local_quality_intercept=5.0, local_quality_slope=0.0, review_quality=0.0)
    audit = world_audit(generate_router_log(QUERIES, world=pinned))
    assert audit["routes_that_ever_win"] == 1


# --- preference supervision -----------------------------------------------------


def test_the_target_follows_the_operating_point():
    """This is what makes conditioning learnable rather than contradictory."""
    rows = generate_router_log(QUERIES)
    quality_only = preference_targets(rows, Preference.at_cost_share(0.0))
    cost_only = preference_targets(rows, Preference.at_cost_share(1.0))
    assert quality_only.numpy().tolist() != cost_only.numpy().tolist()


def test_the_quality_extreme_escalates_and_the_cost_extreme_does_not():
    rows = generate_router_log(QUERIES)
    quality_only = Counter(preference_targets(rows, Preference.at_cost_share(0.0)).numpy().tolist())
    cost_only = Counter(preference_targets(rows, Preference.at_cost_share(1.0)).numpy().tolist())
    review_index = list(ROUTE_ORDER).index(Route.REVIEW)
    local_index = list(ROUTE_ORDER).index(Route.LOCAL)
    assert quality_only.get(review_index, 0) > cost_only.get(review_index, 0)
    assert cost_only.get(local_index, 0) >= quality_only.get(local_index, 0)


def test_rows_without_counterfactuals_keep_their_own_label():
    from minijev.schema import JevSample

    rows = [JevSample(query="没有反事实表", confidence=1.0, route=Route.CLOUD)]
    targets = preference_targets(rows, Preference.at_cost_share(1.0))
    assert targets.numpy().flatten().tolist() == [list(ROUTE_ORDER).index(Route.CLOUD)]


def test_the_sampling_schedule_covers_both_endpoints():
    """Uniform draws alone under-train the ends, where the optimal policy changes fastest."""
    tensor = sample_preferences(400, random.Random(0))
    shares = [row[1] for row in tensor.numpy().tolist()]
    assert any(share == 0.0 for share in shares), "the quality-only end must be sampled"
    assert any(share >= 0.9 for share in shares), "the cost-heavy end must be sampled"
    assert all(0.0 <= share <= 1.0 for share in shares)
    assert 0.0 in _PREFERENCE_GRID and 1.0 in _PREFERENCE_GRID


def test_the_sampling_schedule_is_deterministic():
    first = sample_preferences(20, random.Random(3)).numpy().tolist()
    second = sample_preferences(20, random.Random(3)).numpy().tolist()
    assert first == second


def test_the_environment_replays_the_generated_log(queries=QUERIES):
    rows = generate_router_log(queries)
    env = build_env(rows)
    assert len(env) == len(queries)
    table = env.reward_table(queries[0])
    assert table, "a generated row must be replayable"
