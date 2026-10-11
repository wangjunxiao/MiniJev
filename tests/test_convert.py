"""The MASSIVE rubric must be exhaustive, deliberate and honest about what it cannot cover."""

from __future__ import annotations

import json

import pytest
from conftest import MASSIVE_ZH_CN

from minijev.convert import (
    LOCAL_INTENTS,
    MASSIVE_RUBRIC,
    RAG_INTENTS,
    REVIEW_INTENTS,
    SUPPORTED_ROUTES,
    UNSUPPORTED_ROUTES,
    MassiveConvertor,
    massive_route,
    unmapped_intents,
)
from minijev.schema import ROUTE_ORDER, Route

requires_massive = pytest.mark.skipif(not MASSIVE_ZH_CN.exists(), reason="MASSIVE zh-CN not downloaded")


def test_rubric_groups_are_disjoint():
    assert not (LOCAL_INTENTS & RAG_INTENTS)
    assert not (LOCAL_INTENTS & REVIEW_INTENTS)
    assert not (RAG_INTENTS & REVIEW_INTENTS)
    assert len(MASSIVE_RUBRIC) == len(LOCAL_INTENTS) + len(RAG_INTENTS) + len(REVIEW_INTENTS)


def test_unknown_intent_is_refused_rather_than_guessed():
    with pytest.raises(KeyError):
        massive_route("definitely_not_an_intent")


def test_cloud_is_declared_unsupported():
    """MASSIVE is a smart-speaker command set; it contains no reasoning-heavy intent."""
    assert Route.CLOUD in UNSUPPORTED_ROUTES
    assert Route.CLOUD not in SUPPORTED_ROUTES
    assert Route.CLOUD not in set(MASSIVE_RUBRIC.values())
    assert set(UNSUPPORTED_ROUTES) | set(SUPPORTED_ROUTES) == set(ROUTE_ORDER)


def test_writes_to_an_on_device_store_are_local_not_retrieval():
    """v1 classified these as rag, which was the main semantic defect."""
    for intent in ("alarm_set", "calendar_set", "lists_createoradd", "lists_remove", "music_likeness"):
        assert massive_route(intent) is Route.LOCAL, intent


def test_reads_of_stored_data_are_rag():
    for intent in ("calendar_query", "lists_query", "email_query"):
        assert massive_route(intent) is Route.RAG, intent


def test_recommendations_are_not_complex_reasoning():
    """v1 put these in cloud; a catalogue lookup is retrieval."""
    for intent in ("recommendation_movies", "recommendation_locations", "recommendation_events"):
        assert massive_route(intent) is Route.RAG, intent


@requires_massive
def test_every_intent_in_the_real_dataset_is_mapped():
    intents = set()
    with MASSIVE_ZH_CN.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                intents.add(str(json.loads(line)["intent"]))
    assert unmapped_intents(intents) == []


@requires_massive
def test_audit_reports_no_unmapped_intents_and_a_cloud_free_distribution():
    audit = MassiveConvertor().audit(MASSIVE_ZH_CN)
    assert audit["unmapped_intents"] == []
    assert audit["route_counts"]["cloud"] == 0
    assert audit["route_share"]["cloud"] == 0.0
    assert audit["rubric"] == "v2"


@requires_massive
def test_converted_rows_carry_provenance(tmp_path):
    convertor = MassiveConvertor()
    counts = convertor.convert(MASSIVE_ZH_CN, "test", tmp_path / "test.jsonl")
    assert sum(counts.values()) == 2974
    rows = [json.loads(line) for line in (tmp_path / "test.jsonl").open(encoding="utf-8") if line.strip()]
    assert rows[0]["source"] == "massive-1.1-zh-CN"
    assert rows[0]["intent"]
    assert rows[0]["scenario"]
    # MASSIVE has no confidence annotation, so the confidence head is untrained here.
    assert {row["confidence"] for row in rows} == {1.0}


# --- RouterBench convertor ---------------------------------------------------


def _routerbench_rows():
    from pathlib import Path

    from minijev.convert import RouterBenchConvertor

    fixture = Path(__file__).resolve().parents[1] / "data" / "external" / "routerbench" / "fixture.jsonl"
    return list(RouterBenchConvertor(quality_threshold=0.8).rows(fixture))


def test_routerbench_derives_one_of_each_route_from_the_fixture():
    rows = _routerbench_rows()
    assert {row.route for row in rows} == set(ROUTE_ORDER)


def test_routerbench_marks_below_bar_models_ineligible():
    rows = {row.id: row for row in _routerbench_rows()}
    rag = rows["rb-rag"]
    assert rag.route is Route.RAG, "the medium model clears 0.8, the small one does not"
    small = next(candidate for candidate in rag.candidates if candidate.quality == pytest.approx(0.55))
    assert small.eligible is False
    assert all(candidate.eligible for candidate in rag.candidates if candidate.quality >= 0.8)


def test_routerbench_review_row_synthesises_a_terminal_review_candidate():
    rows = {row.id: row for row in _routerbench_rows()}
    review = rows["rb-review"]
    assert review.route is Route.REVIEW
    assert review.metadata["selected_model"] is None
    assert all(not candidate.eligible for candidate in review.candidates if candidate.route is not Route.REVIEW)
    terminal = [candidate for candidate in review.candidates if candidate.route is Route.REVIEW]
    assert len(terminal) == 1
    assert terminal[0].cost is not None, "the synthesized candidate carries the declared human-review cost prior"


def test_routerbench_reward_table_never_prefers_an_ineligible_model():
    from minijev.convert import ROUTERBENCH_REVIEW_COST
    from minijev.rewards import OfflineRouterEnv, compute_reward

    env = OfflineRouterEnv(_routerbench_rows())
    table = env.reward_table("Give a definitive medical diagnosis from this short message.")
    # The terminal review candidate carries the declared human-review prior, so it
    # wins; every recorded model is below the bar and scores -inf.
    assert table[Route.REVIEW] == pytest.approx(compute_reward(quality=0.8, cost=ROUTERBENCH_REVIEW_COST))
    assert table[Route.CLOUD] == float("-inf")
    assert max(table, key=table.__getitem__) is Route.REVIEW


def test_routerbench_audit_reports_the_derivation_policy():
    from pathlib import Path

    from minijev.convert import RouterBenchConvertor

    fixture = Path(__file__).resolve().parents[1] / "data" / "external" / "routerbench" / "fixture.jsonl"
    audit = RouterBenchConvertor(quality_threshold=0.8).audit(fixture)
    assert audit["quality_threshold"] == 0.8
    assert audit["review_rows_with_synthesized_candidate"] == 1
    assert audit["with_counterfactuals"] == 4
    assert audit["route_counts"] == {"local": 1, "rag": 1, "cloud": 1, "review": 1}
