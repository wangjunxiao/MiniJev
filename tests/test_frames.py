"""The frame bank is the routing policy, so it must be impossible to drift.

Three invariants matter here, and each of them was a real failure mode of the
previous hard-case design:

1. **The route is derived, not asserted.** If a frame and its route could disagree,
   the bank would be a list of opinions. `route_from_frame` is the single rule.
2. **The benchmark is never trained on.** Boundary cases only measure something if
   the model has not seen them, and the old bank put every hard case into the
   training pool *and* the test pool.
3. **A frame pair differs in the slot that decides it.** If the two phrasings of a
   contrast pair share the deciding slot, the pair is not testing the boundary.
"""

from __future__ import annotations

import pytest

from minijev.frames import (
    BENCHMARK_FRAME_IDS,
    FRAME_BANK,
    FRAME_IDS,
    TRAIN_FRAME_IDS,
    Derivation,
    Risk,
    RouteFrame,
    Source,
    assert_no_text_overlap,
    distinct_text_count,
    explain_frame,
    frame_audit,
    generate_benchmark,
    generate_hard_cases,
    route_from_frame,
)
from minijev.schema import ROUTE_ORDER, Route


def _frame(frame_id: str) -> RouteFrame:
    return next(case[1] for case in FRAME_BANK if case[0] == frame_id)


# --- the derivation rule -----------------------------------------------------


def test_risk_dominates_everything_else():
    frame = RouteFrame("服用", "这个药", Source.NONE, Derivation.EXTRACT, steps=9, risk=Risk.HIGH)
    assert route_from_frame(frame) is Route.REVIEW
    assert "risk=high" in explain_frame(frame)


def test_underspecified_goes_to_review():
    frame = RouteFrame("处理", "（未指明对象）", Source.NONE, Derivation.EXTRACT, underspecified=True)
    assert route_from_frame(frame) is Route.REVIEW


def test_construction_goes_to_cloud_even_without_an_external_source():
    frame = RouteFrame("写", "一首诗", Source.NONE, Derivation.CONSTRUCT)
    assert route_from_frame(frame) is Route.CLOUD


def test_multi_step_goes_to_cloud_even_when_extractive():
    frame = RouteFrame("列", "两个文件的差异", Source.GIVEN_TEXT, Derivation.EXTRACT, steps=2)
    assert route_from_frame(frame) is Route.CLOUD


def test_external_source_goes_to_rag():
    frame = RouteFrame("查", "报销记录", Source.USER_DATA, Derivation.EXTRACT)
    assert route_from_frame(frame) is Route.RAG


def test_everything_else_is_local():
    frame = RouteFrame("算", "2 加 2", Source.NONE, Derivation.EXTRACT)
    assert route_from_frame(frame) is Route.LOCAL


def test_an_asserted_route_cannot_be_stored():
    """RouteFrame has no route field to disagree with the rule."""
    assert not hasattr(RouteFrame("算", "2 加 2", Source.NONE, Derivation.EXTRACT), "route")


# --- the contrast pairs ------------------------------------------------------


def test_the_spec_pair_differs_in_source_and_derivation_not_in_predicate():
    """分析产品手册... is rag; 分析这套系统为什么存在并发问题 is cloud. Same 谓宾."""
    manual, system = _frame("f01"), _frame("f02")
    assert manual.predicate == system.predicate == "分析"
    assert route_from_frame(manual) is Route.RAG
    assert route_from_frame(system) is Route.CLOUD
    assert manual.source is not system.source
    assert manual.derivation is not system.derivation


def test_a_pair_sharing_every_deciding_slot_does_not_span_two_routes():
    """If two frames agree on all deciding slots they must agree on the route."""
    for left_id in FRAME_IDS:
        for right_id in FRAME_IDS:
            if left_id >= right_id:
                continue
            left, right = _frame(left_id), _frame(right_id)
            same_deciding = (
                left.risk is right.risk
                and left.underspecified == right.underspecified
                and left.derivation is right.derivation
                and (left.steps >= 2) == (right.steps >= 2)
                and (left.source in {Source.USER_DATA, Source.DOCUMENT, Source.SERVICE})
                == (right.source in {Source.USER_DATA, Source.DOCUMENT, Source.SERVICE})
            )
            if same_deciding:
                assert route_from_frame(left) is route_from_frame(right), f"{left_id} vs {right_id}"


def test_the_bank_covers_every_route():
    routes = {route_from_frame(case[1]) for case in FRAME_BANK}
    assert routes == set(ROUTE_ORDER)


# --- the benchmark split -----------------------------------------------------


def test_benchmark_and_training_frames_are_disjoint():
    assert not (BENCHMARK_FRAME_IDS & set(TRAIN_FRAME_IDS))
    assert set(FRAME_IDS) == BENCHMARK_FRAME_IDS | set(TRAIN_FRAME_IDS)


def test_the_generator_never_emits_a_benchmark_frame():
    rows = generate_hard_cases(variants=3)
    assert {row.metadata["frame_id"] for row in rows}.isdisjoint(BENCHMARK_FRAME_IDS)


def test_the_benchmark_never_emits_a_training_frame():
    rows = generate_benchmark(variants=3)
    assert {row.metadata["frame_id"] for row in rows} <= BENCHMARK_FRAME_IDS


def test_no_query_text_is_shared_between_the_two_files():
    train_texts = {row.query for row in generate_hard_cases(variants=3)}
    bench_texts = {row.query for row in generate_benchmark(variants=3)}
    assert train_texts.isdisjoint(bench_texts)


def test_the_benchmark_shares_no_wording_with_the_training_corpus():
    """A surface collision is as fatal as a group leak: the row would measure recall.

    Two underspecified frames originally reused ambiguous phrasings verbatim from the
    template bank, so those benchmark rows had already been trained on. Training frames
    *are* part of the training corpus, so only the benchmark side is asserted here.
    """
    from minijev.synthetic import SyntheticOptions, generate_dataset

    template_rows = generate_dataset(SyntheticOptions(seed=1, samples_per_route=250))
    assert_no_text_overlap(generate_benchmark(variants=3), template_rows)


def test_the_extra_variant_is_a_tail_variant_not_an_independent_phrasing():
    """Counting padded rows as independent samples would overstate the benchmark."""
    from minijev.frames import _texts as build_texts

    texts = build_texts("f01", 3, None)
    assert len(texts) == 3
    assert len(set(texts)) == 3
    assert texts[2].startswith(texts[0])
    assert texts[2] != texts[0]

    counted = distinct_text_count(variants=3)
    assert counted["rows"] > counted["hand_written_texts"] >= counted["groups"]


def test_distinct_text_count_reports_groups_and_hand_written_phrasings():
    counted = distinct_text_count(variants=3)
    assert counted["groups"] == len(BENCHMARK_FRAME_IDS)
    assert counted["hand_written_texts"] == 2 * len(BENCHMARK_FRAME_IDS)
    assert counted["distinct_texts"] == counted["rows"], "every emitted row must be unique text"


def test_the_benchmark_is_large_enough_to_be_measurable():
    """The old benchmark was 12 rows, where one row is 8 points of noise."""
    rows = generate_benchmark(variants=3)
    assert len(rows) >= 60, "a benchmark this small cannot distinguish encoders"


def test_every_route_is_represented_in_the_benchmark_with_several_rows():
    rows = generate_benchmark(variants=3)
    per_route = {route: sum(1 for row in rows if row.route is route) for route in Route}
    assert all(count >= 3 for count in per_route.values()), per_route


def test_the_benchmark_is_route_balanced_at_the_frame_level():
    """Counted in frames, not rows: rows are a multiple of the variant count.

    Exact balance is achievable and is asserted, because a benchmark that is 30% one
    route makes per-route recall look better than it is.
    """
    from collections import Counter

    from minijev.frames import _frame as bank_frame
    from minijev.frames import route_from_frame

    per_route: Counter[str] = Counter()
    for frame_id in BENCHMARK_FRAME_IDS:
        per_route[route_from_frame(bank_frame(frame_id)).value] += 1
    assert max(per_route.values()) - min(per_route.values()) <= 1, dict(per_route)


def test_the_benchmark_row_counts_stay_within_one_frame_of_each_other():
    rows = generate_benchmark(variants=3)
    per_route = {route: sum(1 for row in rows if row.route is route) for route in Route}
    largest, smallest = max(per_route.values()), min(per_route.values())
    assert largest - smallest <= 3, per_route


# --- grouping and labels -----------------------------------------------------


def test_each_frame_is_one_semantic_group_with_distinct_variants():
    rows = generate_benchmark(variants=3)
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(str(row.metadata["semantic_group_id"]), set()).add(row.query)
    assert len(groups) == len(BENCHMARK_FRAME_IDS)
    assert all(len(texts) == 3 for texts in groups.values())


def test_boundary_rows_carry_majority_agreement_not_certainty():
    rows = generate_hard_cases(variants=3)
    assert {row.confidence for row in rows} == {0.67}


def test_every_row_points_at_the_slot_that_decided_it():
    for row in generate_hard_cases(variants=3):
        assert row.metadata["route_rule"], "the deciding slot must be recorded for error analysis"
        assert row.metadata["boundary"] is True
        assert "derived" in str(row.metadata["label_note"])


def test_the_frame_round_trips_through_json():
    import json

    from minijev.schema import JevSample

    row = generate_benchmark(variants=1)[0]
    payload = json.loads(json.dumps(row.to_dict(), ensure_ascii=False))
    restored = JevSample.from_dict(payload)
    frame = RouteFrame.from_dict(restored.metadata["frame"])
    assert route_from_frame(frame) is restored.route


# --- construction guards -----------------------------------------------------


def test_a_frame_cannot_have_a_negative_step_count():
    with pytest.raises(ValueError):
        RouteFrame("算", "2 加 2", Source.NONE, Derivation.EXTRACT, steps=-1)


def test_a_frame_cannot_have_an_empty_predicate_or_object():
    with pytest.raises(ValueError):
        RouteFrame("", "2 加 2", Source.NONE, Derivation.EXTRACT)
    with pytest.raises(ValueError):
        RouteFrame("算", "", Source.NONE, Derivation.EXTRACT)


def test_the_audit_reports_both_splits_and_the_slot_distribution():
    audit = frame_audit()
    assert audit["frames"] == len(FRAME_BANK)
    assert audit["benchmark_frames"] == len(BENCHMARK_FRAME_IDS)
    assert audit["train_frames"] == len(TRAIN_FRAME_IDS)
    assert sum(audit["route_counts_all"].values()) == len(FRAME_BANK)
    assert set(audit["sources"]) == {source.value for source in Source}
    assert set(audit["derivations"]) == {derivation.value for derivation in Derivation}


def test_source_and_derivation_are_both_load_bearing():
    """Neither slot alone explains the bank: they split the routes differently."""
    audit = frame_audit()
    assert len(audit["sources"]) >= 4, "a bank with one source cannot test the source slot"
    assert set(audit["derivations"]) == {"extract", "construct"}
    assert audit["underspecified"] >= 3, "review needs more than one kind of trigger"
