"""The split is the measurement instrument; if it leaks, every number above it is worthless."""

from __future__ import annotations

import pytest

from minijev.data import assert_no_group_leak, describe, group_key, group_leaks, normalize_query, split_dataset
from minijev.schema import JevSample, Route
from minijev.synthetic import SyntheticOptions, generate_dataset


def _rows(route: Route, group: str, count: int) -> list[JevSample]:
    return [
        JevSample(query=f"{group} 的第 {index} 个说法", confidence=1.0, route=route, metadata={"semantic_group_id": group}) for index in range(count)
    ]


def _leaky_dataset() -> list[JevSample]:
    rows: list[JevSample] = []
    for route in Route:
        for group_index in range(6):
            rows.extend(_rows(route, f"{route.value}:g{group_index}", 4))
    return rows


def test_normalize_query_collapses_slot_values():
    assert normalize_query("计算 12 加 34") == normalize_query("计算 45 加 67")
    assert normalize_query("你好（示例 4）") == normalize_query("你好")
    assert normalize_query("查一下天气") != normalize_query("设计一个系统")


def test_group_key_prefers_an_explicit_group_id():
    sample = JevSample(query="随便什么", confidence=1.0, route=Route.LOCAL, metadata={"semantic_group_id": "tpl-9"})
    assert group_key(sample) == "tpl-9"


def test_group_aware_split_does_not_leak():
    train, validation, test = split_dataset(_leaky_dataset(), seed=3)
    assert not group_leaks((train, validation, test))
    assert_no_group_leak((train, validation, test))
    assert len(train) + len(validation) + len(test) == 96


def test_row_level_split_does_leak_and_the_guard_catches_it():
    """The guard is only worth having if it can fail."""
    train, validation, test = split_dataset(_leaky_dataset(), seed=3, by_group=False)
    assert group_leaks((train, validation, test)), "row-level splitting is expected to leak; if it does not, this test lost its meaning"


def test_assert_raises_on_a_hand_built_leak():
    shared = JevSample(query="同一个模板 1", confidence=1.0, route=Route.LOCAL, metadata={"semantic_group_id": "shared"})
    other = JevSample(query="别的 2", confidence=1.0, route=Route.RAG, metadata={"semantic_group_id": "other"})
    with pytest.raises(AssertionError):
        assert_no_group_leak(([shared], [other, shared]))


def test_synthetic_dataset_gives_every_row_a_group():
    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=20))
    assert samples
    assert all(sample.metadata.get("semantic_group_id") for sample in samples)
    groups = {group_key(sample) for sample in samples}
    assert len(groups) > 1


def test_hard_boundary_cases_are_included_grouped_and_contested():
    """Boundary rows come from the frame bank: three phrasings, one group, majority agreement."""
    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=5))
    hard = [sample for sample in samples if str(sample.metadata["semantic_group_id"]).startswith("frame:")]
    assert hard, "frames are on by default"
    assert {sample.route for sample in hard} == set(Route)
    assert {sample.confidence for sample in hard} == {0.67}, "boundary cases are contested by construction"
    groups: dict[str, list[JevSample]] = {}
    for sample in hard:
        groups.setdefault(group_key(sample), []).append(sample)
    assert all(len(members) == 3 for members in groups.values()), "each frame carries three phrasings"


def test_hard_cases_can_be_switched_off():
    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=5, include_hard_cases=False))
    assert not any(str(sample.metadata["semantic_group_id"]).startswith("frame:") for sample in samples)


def test_the_synthetic_dataset_never_contains_a_benchmark_frame():
    """Training data and the held-out boundary benchmark must not overlap."""
    from minijev.frames import BENCHMARK_FRAME_IDS

    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=5))
    frame_ids = {str(sample.metadata.get("frame_id")) for sample in samples if sample.metadata.get("frame_id")}
    assert frame_ids.isdisjoint(BENCHMARK_FRAME_IDS)


def test_synthetic_confidence_now_has_a_real_spread():
    """Hard cases carry majority agreement, so the label is no longer a single value."""
    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=20))
    values = {sample.confidence for sample in samples}
    assert values == {1.0, 0.67}


def test_synthetic_domain_is_not_noise():
    """Domain must be a property of the template, not a coin flip."""
    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=20))
    by_query: dict[str, set[str]] = {}
    for sample in samples:
        by_query.setdefault(normalize_query(sample.query), set()).add(str(sample.metadata["domain"]))
    assert all(len(domains) == 1 for domains in by_query.values()), "the same template got more than one domain"

    arithmetic = [sample for sample in samples if sample.query.startswith("计算 ")]
    assert arithmetic, "expected arithmetic templates in the generated set"
    assert {sample.metadata["domain"] for sample in arithmetic} == {"calculation"}


def test_describe_reports_the_fields_that_expose_dead_labels():
    samples = generate_dataset(SyntheticOptions(seed=1, samples_per_route=20))
    stats = describe(samples)
    assert stats["samples"] == len(samples)
    assert stats["semantic_groups"] > 0
    assert stats["with_counterfactuals"] == 0
    assert set(stats["route_share"]) == {route.value for route in Route}
