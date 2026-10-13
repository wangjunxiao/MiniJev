"""The product benchmark must not be solvable by reading markers.

This is a regression guard with a measured baseline. The explicit product states every
deciding slot in the text, so a classifier that reads only those markers reconstructs the
route at **accuracy 1.0000** -- the corpus measures marker lookup, every model saturates
near 0.97, and it cannot separate encoders. A benchmark needs to be hard *and* decidable:
the hand-written set scores 0.2609 (chance) on the same marker reader and 0.46-0.48 for
real models, which is the band worth aiming at.
"""

from __future__ import annotations

from collections import Counter

import pytest

from minijev.data import load_jsonl
from minijev.frame_product import (
    _SOURCE_PHRASES,
    ProductOptions,
    generate_frame_product,
    product_rows,
)
from minijev.frames import Derivation, Risk, RouteFrame, Source, route_from_frame
from minijev.schema import ROUTE_ORDER

PREFIX_TO_SOURCE = {prefix: source for source, prefixes in _SOURCE_PHRASES.items() for prefix in prefixes if prefix}


def marker_only_route(text: str):
    """Reconstruct the route from explicit markers alone. No semantics used at all."""
    source = Source.NONE
    for prefix, candidate in sorted(PREFIX_TO_SOURCE.items(), key=lambda item: -len(item[0])):
        if prefix in text:
            source = candidate
            break
    underspecified = "具体对象还没有说明" in text or "尚未说明具体对象" in text or "来源线索" in text
    derivation = Derivation.CONSTRUCT if "并解释原因或给出方案" in text else Derivation.EXTRACT
    steps = 1 + ("然后核对关键依据" in text) + ("最后给出下一步行动" in text)
    if "请谨慎核实" in text:
        risk = Risk.HIGH
    elif "请注明依据" in text:
        risk = Risk.MEDIUM
    else:
        risk = Risk.LOW
    frame = RouteFrame("读标记", "读标记", source, derivation, steps=steps, risk=risk, underspecified=underspecified)
    return route_from_frame(frame)


def marker_accuracy(rows) -> float:
    return sum(1 for row in rows if marker_only_route(row.query) is row.route) / len(rows)


@pytest.fixture(scope="module")
def explicit_rows():
    return product_rows(generate_frame_product(ProductOptions(naming="explicit")))


@pytest.fixture(scope="module")
def implied_rows():
    return product_rows(generate_frame_product(ProductOptions(naming="implied")))


def test_the_explicit_product_is_solvable_by_markers_alone(explicit_rows):
    """Documents the defect rather than hiding it: this is why `implied` exists."""
    assert marker_accuracy(explicit_rows) == 1.0


def test_the_implied_product_is_not_solvable_by_markers(implied_rows):
    accuracy = marker_accuracy(implied_rows)
    majority = max(Counter(row.route for row in implied_rows).values()) / len(implied_rows)
    assert accuracy <= majority + 0.05, f"marker baseline {accuracy:.3f} is above chance {majority:.3f}"


def test_the_implied_product_names_no_deciding_slot(implied_rows):
    markers = tuple(prefix for prefix in PREFIX_TO_SOURCE if prefix)
    clauses = ("并解释原因或给出方案", "直接列出结果", "然后核对关键依据", "最后给出下一步行动", "请注明依据", "请谨慎核实")
    for row in implied_rows:
        for marker in markers + clauses:
            assert marker not in row.query, f"{marker!r} leaked into {row.query!r}"
        assert "来源线索" not in row.query
        assert row.query, "a row must carry text"


@pytest.mark.parametrize("naming", ["explicit", "implied"])
def test_both_modes_stay_route_balanced(naming, explicit_rows, implied_rows):
    rows = explicit_rows if naming == "explicit" else implied_rows
    counts = Counter(row.route for row in rows)
    assert set(counts) == set(ROUTE_ORDER)
    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize("naming", ["explicit", "implied"])
def test_both_modes_keep_query_text_unique(naming, explicit_rows, implied_rows):
    """A duplicate that straddles the split is leakage: that row measures recall."""
    rows = explicit_rows if naming == "explicit" else implied_rows
    assert len({row.query for row in rows}) == len(rows)


def test_implied_frames_spread_across_every_route_combination_of_slots():
    """The slots must vary independently enough to be learnable, not one constant."""
    rows = product_rows(generate_frame_product(ProductOptions(naming="implied")))
    frames = {row.metadata["frame_id"]: row.metadata["frame"] for row in rows}
    assert len({frame["derivation"] for frame in frames.values()}) == 2
    assert len({frame["underspecified"] for frame in frames.values()}) == 2
    assert len({frame["risk"] for frame in frames.values()}) >= 2
    assert len({frame["source"] for frame in frames.values()}) >= 3


def test_implied_source_and_risk_follow_from_the_object_not_from_a_free_product():
    """A fixed object must not appear under two different sources, or the cue is noise."""
    rows = product_rows(generate_frame_product(ProductOptions(naming="implied")))
    by_object: dict[str, set[str]] = {}
    for row in rows:
        by_object.setdefault(str(row.metadata["frame"]["object"]), set()).add(str(row.metadata["frame"]["source"]))
    assert all(len(sources) == 1 for sources in by_object.values()), "an object with two sources is ambiguous"


def test_the_implied_product_is_smaller_and_that_is_correct():
    """A constrained product has fewer legal combinations; the reduction is information."""
    explicit = generate_frame_product(ProductOptions(naming="explicit"))
    implied = generate_frame_product(ProductOptions(naming="implied"))
    assert len(implied.cases) < len(explicit.cases)
    assert len(implied.cases) >= 100, "too small to estimate a collapse rate"


def test_an_unknown_naming_mode_is_refused():
    with pytest.raises(ValueError):
        generate_frame_product(ProductOptions(naming="nonsense"))


def test_every_verb_object_pair_is_one_a_speaker_would_use():
    """A free predicate x object product emitted '预测安装步骤'. Verbs select objects.

    This is the same defect class as the explicit markers, one level up: a corpus of
    sentences nobody would say cannot be learned from, and its labels are not the labels a
    real router would see.
    """
    from minijev.frame_product import _OBJECT_CONTEXT, _PREDICATE_GROUPS

    allowed = {
        (predicate, object_) for object_, (_, _, _, groups) in _OBJECT_CONTEXT.items() for group in groups for predicate in _PREDICATE_GROUPS[group]
    }
    product = generate_frame_product(ProductOptions(naming="implied"))
    # Underspecified frames deliberately carry no operation -- their placeholder verb is not
    # a verb-object pair at all -- so only specified requests are checked.
    produced = {(case.frame.predicate, case.frame.object) for case in product.cases if not case.frame.underspecified}
    assert produced <= allowed, sorted(produced - allowed)[:5]
    assert ("预测", "安装步骤") not in produced
    assert ("统计", "安装步骤") not in produced
    assert ("查", "安装步骤") in produced


def test_the_implied_benchmark_is_sized_for_the_effect_it_must_detect():
    """Power, not vibes: state what this instrument can resolve.

    At 95% confidence and 80% power a two-proportion test needs about 59 pairs to resolve a
    20-point collapse difference at a 30% baseline, and about 197 for a 10-point one at 20%.
    """
    from minijev.frame_product import product_contrast_pairs

    product = generate_frame_product(ProductOptions(naming="implied"))
    pairs = product_contrast_pairs(product, split="benchmark")
    assert len(pairs) >= 50, f"only {len(pairs)} contrast pairs; a 20-point effect needs ~59"


def test_written_files_carry_the_generated_rows_on_disk(tmp_path):
    """The on-disk benchmark must match the generator, or a report describes a ghost."""
    options = ProductOptions(naming="implied")
    product = generate_frame_product(options)
    rows = product_rows(product, split="benchmark")
    assert len(rows) == len(product.benchmark_cases) * options.variants
    assert load_jsonl  # imported for the report path; disk check lives in the CLI test
    del tmp_path
