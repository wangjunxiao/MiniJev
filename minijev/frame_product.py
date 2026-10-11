"""Deterministic slot cross-product generation for boundary frames.

The hand-written frame bank is useful for explaining the routing rule, but it is
too small to estimate a collapse rate precisely.  This module expands the same
slots into a controlled product and keeps the route label derived by
``frames.route_from_frame``.  It deliberately lives beside ``frames.py`` rather
than mutating ``FRAME_BANK``: the original bank and its benchmark remain a
stable, human-readable reference set.

The generator selects an equal number of frames for each route from a larger
valid product, then holds out complete semantic groups.  A generated row carries
the complete frame in metadata, so a later audit can recompute the route and
verify that no asserted label slipped into the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Any

from .frames import Derivation, Risk, RouteFrame, Source, deciding_slots_differing, route_from_frame
from .schema import ROUTE_ORDER, JevSample, Route, save_jsonl


@dataclass(frozen=True)
class ProductOptions:
    """Controls for the deterministic product.

    ``target_per_route`` is the number of selected frames before the benchmark
    split.  The default produces 1,280 frames (320 per route), with 25% held out
    as a new benchmark and the remaining 960 available for training.
    """

    target_per_route: int = 320
    benchmark_fraction: float = 0.25
    variants: int = 3
    seed: int = 17

    def __post_init__(self) -> None:
        if self.target_per_route <= 0:
            raise ValueError("target_per_route must be positive")
        if not 0.0 < self.benchmark_fraction < 1.0:
            raise ValueError("benchmark_fraction must be between 0 and 1")
        if self.variants <= 0:
            raise ValueError("variants must be positive")


@dataclass(frozen=True)
class ProductFrameCase:
    """One generated frame and its surface variants."""

    frame_id: str
    frame: RouteFrame
    texts: tuple[str, ...]


@dataclass(frozen=True)
class FrameProduct:
    """Selected product cases and a group-aware train/benchmark split."""

    cases: tuple[ProductFrameCase, ...]
    benchmark_ids: frozenset[str]
    options: ProductOptions

    @property
    def train_ids(self) -> frozenset[str]:
        return frozenset(case.frame_id for case in self.cases if case.frame_id not in self.benchmark_ids)

    @property
    def benchmark_cases(self) -> tuple[ProductFrameCase, ...]:
        return tuple(case for case in self.cases if case.frame_id in self.benchmark_ids)

    @property
    def train_cases(self) -> tuple[ProductFrameCase, ...]:
        return tuple(case for case in self.cases if case.frame_id not in self.benchmark_ids)


def product_contrast_pairs(product_set: FrameProduct, *, split: str = "benchmark") -> tuple[tuple[str, str], ...]:
    """Find a deterministic disjoint set of one-slot contrast pairs.

    The hand-written benchmark stores pairs explicitly.  Product frames are made
    by a Cartesian product, so the same relation can be derived: two frames are a
    contrast when exactly one deciding slot differs and their derived routes are
    different.  Greedy matching keeps pair outcomes independent enough for a
    useful collapse-rate estimate instead of counting every possible cross-pair.
    """

    if split == "benchmark":
        cases = product_set.benchmark_cases
    elif split == "train":
        cases = product_set.train_cases
    elif split == "all":
        cases = product_set.cases
    else:
        raise ValueError("split must be 'all', 'train' or 'benchmark'")
    candidates: list[tuple[str, str, str]] = []
    for left, right in combinations(sorted(cases, key=lambda case: case.frame_id), 2):
        differing = deciding_slots_differing(left.frame, right.frame)
        if len(differing) == 1 and route_from_frame(left.frame) is not route_from_frame(right.frame):
            candidates.append((differing[0], left.frame_id, right.frame_id))
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _, left_id, right_id in sorted(candidates):
        if left_id in used or right_id in used:
            continue
        pairs.append((left_id, right_id))
        used.update((left_id, right_id))
    return tuple(pairs)


# The source phrase is intentionally visible in the query.  It gives the model
# a learnable cue for the deciding slot while retaining enough surface variation
# in predicates and objects to prevent an id-only lookup.
_SOURCE_PHRASES: dict[Source, tuple[str, str]] = {
    Source.GIVEN_TEXT: ("给定文本中的", "输入里的"),
    Source.DEVICE: ("设备里的", "本机中的"),
    Source.USER_DATA: ("我的数据中的", "我的记录里的"),
    Source.DOCUMENT: ("知识库中的", "产品资料里的"),
    Source.SERVICE: ("在线服务返回的", "外部服务里的"),
    Source.NONE: ("", "已知的"),
}

_PREDICATES: tuple[str, ...] = (
    "查",
    "统计",
    "列出",
    "核对",
    "翻译",
    "转成",
    "读取",
    "提取",
    "分析",
    "比较",
    "总结",
    "规划",
    "解释",
    "评估",
    "设计",
    "预测",
)

_OBJECTS: tuple[str, ...] = (
    "安装步骤",
    "报销明细",
    "错误码含义",
    "季度数据",
    "合同条款",
    "客户名单",
    "会议记录",
    "部署日志",
    "预算变化",
    "服务用量",
    "航班状态",
    "天气信息",
    "产品规格",
    "故障原因",
    "实验结果",
    "培训材料",
    "项目进度",
    "库存数量",
    "权限记录",
    "接口响应",
)

_SOURCES: tuple[Source, ...] = tuple(Source)
_DERIVATIONS: tuple[Derivation, ...] = tuple(Derivation)
_STEPS: tuple[int, ...] = (1, 2, 3)
_RISKS: tuple[Risk, ...] = tuple(Risk)


def _source_text(source: Source, object_: str, alternate: bool = False) -> str:
    prefix = _SOURCE_PHRASES[source][1 if alternate else 0]
    return f"{prefix}{object_}" if prefix else object_


def _build_text(
    predicate: str,
    object_: str,
    source: Source,
    derivation: Derivation,
    steps: int,
    risk: Risk,
    underspecified: bool,
    *,
    alternate: bool = False,
) -> str:
    if underspecified:
        source_hint = f"来源线索：{source.value}"
        wording = "尚未说明具体对象" if alternate else "具体对象还没有说明"
        return f"请{predicate}一下，关于{object_}的{wording}（{source_hint}；预计{steps}步；{risk.value}风险；{derivation.value}）"

    subject = _source_text(source, object_, alternate)
    text = f"{predicate}{subject}"
    if derivation is Derivation.CONSTRUCT:
        text += "，并解释原因或给出方案"
    else:
        text += "，直接列出结果"
    if steps >= 2:
        text += "，然后核对关键依据"
    if steps >= 3:
        text += "，最后给出下一步行动"
    if risk is Risk.MEDIUM:
        text += "（请注明依据）"
    elif risk is Risk.HIGH:
        text += "（请谨慎核实）"
    return text


def _variants(frame: RouteFrame, *, variants: int) -> tuple[str, ...]:
    texts = [
        _build_text(frame.predicate, frame.object, frame.source, frame.derivation, frame.steps, frame.risk, frame.underspecified),
        _build_text(frame.predicate, frame.object, frame.source, frame.derivation, frame.steps, frame.risk, frame.underspecified, alternate=True),
    ]
    suffixes = ("请按这些条件回答。", "结果请说清楚。", "需要时说明判断依据。")
    index = 0
    while len(texts) < variants:
        texts.append(texts[0] + suffixes[index % len(suffixes)])
        index += 1
    return tuple(texts[:variants])


def _candidate_cases() -> Iterable[ProductFrameCase]:
    """Yield the legal slot product before route-balanced selection.

    An underspecified request names a topic ("关于安装步骤的") but never states what to do
    with it. `derivation` and `steps` are therefore *undefined* for such a frame, not
    merely unstated: there is no operation to classify as extract-vs-construct and no
    step count to count. Pinning them to their canonical values (EXTRACT, 1) keeps the
    annotation honest, and it also makes the surface text an injective function of the
    frame. Left free, the six combinations of derivation and steps would emit six
    identical queries, and a duplicate that straddles the train/benchmark split is
    leakage -- the row would measure recall instead of routing.
    """

    serial = 0
    for predicate, object_, source, derivation, steps, risk, underspecified in product(
        _PREDICATES, _OBJECTS, _SOURCES, _DERIVATIONS, _STEPS, _RISKS, (False, True)
    ):
        # A request without a specified object is meaningful only as a review
        # case; keeping it out of other route strata avoids synthetic contradictions.
        if underspecified and (risk is not Risk.MEDIUM or derivation is not Derivation.EXTRACT or steps != 1):
            continue
        frame = RouteFrame(
            predicate,
            object_,
            source,
            derivation,
            steps=steps,
            risk=risk,
            underspecified=underspecified,
            domain="slot_product",
        )
        frame_id = f"xp{serial:06d}"
        serial += 1
        variants = _variants(frame, variants=3)
        if len(set(variants)) != len(variants):
            raise ValueError(f"{frame_id}: variants are not distinct, so the text does not determine the frame")
        yield ProductFrameCase(frame_id, frame, variants)


def _stratum(case: ProductFrameCase) -> tuple[str, ...]:
    frame = case.frame
    return (
        frame.source.value,
        frame.derivation.value,
        str(min(frame.steps, 3)),
        frame.risk.value,
        str(frame.underspecified),
    )


def _select_route(candidates: Sequence[ProductFrameCase], quota: int, seed: int) -> list[ProductFrameCase]:
    if len(candidates) < quota:
        raise ValueError(f"only {len(candidates)} candidates available for a route; need {quota}")
    by_stratum: dict[tuple[str, ...], list[ProductFrameCase]] = defaultdict(list)
    for case in candidates:
        by_stratum[_stratum(case)].append(case)
    rng = random.Random(seed)
    for rows in by_stratum.values():
        rows.sort(key=lambda case: case.frame_id)
        rng.shuffle(rows)

    selected: list[ProductFrameCase] = []
    strata = sorted(by_stratum)
    cursor = 0
    while len(selected) < quota:
        key = strata[cursor % len(strata)]
        rows = by_stratum[key]
        if rows:
            selected.append(rows.pop())
        cursor += 1
        if cursor >= len(strata) and not any(by_stratum[key] for key in strata):
            break
    if len(selected) != quota:
        raise ValueError(f"stratified selection stopped at {len(selected)} of {quota}")
    return selected


@lru_cache(maxsize=8)
def _generate_cached(options: ProductOptions) -> FrameProduct:
    """Generate a balanced, deterministic frame product and group split."""

    candidates_by_route: dict[Route, list[ProductFrameCase]] = defaultdict(list)
    for case in _candidate_cases():
        candidates_by_route[route_from_frame(case.frame)].append(case)

    selected: list[ProductFrameCase] = []
    for index, route in enumerate(ROUTE_ORDER):
        selected.extend(_select_route(candidates_by_route[route], options.target_per_route, options.seed + index))
    selected.sort(key=lambda case: case.frame_id)

    # Hashing the semantic group, rather than rows or text variants, guarantees
    # that all phrasings of a frame land on the same side of the split.  Selecting
    # an exact count within each route keeps the held-out benchmark balanced.
    benchmark_ids: set[str] = set()
    for route in ROUTE_ORDER:
        route_cases = [case for case in selected if route_from_frame(case.frame) is route]
        holdout_count = min(len(route_cases) - 1, max(1, round(len(route_cases) * options.benchmark_fraction)))
        ranked = sorted(route_cases, key=lambda case: hashlib.sha256(f"{options.seed}:{case.frame_id}".encode()).hexdigest())
        benchmark_ids.update(case.frame_id for case in ranked[:holdout_count])
    return FrameProduct(tuple(selected), frozenset(benchmark_ids), options)


def generate_frame_product(options: ProductOptions | None = None) -> FrameProduct:
    """Generate a balanced, deterministic frame product and group split."""

    return _generate_cached(options or ProductOptions())


def product_rows(product_set: FrameProduct, *, split: str = "all", variants: int | None = None) -> list[JevSample]:
    """Materialise generated cases as schema rows for ``train`` or ``benchmark``."""

    if split not in {"all", "train", "benchmark"}:
        raise ValueError("split must be 'all', 'train' or 'benchmark'")
    count = product_set.options.variants if variants is None else variants
    if count <= 0:
        raise ValueError("variants must be positive")
    rows: list[JevSample] = []
    for case in product_set.cases:
        held_out = case.frame_id in product_set.benchmark_ids
        if split == "train" and held_out or split == "benchmark" and not held_out:
            continue
        route = route_from_frame(case.frame)
        for index, text in enumerate(_variants(case.frame, variants=count)):
            rows.append(
                JevSample(
                    query=text,
                    confidence=0.67 if count > 1 else 1.0,
                    route=route,
                    id=f"{case.frame_id}_{index}",
                    source="slot-product-v1",
                    metadata={
                        "semantic_group_id": f"frame-product:{case.frame_id}",
                        "frame": case.frame.to_dict(),
                        "frame_id": case.frame_id,
                        "route_rule": _route_rule(case.frame),
                        "boundary": True,
                        "label_note": "route derived from the frame by frames.route_from_frame, not asserted",
                        "product_split": "benchmark" if held_out else "train",
                        "generator": "minijev.frame_product",
                    },
                )
            )
    return rows


def _route_rule(frame: RouteFrame) -> str:
    # Reuse the canonical explanation without exposing another route policy.
    from .frames import explain_frame

    return explain_frame(frame)


def product_audit(product_set: FrameProduct) -> dict[str, Any]:
    """Return counts that make product balance and split integrity inspectable."""

    all_cases = product_set.cases
    benchmark_ids = product_set.benchmark_ids
    route_counts = Counter(route_from_frame(case.frame).value for case in all_cases)
    benchmark_counts = Counter(route_from_frame(case.frame).value for case in product_set.benchmark_cases)
    train_counts = Counter(route_from_frame(case.frame).value for case in product_set.train_cases)
    rows = product_rows(product_set)
    train_rows = product_rows(product_set, split="train")
    benchmark_rows = product_rows(product_set, split="benchmark")
    return {
        "frames": len(all_cases),
        "train_frames": len(product_set.train_cases),
        "benchmark_frames": len(product_set.benchmark_cases),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "benchmark_rows": len(benchmark_rows),
        "route_counts": {route.value: route_counts.get(route.value, 0) for route in ROUTE_ORDER},
        "train_route_counts": {route.value: train_counts.get(route.value, 0) for route in ROUTE_ORDER},
        "benchmark_route_counts": {route.value: benchmark_counts.get(route.value, 0) for route in ROUTE_ORDER},
        "source_counts": dict(Counter(case.frame.source.value for case in all_cases)),
        "derivation_counts": dict(Counter(case.frame.derivation.value for case in all_cases)),
        "risk_counts": dict(Counter(case.frame.risk.value for case in all_cases)),
        "steps_counts": dict(Counter(case.frame.steps for case in all_cases)),
        "underspecified_frames": sum(case.frame.underspecified for case in all_cases),
        "unique_queries": len({row.query for row in rows}),
        "minimal_pairs": len(product_contrast_pairs(product_set)),
        "benchmark_ids": sorted(benchmark_ids),
    }


def write_product(output_dir: str | Path, options: ProductOptions | None = None) -> dict[str, Any]:
    """Write train, benchmark and manifest files and return the manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    product_set = generate_frame_product(options)
    train_rows = product_rows(product_set, split="train")
    benchmark_rows = product_rows(product_set, split="benchmark")
    save_jsonl(train_rows, destination / "frame_product_train.jsonl")
    save_jsonl(benchmark_rows, destination / "frame_product_benchmark.jsonl")
    manifest = {"version": "slot-product-v1", "options": vars(product_set.options), "audit": product_audit(product_set)}
    (destination / "frame_product_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.frame_product", description="Generate a balanced slot cross-product")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-per-route", type=int, default=ProductOptions.target_per_route)
    parser.add_argument("--benchmark-fraction", type=float, default=ProductOptions.benchmark_fraction)
    parser.add_argument("--variants", type=int, default=ProductOptions.variants)
    parser.add_argument("--seed", type=int, default=ProductOptions.seed)
    parser.add_argument("--audit", action="store_true", help="print the manifest without writing files")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    options = ProductOptions(args.target_per_route, args.benchmark_fraction, args.variants, args.seed)
    product_set = generate_frame_product(options)
    manifest = {"version": "slot-product-v1", "options": vars(options), "audit": product_audit(product_set)}
    if args.audit:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    written = write_product(args.output_dir, options)
    print(json.dumps({key: value for key, value in written["audit"].items() if key != "benchmark_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
