"""Read, write, split and describe datasets. Understands only the schema.

The split here is the important part. Splitting rows at random lets two phrasings
of the same underlying question land on both sides of the boundary, which turns a
test score into a measure of memorisation. Splitting *semantic groups* is what
makes a test score mean something, so that is the default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from .schema import ROUTE_ORDER, JevSample, Route, load_jsonl, save_jsonl

GROUP_FIELDS = ("semantic_group_id", "group_id", "template_id")
SAMPLE_SUFFIX = re.compile(r"（示例\s*\d+）")


def normalize_query(query: str) -> str:
    """Collapse a query to a form that ignores the parts a template varies.

    Digits become `#` and the synthetic generator's `（示例 N）` suffix is dropped,
    so "计算 12 加 34" and "计算 45 加 67" land in the same group.
    """
    collapsed = SAMPLE_SUFFIX.sub("", query)
    collapsed = re.sub(r"\d+", "#", collapsed)
    return " ".join(collapsed.split()).lower()


def group_key(sample: JevSample) -> str:
    """Resolve the semantic group of a sample.

    An explicit `semantic_group_id` in metadata always wins; otherwise the query is
    normalised. The fallback is weaker than a real group id, so generators should
    emit one.
    """
    for field in GROUP_FIELDS:
        value = sample.metadata.get(field)
        if value:
            return str(value)
    return normalize_query(sample.query)


def group_routes(samples: Sequence[JevSample]) -> dict[str, Route]:
    """Assign each group the majority route of its members."""
    tally: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        tally[group_key(sample)][sample.route] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in tally.items()}


def split_dataset(
    samples: Sequence[JevSample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 0,
    *,
    by_group: bool = True,
) -> tuple[list[JevSample], list[JevSample], list[JevSample]]:
    """Stratify by route, and by semantic group unless `by_group` is disabled.

    Every member of a group lands in exactly one split, so no template is shared
    between train and test.
    """
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must leave a test split")

    rng = random.Random(seed)
    if not by_group:
        buckets: dict[Route, list[list[JevSample]]] = defaultdict(list)
        for sample in samples:
            buckets[sample.route].append([sample])
    else:
        routes = group_routes(samples)
        members: dict[str, list[JevSample]] = defaultdict(list)
        for sample in samples:
            members[group_key(sample)].append(sample)
        buckets = defaultdict(list)
        for key, rows in members.items():
            buckets[routes[key]].append(rows)

    train: list[JevSample] = []
    validation: list[JevSample] = []
    test: list[JevSample] = []
    for route in ROUTE_ORDER:
        groups = buckets.get(route, [])
        rng.shuffle(groups)
        train_end = max(1, int(len(groups) * train_ratio))
        validation_end = train_end + max(1, int(len(groups) * validation_ratio))
        for rows in groups[:train_end]:
            train.extend(rows)
        for rows in groups[train_end:validation_end]:
            validation.extend(rows)
        for rows in groups[validation_end:]:
            test.extend(rows)

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def group_leaks(splits: Iterable[Sequence[JevSample]]) -> set[str]:
    """Return semantic groups that appear in more than one split."""
    seen: dict[str, int] = defaultdict(int)
    for index, split in enumerate(splits):
        for key in {group_key(sample) for sample in split}:
            seen[key] |= 1 << index
    return {key for key, mask in seen.items() if bin(mask).count("1") > 1}


def assert_no_group_leak(splits: Iterable[Sequence[JevSample]]) -> None:
    leaked = group_leaks(splits)
    if leaked:
        sample = sorted(leaked)[:5]
        raise AssertionError(f"{len(leaked)} semantic group(s) span multiple splits, e.g. {sample}")


def route_distribution(samples: Sequence[JevSample]) -> dict[str, int]:
    counts = Counter(sample.route.value for sample in samples)
    return {route.value: counts.get(route.value, 0) for route in ROUTE_ORDER}


def confidence_distribution(samples: Sequence[JevSample]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(sample.confidence for sample in samples).items(), reverse=True)}


def describe(samples: Sequence[JevSample], name: str = "dataset") -> dict[str, object]:
    """Summary statistics, including the fields that expose dead labels."""
    total = len(samples) or 1
    routes = route_distribution(samples)
    confidences = [sample.confidence for sample in samples]
    groups = {group_key(sample) for sample in samples}
    return {
        "name": name,
        "samples": len(samples),
        "unique_queries": len({sample.query for sample in samples}),
        "semantic_groups": len(groups),
        "rows_per_group": round(len(samples) / max(len(groups), 1), 2),
        "route_counts": routes,
        "route_share": {route: round(count / total, 4) for route, count in routes.items()},
        "majority_route": max(routes, key=lambda key: routes[key]) if samples else None,
        "majority_share": round(max(routes.values()) / total, 4) if samples else 0.0,
        "confidence_values": confidence_distribution(samples),
        "confidence_mean": round(sum(confidences) / total, 4) if samples else 0.0,
        "confidence_spread": round(max(confidences) - min(confidences), 4) if samples else 0.0,
        "with_counterfactuals": sum(1 for sample in samples if sample.has_counterfactuals),
    }


def load_many(paths: Iterable[str | Path]) -> list[JevSample]:
    samples: list[JevSample] = []
    for path in paths:
        samples.extend(load_jsonl(path))
    return samples


def _command_split(args: argparse.Namespace) -> None:
    samples = load_jsonl(args.input)
    train, validation, test = split_dataset(samples, args.train_ratio, args.validation_ratio, args.seed, by_group=not args.by_row)
    assert_no_group_leak((train, validation, test))
    # Names match the defaults `minijev.training` reads.
    for name, rows in (("train", train), ("val", validation), ("test", test)):
        save_jsonl(rows, args.output_dir / f"{name}.jsonl")
        print(f"{name}: {len(rows)} samples, {len({group_key(row) for row in rows})} groups")
    print(f"group leak check: passed ({len({group_key(row) for row in samples})} groups total)")


def _command_stats(args: argparse.Namespace) -> None:
    for path in args.input:
        print(json.dumps(describe(load_jsonl(path), Path(path).name), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.data", description="Dataset utilities for MiniJev")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="split JSONL by semantic group and route")
    split.add_argument("input", type=Path)
    split.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--validation-ratio", type=float, default=0.15)
    split.add_argument("--by-row", action="store_true", help="disable group-aware splitting (leaks templates; for comparison only)")
    split.set_defaults(func=_command_split)

    stats = sub.add_parser("stats", help="describe one or more JSONL datasets")
    stats.add_argument("input", type=Path, nargs="+")
    stats.set_defaults(func=_command_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
