"""Boundary-benchmark evaluation, including the pairing analysis.

Overall accuracy on the boundary benchmark answers a weak question: a model can score
well by learning whichever surface cue dominates and still fail every contrast. The
pairing analysis answers the question the benchmark exists for:

* **both correct** -- the model separated the two members of a pair;
* **collapsed** -- it predicted the *same* route for both, so it cannot see the slot that
  distinguishes them at all, however well it does on the benchmark overall;
* **split** -- it got one right, which is the ambiguous case that one more sample would
  flip.

`collapse rate` is the headline: it is the fraction of pairs where the deciding slot is
invisible to the model. For **minimal pairs** (exactly one deciding slot differs) a
collapse is unambiguous evidence about that one slot.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import load_jsonl
from .evaluation import classification_report
from .frames import CONTRAST_PAIRS, RouteFrame, deciding_slots_differing, load_frame
from .routers import Router
from .schema import ROUTE_ORDER, JevSample, Route


@dataclass(frozen=True)
class PairOutcome:
    """What the model did with one contrast pair."""

    left: str
    right: str
    left_route: Route
    right_route: Route
    left_predicted: Route
    right_predicted: Route
    deciding_slots: tuple[str, ...]

    @property
    def minimal(self) -> bool:
        return len(self.deciding_slots) == 1

    @property
    def both_correct(self) -> bool:
        return self.left_predicted is self.left_route and self.right_predicted is self.right_route

    @property
    def neither_correct(self) -> bool:
        return self.left_predicted is not self.left_route and self.right_predicted is not self.right_route

    @property
    def collapsed(self) -> bool:
        """One answer for two different questions: the deciding slot was not read."""
        return self.left_predicted is self.right_predicted

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "deciding_slots": list(self.deciding_slots),
            "minimal": self.minimal,
            "expected": [self.left_route.value, self.right_route.value],
            "predicted": [self.left_predicted.value, self.right_predicted.value],
            "both_correct": self.both_correct,
            "neither_correct": self.neither_correct,
            "collapsed": self.collapsed,
        }


def _group_predictions(samples: Sequence[JevSample], predicted: Sequence[Route]) -> dict[str, Route]:
    by_group: dict[str, Route] = {}
    for sample, route in zip(samples, predicted):
        group = str(sample.metadata.get("semantic_group_id", ""))
        by_group.setdefault(group, route)
    return by_group


def evaluate_boundary(router: Router, samples: Sequence[JevSample]) -> dict[str, Any]:
    """Accuracy, per-slot accuracy and pair separation on a boundary benchmark."""
    decisions = router.decide([sample.query for sample in samples])
    predicted = [decision.route for decision in decisions]
    actual = [sample.route for sample in samples]

    present = tuple(sorted(set(actual), key=lambda route: route.value))
    report = classification_report(actual, predicted, labels=present)

    per_route_recall: dict[str, float] = {}
    for route in present:
        rows = [1.0 if p is route else 0.0 for p, a in zip(predicted, actual) if a is route]
        per_route_recall[route.value] = sum(rows) / len(rows) if rows else 0.0

    # Accuracy grouped by the slot that decided the route, so a failure names a slot.
    by_group = _group_predictions(samples, predicted)
    slot_accuracy: dict[str, dict[str, float]] = {}
    for slot in ("risk", "underspecified", "derivation", "steps", "source_class"):
        buckets: dict[str, list[bool]] = {}
        for sample in samples:
            frame = load_frame(sample)
            if frame is None:
                continue
            key = _slot_value(frame, slot)
            buckets.setdefault(key, []).append(by_group.get(str(sample.metadata.get("semantic_group_id", ""))) is sample.route)
        slot_accuracy[slot] = {key: (sum(values) / len(values) if values else 0.0) for key, values in buckets.items()}

    outcomes = _pair_outcomes(samples, by_group)
    minimal = [outcome for outcome in outcomes if outcome.minimal]

    def summarise(subset: Sequence[PairOutcome]) -> dict[str, Any]:
        if not subset:
            return {"pairs": 0, "both_correct": 0, "collapsed": 0, "neither_correct": 0}
        return {
            "pairs": len(subset),
            "both_correct": sum(1 for outcome in subset if outcome.both_correct) / len(subset),
            "collapsed": sum(1 for outcome in subset if outcome.collapsed) / len(subset),
            "neither_correct": sum(1 for outcome in subset if outcome.neither_correct) / len(subset),
        }

    return {
        "rows": len(samples),
        "groups": len(by_group),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "per_route_recall": per_route_recall,
        "slot_accuracy": slot_accuracy,
        "pairs": summarise(outcomes),
        "minimal_pairs": summarise(minimal),
        "outcomes": [outcome.to_dict() for outcome in outcomes],
    }


def _slot_value(frame: RouteFrame, slot: str) -> str:
    from .frames import EXTERNAL_SOURCES

    if slot == "source_class":
        return "external" if frame.source in EXTERNAL_SOURCES else "internal"
    if slot == "steps":
        return "1" if frame.steps <= 1 else "2+"
    value = getattr(frame, slot)
    return str(getattr(value, "value", value))


def _pair_outcomes(samples: Sequence[JevSample], by_group: dict[str, Route]) -> list[PairOutcome]:
    """Join the two halves of every contrast pair that is actually in the benchmark."""
    expected: dict[str, tuple[Route, RouteFrame]] = {}
    for sample in samples:
        frame = load_frame(sample)
        if frame is None:
            continue
        expected[str(sample.metadata.get("frame_id"))] = (sample.route, frame)

    outcomes: list[PairOutcome] = []
    for left_id, right_id in CONTRAST_PAIRS:
        if left_id not in expected or right_id not in expected:
            continue
        left_route, left_frame = expected[left_id]
        right_route, right_frame = expected[right_id]
        left_predicted = by_group.get(f"frame:{left_id}")
        right_predicted = by_group.get(f"frame:{right_id}")
        if left_predicted is None or right_predicted is None:
            continue
        outcomes.append(
            PairOutcome(
                left=left_id,
                right=right_id,
                left_route=left_route,
                right_route=right_route,
                left_predicted=left_predicted,
                right_predicted=right_predicted,
                deciding_slots=deciding_slots_differing(left_frame, right_frame),
            )
        )
    return outcomes


def format_report(name: str, result: Mapping[str, Any]) -> str:
    lines = [f"{name}", f"  rows={result['rows']} groups={result['groups']} acc={result['accuracy']:.4f} mF1={result['macro_f1']:.4f}"]
    recall = " ".join(f"{route}={result['per_route_recall'].get(route.value, 0.0):.3f}" for route in ROUTE_ORDER)
    lines.append(f"  per-route recall: {recall}")
    pairs = result["pairs"]
    minimal = result["minimal_pairs"]
    lines.append(
        f"  pairs: {pairs['pairs']}  both_correct={pairs['both_correct']:.1%}  collapsed={pairs['collapsed']:.1%}  "
        f"neither={pairs['neither_correct']:.1%}"
    )
    lines.append(f"  minimal pairs: {minimal['pairs']}  both_correct={minimal['both_correct']:.1%}  collapsed={minimal['collapsed']:.1%}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.boundary", description="Evaluate a checkpoint on the boundary benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("data/hard_benchmark.jsonl"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--only-minimal", action="store_true", help="print one line per minimal pair instead of a summary")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .inference import load_router

    args = build_parser().parse_args(argv)
    samples = load_jsonl(args.benchmark)
    result = evaluate_boundary(load_router(args.checkpoint), samples)

    if args.only_minimal:
        for outcome in result["outcomes"]:
            if not outcome["minimal"]:
                continue
            mark = "OK " if outcome["both_correct"] else ("COLLAPSE" if outcome["collapsed"] else "SPLIT")
            print(
                f"  {mark:8s} {outcome['left']}/{outcome['right']} "
                f"[{','.join(outcome['deciding_slots'])}] expected={outcome['expected']} predicted={outcome['predicted']}"
            )
        return

    print(format_report(args.checkpoint.name, result))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
