"""Evidence contracts for turning routing outcomes into defensible reward.

``rewards.py`` defines how numbers are combined.  This module answers the more
important question: where did those numbers come from, and are they complete
enough to optimise against?

Two evidence streams are kept separate because their units are different:

* action evidence records quality, money, latency and availability for every
  counterfactual action;
* misroute review records the harm of choosing route B when route A was required.

An ordinal human harm score is not silently converted to dollars.  The audit
reports both, and a later product decision must explicitly define any conversion.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import load_jsonl
from .schema import ROUTE_ORDER, JevSample, Route

EVIDENCE_KINDS = ("production", "benchmark", "human_elicited", "declared", "synthetic")
REQUIRED_ACTION_FIELDS = ("quality", "cost", "latency_ms", "available")
MISROUTE_SCHEMA_VERSION = "misroute-review-v1"
HARM_SCALE = {
    0: "no material impact",
    1: "minor inconvenience or cheap retry",
    2: "meaningful delay, wasted work, or avoidable spend",
    3: "serious user/business impact requiring intervention",
    4: "safety, legal, financial, or irreversible harm",
}


def _model_key(candidate: Any, index: int) -> str:
    return str(candidate.model or f"{candidate.route.value}:{index}")


def action_evidence_audit(samples: Iterable[JevSample]) -> dict[str, Any]:
    """Check whether every candidate field has explicit provenance.

    Converted rows carry ``metadata.candidate_evidence[model][field]`` with an
    evidence kind and source.  Old rows remain loadable, but audit as missing;
    backwards compatibility must not be mistaken for evidence.
    """
    rows = list(samples)
    counts = {field: Counter() for field in REQUIRED_ACTION_FIELDS}
    missing_examples: list[dict[str, str]] = []
    candidate_count = 0
    rows_with_two_actions = 0
    rows_with_all_routes = 0
    for sample in rows:
        if len(sample.candidates) >= 2:
            rows_with_two_actions += 1
        if {candidate.route for candidate in sample.candidates} == set(ROUTE_ORDER):
            rows_with_all_routes += 1
        evidence = sample.metadata.get("candidate_evidence", {})
        evidence = evidence if isinstance(evidence, Mapping) else {}
        for index, candidate in enumerate(sample.candidates):
            candidate_count += 1
            key = _model_key(candidate, index)
            fields = evidence.get(key, {})
            fields = fields if isinstance(fields, Mapping) else {}
            for field in REQUIRED_ACTION_FIELDS:
                entry = fields.get(field)
                kind = entry.get("kind") if isinstance(entry, Mapping) else None
                if kind not in EVIDENCE_KINDS:
                    counts[field]["missing"] += 1
                    if len(missing_examples) < 20:
                        missing_examples.append({"sample_id": str(sample.id or sample.query), "candidate": key, "field": field})
                else:
                    counts[field][str(kind)] += 1

    by_field = {field: {kind: counts[field].get(kind, 0) for kind in (*EVIDENCE_KINDS, "missing")} for field in REQUIRED_ACTION_FIELDS}
    no_missing = candidate_count > 0 and all(values["missing"] == 0 for values in by_field.values())
    benchmark_or_better = candidate_count > 0 and all(values["benchmark"] + values["production"] == candidate_count for values in by_field.values())
    production_complete = candidate_count > 0 and all(values["production"] == candidate_count for values in by_field.values())
    return {
        "rows": len(rows),
        "candidates": candidate_count,
        "rows_with_at_least_two_actions": rows_with_two_actions,
        "rows_with_all_four_routes": rows_with_all_routes,
        "evidence_by_field": by_field,
        "all_fields_have_provenance": no_missing,
        "offline_counterfactual_ready": benchmark_or_better and rows_with_two_actions == len(rows),
        "production_reward_ready": production_complete and rows_with_all_routes == len(rows),
        "missing_examples": missing_examples,
        "note": "Benchmark evidence supports offline experiments; only production evidence supports deployment reward claims.",
    }


@dataclass(frozen=True)
class MisrouteJudgment:
    """One human or observed assessment of a required→chosen route error."""

    sample_id: str
    query: str
    required_route: Route
    chosen_route: Route
    status: str = "pending"
    harm_score: float | None = None
    direct_cost: float | None = None
    delay_minutes: float | None = None
    reviewer_id: str | None = None
    rationale: str | None = None
    evidence_kind: str = "human_elicited"
    source_model: str | None = None
    model_confidence: float | None = None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        if self.required_route is self.chosen_route:
            raise ValueError("a misroute judgment needs two different routes")
        if self.status not in {"pending", "reviewed", "excluded"}:
            raise ValueError("status must be pending, reviewed, or excluded")
        if self.evidence_kind not in {"human_elicited", "production"}:
            raise ValueError("misroute evidence must be human_elicited or production")
        if self.harm_score is not None and (not math.isfinite(self.harm_score) or not 0.0 <= self.harm_score <= 4.0):
            raise ValueError("harm_score must be in [0, 4]")
        for name, value in (("direct_cost", self.direct_cost), ("delay_minutes", self.delay_minutes)):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.status == "reviewed" and (self.harm_score is None or not self.reviewer_id or not self.rationale):
            raise ValueError("reviewed judgments need harm_score, reviewer_id, and rationale")

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "MisrouteJudgment":
        return cls(
            sample_id=str(row["sample_id"]),
            query=str(row["query"]),
            required_route=Route(str(row["required_route"])),
            chosen_route=Route(str(row["chosen_route"])),
            status=str(row.get("status", "pending")),
            harm_score=None if row.get("harm_score") is None else float(row["harm_score"]),
            direct_cost=None if row.get("direct_cost") is None else float(row["direct_cost"]),
            delay_minutes=None if row.get("delay_minutes") is None else float(row["delay_minutes"]),
            reviewer_id=None if row.get("reviewer_id") is None else str(row["reviewer_id"]),
            rationale=None if row.get("rationale") is None else str(row["rationale"]),
            evidence_kind=str(row.get("evidence_kind", "human_elicited")),
            source_model=None if row.get("source_model") is None else str(row["source_model"]),
            model_confidence=None if row.get("model_confidence") is None else float(row["model_confidence"]),
            frame_id=None if row.get("frame_id") is None else str(row["frame_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MISROUTE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "query": self.query,
            "required_route": self.required_route.value,
            "chosen_route": self.chosen_route.value,
            "status": self.status,
            "harm_score": self.harm_score,
            "direct_cost": self.direct_cost,
            "delay_minutes": self.delay_minutes,
            "reviewer_id": self.reviewer_id,
            "rationale": self.rationale,
            "evidence_kind": self.evidence_kind,
            "source_model": self.source_model,
            "model_confidence": self.model_confidence,
            "frame_id": self.frame_id,
            "harm_scale": HARM_SCALE,
        }


def build_misroute_queue(report: Mapping[str, Any], *, per_cell: int = 5, source_model: str = "jev") -> list[MisrouteJudgment]:
    """Select high-confidence disagreements, balanced by confusion cell."""
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    disagreements = report.get("disagreements", [])
    if not isinstance(disagreements, list):
        raise ValueError("report.disagreements must be a list")
    cells: dict[tuple[Route, Route], list[Mapping[str, Any]]] = {}
    for row in disagreements:
        if not isinstance(row, Mapping):
            continue
        required = Route(str(row["ground_truth"]))
        chosen = Route(str(row["jev_prediction"]))
        if required is chosen:
            continue
        cells.setdefault((required, chosen), []).append(row)
    queue: list[MisrouteJudgment] = []
    for (required, chosen), rows in sorted(cells.items(), key=lambda item: (item[0][0].value, item[0][1].value)):
        ranked = sorted(rows, key=lambda row: (-float(row.get("jev_confidence", 0.0)), str(row.get("sample_id", ""))))
        for row in ranked[:per_cell]:
            queue.append(
                MisrouteJudgment(
                    sample_id=str(row["sample_id"]),
                    query=str(row["query"]),
                    required_route=required,
                    chosen_route=chosen,
                    source_model=source_model,
                    model_confidence=float(row.get("jev_confidence", 0.0)),
                    frame_id=None if row.get("frame_id") is None else str(row["frame_id"]),
                )
            )
    return queue


def misroute_audit(judgments: Iterable[MisrouteJudgment], *, minimum_per_cell: int = 2, minimum_reviewers: int = 2) -> dict[str, Any]:
    """Aggregate harm without pretending sparse elicitation is a calibrated cost."""
    if minimum_per_cell <= 0 or minimum_reviewers <= 0:
        raise ValueError("minimum_per_cell and minimum_reviewers must be positive")
    rows = list(judgments)
    reviewed = [row for row in rows if row.status == "reviewed"]
    observed_cells = {(row.required_route, row.chosen_route) for row in rows}
    required_cells = {(required, chosen) for required in ROUTE_ORDER for chosen in ROUTE_ORDER if required is not chosen}
    matrix: dict[str, dict[str, Any]] = {}
    ready_cells = 0
    for required, chosen in sorted(required_cells, key=lambda cell: (cell[0].value, cell[1].value)):
        members = [row for row in reviewed if row.required_route is required and row.chosen_route is chosen]
        reviewers = {row.reviewer_id for row in members if row.reviewer_id}
        harms = [float(row.harm_score) for row in members if row.harm_score is not None]
        costs = [float(row.direct_cost) for row in members if row.direct_cost is not None]
        delays = [float(row.delay_minutes) for row in members if row.delay_minutes is not None]
        ready = len(harms) >= minimum_per_cell and len(reviewers) >= minimum_reviewers
        ready_cells += int(ready)
        matrix[f"{required.value}>{chosen.value}"] = {
            "judgments": len(members),
            "reviewers": len(reviewers),
            "mean_harm_score": sum(harms) / len(harms) if harms else None,
            "mean_direct_cost": sum(costs) / len(costs) if costs else None,
            "mean_delay_minutes": sum(delays) / len(delays) if delays else None,
            "ready": ready,
        }
    production_rows = sum(row.evidence_kind == "production" for row in reviewed)
    observed_ready = bool(observed_cells) and all(matrix[f"{required.value}>{chosen.value}"]["ready"] for required, chosen in observed_cells)
    full_ready = ready_cells == len(required_cells)
    return {
        "rows": len(rows),
        "pending": sum(row.status == "pending" for row in rows),
        "reviewed": len(reviewed),
        "excluded": sum(row.status == "excluded" for row in rows),
        "observed_confusion_cells": len(observed_cells),
        "required_confusion_cells": len(required_cells),
        "missing_confusion_cells": sorted(f"{required.value}>{chosen.value}" for required, chosen in required_cells - observed_cells),
        "ready_cells": ready_cells,
        "minimum_per_cell": minimum_per_cell,
        "minimum_reviewers": minimum_reviewers,
        "observed_cells_ready": observed_ready,
        "harm_matrix_ready": full_ready,
        "production_cost_ready": full_ready
        and bool(reviewed)
        and production_rows == len(reviewed)
        and all(row.direct_cost is not None for row in reviewed),
        "matrix": matrix,
        "warning": "Ordinal harm, money, and delay are separate units; no conversion is implied.",
    }


def load_judgments(path: str | Path) -> list[MisrouteJudgment]:
    rows: list[MisrouteJudgment] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(MisrouteJudgment.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid misroute judgment at {path}:{line_number}: {error}") from error
    return rows


def save_judgments(rows: Iterable[MisrouteJudgment], path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; review queues are not overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.reward_data", description="Audit evidence behind MiniJev reward")
    subparsers = parser.add_subparsers(dest="command", required=True)
    actions = subparsers.add_parser("action-audit", help="audit candidate quality/cost/latency/availability provenance")
    actions.add_argument("--data", type=Path, required=True)
    actions.add_argument("--output", type=Path)
    queue = subparsers.add_parser("misroute-queue", help="create a balanced human-review queue from a Jev report")
    queue.add_argument("--report", type=Path, required=True)
    queue.add_argument("--output", type=Path, required=True)
    queue.add_argument("--per-cell", type=int, default=5)
    audit = subparsers.add_parser("misroute-audit", help="audit a filled human misroute review queue")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--minimum-per-cell", type=int, default=2)
    audit.add_argument("--minimum-reviewers", type=int, default=2)
    return parser


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"wrote {output}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "action-audit":
        _emit(action_evidence_audit(load_jsonl(args.data)), args.output)
    elif args.command == "misroute-queue":
        report = json.loads(args.report.read_text(encoding="utf-8"))
        rows = build_misroute_queue(report, per_cell=args.per_cell)
        save_judgments(rows, args.output)
        print(f"wrote {len(rows)} review rows to {args.output}")
    else:
        _emit(
            misroute_audit(load_judgments(args.data), minimum_per_cell=args.minimum_per_cell, minimum_reviewers=args.minimum_reviewers),
            args.output,
        )


if __name__ == "__main__":
    main()
