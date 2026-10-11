"""Convert heterogeneous external sources into the MiniJev data contract.

No dataset ships MiniJev's four routes, so every external source needs an explicit,
reviewable rubric. This module holds those rubrics in one place, next to the
reasoning for them, instead of scattering `if` branches through evaluation code.

The MASSIVE rubric below is a *pilot rubric*, not ground truth. MASSIVE annotates
intents; it does not annotate routing requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .schema import ROUTE_ORDER, Candidate, JevSample, Route

ROUTERBENCH_SOURCE = "routerbench"
ROUTERBENCH_SCORE_FIELDS = ("score", "performance", "quality", "accuracy", "reward")
ROUTERBENCH_COST_FIELDS = ("total_cost", "cost", "price")
ROUTERBENCH_LATENCY_FIELDS = ("latency_ms", "latency", "time_ms")

# Priors for the terminal human-review action when no recorded model qualifies.
# RouterBench has no column for it; these are declared placeholders, not
# observations, and must be recalibrated against real review costs before any
# reward-trained policy is trusted.
ROUTERBENCH_REVIEW_COST = 2.0
ROUTERBENCH_REVIEW_LATENCY_MS = 3_600_000.0


def _number(value: object) -> float | None:
    """Parse RouterBench numeric cells without accepting missing values as zero."""
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _first_number(row: dict[str, object], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _prompt_from_row(row: dict[str, object]) -> str:
    for key in ("prompt", "query", "origin_query", "input", "state", "question"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("RouterBench row has no non-empty prompt/query/state field")


def _load_records(path: str | Path) -> list[dict[str, object]]:
    """Load the JSONL/JSON/CSV forms published by RouterBench and mirrors."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("data", "records", "rows", "examples"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key]]
        raise ValueError("JSON RouterBench input must be a list or contain data/records/rows/examples")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("reading RouterBench parquet requires pandas and pyarrow") from exc
        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    raise ValueError(f"unsupported RouterBench file type {path.suffix!r}; use .jsonl, .json, .csv, or .parquet")


def _model_cells(row: dict[str, object]) -> dict[str, dict[str, float | None]]:
    """Extract model score/cost cells from nested and RouterBench wide rows."""
    models: dict[str, dict[str, float | None]] = {}
    nested = row.get("models") or row.get("model_scores") or row.get("results")
    cost_map = row.get("model_costs") or row.get("costs")
    latency_map = row.get("model_latencies") or row.get("latencies")
    if isinstance(nested, dict):
        for model, value in nested.items():
            cells = value if isinstance(value, dict) else {"score": value}
            model_cost = cost_map.get(model) if isinstance(cost_map, dict) else None
            model_latency = latency_map.get(model) if isinstance(latency_map, dict) else None
            models[str(model)] = {
                "score": _first_number(cells, ROUTERBENCH_SCORE_FIELDS),
                "cost": _first_number(cells, ROUTERBENCH_COST_FIELDS) if isinstance(cells, dict) else _number(model_cost),
                "latency_ms": _first_number(cells, ROUTERBENCH_LATENCY_FIELDS) if isinstance(cells, dict) else _number(model_latency),
            }
    for key, value in row.items():
        match = re.match(r"^(.+)\|([^|]+)$", str(key))
        if not match:
            continue
        model, field = match.groups()
        field = field.lower()
        kind = (
            "score"
            if field in ROUTERBENCH_SCORE_FIELDS
            else "cost"
            if field in ROUTERBENCH_COST_FIELDS
            else "latency_ms"
            if field in ROUTERBENCH_LATENCY_FIELDS
            else None
        )
        if kind is None:
            continue
        models.setdefault(model, {"score": None, "cost": None, "latency_ms": None})[kind] = _number(value)
    return models


class Convertor(ABC):
    """Normalise one external source into `JevSample` rows."""

    name: str = "convertor"
    source: str = "unknown"

    @abstractmethod
    def rows(self, path: str | Path, partition: str) -> Iterator[JevSample]: ...

    def convert(self, path: str | Path, partition: str, destination: str | Path) -> Counter:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        counts: Counter = Counter()
        with destination.open("w", encoding="utf-8") as handle:
            for sample in self.rows(path, partition):
                counts[sample.route.value] += 1
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        return counts


@dataclass
class RouterBenchConvertor(Convertor):
    """Derive MiniJev routes from RouterBench quality/cost observations.

    RouterBench has no Jev route labels. A row is sent to the cheapest model whose
    quality reaches ``quality_threshold``. The cheapest, second-cheapest and
    remaining qualifying tiers become local, rag and cloud. If no model qualifies,
    the row becomes review. This is an explicit evaluation policy, not ground truth.
    """

    name: str = "routerbench"
    source: str = ROUTERBENCH_SOURCE
    quality_threshold: float = 0.8

    def rows(self, path: str | Path, partition: str = "all") -> Iterator[JevSample]:
        records = _load_records(path)
        for index, row in enumerate(records):
            if not self._partition_matches(row, partition):
                continue
            models = _model_cells(row)
            scored = [(name, cells) for name, cells in models.items() if cells.get("score") is not None]
            if not scored:
                raise ValueError(f"RouterBench row {index} has no model score fields")
            ranked = sorted(scored, key=lambda item: (item[1].get("cost") is None, item[1].get("cost") or 0.0, item[0]))
            qualifying = [(name, cells) for name, cells in ranked if float(cells["score"]) >= self.quality_threshold]
            if not qualifying:
                route = Route.REVIEW
                selected_name = None
            else:
                selected_name = qualifying[0][0]
                tier = next(position for position, (name, _) in enumerate(ranked) if name == selected_name)
                route = (Route.LOCAL, Route.RAG, Route.CLOUD)[min(tier, 2)]
            candidates = tuple(self._candidate(cells, position, ranked) for position, (_, cells) in enumerate(ranked))
            if route is Route.REVIEW:
                # No recorded model clears the bar, so the honest counterfactual is
                # the terminal review action with declared placeholders for its cost.
                candidates = candidates + (
                    Candidate(
                        route=Route.REVIEW,
                        reward=self.quality_threshold,
                        quality=self.quality_threshold,
                        cost=ROUTERBENCH_REVIEW_COST,
                        latency_ms=ROUTERBENCH_REVIEW_LATENCY_MS,
                        propensity=None,
                        eligible=True,
                    ),
                )
            sample_id = row.get("sample_id") or row.get("id") or row.get("uid") or f"routerbench_{index}"
            metadata = {
                "quality_threshold": self.quality_threshold,
                "selected_model": selected_name,
                "route_policy": "cheapest_qualifying_model_v1",
                "model_scores": {name: cells.get("score") for name, cells in ranked},
                "model_costs": {name: cells.get("cost") for name, cells in ranked},
            }
            for key in ("eval_name", "source", "category", "task", "split"):
                if row.get(key) is not None:
                    metadata[f"routerbench_{key}"] = row[key]
            yield JevSample(
                query=_prompt_from_row(row),
                confidence=1.0,
                route=route,
                candidates=candidates,
                id=str(sample_id),
                source=self.source,
                metadata=metadata,
            )

    @staticmethod
    def _partition_matches(row: dict[str, object], partition: str) -> bool:
        if partition in {"all", "*"}:
            return True
        value = row.get("split", row.get("partition"))
        if value is None:
            return True
        aliases = {"dev": "val", "validation": "val", "development": "val"}
        return aliases.get(str(value).lower(), str(value).lower()) == aliases.get(partition.lower(), partition.lower())

    def _candidate(self, cells: dict[str, float | None], position: int, ranked: list[tuple[str, dict[str, float | None]]]) -> Candidate:
        score = float(cells["score"] or 0.0)
        cost = cells.get("cost")
        # Uniform propensity is a placeholder: it pretends the data was collected
        # under a random router. Off-policy evaluation must replace it with the
        # true behaviour-policy probabilities before any IPS estimate is trusted.
        propensity = 1.0 / len(ranked) if ranked else None
        tier_route = (Route.LOCAL, Route.RAG, Route.CLOUD)[min(position, 2)]
        return Candidate(
            route=tier_route,
            reward=score,
            quality=score,
            cost=cost,
            latency_ms=cells.get("latency_ms"),
            propensity=propensity,
            eligible=score >= self.quality_threshold,
        )

    def audit(self, path: str | Path, partition: str = "all") -> dict[str, object]:
        rows = list(self.rows(path, partition))
        counts = Counter(row.route.value for row in rows)
        return {
            "source": self.source,
            "rows": len(rows),
            "quality_threshold": self.quality_threshold,
            "route_policy": "cheapest_qualifying_model_v1",
            "route_counts": {route.value: counts.get(route.value, 0) for route in ROUTE_ORDER},
            "with_counterfactuals": sum(row.has_counterfactuals for row in rows),
            "review_rows_with_synthesized_candidate": counts.get(Route.REVIEW.value, 0),
        }


# ---------------------------------------------------------------------------
# MASSIVE zh-CN -> MiniJev route rubric (v2)
# ---------------------------------------------------------------------------
#
# The rule, stated once so it can be argued with:
#
#   local   the request can be served by the device itself -- controlling it,
#           writing to an on-device store, or answering from built-in knowledge.
#   rag     serving it requires fetching from an external service or from stored
#           user data, but no multi-step reasoning is involved.
#   cloud   serving it requires multi-step reasoning, planning or creation.
#   review  the intent is ambiguous, underspecified, or high-risk.
#
# v1 (the original pilot rubric) sent every non-device intent to rag and used
# cloud for the three `recommendation_*` intents. Two things were wrong with it:
#
#   1. Pure writes to on-device stores were classified as retrieval. Setting an
#      alarm already sat in local while adding a list item sat in rag, which is
#      inconsistent -- neither fetches anything.
#   2. `recommendation_*` is a catalog lookup, not complex reasoning. It does not
#      belong in cloud.
#
# What v2 cannot fix: MASSIVE contains no intent that requires multi-step
# reasoning or high-stakes judgement. `cloud` therefore has **no samples at all**
# and `review` rests on a single intent. See `SUPPORTED_ROUTES`.

LOCAL_INTENTS = frozenset(
    {
        # Device state and settings.
        "alarm_query",
        "alarm_remove",
        "alarm_set",
        "audio_volume_down",
        "audio_volume_mute",
        "audio_volume_other",
        "audio_volume_up",
        # Writes to an on-device store. The matching _query intents are reads, and
        # reads are rag below -- that split is the main v1 -> v2 correction.
        "calendar_remove",
        "calendar_set",
        "lists_createoradd",
        "lists_remove",
        "music_dislikeness",
        "music_likeness",
        "music_settings",
        # Built-in knowledge: the device clock, arithmetic, canned social replies.
        "datetime_convert",
        "datetime_query",
        "general_greet",
        "general_joke",
        "qa_maths",
        # Direct device control.
        "iot_cleaning",
        "iot_coffee",
        "iot_hue_lightchange",
        "iot_hue_lightdim",
        "iot_hue_lightoff",
        "iot_hue_lighton",
        "iot_hue_lightup",
        "iot_wemo_off",
        "iot_wemo_on",
        # Playback. A streaming catalogue is involved, but "play this" is the
        # canonical device function and requires no lookup the user must wait on.
        "play_audiobook",
        "play_game",
        "play_music",
        "play_podcasts",
        "play_radio",
    }
)

RAG_INTENTS = frozenset(
    {
        # Reads of stored user data.
        "calendar_query",
        "email_query",
        "email_querycontact",
        "lists_query",
        # Intents that reach an external service, including the write-shaped ones.
        # `email_sendemail` and `social_post` write, but they write to a service
        # off the device, so they are not local.
        "email_addcontact",
        "email_sendemail",
        "social_post",
        "social_query",
        "takeaway_order",
        "takeaway_query",
        "transport_query",
        "transport_taxi",
        "transport_ticket",
        "transport_traffic",
        # Live or encyclopaedic lookups.
        "cooking_query",
        "cooking_recipe",
        "music_query",
        "news_query",
        "qa_currency",
        "qa_definition",
        "qa_factoid",
        "qa_stock",
        "weather_query",
        # Catalog lookups. Moved out of cloud in v2.
        "recommendation_events",
        "recommendation_locations",
        "recommendation_movies",
    }
)

REVIEW_INTENTS = frozenset(
    {
        # MASSIVE's "odd utterance" bucket: underspecified or unparseable requests.
        # This is the only genuinely review-shaped intent in the dataset.
        "general_quirky",
    }
)

# Routes this source can actually evidence. `cloud` is deliberately absent.
SUPPORTED_ROUTES: tuple[Route, ...] = (Route.LOCAL, Route.RAG, Route.REVIEW)
UNSUPPORTED_ROUTES: tuple[Route, ...] = tuple(route for route in ROUTE_ORDER if route not in SUPPORTED_ROUTES)

MASSIVE_RUBRIC: dict[str, Route] = {
    **{intent: Route.LOCAL for intent in LOCAL_INTENTS},
    **{intent: Route.RAG for intent in RAG_INTENTS},
    **{intent: Route.REVIEW for intent in REVIEW_INTENTS},
}

MASSIVE_SOURCE = "massive-1.1-zh-CN"
MASSIVE_CONFIDENCE = 1.0


def massive_route(intent: str) -> Route:
    """Map a MASSIVE intent to a route, refusing to guess."""
    try:
        return MASSIVE_RUBRIC[intent]
    except KeyError:
        raise KeyError(f"no rubric entry for MASSIVE intent {intent!r}; add it to LOCAL/RAG/REVIEW_INTENTS deliberately") from None


def unmapped_intents(intents: Iterable[str]) -> list[str]:
    return sorted({intent for intent in intents if intent not in MASSIVE_RUBRIC})


@dataclass
class MassiveConvertor(Convertor):
    """MASSIVE 1.1 JSONL -> MiniJev JSONL, via the v2 rubric.

    `confidence` is set to 1.0 because MASSIVE has no confidence annotation. The
    confidence head is therefore untrained on this source -- report route metrics
    only, and read `minijev.evaluation.ConfidenceAudit` before trusting a
    confidence number.
    """

    name: str = "massive"
    source: str = MASSIVE_SOURCE
    confidence: float = MASSIVE_CONFIDENCE

    def rows(self, path: str | Path, partition: str) -> Iterator[JevSample]:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("partition") != partition:
                    continue
                intent = str(row["intent"])
                yield JevSample(
                    query=str(row["utt"]),
                    confidence=self.confidence,
                    route=massive_route(intent),
                    id=f"massive_{row['id']}",
                    source=self.source,
                    metadata={"intent": intent, "scenario": row.get("scenario")},
                )

    def audit(self, path: str | Path) -> dict[str, object]:
        """Report rubric coverage and the resulting distribution, before converting."""
        intents: Counter = Counter()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    intents[str(json.loads(line)["intent"])] += 1
        missing = unmapped_intents(intents)
        per_route: Counter = Counter()
        for intent, count in intents.items():
            route = MASSIVE_RUBRIC.get(intent)
            if route is not None:
                per_route[route.value] += count
        total = sum(per_route.values()) or 1
        return {
            "source": self.source,
            "rubric": "v2",
            "distinct_intents": len(intents),
            "unmapped_intents": missing,
            "route_counts": {route.value: per_route.get(route.value, 0) for route in ROUTE_ORDER},
            "route_share": {route.value: round(per_route.get(route.value, 0) / total, 4) for route in ROUTE_ORDER},
            "supported_routes": [route.value for route in SUPPORTED_ROUTES],
            "unsupported_routes": [route.value for route in UNSUPPORTED_ROUTES],
        }


CONVERTORS: dict[str, type[Convertor]] = {"massive": MassiveConvertor, "routerbench": RouterBenchConvertor}


def _command_audit(args: argparse.Namespace) -> None:
    print(json.dumps(MassiveConvertor().audit(args.data), ensure_ascii=False, indent=2))


def _command_convert(args: argparse.Namespace) -> None:
    convertor = MassiveConvertor()
    for partition, name in (("train", "train"), ("dev", "val"), ("test", "test")):
        counts = convertor.convert(args.data, partition, args.output_dir / f"massive_{name}.jsonl")
        total = sum(counts.values())
        detail = " ".join(f"{route.value}={counts.get(route.value, 0)}" for route in ROUTE_ORDER)
        print(f"{name:5s} {total:6d}  {detail}")


def _command_routerbench(args: argparse.Namespace) -> None:
    convertor = RouterBenchConvertor(quality_threshold=args.quality_threshold)
    if args.audit:
        print(json.dumps(convertor.audit(args.data, args.partition), ensure_ascii=False, indent=2))
        return
    counts = convertor.convert(args.data, args.partition, args.output)
    detail = " ".join(f"{route.value}={counts.get(route.value, 0)}" for route in ROUTE_ORDER)
    print(f"converted {sum(counts.values())} RouterBench rows: {detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.convert", description="Convert external datasets into MiniJev JSONL")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="show rubric coverage without writing anything")
    audit.add_argument("--data", type=Path, required=True)
    audit.set_defaults(func=_command_audit)

    convert = sub.add_parser("convert", help="write MiniJev JSONL splits")
    convert.add_argument("--data", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.set_defaults(func=_command_convert)

    routerbench = sub.add_parser("routerbench", help="derive routes from RouterBench quality/cost records")
    routerbench.add_argument("--data", type=Path, required=True)
    routerbench.add_argument("--output", type=Path, default=Path("data/external/routerbench/routerbench.jsonl"))
    routerbench.add_argument("--partition", default="all", help="all, train, val/dev, or test when the file carries a split field")
    routerbench.add_argument("--quality-threshold", type=float, default=0.8)
    routerbench.add_argument("--audit", action="store_true", help="report derived labels without writing JSONL")
    routerbench.set_defaults(func=_command_routerbench)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
