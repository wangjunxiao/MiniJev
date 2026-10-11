"""Inference pipeline entry point.

Loads a checkpoint by default. The previous CLI constructed a randomly initialised
model and never loaded weights, so its output was unrelated to anything trained --
that is fixed here, and a missing checkpoint is an error rather than a silent
fallback to noise.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .confidence import DEFAULT_MEASURE, distribution_shape
from .config import DEFAULT_CHECKPOINT, JevConfig
from .policies import Policy, RiskLadderPolicy
from .routers import NeuralRouter
from .schema import Decision, Route


class CheckpointMissing(FileNotFoundError):
    """Raised when inference is asked for weights that do not exist."""


def load_router(
    checkpoint: str | Path | None = None,
    *,
    policy: Policy | None = None,
    config: JevConfig | None = None,
    allow_untrained: bool = False,
) -> NeuralRouter:
    """Build a neural router backed by real weights.

    Pass `allow_untrained=True` only to smoke-test the execution path; the scores
    it returns are then meaningless.
    """
    path = Path(DEFAULT_CHECKPOINT if checkpoint is None else checkpoint)
    if not path.exists():
        if not allow_untrained:
            raise CheckpointMissing(f"no checkpoint at {path}; train one first or pass --allow-untrained")
        path_value: str | None = None
    else:
        path_value = str(path)
    return NeuralRouter(config, checkpoint=path_value, policy=policy)


def predict(queries: Sequence[str], *, router: NeuralRouter | None = None, checkpoint: str | Path | None = None) -> list[Decision]:
    active = router if router is not None else load_router(checkpoint)
    return active.decide(list(queries))


def _decision_payload(query: str, decision: Decision) -> dict[str, object]:
    return {
        "query": query,
        "route": decision.route.value,
        "confidence": decision.confidence,
        "confidence_measure": decision.confidence_measure,
        "route_probs": {route.value: probability for route, probability in zip(Route, decision.route_probs)},
        "route_logits": list(decision.route_logits),
        "distribution": distribution_shape(decision.route_probs, decision.confidence_measure),
        "policy": decision.policy,
        "model_preferred": decision.top_route.value,
        "escalated_from": None if decision.escalated_from is None else decision.escalated_from.value,
        "log_prob": decision.log_prob(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.inference", description="Route queries with a trained MiniJev model")
    parser.add_argument("query", nargs="*", help="query text; omit to read --queries-file")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--queries-file", type=Path, default=None, help="one query per line")
    parser.add_argument("--measure", default=DEFAULT_MEASURE, help="confidence measure for the policy")
    parser.add_argument("--caution", type=float, default=1.0, help="scale every route gate; >1 escalates more often")
    parser.add_argument("--allow-untrained", action="store_true", help="run with random weights; scores are meaningless")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    queries = list(args.query)
    if args.queries_file is not None:
        queries.extend(line.strip() for line in args.queries_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not queries:
        raise SystemExit("provide at least one query, or --queries-file")

    policy = RiskLadderPolicy(measure=args.measure).scaled(args.caution)
    router = load_router(args.checkpoint, policy=policy, allow_untrained=args.allow_untrained)
    payload = {
        "checkpoint": router.checkpoint,
        "policy": router.policy.name,
        "decisions": [_decision_payload(query, decision) for query, decision in zip(queries, router.decide(queries))],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
