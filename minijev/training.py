"""Training pipeline entry point.

Shares the schema, tokenizer, model, objectives and evaluation with inference;
nothing here is imported by inference. The loop itself is deliberately unchanged --
the differences from the first version are all in what it reports, because the
previous report could not distinguish a working confidence signal from a dead one.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tinygrad import Context, Tensor, dtypes
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters

from .confidence import DEFAULT_MEASURE, distribution_shape
from .config import DATA_DIR, DEFAULT_CHECKPOINT, DEFAULT_METRICS, JevConfig
from .data import describe
from .evaluation import classification_report, confidence_audit, majority_baseline
from .frames import slot_targets
from .model import TinyJev, load_checkpoint, save_checkpoint
from .objectives import slot_loss, supervised_loss
from .policies import Policy, RiskLadderPolicy
from .schema import ROUTE_INDEX, JevSample, Route, RouteScores
from .tokenizer import TURN_SEPARATOR_ID, ByteTokenizer

DEFAULT_LABELS: tuple[Route, ...] = (Route.LOCAL, Route.RAG)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.003
    embed_dim: int = 64
    hidden_dim: int = 64
    encoder_type: str = "pooled"
    num_layers: int = 1
    num_heads: int = 4
    ffn_dim: int = 128
    preserve_tail: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    log_every: int = 5
    seed: int = 7
    weight_by_agreement: bool = True
    label_smoothing: float = 0.0
    use_context: bool = False
    aux_weight: float = 0.0
    """Weight on the deciding-slot auxiliary loss. 0 disables the heads entirely."""

    def __post_init__(self) -> None:
        for field_name in (
            "epochs",
            "batch_size",
            "embed_dim",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "patience",
            "log_every",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.encoder_type not in {"pooled", "transformer"}:
            raise ValueError("encoder_type must be 'pooled' or 'transformer'")


def batches(samples: Sequence[JevSample], batch_size: int, order: Sequence[int] | None = None) -> Iterator[list[JevSample]]:
    indices = list(order) if order is not None else list(range(len(samples)))
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start : start + batch_size]]


def tensors(
    tokenizer: ByteTokenizer, samples: Sequence[JevSample], *, weight_by_agreement: bool = True
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Encode a batch. The agreement label becomes a sample weight, never a target.

    Each sample's `context` turns are placed before its query, so a row whose route
    depends on earlier state is encoded with that state. A batch with no context encodes
    exactly as before, so single-turn and context rows can share one corpus.
    """
    turns = [tuple(sample.context) + (sample.query,) for sample in samples]
    if any(sample.context for sample in samples):
        input_ids, attention_mask = tokenizer.batch_turns(turns)
    else:
        input_ids, attention_mask = tokenizer.batch([sample.query for sample in samples])
    routes = Tensor([ROUTE_INDEX[sample.route] for sample in samples], dtype=dtypes.int32)
    weights = Tensor([sample.confidence for sample in samples], dtype=dtypes.float32) if weight_by_agreement else None
    return input_ids, attention_mask, routes, weights


def model_scores(model: TinyJev, tokenizer: ByteTokenizer, samples: Sequence[JevSample], batch_size: int) -> list[RouteScores]:
    scores: list[RouteScores] = []
    for batch in batches(samples, batch_size):
        input_ids, attention_mask, _, _ = tensors(tokenizer, batch)
        outputs = model(input_ids, attention_mask)
        logits = outputs.route_logits.numpy().tolist()
        scores.extend(RouteScores(route_logits=tuple(float(x) for x in row)) for row in logits)
    return scores


def evaluate_split(
    model: TinyJev,
    tokenizer: ByteTokenizer,
    samples: Sequence[JevSample],
    batch_size: int,
    policy: Policy,
    *,
    name: str,
) -> dict[str, object]:
    scores = model_scores(model, tokenizer, samples, batch_size)
    actual = [sample.route for sample in samples]
    decisions = [policy.decide(value) for value in scores]
    predicted = [decision.route for decision in decisions]
    report = classification_report(actual, predicted, name=name)
    report["restricted"] = classification_report(actual, predicted, labels=DEFAULT_LABELS, name=f"{name}:supported", scope_to_labels=True)
    report["confidence"] = confidence_audit(
        [decision.confidence for decision in decisions], [guess is truth for truth, guess in zip(actual, predicted)]
    )
    report["confidence"]["measure"] = getattr(policy, "measure", DEFAULT_MEASURE)
    report["confidence"]["escalation_rate"] = sum(1 for decision in decisions if decision.escalated) / (len(decisions) or 1)
    shapes = [distribution_shape(decision.route_probs, getattr(policy, "measure", DEFAULT_MEASURE)) for decision in decisions]
    report["distribution"] = {
        "mean_effective_options": sum(shape["effective_options"] for shape in shapes) / (len(shapes) or 1),
        "mean_top_probability": sum(shape["top_probability"] for shape in shapes) / (len(shapes) or 1),
        "measure": getattr(policy, "measure", DEFAULT_MEASURE),
    }
    return report


def fit(
    train_samples: Sequence[JevSample],
    validation_samples: Sequence[JevSample],
    config: TrainingConfig,
    *,
    output: str | Path = DEFAULT_CHECKPOINT,
    policy: Policy | None = None,
) -> tuple[TinyJev, dict[str, object]]:
    """Train, select by validation macro-F1, and return the best model plus its log."""
    random.seed(config.seed)
    Tensor.manual_seed(config.seed)
    policy = policy if policy is not None else RiskLadderPolicy()

    jev_config = JevConfig(
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        preserve_tail=config.preserve_tail,
        encoder_type=config.encoder_type,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        **({"vocab_size": TURN_SEPARATOR_ID + 1, "turn_sep_id": TURN_SEPARATOR_ID} if config.use_context else {}),
        aux_slots=config.aux_weight > 0.0,
    )
    tokenizer = ByteTokenizer(jev_config)
    model = TinyJev(jev_config)
    optimizer = optim.Adam(get_parameters(model), lr=config.learning_rate)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    parameters = get_parameters(model)
    with Context(TRAINING=1):
        for epoch in range(1, config.epochs + 1):
            order = list(range(len(train_samples)))
            random.shuffle(order)
            loss_sum = 0.0
            for batch in batches(train_samples, config.batch_size, order):
                input_ids, attention_mask, routes, weights = tensors(tokenizer, batch, weight_by_agreement=config.weight_by_agreement)
                outputs = model(input_ids, attention_mask)
                loss = supervised_loss(outputs, routes, sample_weights=weights, label_smoothing=config.label_smoothing)
                if config.aux_weight > 0.0:
                    # The route is a function of the deciding slots, so predicting those
                    # slots is a denser version of the same question. The auxiliary term
                    # is added, never substituted: the route head is still trained end to
                    # end, and the decision path still reads only `route_logits`.
                    columns, mask = slot_targets(batch)
                    aux_targets = {slot: Tensor(values, dtype=dtypes.int32) for slot, values in columns.items()}
                    auxiliary = slot_loss(outputs, aux_targets, Tensor(mask, dtype=dtypes.float32))
                    if auxiliary is not None:
                        loss = loss + auxiliary * config.aux_weight
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_sum += loss.item() * len(batch)

            validation = evaluate_split(model, tokenizer, validation_samples, config.batch_size, policy, name="validation")
            macro_f1 = float(validation["macro_f1"])
            improved = macro_f1 > best_macro_f1 + config.min_delta
            if improved:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(model, output)
            else:
                stale_epochs += 1

            total = len(train_samples) or 1
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": loss_sum / total,
                    "val_accuracy": float(validation["accuracy"]),
                    "val_macro_f1": macro_f1,
                    "val_confidence_mean": float(validation["confidence"]["mean"]),  # type: ignore[index]
                    "val_escalation_rate": float(validation["confidence"]["escalation_rate"]),  # type: ignore[index]
                }
            )
            if epoch == 1 or epoch % config.log_every == 0 or improved or stale_epochs >= config.patience:
                marker = " *" if improved else ""
                print(
                    f"epoch={epoch:3d} loss={history[-1]['loss']:.4f} val_acc={validation['accuracy']:.4f} "
                    f"val_macro_f1={macro_f1:.4f} conf={history[-1]['val_confidence_mean']:.4f} "
                    f"escalated={history[-1]['val_escalation_rate']:.3f}{marker}"
                )
            if stale_epochs >= config.patience:
                print(f"early_stop epoch={epoch} patience={config.patience}")
                break

    load_checkpoint(model, output)
    log: dict[str, object] = {
        "config": {**asdict(config), "best_epoch": best_epoch, "parameters": sum(parameter.numel() for parameter in parameters)},
        "device": str(parameters[0].device),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output.resolve()),
        "policy": policy.name,
        "train": describe(train_samples, "train"),
        "validation": evaluate_split(model, tokenizer, validation_samples, config.batch_size, policy, name="validation"),
        "history": history,
    }
    return model, log


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minijev.training", description="Train the MiniJev decision model")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=DATA_DIR / "val.jsonl")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--embed-dim", type=positive_int, default=64)
    parser.add_argument("--hidden-dim", type=positive_int, default=64)
    parser.add_argument("--encoder-type", choices=("pooled", "transformer"), default="pooled")
    parser.add_argument("--num-layers", type=positive_int, default=1)
    parser.add_argument("--num-heads", type=positive_int, default=4)
    parser.add_argument("--ffn-dim", type=positive_int, default=128)
    parser.add_argument(
        "--no-preserve-tail",
        action="store_true",
        help="truncate long UTF-8 byte sequences from the end instead of retaining head and tail",
    )
    parser.add_argument("--patience", type=positive_int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--log-every", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--no-agreement-weights",
        action="store_true",
        help="ignore the annotator-agreement label instead of weighting rows by it",
    )
    parser.add_argument("--measure", default=DEFAULT_MEASURE, help="confidence measure for the decision policy")
    parser.add_argument("--caution", type=float, default=1.0, help="scale every route gate; >1 escalates more often")
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.0,
        help="weight on the deciding-slot auxiliary loss; 0 leaves the model without aux heads",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="enable turn separators so each sample's preceding turns are encoded before its request",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .data import load_jsonl

    args = build_parser().parse_args(argv)
    train_samples = load_jsonl(args.train)
    validation_samples = load_jsonl(args.validation)
    test_samples = load_jsonl(args.test)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        preserve_tail=not args.no_preserve_tail,
        patience=args.patience,
        min_delta=args.min_delta,
        log_every=args.log_every,
        seed=args.seed,
        weight_by_agreement=not args.no_agreement_weights,
        label_smoothing=args.label_smoothing,
        use_context=args.context,
        aux_weight=args.aux_weight,
    )
    policy = RiskLadderPolicy(measure=args.measure).scaled(args.caution)

    print(f"train={len(train_samples)} validation={len(validation_samples)} test={len(test_samples)}")
    model, log = fit(train_samples, validation_samples, config, output=args.output, policy=policy)
    # Rebuild the same config the checkpoint carries, so evaluation tokenises exactly the
    # way training did. A context model and a plain model read different bytes.
    tokenizer = ByteTokenizer(model.config)

    log["test"] = evaluate_split(model, tokenizer, test_samples, config.batch_size, policy, name="test")
    log["test"]["majority_baseline"] = majority_baseline(test_samples, training_samples=train_samples)
    log["test"]["majority_baseline_supported"] = majority_baseline(
        test_samples, labels=DEFAULT_LABELS, training_samples=train_samples, scope_to_labels=True
    )

    text = json.dumps(log, ensure_ascii=False, indent=2)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
