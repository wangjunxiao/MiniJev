"""Explicit model shapes and filesystem layout. Data only, no behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import ROUTE_ORDER

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
EXTERNAL_DIR = DATA_DIR / "external"

DEFAULT_CHECKPOINT = MODEL_DIR / "minijev_phase1.safetensors"
DEFAULT_METRICS = MODEL_DIR / "minijev_phase1_metrics.json"


@dataclass(frozen=True)
class JevConfig:
    """Configuration kept small and explicit so shapes are easy to inspect."""

    vocab_size: int = 257  # 0 is padding; 1..256 are byte values
    max_seq_len: int = 128
    embed_dim: int = 64
    hidden_dim: int = 64
    num_routes: int = len(ROUTE_ORDER)
    pad_id: int = 0
    preserve_tail: bool = True
    encoder_type: str = "pooled"
    num_layers: int = 1
    num_heads: int = 4
    ffn_dim: int = 128
    turn_sep_id: int = -1
    """Turn-boundary id, or -1 for a single-turn checkpoint.

    -1 keeps `vocab_size=257` and is what every pre-context checkpoint carries, so those
    still load. Enabling it means `vocab_size=258` and travels in the sidecar.
    """

    aux_slots: bool = False
    """Whether the model carries auxiliary heads for the deciding slots.

    Part of the checkpoint shape, so a checkpoint states whether it was trained with
    them. The route decision never reads the heads, so an aux checkpoint and a plain one
    answer identically once loaded; only training differs.
    """

    def __post_init__(self) -> None:
        if self.encoder_type not in {"pooled", "transformer"}:
            raise ValueError("encoder_type must be 'pooled' or 'transformer'")
        if self.num_layers <= 0 or self.num_heads <= 0 or self.ffn_dim <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.turn_sep_id >= self.vocab_size:
            raise ValueError("turn_sep_id must be inside the embedding table; bump vocab_size")

    @property
    def uses_context(self) -> bool:
        """Whether this config was built to read turn separators."""
        return self.turn_sep_id >= 0


def context_config(**overrides: Any) -> JevConfig:
    """A config with turn separators enabled, for context-aware training."""
    from .tokenizer import TURN_SEPARATOR_ID

    overrides.setdefault("vocab_size", TURN_SEPARATOR_ID + 1)
    overrides.setdefault("turn_sep_id", TURN_SEPARATOR_ID)
    return JevConfig(**overrides)
