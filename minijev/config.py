"""Explicit model shapes and filesystem layout. Data only, no behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    def __post_init__(self) -> None:
        if self.encoder_type not in {"pooled", "transformer"}:
            raise ValueError("encoder_type must be 'pooled' or 'transformer'")
        if self.num_layers <= 0 or self.num_heads <= 0 or self.ffn_dim <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
