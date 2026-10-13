"""The tinygrad network and its checkpoint format.

This module only does arithmetic: bytes in, a score vector out.

There is deliberately **one** head. An earlier version had a second head that
predicted the dataset's annotator-agreement label and called its output
"confidence". That was a category error, and it failed in the way that error
always fails: the labels were almost constant, so the head learned the constant
and its output had a spread of 0.017 across 226 test rows. Confidence is a
statistic of the route distribution, computed in `minijev.confidence`, because the
route head still discriminates even when a regression target does not.

It also does not know what a route means, so the same forward pass serves
supervised training, inference and any future policy-gradient step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tinygrad import Tensor
from tinygrad.nn import Embedding, LayerNorm, Linear
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save

from .config import JevConfig
from .frames import slot_vocabulary


@dataclass
class JevOutputs:
    """Raw model outputs before any policy converts them into a route.

    `slot_logits` is present only when the config enabled auxiliary supervision. The
    inference path never reads it: the decision comes from `route_logits` alone, so the
    auxiliary heads can be ignored at deployment without changing a single decision.
    """

    route_logits: Tensor
    slot_logits: dict[str, Tensor] | None = None


class TransformerBlock:
    """Small bidirectional self-attention block for the context ablation."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = Linear(embed_dim, embed_dim * 3)
        self.out = Linear(embed_dim, embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn1 = Linear(embed_dim, ffn_dim)
        self.ffn2 = Linear(ffn_dim, embed_dim)

    def __call__(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        batch, sequence, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, sequence, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim**0.5)
        key_mask = (1.0 - attention_mask).reshape(batch, 1, 1, sequence)
        scores = scores + key_mask * -1e9
        weights = scores.softmax(axis=-1)
        attended = (weights @ value).permute(0, 2, 1, 3).reshape(batch, sequence, self.embed_dim)
        hidden = self.norm1(hidden + self.out(attended))
        hidden = self.norm2(hidden + self.ffn2(self.ffn1(hidden).relu()))
        return hidden


class TinyJev:
    """Pooled baseline or optional tiny Transformer encoder plus a route head."""

    def __init__(self, config: JevConfig | None = None):
        self.config = config or JevConfig()
        self.embedding = Embedding(self.config.vocab_size, self.config.embed_dim)
        self.position_embedding = Embedding(self.config.max_seq_len, self.config.embed_dim) if self.config.encoder_type == "transformer" else None
        self.transformer = (
            [TransformerBlock(self.config.embed_dim, self.config.num_heads, self.config.ffn_dim) for _ in range(self.config.num_layers)]
            if self.config.encoder_type == "transformer"
            else []
        )
        self.encoder = Linear(self.config.embed_dim, self.config.hidden_dim) if self.config.encoder_type == "pooled" else None
        # Keep this projection as part of the model. Creating it in __call__ would
        # reset its weights on every batch and omit it from checkpoints.
        self.transformer_projection = Linear(self.config.embed_dim, self.config.hidden_dim) if self.config.encoder_type == "transformer" else None
        # The head reads the operating point as well as the query when the config asks for
        # one, so the same weights serve a whole family of quality-versus-cost trade-offs.
        self.route_head = Linear(self.config.hidden_dim + self.config.preference_dim, self.config.num_routes)
        # Auxiliary heads, one per deciding slot. They are small (a few hundred
        # parameters in total) and exist only to shape the shared representation during
        # training; the decision path never reads them.
        self.slot_heads = (
            {slot: Linear(self.config.hidden_dim, len(vocabulary)) for slot, vocabulary in slot_vocabulary().items()} if self.config.aux_slots else {}
        )

    def __call__(self, input_ids: Tensor, attention_mask: Tensor, preference: Tensor | None = None) -> JevOutputs:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have shape [batch, sequence]")

        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        if self.config.encoder_type == "transformer":
            positions = Tensor.arange(input_ids.shape[1], dtype=input_ids.dtype).unsqueeze(0)
            hidden = hidden + self.position_embedding(positions)
            for block in self.transformer:
                hidden = block(hidden, attention_mask)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).maximum(1.0)
            shared = self.transformer_projection(pooled).relu()
        else:
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).maximum(1.0)
            shared = self.encoder(pooled).relu()

        head_input = self._condition(shared, preference)
        slot_logits = {slot: head(shared) for slot, head in self.slot_heads.items()} if self.slot_heads else None
        return JevOutputs(route_logits=self.route_head(head_input), slot_logits=slot_logits)

    def _condition(self, shared: Tensor, preference: Tensor | None) -> Tensor:
        """Attach the operating point to the pooled representation.

        A config that declares a preference width but receives no vector is a wiring bug, not
        a request for the default: silently substituting one would make an unconditional run
        look conditional in the logs. The reverse is also refused, so a checkpoint cannot be
        fed a preference it has no weights for.
        """
        if not self.config.uses_preference:
            if preference is not None:
                raise ValueError("this model was built without preference conditioning; rebuild it with preference_dim > 0")
            return shared
        if preference is None:
            raise ValueError(f"this model expects a preference vector of width {self.config.preference_dim}")
        if preference.ndim != 2 or preference.shape[1] != self.config.preference_dim:
            raise ValueError(f"preference must have shape [batch, {self.config.preference_dim}], got {tuple(preference.shape)}")
        if preference.shape[0] != shared.shape[0]:
            raise ValueError("preference must have one row per sample")
        return shared.cat(preference, dim=1)


def config_path(checkpoint: str | Path) -> Path:
    """Sidecar file holding the shapes a checkpoint was trained with."""
    return Path(checkpoint).with_suffix(".config.json")


def save_checkpoint(model: object, path: str | Path) -> None:
    """Save model tensors in tinygrad's portable safetensors format.

    A safetensors file stores tensors and nothing else, so the shapes have to be
    written separately. Without this sidecar a checkpoint trained at a non-default
    width cannot be loaded again -- the loader would build a network of the wrong
    shape and fail.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_save(get_state_dict(model), str(destination))

    config = getattr(model, "config", None)
    if isinstance(config, JevConfig):
        config_path(destination).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_checkpoint_config(path: str | Path) -> JevConfig | None:
    """Read the sidecar written by `save_checkpoint`, or None when it is absent."""
    sidecar = config_path(path)
    if not sidecar.exists():
        return None
    return JevConfig(**json.loads(sidecar.read_text(encoding="utf-8")))


def load_checkpoint(model: object, path: str | Path) -> None:
    load_state_dict(model, safe_load(str(path)), verbose=False)
