from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor, dtypes

from .config import JevConfig


@dataclass(frozen=True)
class EncodedText:
    input_ids: list[int]
    attention_mask: list[int]


class ByteTokenizer:
    """A deterministic UTF-8 tokenizer with no vocabulary training step."""

    def __init__(self, config: JevConfig | None = None):
        self.config = config or JevConfig()

    def encode(self, text: str) -> EncodedText:
        raw = list(text.encode("utf-8"))
        if len(raw) > self.config.max_seq_len:
            if self.config.preserve_tail:
                # Literary and document prompts put the decisive task at the end.
                # Keeping both ends preserves the state cue and the requested action.
                head = self.config.max_seq_len // 2
                raw = raw[:head] + raw[-(self.config.max_seq_len - head) :]
            else:
                raw = raw[: self.config.max_seq_len]
        ids = [byte + 1 for byte in raw]
        mask = [1] * len(ids)
        padding = self.config.max_seq_len - len(ids)
        return EncodedText(ids + [self.config.pad_id] * padding, mask + [0] * padding)

    def batch(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        encoded = [self.encode(text) for text in texts]
        ids = Tensor([item.input_ids for item in encoded], dtype=dtypes.int32)
        mask = Tensor([item.attention_mask for item in encoded], dtype=dtypes.float32)
        return ids, mask
