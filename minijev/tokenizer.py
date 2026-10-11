from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tinygrad import Tensor, dtypes

from .config import JevConfig

TURN_SEPARATOR_ID = 257
"""The id that marks a turn boundary, when the config enables one.

`vocab_size` is 257 without it (0 padding, 1..256 byte values), so enabling turn
separators means a 258-row embedding. That is why the flag lives in the config and
travels in the checkpoint sidecar: a single-turn checkpoint keeps 257 rows and still
loads, and a context checkpoint states that it needs the extra row.
"""


@dataclass(frozen=True)
class EncodedText:
    input_ids: list[int]
    attention_mask: list[int]


class ByteTokenizer:
    """A deterministic UTF-8 tokenizer with no vocabulary training step.

    It also encodes *turns*. A routing decision often depends on preceding state rather
    than on the request alone. Turns are joined with an explicit separator id so the
    boundary cannot be confused with content, and the request is placed last so
    `preserve_tail` keeps it.
    """

    def __init__(self, config: JevConfig | None = None):
        self.config = config or JevConfig()

    def encode(self, text: str) -> EncodedText:
        """Encode one single-turn request."""
        return self.encode_turns((text,))

    def encode_turns(self, turns: Sequence[str]) -> EncodedText:
        """Encode preceding turns followed by the request, in order.

        The final turn is the request; earlier turns are state. Empty turns are dropped
        rather than emitting a separator with nothing on one side of it. When every turn
        is empty the result is a fully padded row, which is what a single blank string
        encoded to before turns existed.
        """
        return self._finalize(self._turn_ids(turns))

    def _turn_ids(self, turns: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for turn in turns:
            if not turn.strip():
                continue
            if ids:
                if self.config.turn_sep_id >= 0:
                    ids.append(self.config.turn_sep_id)
                else:
                    # No separator configured: fall back to a newline byte, which is at
                    # least a visible boundary. A checkpoint trained without separators
                    # cannot use context anyway; this keeps it loadable rather than
                    # crashing, so the benchmark can report the collapse.
                    ids.extend(byte + 1 for byte in b"\n")
            ids.extend(byte + 1 for byte in turn.encode("utf-8"))
        return ids

    def _finalize(self, ids: list[int]) -> EncodedText:
        if len(ids) > self.config.max_seq_len:
            if self.config.preserve_tail:
                # Document prompts put the decisive task at the end. Keeping both ends
                # preserves the opening state cue and the requested action.
                head = self.config.max_seq_len // 2
                ids = ids[:head] + ids[-(self.config.max_seq_len - head) :]
            else:
                ids = ids[: self.config.max_seq_len]
        mask = [1] * len(ids)
        padding = self.config.max_seq_len - len(ids)
        return EncodedText(ids + [self.config.pad_id] * padding, mask + [0] * padding)

    def batch(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        return self._stack([self.encode(text) for text in texts])

    def batch_turns(self, turn_sequences: Sequence[Sequence[str]]) -> tuple[Tensor, Tensor]:
        """Batch one turn sequence per sample."""
        return self._stack([self.encode_turns(turns) for turns in turn_sequences])

    @staticmethod
    def _stack(encoded: list[EncodedText]) -> tuple[Tensor, Tensor]:
        if not encoded:
            raise ValueError("cannot batch an empty list")
        ids = Tensor([item.input_ids for item in encoded], dtype=dtypes.int32)
        mask = Tensor([item.attention_mask for item in encoded], dtype=dtypes.float32)
        return ids, mask
