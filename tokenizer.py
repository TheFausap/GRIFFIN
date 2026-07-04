"""
Tokenizers for the Griffin / CG-LRU experiments.
================================================

Two interchangeable tokenizers behind one interface:

    ByteTokenizer  -- UTF-8 bytes. Fixed vocab of 256, no vocab-building, no
                      rare-Unicode problem. Sequences run ~1.0-1.5x longer than
                      char-level for Latin text (more for other scripts). This
                      is the regime where Griffin's O(T) recurrence + O(T*window)
                      local attention are supposed to pay off vs a Transformer.

    CharTokenizer  -- one id per distinct character in the corpus. The original
                      baseline; vocab size depends on the data.

Each tokenizer exposes:
    .kind         : "byte" | "char"
    .vocab_size   : int
    .encode(str)  -> list[int]
    .decode(ids)  -> str          (byte-level uses errors="replace" so invalid
                                    UTF-8 from an untrained sampler is visible,
                                    not crashing)
    .state()      -> dict          (json/pickle-able; goes in the checkpoint)

Rebuild a tokenizer from a checkpoint with tokenizer_from_state(state_dict).
"""

from __future__ import annotations


class ByteTokenizer:
    kind = "byte"
    vocab_size = 256

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        # Mask to a byte range in case a logit argmax ever lands out of [0,255].
        raw = bytes(int(i) & 0xFF for i in ids)
        return raw.decode("utf-8", errors="replace")

    def state(self) -> dict:
        return {"kind": "byte"}


class CharTokenizer:
    kind = "char"

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def state(self) -> dict:
        return {"kind": "char", "chars": self.chars}


def build_tokenizer(kind: str, text: str = "") -> "ByteTokenizer | CharTokenizer":
    """Fresh tokenizer. Byte needs nothing; char derives its vocab from `text`."""
    if kind == "byte":
        return ByteTokenizer()
    if kind == "char":
        return CharTokenizer(sorted(set(text)))
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


def tokenizer_from_state(state: dict) -> "ByteTokenizer | CharTokenizer":
    """Reconstruct a tokenizer saved in a checkpoint. Backward-compatible with
    older checkpoints that stored a raw {char: id} map under 'stoi'."""
    if state.get("kind") == "byte":
        return ByteTokenizer()
    if state.get("kind") == "char":
        return CharTokenizer(state["chars"])
    raise ValueError(f"unrecognized tokenizer state: {list(state)[:5]}")
