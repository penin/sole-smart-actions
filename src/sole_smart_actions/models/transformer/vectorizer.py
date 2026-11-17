"""Pure-Python replacement for ``keras.layers.TextVectorization`` (TensorFlow-only preprocessing).

Matches the original notebook's settings: ``standardize="lower_and_strip_punctuation"``,
``split="whitespace"``, ``output_mode="int"``, ``max_tokens=2000``, ``output_sequence_length=16``.
Index 0 is padding and 1 is ``[UNK]``; the vocabulary is learned only from the fitting texts.
Frequency ties are broken alphabetically so the vocabulary is fully deterministic.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

# Keras' DEFAULT_STRIP_REGEX
_PUNCT = re.compile(r'[!"#$%&()\*\+,\-\./:;<=>?@\[\\\]^_`{|}~\']')
PAD, OOV = "", "[UNK]"


def standardize(text: str) -> str:
    return _PUNCT.sub("", str(text).lower())


def tokenize(text: str) -> list[str]:
    return standardize(text).split()


class TextVectorizer:
    def __init__(self, max_tokens: int = 2000, sequence_length: int = 16, vocabulary: list[str] | None = None):
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.vocabulary: list[str] = []
        self.index: dict[str, int] = {}
        if vocabulary is not None:
            self.set_vocabulary(vocabulary)

    def set_vocabulary(self, vocabulary: list[str]) -> None:
        if vocabulary[:2] != [PAD, OOV]:
            vocabulary = [PAD, OOV] + [t for t in vocabulary if t not in (PAD, OOV)]
        self.vocabulary = list(vocabulary)
        self.index = {t: i for i, t in enumerate(self.vocabulary)}

    def adapt(self, texts) -> "TextVectorizer":
        counts = Counter(tok for t in texts for tok in tokenize(t))
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.set_vocabulary([PAD, OOV] + [t for t, _ in ranked[: self.max_tokens - 2]])
        return self

    def __call__(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.sequence_length), dtype=np.int32)
        for i, t in enumerate(texts):
            ids = [self.index.get(tok, 1) for tok in tokenize(t)][: self.sequence_length]
            out[i, : len(ids)] = ids
        return out

    def unknown_share(self, texts) -> float:
        ids = self(texts)
        real = ids > 0
        return float((ids[real] == 1).mean()) if real.any() else 0.0
