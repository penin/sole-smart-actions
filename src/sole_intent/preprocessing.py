"""Minimal, reproducible text preprocessing and the Bag-of-Words vectorizer factory.

Deliberately conservative: Unicode normalization, lowercasing and whitespace collapsing only.
No stemming, lemmatization or stop-word removal, and numbers / currency symbols are kept,
because "send 50" vs "send 500" or "$" may carry meaning for a wallet assistant.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_TOKEN_PATTERN = r"(?u)\$|\b\w+\b"
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_text(text, unicode_form: str = "NFKC", lowercase: bool = True,
                   collapse_whitespace: bool = True) -> str:
    """Normalize one utterance. Missing values become the empty string."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = unicodedata.normalize(unicode_form, str(text))
    if lowercase:
        text = text.lower()
    if collapse_whitespace:
        text = _WHITESPACE.sub(" ", text).strip()
    return text


def tokenize(text: str, token_pattern: str = DEFAULT_TOKEN_PATTERN) -> list[str]:
    """Tokenize already-normalized text exactly as the vectorizer does."""
    return re.findall(token_pattern, text)


def canonical_key(text) -> str:
    """Punctuation-insensitive key used for duplicate / leakage auditing (not for modelling)."""
    return " ".join(tokenize(normalize_text(text)))


def template_key(text) -> str:
    """canonical_key with numbers masked, so "send 50 to jone" == "send 500 to jone"."""
    return _NUMBER.sub("<num>", canonical_key(text))


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Stateless pipeline step wrapping :func:`normalize_text`."""

    def __init__(self, unicode_form: str = "NFKC", lowercase: bool = True,
                 collapse_whitespace: bool = True):
        self.unicode_form = unicode_form
        self.lowercase = lowercase
        self.collapse_whitespace = collapse_whitespace

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable) -> list[str]:
        return [normalize_text(t, self.unicode_form, self.lowercase, self.collapse_whitespace)
                for t in X]


def build_vectorizer(ngram_range=(1, 1), binary: bool = False,
                     token_pattern: str = DEFAULT_TOKEN_PATTERN) -> CountVectorizer:
    # Lowercasing is handled by TextNormalizer so the vectorizer sees normalized text only.
    return CountVectorizer(lowercase=False, token_pattern=token_pattern,
                           ngram_range=tuple(ngram_range), binary=binary)
