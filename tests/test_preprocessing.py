import numpy as np
import pytest

from sole_intent.preprocessing import (TextNormalizer, build_vectorizer, canonical_key, normalize_text,
                                       template_key, tokenize)


@pytest.mark.parametrize("raw, expected", [
    ("Send $50 to Jone", "send $50 to jone"),
    ("  pay   my\tEFL\nbill ", "pay my efl bill"),
    ("ＳＥＮＤ　５０", "send 50"),          # full-width characters -> NFKC
    ("", ""),
    ("   ", ""),
    (None, ""),
    (float("nan"), ""),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_is_idempotent(splits):
    for t in splits.train["text"].head(200):
        once = normalize_text(t)
        assert normalize_text(once) == once


def test_numbers_and_currency_preserved():
    assert tokenize(normalize_text("Send $500 to Jone")) == ["send", "$", "500", "to", "jone"]
    assert tokenize("a 5 b") == ["a", "5", "b"]  # single characters are kept


def test_transformer_matches_function():
    texts = ["Hello  World", "SEND $5", None]
    assert TextNormalizer().fit(texts).transform(texts) == [normalize_text(t) for t in texts]


def test_audit_keys():
    assert canonical_key("Send $50 to Jone?") == canonical_key("send $50  to jone")
    assert template_key("send 50 to jone") == template_key("send 500 to jone")
    assert template_key("send 50 to jone") != template_key("send 50 to mere")


def test_vectorizer_settings():
    v = build_vectorizer((1, 2), binary=True)
    assert v.ngram_range == (1, 2) and v.binary is True and v.lowercase is False


def test_vocabulary_fitted_only_on_training(fitted_pipeline, splits):
    vocab = set(fitted_pipeline.named_steps["bow"].vocabulary_)
    train_tokens = {tok for t in splits.train["text"] for tok in tokenize(normalize_text(t))}
    held_out_tokens = {tok for f in (splits.validation, splits.test, splits.unknown)
                       for t in f["text"] for tok in tokenize(normalize_text(t))}
    held_out_only = held_out_tokens - train_tokens
    assert held_out_only, "fixture sanity: held-out data should contain unseen words"
    assert vocab <= train_tokens
    assert not vocab & held_out_only


def test_validation_and_test_use_training_vocabulary(fitted_pipeline, splits):
    vec = fitted_pipeline.named_steps["bow"]
    size = len(vec.vocabulary_)
    normalized = fitted_pipeline.named_steps["normalize"].transform(splits.test["text"])
    X = vec.transform(normalized)
    assert X.shape == (len(splits.test), size)
    assert len(vec.vocabulary_) == size  # transform must not grow the vocabulary
    assert np.all(X.data > 0)
