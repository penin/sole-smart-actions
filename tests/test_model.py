import copy

import numpy as np
import pandas as pd
import pytest

from sole_intent import train as train_mod
from sole_intent.train import (build_pipeline, fit_pipeline, load_bundle, param_grid, reorder_proba,
                               save_bundle, tune_hyperparameters)

EDGE_CASES = ["", "   ", "!!!???", "12345", "$$$", "zzqx flurble unseenword", "SEND $50 TO JONE",
              "send $50 to jone", "Pay My EFL Bill", "x" * 500]


def test_probability_shape_and_range(fitted_pipeline, splits, labels):
    proba = reorder_proba(fitted_pipeline, splits.test["text"], labels)
    assert proba.shape == (len(splits.test), len(labels))
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_class_order_is_canonical(fitted_pipeline, splits, labels):
    assert set(fitted_pipeline.classes_) == set(labels)
    proba = reorder_proba(fitted_pipeline, splits.validation["text"], labels)
    native = fitted_pipeline.predict(list(splits.validation["text"]))
    assert list(np.asarray(labels, dtype=object)[proba.argmax(axis=1)]) == list(native)


def test_edge_cases_produce_valid_probabilities(fitted_pipeline, labels):
    proba = reorder_proba(fitted_pipeline, EDGE_CASES, labels)
    assert proba.shape == (len(EDGE_CASES), len(labels))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    # capitalization must not change the prediction
    np.testing.assert_allclose(proba[6], proba[7])
    # empty, whitespace and punctuation-only inputs all reduce to the same empty bag of words
    np.testing.assert_allclose(proba[0], proba[1])
    np.testing.assert_allclose(proba[0], proba[2])


def test_serialization_roundtrip(fitted_pipeline, splits, labels, tmp_path):
    path = save_bundle(fitted_pipeline, tmp_path / "m.joblib", class_order=labels, threshold=0.42,
                       metadata={"note": "test"})
    assert path.exists() and path.with_suffix(".metadata.json").exists()
    bundle = load_bundle(path)
    assert bundle["class_order"] == labels and bundle["rejection_threshold"] == 0.42
    texts = list(splits.test["text"]) + EDGE_CASES
    np.testing.assert_array_equal(reorder_proba(fitted_pipeline, texts, labels),
                                  reorder_proba(bundle["pipeline"], texts, labels))
    assert list(fitted_pipeline.predict(texts)) == list(bundle["pipeline"].predict(texts))


def test_load_bundle_rejects_incomplete(tmp_path):
    import joblib
    joblib.dump({"pipeline": None}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        load_bundle(tmp_path / "bad.joblib")


def test_fit_rejects_missing_classes(cfg, labels, default_params, splits):
    subset = splits.train[splits.train["intent"] != "CHECK_SPENDING"]
    with pytest.raises(ValueError, match="do not match"):
        fit_pipeline(build_pipeline(default_params, cfg), subset["text"], subset["intent"], labels)


def test_training_is_deterministic(cfg, labels, default_params, splits):
    a, _ = fit_pipeline(build_pipeline(default_params, cfg), splits.train["text"], splits.train["intent"], labels)
    b, _ = fit_pipeline(build_pipeline(default_params, cfg), splits.train["text"], splits.train["intent"], labels)
    np.testing.assert_array_equal(a.named_steps["clf"].coef_, b.named_steps["clf"].coef_)


def test_param_grid_covers_required_settings(cfg):
    grid = param_grid(cfg)
    assert {g["ngram_range"] for g in grid} == {(1, 1), (1, 2)}
    assert {g["C"] for g in grid} == {0.1, 1.0, 10.0}
    assert len(grid) == 12


def test_tuning_never_sees_test_data(cfg, labels, splits, monkeypatch):
    """Spy on every fit during tuning: only training rows may be fitted, and the test split is
    never passed to the tuner at all."""
    seen_fit, seen_scored = [], []
    real_fit = train_mod.fit_pipeline
    real_reorder = train_mod.reorder_proba

    def spy_fit(pipeline, texts, y, class_order):
        seen_fit.append(list(texts))
        return real_fit(pipeline, texts, y, class_order)

    def spy_reorder(pipeline, texts, class_order):
        seen_scored.append(list(texts))
        return real_reorder(pipeline, texts, class_order)

    monkeypatch.setattr(train_mod, "fit_pipeline", spy_fit)
    monkeypatch.setattr(train_mod, "reorder_proba", spy_reorder)
    small = copy.deepcopy(cfg)
    small["tuning"]["grid"] = {"ngram_range": [[1, 1]], "C": [0.1, 1.0], "binary": [False]}
    results, best = tune_hyperparameters(splits.train, splits.validation, small, labels)

    train_texts = list(splits.train["text"])
    assert all(t == train_texts for t in seen_fit)
    assert all(t == list(splits.validation["text"]) for t in seen_scored)
    test_only = set(splits.test["text"]) - set(splits.train["text"]) - set(splits.validation["text"])
    assert not test_only & {x for t in seen_fit + seen_scored for x in t}
    assert results.iloc[0]["validation_macro_f1"] == results["validation_macro_f1"].max()
    assert best["C"] == results.iloc[0]["C"]


def test_tuning_signature_has_no_test_argument():
    import inspect
    params = inspect.signature(tune_hyperparameters).parameters
    assert "test" not in params and list(params)[:2] == ["train", "validation"]
