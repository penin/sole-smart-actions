"""Common model interface, artifact loading, published-result schemas and reproducible lightweight training."""

import numpy as np
import pandas as pd
import pytest

from sole_smart_actions import MODEL_NAMES
from sole_smart_actions.config import load_config, resolve
from sole_smart_actions.data import load_splits
from sole_smart_actions.reproducibility import compare_results, verify_result_schema
from sole_smart_actions.training import artifact_dir, load_trained, train_model, verify_artifact_hashes
from sole_smart_actions.utils import load_json

LABELS = load_config()["labels"]
TEXTS = ["Send $50 to Jone", "Pay my EFL bill", "What's the weather tomorrow?", "", "$!?", "x" * 600, "Sénd mōney!!"]


def available(name):
    return (artifact_dir(name) / "model_card.json").exists()


models = [pytest.param(m, marks=pytest.mark.skipif(not available(m), reason=f"{m} not trained")) for m in MODEL_NAMES]


@pytest.fixture(scope="module")
def loaded():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_trained(name)
        return cache[name]
    return get


@pytest.mark.parametrize("name", models)
def test_common_interface_shapes_order_and_normalisation(loaded, name):
    model, card = loaded(name)
    p = model.predict_proba(TEXTS)
    assert p.shape == (len(TEXTS), 9) and model.class_order == LABELS
    assert np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()
    assert model.predict(TEXTS) == [LABELS[i] for i in p.argmax(1)]
    assert model.predict_proba([]).shape == (0, 9)
    meta = model.get_model_metadata()
    assert meta["name"] == name and meta["class_order"] == LABELS and meta["n_parameters"] > 0


@pytest.mark.parametrize("name", models)
def test_artifact_is_primary_protocol_and_hashes_match(loaded, name):
    _, card = loaded(name)
    assert card["protocol"] == "primary" and card["trained_on"] == ["train"] and card["primary_comparison"]
    assert card["dataset"]["sha256"] == load_config()["dataset"]["sha256"]
    assert verify_artifact_hashes(artifact_dir(name))["ok"]


@pytest.mark.parametrize("name", models)
def test_published_results_have_standard_schema(name):
    if not (resolve("results") / name / "experiment_metadata.json").exists():
        pytest.skip("not evaluated")
    report = verify_result_schema(resolve("results"), name)
    assert report["ok"], report["problems"]
    meta = load_json(resolve("results") / name / "experiment_metadata.json")
    assert meta["headline"]["raw_test_n"] == 270
    preds = pd.read_csv(resolve("results") / name / "test_predictions.csv")
    assert meta["headline"]["raw_test_correct"] == int(preds["correct_raw"].sum())


@pytest.mark.parametrize("name", [pytest.param(m, marks=pytest.mark.skipif(not available(m), reason="not trained"))
                                  for m in ("bow", "tfidf")])
def test_lightweight_models_retrain_bit_for_bit(tmp_path, loaded, name):
    """Reproducible training: retraining from the saved configuration gives identical probabilities."""
    out = train_model(name, artifacts_root=tmp_path)
    fresh, card = load_trained(name, out)
    committed, ccard = loaded(name)
    splits, _ = load_splits(load_config())
    for _, df in splits.items():
        assert np.array_equal(fresh.predict_proba(df["text"]), committed.predict_proba(df["text"]))
    assert card["selection"]["params"] == ccard["selection"]["params"]


def test_bow_primary_equals_historical_v5_model(loaded):
    if not available("bow"):
        pytest.skip("bow not trained")
    from sole_intent.train import load_bundle, reorder_proba
    hist = load_bundle(resolve("artifacts/history/bow_v5/bow_logreg_v5_tuned_v1.0.0.joblib"))
    model, _ = loaded("bow")
    splits, _ = load_splits(load_config())
    assert np.array_equal(model.predict_proba(splits.test["text"]),
                          reorder_proba(hist["pipeline"], splits.test["text"], hist["class_order"]))


def test_compare_results_exact_mode(tmp_path):
    if not (resolve("results") / "bow" / "test_predictions.csv").exists():
        pytest.skip("bow not evaluated")
    import shutil
    shutil.copytree(resolve("results") / "bow", tmp_path / "bow")
    assert compare_results(resolve("results"), tmp_path, "bow")["ok"]
    df = pd.read_csv(tmp_path / "bow" / "test_predictions.csv")
    df.loc[0, "prob_TRANSFER_MONEY"] += 1e-6
    df.to_csv(tmp_path / "bow" / "test_predictions.csv", index=False)
    assert not compare_results(resolve("results"), tmp_path, "bow")["ok"]  # exact mode tolerates nothing


def test_missing_artifact_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m scripts.train --model distilbert"):
        load_trained("distilbert", tmp_path)
