import json
import sys

import numpy as np
import pytest

from sole_intent import predict as predict_mod
from sole_intent.predict import predict_texts
from sole_intent.rejection import UNKNOWN_INTENT
from sole_intent.train import save_bundle


@pytest.fixture
def bundle(fitted_pipeline, labels):
    return {"pipeline": fitted_pipeline, "class_order": labels, "rejection_threshold": 0.3, "metadata": {}}


def test_predict_output_structure(bundle, labels):
    out = predict_texts(["send $50 to jone", "who won the rugby"], bundle)
    assert len(out) == 2
    for r in out:
        assert set(r) == {"text", "predicted_intent", "final_intent", "rejected", "max_confidence",
                          "threshold", "probabilities", "application"}
        assert list(r["probabilities"]) == labels
        assert sum(r["probabilities"].values()) == pytest.approx(1.0)
        assert r["max_confidence"] == pytest.approx(max(r["probabilities"].values()))
        assert r["rejected"] == (r["max_confidence"] < 0.3) == (r["final_intent"] == UNKNOWN_INTENT)


def test_obvious_commands(bundle):
    out = {r["text"]: r for r in predict_texts(["send $50 to jone", "pay my efl bill"], bundle, threshold=0.0)}
    assert out["send $50 to jone"]["final_intent"] == "TRANSFER_MONEY"
    assert out["pay my efl bill"]["final_intent"] == "BILL_PAY"


def test_threshold_override(bundle):
    assert all(r["rejected"] for r in predict_texts(["send $50 to jone"], bundle, threshold=1.01))
    assert not any(r["rejected"] for r in predict_texts(["send $50 to jone"], bundle, threshold=0.0))


@pytest.mark.parametrize("text", ["", "   ", "?!.,", "0000", "qwertyuiop asdf", "PaY mY BiLl"])
def test_edge_case_inputs(bundle, text):
    r = predict_texts([text], bundle)[0]
    assert 0 <= r["max_confidence"] <= 1
    assert r["final_intent"] in set(bundle["class_order"]) | {UNKNOWN_INTENT}


def test_unseen_words_only_falls_back_to_prior(bundle):
    """A text with no known tokens gives the same distribution as an empty string."""
    a, b = predict_texts(["qwertyuiop zxcvbnm", ""], bundle)
    np.testing.assert_allclose(list(a["probabilities"].values()), list(b["probabilities"].values()))


def test_cli_end_to_end(fitted_pipeline, labels, tmp_path, monkeypatch, capsys):
    path = save_bundle(fitted_pipeline, tmp_path / "cli.joblib", class_order=labels, threshold=0.2, metadata={})
    monkeypatch.setattr(sys, "argv", ["predict", "--model", str(path), "--json", "send $20 to Mere"])
    predict_mod.main()
    result = json.loads(capsys.readouterr().out)
    assert result[0]["text"] == "send $20 to Mere" and result[0]["threshold"] == 0.2


def test_threshold_precedence_cli_over_config_over_bundle(fitted_pipeline, labels, tmp_path, monkeypatch, capsys):
    path = save_bundle(fitted_pipeline, tmp_path / "p.joblib", class_order=labels, threshold=0.2, metadata={})
    cfg = tmp_path / "inference.yaml"
    cfg.write_text(f"inference:\n  model_path: '{path.as_posix()}'\n  threshold: 0.35\n", encoding="utf-8")

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["predict", "--json", *extra, "send $20 to Mere"])
        predict_mod.main()
        return json.loads(capsys.readouterr().out)[0]

    assert run("--model", str(path))["threshold"] == 0.2                      # bundle
    assert run("--config", str(cfg))["threshold"] == 0.35                     # config file
    assert run("--config", str(cfg), "--threshold", "0.5")["threshold"] == 0.5  # CLI wins
    r = run("--config", str(cfg), "--threshold", "1.01")
    assert r["final_intent"] == UNKNOWN_INTENT and r["application"]["action"] == "FALLBACK_HELP_OR_CLARIFY"
    assert r["application"]["suggested_intent"] == r["predicted_intent"]
