"""Bag of Words + Logistic Regression adapter around the original ``sole_intent`` implementation.

Preprocessing, vectorizer, classifier, grid and selection rule are the existing project's code, unchanged.
The artifact is the same joblib bundle format as the historical V4/V5 models, so the demo can explain it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sole_intent.train import build_pipeline, fit_pipeline, reorder_proba, tune_hyperparameters

from ..base import IntentModel

ARTIFACT = "model.joblib"


def _legacy_cfg(cfg: dict) -> dict:
    """The configuration shape expected by sole_intent (config/history/bow_v5.yaml)."""
    return {"preprocessing": cfg["preprocessing"],
            "model": {"solver": cfg["estimator"]["solver"], "max_iter": cfg["estimator"]["max_iter"]},
            "tuning": cfg["tuning"], "experiment": {"random_seed": cfg["model"]["seed"]}}


class BowModel(IntentModel):
    name = "bow"
    framework = "scikit-learn"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.pipeline = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "BowModel":
        legacy = _legacy_cfg(self.cfg)
        table, best = tune_hyperparameters(train, validation, legacy, self.class_order)
        self.search_table = table
        self.pipeline, self.training_seconds = fit_pipeline(build_pipeline(best, legacy), train["text"],
                                                            train["intent"], self.class_order)
        self.selection = {"params": {"ngram_range": list(best["ngram_range"]), "C": best["C"],
                                     "binary": best["binary"]},
                          "rule": "highest validation macro F1; ties -> lowest validation log loss",
                          "validation_macro_f1": float(table.iloc[0]["validation_macro_f1"]),
                          "validation_log_loss": float(table.iloc[0]["validation_log_loss"]),
                          "n_configurations": int(len(table))}
        self.trained_on = ["train"]
        return self

    def refit(self, data: pd.DataFrame) -> "BowModel":
        p = self.selection["params"]
        params = {"ngram_range": tuple(p["ngram_range"]), "C": p["C"], "binary": p["binary"]}
        self.pipeline, self.training_seconds = fit_pipeline(build_pipeline(params, _legacy_cfg(self.cfg)),
                                                            data["text"], data["intent"], self.class_order)
        self.trained_on = sorted(set(data["split"]))
        return self

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        return reorder_proba(self.pipeline, texts, self.class_order)

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT
        # Same keys as sole_intent.train.save_bundle; the threshold lives in the evaluation results.
        joblib.dump({"pipeline": self.pipeline, "class_order": self.class_order, "rejection_threshold": None,
                     "metadata": {"selection": self.selection}}, path)
        return [path]

    @classmethod
    def load(cls, directory: Path, cfg: dict) -> "BowModel":
        m = cls(cfg)
        bundle = joblib.load(Path(directory) / ARTIFACT)
        if list(bundle["class_order"]) != m.class_order:
            raise ValueError("BoW artifact class order differs from config")
        m.pipeline = bundle["pipeline"]
        m.selection = bundle.get("metadata", {}).get("selection", {})
        return m

    def n_parameters(self) -> int | None:
        clf = self.pipeline.named_steps["clf"]
        return int(clf.coef_.size + clf.intercept_.size)

    def get_model_metadata(self) -> dict:
        meta = super().get_model_metadata()
        meta["vocabulary_size"] = len(self.pipeline.named_steps["bow"].vocabulary_)
        return meta
