"""TF-IDF + Logistic Regression, ported from the TF-IDF teammate's ``src/run_experiments.py``.

Unchanged from the original: the two representations (word TF-IDF; word 1-2-gram + char_wb n-gram TF-IDF
via FeatureUnion), sublinear tf, lowercase, the C grid, seed 42, ``max_iter=5000`` and validation-only
selection. Change: the original chose between the two representations with 5-fold CV on train+validation;
here the primary choice uses validation macro F1 only (ties -> first in grid order, the original tie rule).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from sole_intent.evaluate import multiclass_log_loss

from ..base import IntentModel

ARTIFACT = "model.joblib"


def build(variant: str, params: dict, seed: int, max_iter: int) -> Pipeline:
    """The teammate's ``build()``: same vectorizers and classifier."""
    if variant == "word":
        vec = TfidfVectorizer(lowercase=True, ngram_range=tuple(params["ngram"]), sublinear_tf=True)
    elif variant == "word_char":
        vec = FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, ngram_range=tuple(params["word_ngram"]), sublinear_tf=True)),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=tuple(params["char"]),
                                     sublinear_tf=True)),
        ])
    else:
        raise ValueError(variant)
    return Pipeline([("vec", vec), ("clf", LogisticRegression(C=params["C"], max_iter=max_iter, random_state=seed))])


def search_space(cfg: dict) -> list[tuple[str, dict]]:
    v = cfg["variants"]
    space = []
    for name in cfg.get("variant_names") or list(v):
        if name == "word":
            space += [("word", {"ngram": list(g), "C": c}) for g in v["word"]["ngram_range"] for c in cfg["C_grid"]]
        elif name == "word_char":
            space += [("word_char", {"word_ngram": list(v["word_char"]["word_ngram_range"]), "char": list(g), "C": c})
                      for g in v["word_char"]["char_ngram_range"] for c in cfg["C_grid"]]
    return space


def _clean(texts) -> list[str]:
    return [str(t).strip() for t in texts]  # the teammate's load_data(): strip only


class TfidfModel(IntentModel):
    name = "tfidf"
    framework = "scikit-learn"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.pipeline: Pipeline | None = None
        self.variant: str | None = None
        self.params: dict | None = None

    def _new(self, variant, params):
        return build(variant, params, self.cfg["model"]["seed"], self.cfg["estimator"]["max_iter"])

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "TfidfModel":
        import time
        xtr, ytr = _clean(train["text"]), train["intent"].tolist()
        xva, yva = _clean(validation["text"]), validation["intent"].tolist()
        rows = []
        for order, (variant, params) in enumerate(search_space(self.cfg)):
            m = self._new(variant, params).fit(xtr, ytr)
            pred = m.predict(xva)
            proba = m.predict_proba(xva)[:, [list(m.classes_).index(c) for c in self.class_order]]
            rows.append({"grid_order": order, "variant": variant,
                         "variant_label": self.cfg["variants"][variant]["label"], "params": params,
                         "validation_macro_f1": f1_score(yva, pred, average="macro"),
                         "validation_accuracy": accuracy_score(yva, pred),
                         "validation_log_loss": multiclass_log_loss(yva, proba, self.class_order),
                         "train_accuracy": accuracy_score(ytr, m.predict(xtr))})
        table = pd.DataFrame(rows)
        table["validation_macro_f1_r"] = table["validation_macro_f1"].round(12)
        # The teammate's rule: best validation macro F1, ties -> first in grid order (simpler model, smaller C).
        table = table.sort_values(["validation_macro_f1_r", "grid_order"],
                                  ascending=[False, True]).drop(columns="validation_macro_f1_r")
        table.insert(0, "rank", range(1, len(table) + 1))
        best = table.iloc[0]
        self.variant, self.params = best["variant"], dict(best["params"])
        start = time.perf_counter()
        self.pipeline = self._new(self.variant, self.params).fit(xtr, ytr)
        self.training_seconds = time.perf_counter() - start
        self.search_table = table.assign(params=table["params"].map(str))
        per_variant = (table.groupby("variant_label", sort=False).first()
                       [["params", "validation_macro_f1", "validation_accuracy"]].reset_index())
        self.selection = {"variant": self.variant, "variant_label": self.cfg["variants"][self.variant]["label"],
                          "params": self.params,
                          "rule": "highest validation macro F1 across both representations; ties -> first in "
                                  "grid order (simpler model, smaller C), as in the original script",
                          "validation_macro_f1": float(best["validation_macro_f1"]),
                          "n_configurations": int(len(table)),
                          "best_per_variant": per_variant.assign(params=per_variant["params"].map(str))
                          .to_dict(orient="records")}
        self.trained_on = ["train"]
        return self

    def refit(self, data: pd.DataFrame) -> "TfidfModel":
        import time
        start = time.perf_counter()
        self.pipeline = self._new(self.variant, self.params).fit(_clean(data["text"]), data["intent"].tolist())
        self.training_seconds = time.perf_counter() - start
        self.trained_on = sorted(set(data["split"]))
        return self

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.pipeline.predict_proba(_clean(texts))
        return proba[:, [list(self.pipeline.classes_).index(c) for c in self.class_order]]

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT
        joblib.dump({"pipeline": self.pipeline, "class_order": self.class_order, "variant": self.variant,
                     "params": self.params, "selection": self.selection}, path)
        return [path]

    @classmethod
    def load(cls, directory: Path, cfg: dict) -> "TfidfModel":
        m = cls(cfg)
        b = joblib.load(Path(directory) / ARTIFACT)
        if list(b["class_order"]) != m.class_order:
            raise ValueError("TF-IDF artifact class order differs from config")
        m.pipeline, m.variant, m.params, m.selection = b["pipeline"], b["variant"], b["params"], b["selection"]
        return m

    def n_parameters(self) -> int | None:
        clf = self.pipeline.named_steps["clf"]
        return int(clf.coef_.size + clf.intercept_.size)

    def get_model_metadata(self) -> dict:
        meta = super().get_model_metadata()
        meta["vocabulary_size"] = int(self.pipeline.named_steps["clf"].coef_.shape[1])
        return meta

