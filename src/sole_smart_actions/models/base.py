"""Common model interface. Every model exposes the same methods and the same nine-class probability order.

    model.fit(train, validation)      -> selection info   (train only; validation for selection)
    model.predict_proba(texts)        -> (n, 9) array, columns in cfg["labels"] order
    model.predict(texts)              -> list of intents (argmax, no threshold)
    model.save(directory) / Model.load(directory, cfg)
    model.get_model_metadata()        -> hyperparameters, sizes, provenance

Model-specific code lives in adapters (``models/<name>/model.py``); evaluation never needs to know which
model it is talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class IntentModel(ABC):
    name: str = ""
    framework: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.class_order: list[str] = list(cfg["labels"])
        self.selection: dict = {}          # selected hyperparameters and why
        self.search_table: pd.DataFrame | None = None
        self.training_seconds: float | None = None
        self.trained_on: list[str] = []

    # ---------------------------------------------------------------- training
    @abstractmethod
    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "IntentModel":
        """Select hyperparameters on ``validation`` and fit the final model on ``train`` only."""

    def refit(self, data: pd.DataFrame) -> "IntentModel":
        """Secondary protocol only: refit the already-selected configuration on ``data`` (train + validation)."""
        raise NotImplementedError(f"{self.name} does not support refitting")

    # ---------------------------------------------------------------- inference
    @abstractmethod
    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw probabilities in ``self.class_order`` column order."""

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, len(self.class_order)))
        proba = np.asarray(self._predict_proba(texts), dtype=np.float64)
        if proba.shape != (len(texts), len(self.class_order)):
            raise ValueError(f"{self.name}: probability shape {proba.shape}, expected "
                             f"({len(texts)}, {len(self.class_order)})")
        return proba

    def predict(self, texts: Sequence[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.class_order[i] for i in proba.argmax(axis=1)]

    # ---------------------------------------------------------------- persistence
    @abstractmethod
    def save(self, directory: Path) -> list[Path]:
        """Write model files into ``directory`` and return their paths."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path, cfg: dict) -> "IntentModel":
        ...

    def n_parameters(self) -> int | None:
        return None

    def get_model_metadata(self) -> dict:
        return {"name": self.name, "display_name": self.cfg["model"]["display_name"],
                "framework": self.framework, "class_order": self.class_order,
                "selection": self.selection, "n_parameters": self.n_parameters(),
                "trained_on": self.trained_on, "training_seconds": self.training_seconds}

    # Batching helper for neural models
    @staticmethod
    def batches(items: list, size: int):
        for i in range(0, len(items), size):
            yield items[i:i + size]
