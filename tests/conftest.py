"""Shared fixtures. The real workbook is used read-only; models are small and fit in well under a second."""

from __future__ import annotations

import numpy as np
import pytest

from sole_intent.data_loader import get_splits, load_dataset
from sole_intent.train import build_pipeline, fit_pipeline
from sole_intent.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def labels(cfg):
    return cfg["labels"]


@pytest.fixture(scope="session")
def df(cfg):
    return load_dataset(cfg)


@pytest.fixture(scope="session")
def splits(df, cfg):
    return get_splits(df, cfg)


@pytest.fixture(scope="session")
def default_params(cfg):
    p = cfg["model"]["default_params"]
    return {"ngram_range": tuple(p["ngram_range"]), "C": p["C"], "binary": p["binary"]}


@pytest.fixture(scope="session")
def fitted_pipeline(splits, cfg, labels, default_params):
    pipe, _ = fit_pipeline(build_pipeline(default_params, cfg), splits.train["text"], splits.train["intent"], labels)
    return pipe


@pytest.fixture
def toy_proba():
    """Three classes, four rows, columns in canonical order [A, B, C]."""
    return np.array([[0.7, 0.2, 0.1],
                     [0.4, 0.35, 0.25],
                     [0.1, 0.1, 0.8],
                     [0.34, 0.33, 0.33]])
