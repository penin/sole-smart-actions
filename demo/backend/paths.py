"""Locations of the project's trained models and evaluation results. The demo reads these; it never writes to them."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_DIR.parent
FRONTEND_DIST = DEMO_DIR / "frontend" / "dist"
FRONTEND_DATA = DEMO_DIR / "frontend" / "src" / "data"

# Primary four-model comparison (python -m scripts.train / scripts.evaluate)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"

# Historical Bag of Words baselines (kept selectable for comparison with earlier dataset versions)
MODEL_PATH_V5 = PROJECT_ROOT / "artifacts" / "history" / "bow_v5" / "bow_logreg_v5_tuned_v1.0.0.joblib"
METRICS_DIR_V5 = PROJECT_ROOT / "results" / "history" / "bow_v5" / "metrics"
MODEL_PATH_V4 = PROJECT_ROOT / "artifacts" / "history" / "bow_v4" / "bow_logreg_v1_tuned_v1.0.0.joblib"
METRICS_DIR_V4 = PROJECT_ROOT / "results" / "history" / "bow_v4" / "metrics"
INFERENCE_CONFIG = PROJECT_ROOT / "config" / "history" / "bow_v4_inference.yaml"

os.environ.setdefault("KERAS_BACKEND", "torch")
# The project packages are normally installed (pip install -e .); fall back to the source tree.
if str(PROJECT_ROOT / "src") not in sys.path:
    try:
        import sole_smart_actions  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
