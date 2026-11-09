"""Locations of the existing project artifacts. The demo reads these; it never writes to them."""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_DIR.parent
FRONTEND_DIST = DEMO_DIR / "frontend" / "dist"
FRONTEND_DATA = DEMO_DIR / "frontend" / "src" / "data"

MODEL_PATH = PROJECT_ROOT / "models" / "bow_logreg_v1_tuned_v1.0.0.joblib"
INFERENCE_CONFIG = PROJECT_ROOT / "config" / "inference.yaml"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# The `sole_intent` package is normally installed (pip install -e .); fall back to the source tree.
if str(PROJECT_ROOT / "src") not in sys.path:
    try:
        import sole_intent  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
