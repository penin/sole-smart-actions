"""Command-line entry points. Run from the repository root, e.g. ``python -m scripts.train --model bow``."""

import os
import sys
from pathlib import Path

# Keras 3 must run on PyTorch (one deep-learning framework for the whole project).
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Make ``src/`` importable even without ``pip install -e .``.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
