import sys
import warnings
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import paths  # noqa: E402
from model_service import BowLogRegModel, build_registry  # noqa: E402

PROTECTED = [paths.MODEL_PATH, paths.INFERENCE_CONFIG, paths.PROJECT_ROOT / "config" / "baseline.yaml",
             paths.PROJECT_ROOT / "Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx",
             *sorted(paths.METRICS_DIR.glob("*.json"))]


@pytest.fixture(scope="session")
def model():
    m = BowLogRegModel()
    m.load()
    assert m.available, m.load_error
    return m


@pytest.fixture(scope="session")
def client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from fastapi.testclient import TestClient
        from main import create_app
    with TestClient(create_app(build_registry())) as c:
        yield c


@pytest.fixture(scope="session")
def protected_hashes():
    from sole_intent.utils import sha256_file
    return {p: sha256_file(p) for p in PROTECTED}
