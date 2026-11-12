import sys
import warnings
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import paths  # noqa: E402
from model_service import BowLogRegModel, build_registry  # noqa: E402

PROTECTED = [paths.MODEL_PATH, paths.MODEL_PATH_V4, paths.INFERENCE_CONFIG, paths.PROJECT_ROOT / "config" / "baseline.yaml",
             paths.PROJECT_ROOT / "Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx",
             paths.PROJECT_ROOT / "Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx",
             *sorted(paths.METRICS_DIR.glob("*.json")), *sorted(paths.METRICS_DIR_V4.glob("*.json"))]

# Both trained models served by the demo: (model file, metrics folder, results folder with saved predictions).
MODEL_SPECS = {
    "v5": (paths.MODEL_PATH, paths.METRICS_DIR, paths.PROJECT_ROOT / "results_v5"),
    "v4": (paths.MODEL_PATH_V4, paths.METRICS_DIR_V4, paths.PROJECT_ROOT / "results"),
}


@pytest.fixture(scope="session")
def model():
    m = BowLogRegModel()
    m.load()
    assert m.available, m.load_error
    return m


@pytest.fixture(scope="session", params=list(MODEL_SPECS))
def any_model(request):
    model_path, metrics_dir, results_dir = MODEL_SPECS[request.param]
    m = BowLogRegModel(model_path, metrics_dir=metrics_dir, model_id=f"test_{request.param}")
    m.load()
    assert m.available, m.load_error
    m.results_dir = results_dir
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
