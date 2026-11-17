import sys
import warnings
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import paths  # noqa: E402
from model_service import BowLogRegModel, build_registry  # noqa: E402

PRIMARY_BOW = paths.ARTIFACTS_DIR / "bow" / "model.joblib"
PROTECTED = [PRIMARY_BOW, paths.MODEL_PATH_V5, paths.MODEL_PATH_V4, paths.INFERENCE_CONFIG,
             paths.PROJECT_ROOT / "config" / "common.yaml", paths.PROJECT_ROOT / "config" / "history" / "bow_v4.yaml",
             paths.PROJECT_ROOT / "data" / "raw" / "Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx",
             paths.PROJECT_ROOT / "data" / "raw" / "Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx",
             *sorted(paths.RESULTS_DIR.glob("*/experiment_metadata.json")),
             *sorted(paths.RESULTS_DIR.glob("*/metrics_test.json")),
             *sorted(paths.ARTIFACTS_DIR.glob("*/model_card.json")),
             *sorted(paths.METRICS_DIR_V5.glob("*.json")), *sorted(paths.METRICS_DIR_V4.glob("*.json"))]

# Bag of Words models served by the demo: (constructor kwargs, folder with saved test_predictions.csv).
MODEL_SPECS = {
    "primary": (dict(model_path=PRIMARY_BOW, results_dir=paths.RESULTS_DIR / "bow"), paths.RESULTS_DIR / "bow"),
    "v5_history": (dict(model_path=paths.MODEL_PATH_V5, metrics_dir=paths.METRICS_DIR_V5),
                   paths.METRICS_DIR_V5.parent / "predictions"),
    "v4_history": (dict(model_path=paths.MODEL_PATH_V4, metrics_dir=paths.METRICS_DIR_V4),
                   paths.METRICS_DIR_V4.parent / "predictions"),
}
NEURAL_AND_TFIDF = ["tfidf", "transformer", "distilbert"]


def trained(name: str) -> bool:
    return (paths.ARTIFACTS_DIR / name / "model_card.json").exists() and \
        (paths.RESULTS_DIR / name / "experiment_metadata.json").exists()


@pytest.fixture(scope="session")
def model():
    m = BowLogRegModel(**MODEL_SPECS["primary"][0])
    m.load()
    assert m.available, m.load_error
    return m


@pytest.fixture(scope="session", params=list(MODEL_SPECS))
def any_model(request):
    kwargs, predictions_dir = MODEL_SPECS[request.param]
    m = BowLogRegModel(**kwargs, model_id=f"test_{request.param}")
    m.load()
    assert m.available, m.load_error
    m.predictions_dir = predictions_dir
    return m


@pytest.fixture(scope="session")
def registry():
    return build_registry()


@pytest.fixture(scope="session")
def client(registry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from fastapi.testclient import TestClient
        from main import create_app
    with TestClient(create_app(registry)) as c:
        yield c


@pytest.fixture(scope="session")
def protected_hashes():
    from sole_intent.utils import sha256_file
    return {p: sha256_file(p) for p in PROTECTED}
