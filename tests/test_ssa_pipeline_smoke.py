"""End-to-end smoke test of the command-line pipeline (classical models only, tiny grids, temporary folders)."""

import json

from scripts import compare as compare_cli
from scripts import evaluate as evaluate_cli
from scripts import train as train_cli
from scripts import validate_data
from sole_smart_actions.reproducibility import verify_result_schema


def test_validate_data_cli(tmp_path):
    assert validate_data.main(["--out", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))["passed"] is True


def test_train_evaluate_compare_smoke(tmp_path):
    a, r = tmp_path / "artifacts", tmp_path / "results"
    for m in ("bow", "tfidf"):
        assert train_cli.main(["--model", m, "--smoke", "--artifacts-root", str(a)]) == 0
    assert evaluate_cli.main(["--model", "all", "--smoke", "--artifacts-root", str(a), "--results-root", str(r),
                              "--no-plots"]) == 0
    for m in ("bow", "tfidf"):
        meta = json.loads((r / m / "experiment_metadata.json").read_text(encoding="utf-8"))
        assert meta["smoke_test"] is True and meta["primary_comparison"] is False
        assert meta["headline"]["raw_test_n"] == 270
        assert "SMOKE TEST" in (r / m / "evaluation_report.md").read_text(encoding="utf-8")
        schema = verify_result_schema(r, m)
        assert [p for p in schema["problems"] if not p.endswith(".png")] == []
    assert not (r / "transformer").exists()  # untrained models are skipped, never faked
    assert compare_cli.main(["--results-root", str(r)]) == 0
    assert "SMOKE TEST RESULTS" in (r / "comparison" / "model_comparison.md").read_text(encoding="utf-8")
