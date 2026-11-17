"""Reproduce the whole study from the saved configurations, without touching the published results.

    python -m scripts.reproduce_all                 # full: retrain all four models (GPU recommended, ~1 h)
    python -m scripts.reproduce_all --verify-only   # no training: re-evaluate the committed artifacts (minutes)
    python -m scripts.reproduce_all --smoke         # pipeline check with tiny grids (minutes; NOT results)
    python -m scripts.reproduce_all --models bow tfidf

Steps: 1 validate dataset -> 2 verify environment -> 3 train (unless --verify-only) -> 4 evaluate with the shared
framework -> 5 standardized reports -> 6 comparison charts -> 7 result-schema checks -> 8 reproducibility checks
(reproduced vs published results/) -> 9 reproduction report.
Everything is written under reproduction/<mode>/ (git-ignored). The full mode also refreshes
results/comparison/reproducibility_report.md.
"""

from __future__ import annotations

import argparse
import shutil
import time

from sole_smart_actions import MODEL_NAMES
from sole_smart_actions.comparison import compare
from sole_smart_actions.config import load_config, rel, resolve
from sole_smart_actions.data import validate_dataset
from sole_smart_actions.evaluation.runner import evaluate_model
from sole_smart_actions.reproducibility import (compare_results, verify_environment, verify_result_schema,
                                                write_reproducibility_report)
from sole_smart_actions.training import artifact_dir, train_model, verify_artifact_hashes
from sole_smart_actions.utils import save_json, utc_now


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = ap.parse_args(argv)
    name = "smoke" if args.smoke else "verify" if args.verify_only else "full"
    root = resolve("reproduction") / name
    a_root, r_root = root / "artifacts", root / "results"
    started = time.perf_counter()
    print(f"== Sole Smart Actions reproduction ({name}) -> {rel(root)}")

    print("[1/9] validating dataset")
    data_report = validate_dataset(load_config())            # raises (stops) on failure
    print("[2/9] verifying environment")
    env = verify_environment()
    for m in env["mismatches"]:
        print(f"      WARN {m['package']}: pinned {m['pinned']}, installed {m['installed']}")
    if not args.verify_only:
        print("[3/9] training")
        if (r_root).exists():
            shutil.rmtree(r_root)
        for m in args.models:
            train_model(m, smoke=args.smoke, artifacts_root=a_root)
    else:
        print("[3/9] training skipped (--verify-only): using committed artifacts/")
    print("[4/9] evaluating")
    for m in args.models:
        src = artifact_dir(m) if args.verify_only else artifact_dir(m, root=a_root)
        evaluate_model(m, artifacts=src, out=r_root / m, label="smoke" if args.smoke else "primary")
    print("[5-6/9] reports and comparison charts")
    compare(r_root, models=args.models)
    print("[7/9] result schemas")
    schema = [verify_result_schema(r_root, m) for m in args.models]
    hashes = [verify_artifact_hashes(artifact_dir(m)) for m in args.models
              if (artifact_dir(m) / "model_card.json").exists()]
    comparisons = None
    notes = [f"Run started from `python -m scripts.reproduce_all{' --' + name if name != 'full' else ''}`; "
             f"took {(time.perf_counter() - started) / 60:.1f} min."]
    if args.smoke:
        notes.append("SMOKE mode: tiny grids and training subsets. These numbers check the pipeline only and are "
                     "not comparable with the published results.")
    else:
        print("[8/9] reproduced vs published")
        comparisons = [compare_results(resolve("results"), r_root, m) for m in args.models
                       if (resolve("results") / m / "experiment_metadata.json").exists()]
        for c in comparisons:
            print(f"      {c['model']:12s} {c['mode']:9s} acc {c['raw_test_accuracy'][0]:.4f}->{c['raw_test_accuracy'][1]:.4f} "
                  f"agreement {c['test_prediction_agreement']:.1%} max|dp| {c['max_abs_probability_difference']:.1e} "
                  f"{'PASS' if c['ok'] else 'FAIL'}")
    print("[9/9] report")
    report = write_reproducibility_report(root / "reproduction_report.md", env=env, schema=schema,
                                          comparisons=comparisons, hashes=hashes, mode=name, data_report=data_report,
                                          notes=notes)
    ok = all(s["ok"] for s in schema) and all(h["ok"] for h in hashes) and all(c["ok"] for c in comparisons or [])
    save_json({"mode": name, "finished_utc": utc_now(), "ok": ok, "environment": env, "schema": schema,
               "artifact_hashes": hashes, "comparisons": comparisons}, root / "reproduction_summary.json")
    if name == "full" and set(args.models) == set(MODEL_NAMES):
        shutil.copyfile(report, resolve("results/comparison/reproducibility_report.md"))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} - report: {rel(report)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
