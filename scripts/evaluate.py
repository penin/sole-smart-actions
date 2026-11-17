"""Evaluate trained models with the shared framework and write the standard result files.

    python -m scripts.evaluate --model all                       # primary: artifacts/<m> -> results/<m>
    python -m scripts.evaluate --model distilbert --label trainval_refit   # secondary -> results/secondary/
    python -m scripts.evaluate --model all --smoke               # smoke artifacts -> reproduction/smoke/results
"""

from __future__ import annotations

import argparse

from sole_smart_actions import MODEL_NAMES
from sole_smart_actions.config import rel
from sole_smart_actions.evaluation.runner import evaluate_model, results_dir
from sole_smart_actions.training import artifact_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=[*MODEL_NAMES, "all"])
    ap.add_argument("--label", default="primary", help="primary, trainval_refit, or an ablation name")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--artifacts-root", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    a_root = args.artifacts_root or ("reproduction/smoke/artifacts" if args.smoke else None)
    r_root = args.results_root or ("reproduction/smoke/results" if args.smoke else None)
    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        protocol = "trainval_refit" if args.label == "trainval_refit" else "primary"
        variant = None if args.label in ("primary", "trainval_refit") else args.label
        src = artifact_dir(name, protocol=protocol, variant=variant, root=a_root)
        if not (src / "model_card.json").exists():
            print(f"[evaluate] skip {name}: no trained artifact in {rel(src)}")
            continue
        meta = evaluate_model(name, artifacts=src, out=results_dir(name, args.label, r_root), label=args.label,
                              make_plots=not args.no_plots)
        h = meta["headline"]
        print(f"  {name:12s} raw test accuracy {h['raw_test_accuracy']:.4f} ({h['raw_test_correct']}/{h['raw_test_n']}) "
              f"macro F1 {h['test_macro_f1']:.4f}  default threshold {h['default_threshold']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
