"""Exploratory confidence-threshold analysis on the frozen Baseline V1 model (no retraining).

    python scripts/evaluate_threshold.py                      # sweep config thresholds, detail for 0.35
    python scripts/evaluate_threshold.py --threshold 0.40     # detailed evaluation for another candidate
    python scripts/evaluate_threshold.py --thresholds 0.2 0.3 0.4 --threshold 0.3

Outputs go to results/threshold_analysis/. Baseline V1 artifacts are never modified (verified by hash).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sole_intent.threshold_analysis import run_threshold_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Candidate threshold for the detailed evaluation (default: config, 0.35)")
    parser.add_argument("--thresholds", nargs="+", default=None,
                        help="Thresholds to compare; 'baseline' = frozen V1 threshold (default: config)")
    parser.add_argument("--config", default=None, help="Analysis YAML (default: config/threshold_analysis.yaml)")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: results/threshold_analysis)")
    parser.add_argument("--recompute", action="store_true",
                        help="Score the saved model instead of reading saved V1 probabilities (still no retraining)")
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    ctx = run_threshold_analysis(args.config, args.thresholds, args.threshold, args.output_dir, args.recompute)
    cols = ["label", "known_coverage", "accepted_accuracy", "end_to_end_known_accuracy", "unknown_rejection_rate",
            "safe_fallback_rate", "unknown_to_help_support", "unknown_to_financial"]
    print(ctx["table"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nOutputs: {ctx['out_dir']}  |  Baseline V1 artifacts unchanged: {ctx['v1_unchanged']}")


if __name__ == "__main__":
    main()
