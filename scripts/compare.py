"""Build results/comparison/ (table, markdown, charts) from the evaluated primary models.

    python -m scripts.compare [--smoke]
"""

from __future__ import annotations

import argparse

from sole_smart_actions.comparison import compare


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args(argv)
    root = args.results_root or ("reproduction/smoke/results" if args.smoke else None)
    df = compare(root)
    cols = ["display_name", "raw_test_accuracy", "test_macro_f1", "default_threshold",
            "unknown_safe_fallback_at_default", "latency_p50_ms", "model_size_mb"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
