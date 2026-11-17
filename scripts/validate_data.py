"""Validate the canonical Dataset V5 workbook. Exits non-zero (and trains nothing) if any check fails.

    python -m scripts.validate_data [--out results/data_validation.json]
"""

from __future__ import annotations

import argparse
import sys

from sole_smart_actions.config import load_config, resolve
from sole_smart_actions.data import validate_dataset
from sole_smart_actions.utils import save_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/data_validation.json", help="where to write the JSON report")
    args = ap.parse_args(argv)
    report = validate_dataset(load_config(), raise_on_failure=False)
    for c in report["checks"]:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['check']}")
    d = report["dataset"]
    print(f"\nDataset {d['path']}  version {d['version']}  rows {d['n_rows']}\nSHA-256 {d['sha256']}")
    save_json(report, resolve(args.out))
    if not report["passed"]:
        print(f"\nDATASET VALIDATION FAILED ({report['n_failed']} checks). Nothing was modified; fix the data "
              "source before training.", file=sys.stderr)
        return 1
    print(f"\nAll {report['n_checks']} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
