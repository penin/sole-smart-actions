"""Run the full Sole Smart Actions BoW baseline workflow.

    python scripts/run_all.py [--config config/baseline.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sole_intent.experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Config YAML (default: config/baseline.yaml)")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
