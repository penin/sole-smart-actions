"""Train a model with its saved configuration (config/common.yaml + config/<model>.yaml).

    python -m scripts.train --model bow|tfidf|transformer|distilbert|all
    python -m scripts.train --model distilbert --protocol trainval_refit     # secondary (teammates' original refit)
    python -m scripts.train --model transformer --variant no_attention       # secondary ablation (needs primary)
    python -m scripts.train --model all --smoke                              # fast pipeline check (not results)

Primary artifacts go to artifacts/<model>/; secondary to artifacts/secondary/<model>_<label>/.
"""

from __future__ import annotations

import argparse

from sole_smart_actions import MODEL_NAMES
from sole_smart_actions.training import PROTOCOLS, train_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=[*MODEL_NAMES, "all"])
    ap.add_argument("--protocol", default="primary", choices=PROTOCOLS)
    ap.add_argument("--variant", default=None, help="secondary ablation, e.g. no_attention / no_glove (transformer)")
    ap.add_argument("--smoke", action="store_true", help="tiny grid + subset; writes to reproduction/smoke/")
    ap.add_argument("--artifacts-root", default=None, help="default artifacts/ (smoke: reproduction/smoke/artifacts)")
    args = ap.parse_args(argv)
    root = args.artifacts_root or ("reproduction/smoke/artifacts" if args.smoke else None)
    for name in MODEL_NAMES if args.model == "all" else [args.model]:
        train_model(name, smoke=args.smoke, protocol=args.protocol, variant=args.variant, artifacts_root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
