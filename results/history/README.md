# Historical Bag of Words runs (preserved, not re-run)

These are the Bag of Words + Logistic Regression experiments that preceded the four-model integration. They are kept
exactly as produced (only absolute paths inside eight JSON records were made repository-relative). They are **not**
part of the primary comparison in `results/comparison/`.

| Folder | Dataset | What it is | Model files |
|---|---|---|---|
| `bow_v4/` | V4 cleaned (`data/raw/Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx`) | Baseline V1: full report, bootstrap CIs, leakage analysis, learning curve; `threshold_analysis/` = exploratory threshold study (candidate 0.35) | `artifacts/history/bow_v4/` |
| `bow_v5/` | V5 deduplicated (canonical) | Same methodology rerun on V5, threshold sweep, `v4_vs_v5_comparison.md` | `artifacts/history/bow_v5/` |

The primary `results/bow/` model is the V5 model reproduced bit-for-bit through the shared framework, so its numbers
match `bow_v5/` (93.7% raw test accuracy) while using the standardized result format.

## Where things used to be

| Before the integration | Now |
|---|---|
| `results/` | `results/history/bow_v4/` |
| `results_v5/` | `results/history/bow_v5/` |
| `models/` | `artifacts/history/bow_v4/` |
| `models_v5/` | `artifacts/history/bow_v5/` |
| `data/processed/`, `data/processed_v5/` | `data/processed/bow_v4/`, `data/processed/bow_v5/` |
| `config/baseline.yaml`, `baseline_v5.yaml`, `threshold_analysis.yaml`, `inference.yaml` | `config/history/bow_v4.yaml`, `bow_v5.yaml`, `bow_v4_threshold_analysis.yaml`, `bow_v4_inference.yaml` |
| `Sole_Smart_Actions_Intent_Training_v*.xlsx` (repository root) | `data/raw/` (byte-identical) |

Paths written inside these historical reports (for example `results_v5/...`) refer to the old layout.

Re-run (writes only to the history folders):

```bash
python scripts/run_all.py --config config/history/bow_v5.yaml
python scripts/evaluate_threshold.py      # V4 threshold study
```
