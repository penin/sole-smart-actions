# Sole Smart Actions

**Natural language intent classification for Sole, a digital wallet in Fiji** (MIT AI Edge course project).

Customers type commands such as *"Send $50 to Jone"*, *"Pay my EFL bill"* or *"Show my recent transactions"*.
The model maps each command to one of 9 wallet intents. When it is not confident enough, it hands off to
a help / clarification fallback instead of guessing.

This repository holds:

1. a **Bag of Words + Logistic Regression baseline**: trained, tuned on validation and fully evaluated;
2. a **model-agnostic evaluation framework** (metrics, rejection protocol, plots, comparable summary rows) that
   the other models must reuse so results are directly comparable;
3. an **exploratory confidence-threshold analysis** of the baseline (`results/threshold_analysis/`);
4. an **interactive web demo** (FastAPI + React) that runs the trained model live (`demo/`).

| Model | Status |
|---|---|
| Bag of Words + Logistic Regression | Baseline V1 trained and evaluated |
| TF-IDF + Logistic Regression | Planned |
| Custom Transformer (from scratch) | Planned |
| DistilBERT (fine-tuned) | Planned |

> Classification only. Nothing here executes payments or connects to live financial systems.
> The dataset is synthetic (3,000 generated utterances) and contains no customer data.

## Quick start

After [installation](#installation-windows-powershell-or-vs-code-terminal), from the repository root:

| I want to… | Command |
|---|---|
| Run the web demo | `.\demo\start-demo.ps1` → http://127.0.0.1:8000 |
| Reproduce training + evaluation | `python scripts/run_all.py` |
| Classify a command in the terminal | `python -m sole_intent.predict 'send $50 to Jone'` |
| Run the threshold analysis | `python scripts/evaluate_threshold.py --threshold 0.35` |
| Run the tests | `pytest -v` and `.\demo\run-tests.ps1` |

## Results at a glance

Read the generated reports for full results:

| File | Contents |
|---|---|
| `results/reports/evaluation_report.md` | Full evaluation (15 sections) |
| `results/reports/error_analysis.md` | Confusions, OOV, word order, leakage, financial safety, diagnostic probes |
| `results/reports/data_quality_report.md` | Dataset audit |
| `results/reports/model_card.md` | Model card |
| `results/reports/experiment_summary.csv` | One row per experiment, shared schema |

## Project layout

```text
sole-smart-actions/
├── Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx   # source data (synthetic; read-only, never modified)
├── config/
│   ├── baseline.yaml             # all training settings: labels, grid, thresholds, seeds
│   ├── inference.yaml            # inference threshold override (null = use the model's saved threshold)
│   └── threshold_analysis.yaml   # exploratory threshold sweep settings
├── src/sole_intent/
│   ├── data_loader.py            # load + validate workbook, predefined splits, dataset hashes
│   ├── audit.py                  # data-quality audit: duplicates, conflicts, leakage, lengths
│   ├── preprocessing.py          # normalization, tokenization, CountVectorizer factory
│   ├── train.py                  # pipeline, validation-based tuning, model bundle save/load
│   ├── evaluate.py               # MODEL-AGNOSTIC metrics, bootstrap CIs, latency, summary rows
│   ├── rejection.py              # MODEL-AGNOSTIC UNKNOWN_INTENT rejection + threshold selection
│   ├── plots.py                  # all figures (model-agnostic inputs)
│   ├── experiment.py             # end-to-end orchestration
│   ├── reports.py                # Markdown report generation
│   ├── predict.py                # inference CLI
│   └── utils.py                  # paths, config, hashing, JSON, environment capture
├── scripts/
│   ├── run_all.py                # single command for the full workflow
│   └── evaluate_threshold.py     # exploratory threshold analysis (no retraining)
├── tests/                        # pytest suite
├── models/                       # versioned joblib bundles + metadata JSON (Baseline V1)
├── data/processed/               # split CSVs, audit detail tables (derived, regenerated)
├── results/
│   ├── metrics/ figures/ predictions/ reports/   # Baseline V1 evaluation outputs
│   └── threshold_analysis/                       # exploratory threshold comparison
└── demo/                         # interactive web demo (see demo/README.md)
    ├── backend/                  # FastAPI: loads the trained model, threshold/fallback decisions
    ├── frontend/                 # React + TypeScript + Vite + Tailwind + Recharts
    └── start-demo.ps1            # one-command launcher
```

## Installation (Windows, PowerShell or VS Code terminal)

Prerequisites: **Python 3.13**, **Git**, and **Node.js LTS** (Node is only needed to build the web demo).
Run all commands from the repository root.

```powershell
git clone https://github.com/<owner>/sole-smart-actions.git
cd sole-smart-actions
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                      # makes `sole_intent` importable (needed to load saved models)
```

`requirements.txt` pins the exact versions used for the reported run (Python 3.13, scikit-learn 1.9.1).
Use the same scikit-learn version: saved `.joblib` models are only guaranteed to load with the version that
created them.

## Training

The baseline is already trained (`models/`) and evaluated (`results/`). To reproduce or retrain it:

```powershell
python scripts/run_all.py
```

This validates and audits the dataset, tunes on validation, fits the default and tuned models, evaluates
train/validation/test, runs the unknown-intent rejection protocol, measures runtime, writes predictions,
figures and reports, and saves the models. It takes about 15 seconds and overwrites previous outputs
(idempotent). Use another config with `python scripts/run_all.py --config config/my_experiment.yaml`.

Training only (tune + save, no evaluation): `python -m sole_intent.train`

> **The committed `models/` and `results/` are the official Baseline V1 artifacts.** Re-running
> `run_all.py` reproduces the same metrics and predictions, but it rewrites timestamps, latency figures and
> machine-specific paths. Don't commit those changes unless you mean to replace Baseline V1:
> `git restore models results data` discards them.

## Web demo

An interactive app that runs the trained model live: type a command, adjust the confidence threshold,
see the probability for each intent and the words that drove the prediction.

```powershell
.venv\Scripts\python -m pip install -r demo\backend\requirements.txt
cd demo\frontend; npm ci; npm run build; cd ..\..
.\demo\start-demo.ps1        # or double-click demo\start-demo.cmd  →  http://127.0.0.1:8000
```

`.\demo\start-dev.ps1` starts a hot-reloading development setup (http://127.0.0.1:5173). See
[`demo/README.md`](demo/README.md) for the architecture, API, threshold/fallback rules, tests and how to add
new models.

## Prediction

```powershell
python -m sole_intent.predict "send $50 to Jone" "pay my EFL bill" "who won the rugby"
python -m sole_intent.predict --json "show my recent transactions"
python -m sole_intent.predict --threshold 0.4 "how much money do I have"
python -m sole_intent.predict --model models\bow_logreg_v1_default_v1.0.0.joblib "send 20 to Mere"
```

In PowerShell, wrap text containing `$` in **single quotes** (`'send $50 to Jone'`), otherwise `$50` is
expanded as a variable.

From Python:

```python
from sole_intent.train import load_bundle
from sole_intent.predict import predict_texts
bundle = load_bundle("models/bow_logreg_v1_tuned_v1.0.0.joblib")
predict_texts(["send $50 to Jone"], bundle)
```

## Confidence-threshold analysis (exploratory, no retraining)

```powershell
python scripts/evaluate_threshold.py                       # thresholds from config/threshold_analysis.yaml, detail for 0.35
python scripts/evaluate_threshold.py --threshold 0.40      # detailed evaluation for another candidate
python scripts/evaluate_threshold.py --thresholds baseline 0.3 0.35 0.4 --threshold 0.3
```

The script reuses Baseline V1's saved probabilities (checked against the saved model), reproduces V1's
published counts at the frozen 0.196 threshold, and writes only to `results/threshold_analysis/`. It refuses
to write into V1 folders and records SHA-256 hashes of all V1 artifacts before and after
(`baseline_v1_manifest.json`). The test and unknown sets were inspected before, so results are
**exploratory**. Read `threshold_<tag>_report.md` for the protocol to select a production threshold.

Model decision vs application action: the model returns an intent or `UNKNOWN_INTENT`; the app maps
`UNKNOWN_INTENT` to `FALLBACK_HELP_OR_CLARIFY` (offer HELP_SUPPORT or ask the customer to rephrase) and any
intent to `ROUTE_<INTENT>`. A genuine HELP_SUPPORT classification is not a rejection.

Inference threshold precedence: `--threshold` > `inference.threshold` in `config/inference.yaml`
(`--config`) > the threshold stored in the model bundle (V1: 0.196).

```powershell
python -m sole_intent.predict --threshold 0.35 'send $50 to Jone'
python -m sole_intent.predict --config config/inference.yaml --json 'who won the rugby'
```

> Note: `scripts/run_all.py` regenerates the Baseline V1 outputs in place. It is deterministic (same
> metrics and predictions), but re-running it rewrites timestamps and latency measurements.

## Tests

```powershell
pytest -v
```

The suite covers dataset loading and schema, split integrity, no UNKNOWN_INTENT in training, preprocessing
consistency, vocabulary fitted on train only, probability shape/range/normalization, class ordering,
serialization round-trip, rejection thresholds, edge-case inputs, hand-computed metric fixtures, and spies
that prove tuning never touches the test split.

## Experimental protocol (shared by all models)

| Rule | Implementation |
|---|---|
| Preserve the supplied splits | `data_loader.get_splits` only selects rows; nothing is re-split or dropped |
| Train on supported intents, `split = train` | 2,160 rows; `UNKNOWN_INTENT` is never a class |
| Model selection | validation macro F1 (ties: validation log loss) |
| Rejection threshold | chosen on **known-intent validation rows only**: 95% coverage target fixed in config before viewing unknowns |
| Final evaluation | test split + 300 `evaluation_only` unknowns, scored once with frozen choices |
| Canonical class order | `labels` in `config/baseline.yaml` (probability columns must follow it) |
| Leakage-controlled view | test rows whose phrase (or number-masked template) appears in train are excluded in `metrics_test_leakage_controlled.json` |
| Duplicate-aware uncertainty | phrase-clustered bootstrap CIs (`bootstrap_ci_test.json`) |

## Reusing the evaluation framework for CNN / LSTM / transformer models

Your model only needs to produce a probability matrix with columns in the canonical class order.

```python
import numpy as np, pandas as pd
from sole_intent.utils import load_config, save_json
from sole_intent.data_loader import load_dataset, get_splits
from sole_intent.evaluate import evaluate_predictions, cluster_bootstrap, measure_latency, summary_row
from sole_intent.rejection import threshold_for_coverage, max_confidence, rejection_outcomes
from sole_intent.preprocessing import canonical_key

cfg = load_config()
labels = cfg["labels"]                                   # canonical order: do not re-sort
s = get_splits(load_dataset(cfg), cfg)                   # identical splits for every model

proba_val  = my_model.predict_proba(s.validation["text"])   # shape (n, 9), columns == labels
proba_test = my_model.predict_proba(s.test["text"])
proba_unk  = my_model.predict_proba(s.unknown["text"])
pred_test  = np.asarray(labels, dtype=object)[proba_test.argmax(1)]

m = evaluate_predictions(s.test["intent"], pred_test, proba_test, labels, "test",
                         "cnn_v1", "cnn_v1", core_labels=cfg["core_labels"])
thr = threshold_for_coverage(max_confidence(proba_val), cfg["rejection"]["target_known_coverage"])
rej = rejection_outcomes(s.test["intent"], proba_test, proba_unk, labels, thr)
ci  = cluster_bootstrap(s.test["intent"], {"cnn_v1": pred_test}, labels,
                        s.test["text"].map(canonical_key))
row = summary_row(model_name="cnn_v1", experiment_id="cnn_v1", test_macro_f1=m["macro_f1"],
                  known_intent_coverage=rej["known_intent_coverage"], ...)
# append `row` to results/reports/experiment_summary.csv
```

Figures can be produced with the same functions in `sole_intent.plots` (e.g. `confusion_matrix_plot`,
`roc_curves_plot`, `rejection_threshold_plot`). They take labels and probabilities, not a specific model.

## Output files

| Location | Files |
|---|---|
| `results/metrics/` | `dataset_audit.{json,csv}`, `metrics_{train,validation,test}.json` (+ `_default_model`, `test_leakage_controlled`, `test_unique_phrases`), `per_class_metrics.csv`, `confusion_matrix{,_normalized}.csv`, `roc_auc_per_class.csv`, `rejection_metrics.json`, `runtime_metrics.json`, `experiment_config.json`, `hyperparameter_search.csv`, `bootstrap_ci_test.json`, `metrics_core8_model.json`, plot data CSVs, `error_*.csv` |
| `results/figures/` | `01`–`13` PNGs at 200 dpi |
| `results/predictions/` | `{train,validation,test,unknown,diagnostic}_predictions.csv`, `word_order_probes.csv` |
| `results/reports/` | `evaluation_report.md`, `error_analysis.md`, `data_quality_report.md`, `model_card.md`, `experiment_summary.csv` |
| `models/` | `bow_logreg_v1_{tuned,default}_v1.0.0.joblib` + `.metadata.json` |

Undefined metrics are written as JSON `null` with a reason in `undefined_metrics`. They are never replaced by 0.

## Reproducibility

- Random seed 42 for every stochastic step (LR solver, CV shuffling, bootstrap, latency sampling, unknown split).
- `experiment_config.json` records the full config, its SHA-256, the dataset file and content SHA-256, package
  versions and hardware. Every model bundle stores the same metadata.
- Latency figures depend on the machine. All other metrics are deterministic for a given package set.

## Collaborating

- Work on a branch per task (`git switch -c feature/tfidf-model`) and open a pull request into `main`.
- Add each new model as its own module and append one row to `results/reports/experiment_summary.csv`
  using `sole_intent.evaluate.summary_row`, so every model is compared on identical splits and metrics.
- Never commit `.venv/`, `node_modules/`, `demo/frontend/dist/`, `.env` files or credentials (see `.gitignore`).
- Run `pytest -v` (and `.\demo\run-tests.ps1` if you touched the demo) before pushing.

## Large model files (Git LFS)

The Baseline V1 models are about 23 KB each and are committed normally. Transformer and DistilBERT
checkpoints will be hundreds of MB, and GitHub rejects files over 100 MB. Track those with
[Git LFS](https://git-lfs.com) **before** their first commit:

```powershell
git lfs install
git lfs track "models/**/*.safetensors" "models/**/*.bin" "models/**/*.pt" "models/**/*.onnx"
git add .gitattributes
```

These extensions are currently in `.gitignore` as a safety net. Remove the matching line there when you
enable LFS for them. Note that LFS storage on GitHub is limited on free plans.
