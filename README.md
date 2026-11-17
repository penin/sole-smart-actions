# Sole Smart Actions

**Natural-language intent classification for the Sole digital wallet (Fiji). MIT AI Edge course project.**

Customers type what they want, for example *"Send $50 to Jone"*, *"Pay my EFL bill"*, *"Show my recent
transactions"* or *"How much money do I have?"*, and the system identifies the intent and opens the matching Sole
screen. This repository compares four machine-learning approaches under one protocol:

1. **Bag of Words + Logistic Regression**
2. **TF-IDF + Logistic Regression**
3. **Custom Transformer** trained from scratch
4. **Fine-tuned DistilBERT**

It also includes an interactive demo that runs all four models side by side.

> **This is a classification experiment, not a payment system.** Nothing in this repository executes transactions,
> connects to financial systems or handles credentials. The demo only simulates navigation, and every money-moving
> request ends in a simulated confirmation.

---

## Contents
1. [Business problem](#1-business-problem) · 2. [Intents](#2-supported-intents) · 3. [Models](#3-model-architectures) ·
4. [Dataset](#4-dataset) · 5. [Methodology](#5-training-and-evaluation-methodology) · 6. [Results](#6-model-comparison-results) ·
7. [Thresholds and unknown intents](#7-confidence-thresholds-and-unknown-intents) · 8. [Installation](#8-installation) ·
9. [Commands](#9-training-evaluation-and-reproduction-commands) · 10. [Demo](#10-interactive-demo) ·
11. [Reproducibility](#11-reproducibility) · 12. [Safety](#12-financial-transaction-safety) · 13. [Team](#13-team-contributions) ·
14. [Limitations](#14-known-limitations) · 15. [Future work](#15-future-improvements) · 16. [Layout](#16-repository-layout)

## 1. Business problem

Sole is a digital wallet used in Fiji. Today a customer has to know *where* an action lives: which menu, tab or screen.
Customers think in goals ("send 50 to Litia", "pay my electricity"), and every extra screen adds friction. That
friction falls hardest on the least experienced users. **Sole Smart Actions** is a single text (or speech) entry point
that predicts the intent and routes the customer to the right workflow. It must also recognise requests it cannot
handle ("book a taxi") and fall back to help instead of guessing. Mistakes are not equal: opening a *money-moving*
screen for the wrong request is worse than opening the balance screen.

## 2. Supported intents

| # | Intent | Example | Moves money? |
|---|---|---|---|
| 1 | `TRANSFER_MONEY` | Send $50 to Jone | yes |
| 2 | `CHECK_BALANCE` | How much money do I have? | |
| 3 | `VIEW_TRANSACTIONS` | Show my recent transactions | |
| 4 | `WITHDRAW_MONEY` | Withdraw $100 | yes |
| 5 | `BILL_PAY` | Pay my EFL bill | yes |
| 6 | `HELP_SUPPORT` | My transfer didn't go through | |
| 7 | `MANAGE_CONTACTS` | Add Jone to my contacts | |
| 8 | `VIEW_GROUP` | Show my savings group | |
| 9 | `CHECK_SPENDING` | How much did I spend this month? | |
| – | `UNKNOWN_INTENT` | What's the weather tomorrow? | evaluation only; never a training class |

## 3. Model architectures

| Model | Representation | Classifier | Selected on validation | Original author |
|---|---|---|---|---|
| Bag of Words + LR | NFKC/lowercase text, word 1–2-gram binary counts (keeps `$` and numbers) | multinomial Logistic Regression, C = 10 | 12 configurations | penin |
| TF-IDF + LR | word 1–2-gram TF-IDF + character 2–5-gram TF-IDF (sublinear tf) | Logistic Regression, C = 10 | 35 configurations, two representations | Chelsea Cheng |
| Custom Transformer | 16 tokens, 100-d GloVe-initialised embeddings + positional embeddings | 2 Transformer blocks × 4 heads, mean pooling, dense head (233k parameters) | 36 configurations, best epoch | Felipe (`friosf`) |
| Fine-tuned DistilBERT | WordPiece, 24 tokens | `distilbert-base-uncased`, top 2 of 6 blocks fine-tuned, classification head (67M parameters) | 3 strategies × 3 learning rates, best epoch | Ricardo (`ricardomanjarresr`) |

All four implement one interface (`src/sole_smart_actions/models/base.py`):
`fit`, `predict_proba` (9 columns, canonical order), `predict`, `save` / `load`, `get_model_metadata`. How each teammate's
code was ported, and what changed, is documented in [docs/model_integration.md](docs/model_integration.md). The
architecture is described in [docs/architecture.md](docs/architecture.md).

```text
command ─► model.predict_proba ─► 9 probabilities ─► max ≥ threshold ? ─┬─ yes ─► intent ─► (simulated) Sole screen
                                                                         └─ no  ─► UNKNOWN_INTENT ─► clarify / Help & Support
money-moving intents ─► explicit, simulated confirmation (nothing is executed)
```

## 4. Dataset

`data/raw/Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx` (Dataset V5, SHA-256 `8dc9c89c…`) is the single
authoritative dataset. It holds **3,000 synthetic commands**: 300 per intent plus 300 unsupported requests. The
workbook defines fixed splits that every model uses unchanged:

| Split | Commands | Use |
|---|---:|---|
| train | 2,160 (240 per intent) | fitting |
| validation | 270 (30 per intent) | model selection and threshold calibration |
| test | 270 (30 per intent) | scored once, at the end |
| evaluation_only | 300 `UNKNOWN_INTENT` | out-of-scope handling only |

`python -m scripts.validate_data` runs 14 checks before any training and stops on failure. The checks cover:
- the file hash
- schema and missing values
- intent and split vocabularies
- split sizes and 300 commands per intent
- per-split class balance
- exact and normalized duplicates and conflicting labels
- overlap between splits

All checks pass on V5. V5 replaced 997 duplicates that existed in V4 (see the workbook's *Replacement Log*).

**Limitations:** the commands are synthetic. There are no exact or normalized duplicates across splits, but many
training and test commands share sentence templates (for example "Send *N* to *Name*"). Test accuracy is therefore
optimistic for real customer language, spelling, code-switching and Fijian names.

## 5. Training and evaluation methodology

| Rule | Value |
|---|---|
| Fit | train split only (2,160) |
| Hyperparameter / epoch selection | validation split only (270) |
| Classes | the same nine, in the same order, for every model; `UNKNOWN_INTENT` is never a training class |
| Default threshold | max class probability; largest validation confidence that keeps ≥ 95% of the 270 validation commands (no test or unknown labels used) |
| Test | the same 270 legitimate commands + 300 unknown commands, scored once |
| Metrics | one shared implementation: raw accuracy, macro precision/recall/F1, weighted F1, per-intent P/R/F1/support, confusion matrix, one-vs-rest ROC-AUC, threshold sweep, latency P50/P95, size, parameters, training time |
| Seeds | BoW / TF-IDF 42; Custom Transformer / DistilBERT 1234 (the teammates' seeds) |

**Raw test accuracy = correct predictions / 270**, before any confidence threshold. Selective accuracy among
accepted commands is reported separately and is never the headline.

**Secondary experiments, labelled and never mixed into the comparison:**
- the teammates' original "refit on train + validation" recipe (`--protocol trainval_refit`)
- the Custom Transformer ablations (`--variant no_attention | no_glove`)
- temperature calibration, a three-tier routing gate, and a one-typo robustness test

## 6. Model comparison results

Generated by `python -m scripts.compare` from the models' actual predictions. Full table, charts and
reproducibility report: [results/comparison/](results/comparison/model_comparison.md).

| Model | Raw test accuracy | Macro F1 | Macro ROC-AUC | Default threshold | Correct routing at default | Unknown: safe fallback | Unknown → money flow | Typo test | Size | Latency P50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bag of Words + LR | 93.7% (253/270) | 0.937 | 0.995 | 0.342 | 93.0% | 56.0% | 23 / 300 | 81.5% | 0.23 MB | 0.19 ms (CPU) |
| TF-IDF + LR | 97.4% (263/270) | 0.974 | 0.997 | 0.404 | 97.0% | 78.3% | 17 / 300 | 95.9% | 0.64 MB | 1.7 ms (CPU) |
| Custom Transformer | 97.4% (263/270) | 0.974 | 1.000 | 0.660 | 94.4% | 32.3% | 17 / 300 | 78.1% | 2.9 MB | 13.6 ms (GPU) |
| Fine-tuned DistilBERT | **97.8%** (264/270) | **0.978** | 1.000 | 0.676 | 96.7% | 62.0% | **14 / 300** | 85.6% | 135 MB | 6.8 ms (GPU) |

**Reading the results.**
- **The top three models are statistically indistinguishable on this test set.** TF-IDF, the Custom Transformer
  and DistilBERT are within one command of each other (263–264 of 270); one command is 0.37 points.
  Bag of Words is 10–11 commands behind.
- **TF-IDF with character n-grams gives the best trade-off here.** It is about 200 times smaller than DistilBERT,
  millisecond-fast on a CPU, and the most robust to typos (95.9%). At its default threshold it also handles the most
  unsupported requests safely (78%).
- **DistilBERT is the most accurate and sends the fewest unknown requests into money flows (14 of 300)**, at roughly
  200 times the size.
- **The Custom Transformer handles out-of-scope requests worst.** Only 32% are rejected or sent to help. It also drops
  to 78% accuracy with one typo per command, because a misspelled word becomes `[UNK]`.
- **No model is safe on unknown requests with a confidence threshold alone.** 14–23 of 300 unsupported commands still
  open a money-moving workflow. Explicit confirmation (section 12) is therefore mandatory.

The teammates' original numbers (for example 99.2% for DistilBERT) used 8 classes, 240 test commands and a
train + validation refit. They are preserved in `research/original_experiments/` but are not comparable.

## 7. Confidence thresholds and unknown intents

Each model's default threshold is calibrated **the same way** on validation (95% coverage). The thresholds differ
because the probability distributions differ: linear models spread probability, neural models are more peaked.
`results/<model>/threshold_sweep.csv` reports the rule at no threshold, at the default, and at 0.25, 0.30, 0.35, 0.40,
0.50 and 0.60. For each threshold it gives:

- known-intent coverage
- accepted commands
- correctly classified accepted commands
- accuracy among accepted commands
- overall correct-routing rate (rejections count as failures)
- rejected legitimate commands
- unknown-command rejection rate
- unknown commands classified as `HELP_SUPPORT`
- unknown safe-fallback rate
- unknown commands routed to financial intents

`UNKNOWN_INTENT` (below threshold) and a genuine `HELP_SUPPORT` prediction are counted separately, although both are
safe outcomes. In the app, `UNKNOWN_INTENT` means *ask the customer to clarify or offer Help & Support*; it never
means "execute the best guess".

## 8. Installation

Python **3.13** (the reference environment is 3.13.15). Clone, then from the repository root:

```powershell
git lfs install                       # once per machine; then
git lfs pull                          # downloads the neural model weights (artifacts/**/*.safetensors, *.h5)
py -3.13 -m venv .venv
.venv\Scripts\activate                # macOS/Linux: source .venv/bin/activate
# NVIDIA GPU (Windows/Linux):
python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
# CPU only / macOS:  python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m scripts.validate_data       # should print "All 14 checks passed."
```

`requirements.lock` pins every package (torch 2.14.1, Keras 3.15.1 on the PyTorch backend, transformers 5.22.1,
scikit-learn 1.9.1, …). `requirements.txt` lists the direct dependencies. The demo frontend additionally needs Node.js
(see `demo/README.md`). The training scripts need the pinned pretrained resources; evaluation and the demo do not:
`python -m scripts.download_pretrained` (GloVe for the Custom Transformer, `distilbert-base-uncased` for DistilBERT;
[data/external/README.md](data/external/README.md)).

## 9. Training, evaluation and reproduction commands

```powershell
python -m scripts.validate_data                     # 14 dataset checks; non-zero exit on failure
python -m scripts.train --model bow                 # ~2 s     -> artifacts/bow/
python -m scripts.train --model tfidf               # ~20 s    -> artifacts/tfidf/
python -m scripts.train --model transformer         # ~30 min on an RTX 4080 Laptop GPU (36 configurations)
python -m scripts.train --model distilbert          # ~5 min on the same GPU (9 configurations)
python -m scripts.evaluate --model all              # -> results/<model>/ (standard files)
python -m scripts.compare                           # -> results/comparison/

python -m scripts.reproduce_all --smoke             # whole pipeline with tiny grids, a few minutes (NOT results)
python -m scripts.reproduce_all --verify-only       # re-evaluate the committed models and check them (minutes)
python -m scripts.reproduce_all                     # retrain all four from scratch and compare with results/ (~40 min GPU)

# Secondary experiments (written to artifacts/secondary/ and results/secondary/)
python -m scripts.train --model distilbert --protocol trainval_refit
python -m scripts.evaluate --model distilbert --label trainval_refit
python -m scripts.train --model transformer --variant no_attention
python -m scripts.evaluate --model transformer --label no_attention

python -m pytest                                    # project tests
.\demo\run-tests.ps1                                # demo backend + frontend tests
```

Each `results/<model>/` folder contains:
- metrics: `metrics_train.json`, `metrics_validation.json`, `metrics_test.json`, `classification_report.csv`
- confusion matrix and ROC: `confusion_matrix.csv`/`.png`, `roc_auc.json`, `roc_curves.png`
- predictions: `test_predictions.csv`, `unknown_predictions.csv`
- thresholds and latency: `threshold_sweep.csv`, `latency_metrics.json`
- provenance and report: `experiment_metadata.json` (hashes, environment, commands), `evaluation_report.md`
- secondary analyses: `secondary_analyses.json`, `robustness_typos.json`

## 10. Interactive demo

```powershell
.\demo\start-demo.cmd          # http://127.0.0.1:8000
```

You can:
- choose Bag of Words, TF-IDF, Custom Transformer, DistilBERT or **Compare All Models**
- type a command or click a sample
- move the threshold slider and see each decision update
- see the top-three probabilities and whether clarification is needed
- step through the simulated Sole workflow

The comparison dashboard reads `results/comparison/model_comparison.json`. Details are in
[demo/README.md](demo/README.md).

## 11. Reproducibility

**What is pinned and recorded**
- **Dataset:** SHA-256 pinned in `config/common.yaml` and checked before every run. Fixed splits only; no model
  re-splits the data.
- **Configuration:** versioned configs (`config/common.yaml` + one per model), with their hashes stored in every
  `model_card.json`.
- **Environment:** exact package pins (`requirements.lock`). Python, OS, CPU, RAM, GPU, CUDA/cuDNN, torch and
  scikit-learn versions are recorded with every training and evaluation run.
- **Seeds and kernels:** fixed seeds; deterministic PyTorch kernels requested (`torch.use_deterministic_algorithms`,
  `CUBLAS_WORKSPACE_CONFIG`).
- **Pretrained resources:** pinned to exact revisions with checksums: DistilBERT at Hugging Face commit `12040acc…`,
  GloVe at mirror revision `1947a5a8…` (SHA-256 `22d27d5f…`).
- **Artifacts:** SHA-256 of every model file, plus the training and evaluation commands and the git commit, in
  `artifacts/<model>/model_card.json` and `results/<model>/experiment_metadata.json`.

**What is verified**
- **Bag of Words and TF-IDF retrain bit-for-bit** (tested). The primary BoW model is also identical to the historical
  V5 run.
- **Neural networks are checked within tolerances**: same accuracy within 3 commands, ≥ 97% identical test
  predictions, macro F1 within 0.012. Identical weights across different GPUs, drivers or library builds are not
  guaranteed.
- **Verified on the reference machine:** `python -m scripts.reproduce_all` retrained all four models from scratch
  (34.5 min on an RTX 4080 Laptop GPU). It reproduced every published model exactly, including the two neural
  networks: identical test predictions and a maximum probability difference of 0.0. All 14 dataset checks, the
  environment-lock check, artifact hashes and result schemas passed. Full report:
  [results/comparison/reproducibility_report.md](results/comparison/reproducibility_report.md).

## 12. Financial transaction safety

- The models only classify text. The demo only simulates navigation, never executes or authorises a transaction, and
  never asks for, accepts or stores a PIN, password or account number.
- Every money-moving intent ends in an explicit confirmation step, and confirming in the demo changes nothing.
  - Ambiguous requests become questions: *"I owe Mary $50"* becomes *"Would you like to send $50 to Mary?"*
  - Below the threshold the app asks the customer to clarify instead of acting.
- Even the best model sends some unsupported requests into money flows (section 6). A production system would need
  authentication inside the official app, amount and recipient confirmation, transaction limits and monitoring.
- Classification accuracy on a synthetic dataset is **not** evidence of production readiness.

## 13. Team contributions

| Model / component | Original author |
|---|---|
| Bag of Words baseline, evaluation framework, demo | penin |
| TF-IDF + Logistic Regression | Chelsea Cheng |
| Custom Transformer | Felipe (`friosf`) |
| Fine-tuned DistilBERT (and the shared benchmarks) | Ricardo (`ricardomanjarresr`) |

Details and sources: [docs/team_contributions.md](docs/team_contributions.md). The originals are preserved in
`research/original_experiments/`.

## 14. Known limitations

- **Synthetic data with shared templates.** All four models probably overstate real-world accuracy, and the test set is
  small (270 commands).
- **Threshold-only out-of-scope detection.** None of the models is trained on unsupported requests; 14–23 of 300 still
  reach a money flow.
- **Typo and name sensitivity.** Word-level models drop unseen words, and the Custom Transformer suffers most from typos.
- **Overconfidence on input with no recognisable words.** For example, the Custom Transformer assigns about 98% to
  `MANAGE_CONTACTS` for an input made only of unknown words or punctuation. A confidence threshold cannot catch this,
  which is another reason for confirmation steps.
- **Latency is hardware-specific.** The neural models were measured on a GPU and the linear models on a CPU.
- **Neural reproducibility is tolerance-based**, not bit-for-bit, across machines.
- **The simulated workflow is illustrative only.** Its entity extraction is a few regular expressions.

## 15. Future improvements

- Collect and label real (anonymised, consented) customer commands, including typos, Fijian names and code-switching.
- Add an explicit out-of-scope class or a dedicated detector, validated on held-out unsupported requests.
- Calibrate probabilities (temperature scaling is already measured in `secondary_analyses.json`) and choose thresholds
  from business costs.
- Add slot extraction for amounts, recipients and billers, with confirmation.
- Distil DistilBERT, or ensemble it with TF-IDF, for CPU and mobile deployment.
- Monitor the routes taken in production and the safe-fallback rate.

## 16. Repository layout

```text
SoleSmartActions/
├── config/               common.yaml + bow/tfidf/transformer/distilbert.yaml; history/ = earlier BoW configs
├── data/raw/             V4 and V5 workbooks (V5 is canonical)      data/external/  pinned downloads (git-ignored)
├── src/sole_smart_actions/   data/ models/ evaluation/ training.py comparison.py reproducibility.py
├── src/sole_intent/      original Bag of Words package (reused by the BoW adapter)
├── scripts/              validate_data, train, evaluate, compare, reproduce_all, download_pretrained, rename_project.ps1
├── artifacts/            trained models (+ secondary/, history/); large weights via Git LFS
├── results/              bow/ tfidf/ transformer/ distilbert/ comparison/ secondary/ history/
├── demo/                 FastAPI backend + React frontend
├── research/original_experiments/   teammates' original notebooks and results
├── docs/                 architecture, model integration, team contributions
├── tests/                project tests
├── requirements.lock  requirements.txt  pyproject.toml
```
