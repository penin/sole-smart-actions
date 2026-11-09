# Sole Smart Actions — Interactive Intent Classification Demo

A local web app that runs the **existing trained Bag of Words + Logistic Regression model** live, so anyone
can type a wallet command and see what the model predicts, how confident it is, how the confidence threshold
changes the final decision, and which words drove the result.

Nothing is simulated: every prediction comes from `predict_proba()` on the saved model, and every metric
in the UI is read from the project's evaluation outputs. The demo never retrains, re-saves or modifies the
model, the dataset, configuration or evaluation results.

---

## 1. Purpose

- Demonstrate the baseline classifier for the MIT AI Edge presentation and for exploratory testing.
- Make the decision process understandable: **raw prediction → confidence → threshold → final intent**.
- Provide a shell that future models (TF-IDF, custom Transformer, DistilBERT) can plug into without
  rebuilding the frontend.

## 2. Architecture

```text
Browser (React + TypeScript + Tailwind + Recharts)
   │  POST /api/predict  ──►  FastAPI (demo/backend/main.py)
   │  POST /api/decide        ├─ model_service.py   ModelRegistry → BowLogRegModel (loaded once at startup)
   │  GET  /api/model-info    │                       uses sole_intent.train.load_bundle / reorder_proba
   │  GET  /api/models        ├─ decision.py        apply_threshold() → wraps sole_intent.rejection rules
   ▼                          └─ schemas.py         request/response validation
Slider changes are re-decided locally from the stored probabilities (no inference), then confirmed by
POST /api/decide. The backend is authoritative: if it ever disagreed, the UI would show the backend result.
```

```text
demo/
├── backend/
│   ├── main.py                       FastAPI app; also serves frontend/dist in presentation mode
│   ├── model_service.py              model registry, BoW model wrapper, explanation, model info
│   ├── decision.py                   apply_threshold(probabilities, threshold, class_order)
│   ├── schemas.py                    Pydantic request/response models
│   ├── paths.py                      locations of the existing artifacts (read-only)
│   ├── generate_threshold_cases.py   writes the shared backend↔frontend decision fixture
│   ├── requirements.txt
│   └── tests/                        pytest: model integration, decision rules, API
├── frontend/
│   ├── src/App.tsx                   state, local re-decision, backend confirmation, history
│   ├── src/components/               Header, CommandInput, ThresholdControl, ResultsPanel,
│   │                                 ConfidenceChart, SampleCommands, Interpretation,
│   │                                 ModelInfoPanel, RecentPredictions
│   ├── src/lib/threshold.ts          TypeScript mirror of decision.py (tested against the fixture)
│   ├── src/data/sampleCommands.json  sample inputs (no expected outputs)
│   └── src/__tests__/                vitest: threshold consistency + UI behaviour
├── e2e/browser_smoke.py              optional real-browser test (Playwright + Edge)
├── start-demo.ps1 / start-demo.cmd   presentation mode (one process, one URL)
├── start-dev.ps1                     development mode (backend + Vite hot reload)
├── start-backend.ps1, start-frontend.ps1, run-tests.ps1
└── README.md
```

## 3. Model loading

| Item | Location / value |
|---|---|
| Model bundle | `models/bow_logreg_v1_tuned_v1.0.0.joblib` (loaded once at startup, read-only, SHA-256 shown in the UI) |
| Vectorizer | the fitted `CountVectorizer` is the `bow` step **inside** the bundle's pipeline (there is no separate file) |
| Preprocessing | the bundle's own `TextNormalizer` step (Unicode NFKC, lowercase, whitespace collapse) |
| Classifier | `LogisticRegression` (`C = 0.1`), the `clf` step of the pipeline |
| Classes | read at runtime from the bundle's `class_order`, validated against `classifier.classes_` (9 intents) |
| Default threshold | `config/inference.yaml` → `inference.threshold` if set; it is `null`, so the bundle's `rejection_threshold` **0.19572 (19.57%)** is used. This is the same precedence as `python -m sole_intent.predict`. |
| Metrics | `results/metrics/metrics_test.json`, `metrics_validation.json`, `rejection_metrics.json`, `runtime_metrics.json`, `bootstrap_ci_test.json` |

If the model cannot be loaded, the server still starts. `/api/health` reports the error, prediction
endpoints return HTTP 503 with the reason, and the UI shows it.

## 4. Prediction pipeline

1. **Input:** the customer command (1–2,000 characters on the API; the UI limits input to 500).
2. **Preprocessing:** the trained `TextNormalizer` step.
3. **Bag of Words:** the fitted `CountVectorizer` (unigrams; keeps `$` and numbers). Words not in its
   250-word vocabulary are dropped; the UI lists them as *unknown words*.
4. **Logistic Regression:** `predict_proba()` returns a probability for each of the 9 intents.
5. **Decision:** `decision.apply_threshold()` applies the selected threshold and fallback rules.

**Explanation panel.** For the top intent *k*, the class score is
`intercept_k + Σ_j count_j × coef_kj`. The panel lists each recognized word's `count × coef` term. Together
with the intercept these terms sum *exactly* to `decision_function()` for that class (asserted in tests).
They show association, not causation. Probabilities come from a softmax over all class scores.

## 5. Threshold and fallback logic

These are the project's existing rules (`src/sole_intent/rejection.py`), wrapped unchanged in `decision.py`:

| Situation | Model decision (final intent) | Application action | UI status |
|---|---|---|---|
| top probability ≥ threshold | the top intent | `ROUTE_<INTENT>` | **Accepted** (teal) |
| top probability **=** threshold | the top intent (equal is accepted) | `ROUTE_<INTENT>` | Accepted |
| top probability < threshold | `UNKNOWN_INTENT` | `FALLBACK_HELP_OR_CLARIFY`: offer Help & Support or ask to rephrase; the top intent is shown as a suggestion, never executed | **Below Threshold** (amber) + **Fallback Applied** (neutral) |
| model predicts `HELP_SUPPORT` with enough confidence | `HELP_SUPPORT` | `ROUTE_HELP_SUPPORT` | Accepted (a real prediction, not a fallback) |

- `UNKNOWN_INTENT` is **not** a class of the model. It exists only as the threshold outcome. Low-confidence
  commands are not relabelled `HELP_SUPPORT`.
- The **raw prediction and confidence never change** with the threshold; only the final intent does.
- Ties go to the first class in the canonical class order (numpy `argmax` behaviour).
- The slider (0–100%, 1% steps) is temporary and per-session. It is never written to any configuration.
  **Reset to Default** restores 19.57%. The default sits between slider steps, so it is shown with
  two decimals.
- **Consistency:** `generate_threshold_cases.py` produces `frontend/src/__tests__/fixtures/threshold_cases.json`
  (573 cases from real model probabilities, including exact `confidence == threshold` and ±1-ulp boundaries).
  The backend tests verify the fixture is current; the frontend tests check `applyThreshold()` matches every case.

## 6. Sample commands

`frontend/src/data/sampleCommands.json` holds 10 categories and 36 commands (the brief's list plus
"Transfer 50 dollars to Jone" for Scenario E). They are inputs only, with no expected outputs. Clicking one
fills the input, runs inference with the current threshold, and updates results, chart, explanation and
history. For the *Unknown / Out-of-Scope* category the UI notes that `UNKNOWN_INTENT` is not a class of the
loaded model; this check is computed from the model's classes at runtime.

**Presentation scenarios.** Deep links also work, e.g.
`http://127.0.0.1:8000/?q=Send%20%2450%20to%20Jone&t=60` (`t` = threshold %).

| Scenario | Try | What it shows (actual model behaviour at 19.57%) |
|---|---|---|
| A Confident | "Send $50 to Jone" | TRANSFER_MONEY, 88.5% |
| B Sensitivity | same, drag slider up | final intent flips to UNKNOWN_INTENT at 89%; raw/confidence unchanged |
| C Unknown | "What's the weather today?" | VIEW_TRANSACTIONS at 29.1%: *accepted* at the default threshold (a model error; "weather" is not in the vocabulary). Rejected from 30%. |
| D Ambiguous | "Payment" | HELP_SUPPORT at 33.2%: accepted at default, fallback at 34%+ |
| E Paraphrase | "Transfer 50 dollars to Jone" | TRANSFER_MONEY, 81.3% |

## 7. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, model_loaded, model_id, error}` |
| GET | `/api/models` | registry: id, name, `available`, `status` (`available` / `coming_soon` / `error`) |
| GET | `/api/model-info?model_id=bow_logreg` | classes, vocabulary, vectorizer/classifier details, default threshold + source, fallback rule, evaluation metrics (null → "Not available") |
| POST | `/api/predict` | `{text, threshold?, model_id?}` → raw prediction, confidence, final intent, status, application action, **all class probabilities**, ranked list, recognized/unknown tokens, explanation, `inference_ms` |
| POST | `/api/decide` | `{probabilities, threshold, model_id?}` → decision only (no inference) |

Interactive docs: `http://127.0.0.1:8000/docs`. Validation errors return 422 with field messages; request
text is never echoed into errors or logs. Unknown model → 404, unavailable model → 503.

## 8. Installation and startup (Windows)

**Prerequisites**
- The project's `.venv` (Python 3.13, scikit-learn 1.9.1). The saved model must be loaded with the same
  scikit-learn version it was trained with.
- Node.js ≥ 20 (LTS from nodejs.org) for building the frontend. The scripts use `npm` from `PATH`, or a
  portable Node in `%LOCALAPPDATA%\Programs\node-portable\` if present. Node is only needed to *build*;
  presentation mode needs only Python once built. The built UI (`frontend/dist/`) is not committed, so
  build it once after cloning.

**Install**
```powershell
cd sole-smart-actions             # the repository root, wherever you cloned it
.venv\Scripts\python -m pip install -r demo\backend\requirements.txt
cd demo\frontend; npm ci; npm run build
```

**Presentation mode (recommended)**: one process, one URL:
```powershell
cd sole-smart-actions\demo
.\start-demo.ps1            # or double-click start-demo.cmd   →  http://127.0.0.1:8000
.\start-demo.ps1 -Rebuild   # rebuild the UI first after frontend changes
```

**Development mode**: hot reload:
```powershell
.\start-dev.ps1             # backend :8000 (new window) + Vite :5173 (proxies /api)
# or separately: .\start-backend.ps1   and   .\start-frontend.ps1
```

| URL | What |
|---|---|
| http://127.0.0.1:8000 | demo UI (presentation mode) |
| http://127.0.0.1:5173 | demo UI (development mode) |
| http://127.0.0.1:8000/docs | API docs |

**Troubleshooting**
- *"Cannot reach the model server"* → the backend isn't running; start `start-demo.ps1` or `start-backend.ps1`.
- *Scripts blocked* → run `powershell -ExecutionPolicy Bypass -File .\start-demo.ps1` or use `start-demo.cmd`.
- *Port in use* → `.\start-demo.ps1 -Port 8010`. The Vite port is fixed at 5173.
- *`ModuleNotFoundError: sole_intent`* → `.venv\Scripts\python -m pip install -e .` from the project root.
- *Model fails to load / version warnings* → use the project `.venv` (scikit-learn 1.9.1).
- *`npm` not found* → install Node LTS or put the portable Node folder on `PATH`.
- *Typing `$50` on the CLI* → irrelevant in the browser. In PowerShell `curl` tests, use single quotes.

## 9. Testing

```powershell
cd demo
.\run-tests.ps1                                   # everything below
cd backend;  ..\..\.venv\Scripts\python -m pytest -q      # 125 tests
cd frontend; npm run typecheck; npm test                  # tsc + 18 vitest tests
python e2e\browser_smoke.py                       # optional: real Edge browser, needs `pip install playwright` + running demo
```

- **Backend:** the existing model and vectorizer load; classes come from the classifier; the default
  threshold matches the project. Predictions are identical to `sole_intent.predict` for all samples and
  equal the saved V1 test predictions. Probabilities map to labels, preprocessing is the trained step,
  and contributions sum exactly to `decision_function`. Also covered: decision boundaries (equal accepted,
  +1 ulp rejected), 0/100% extremes, HELP_SUPPORT vs fallback, probabilities not mutated, agreement with
  `apply_rejection` on random inputs, a current fixture, malformed/long/blank input, unknown and
  unavailable models, load failure → 503, and model/config/metrics files unchanged after use.
- **Frontend:** `applyThreshold` agrees with all 573 backend cases. Tests cover the slider range and
  default, re-decision with no new inference, reset, backend-authority fallback, blank input sends
  nothing, Enter submits and input is kept, the 500-char limit, server and network errors, all 36 samples
  clickable, the "not a class" notice, threshold kept on sample click, and history add/restore (current
  threshold, no inference)/cap 10/clear.
- **Browser smoke:** scenarios A–E on the live model, plus no horizontal overflow at 390 px and 768 px.

## 10. Adding future models

1. Implement a class in `backend/model_service.py` with `id`, `name`, `available`, `default_threshold`,
   `class_order`, `summary()`, `info()` and `predict(text) → RawPrediction` (probabilities for every class in
   `class_order`, plus an `explanation` dict with at least `recognized_tokens`/`unknown_tokens`, which may
   be empty for subword models).
2. Replace its `PlannedModel(...)` entry in `build_registry()`.
3. Nothing else changes: thresholding (`decision.apply_threshold`), the API, the chart and the dropdown are
   model-agnostic. The interpretation panel expects BoW-style contributions; give transformer models their
   own explanation shape and a matching component.
4. For side-by-side comparison, call `/api/predict` with each `model_id` on the same command; the response
   already includes `inference_ms`, and `/api/model-info` carries each model's accuracy and macro F1.

## Limitations

- The model is trained on synthetic commands and is under-confident (held-out test: mean top probability
  49% vs accuracy 79%). Confidence values are model estimates, not guarantees.
- At the default 19.57% threshold many out-of-scope commands are **accepted** as some intent (the project's
  evaluation found 14.7% unknown rejection). The demo shows this honestly; it does not hide model errors.
- Bag of Words ignores word order and drops unseen words (e.g. "weather"), as the explanation panel shows.
- `inference_ms` in the UI includes the explanation computation and the first call's warm-up. Use the
  benchmark latency in Model Information for performance claims.
- Session history lives only in browser memory and is lost on reload by design.
