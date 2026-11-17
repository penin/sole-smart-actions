# Sole Smart Actions — interactive four-model demo

A local web app that runs customer commands ("Send $50 to Jone", "Pay my EFL bill") through the four trained intent
classifiers: Bag of Words, TF-IDF, Custom Transformer and fine-tuned DistilBERT. It shows each model's real
probabilities, the confidence-threshold decision and an **illustrative** Sole workflow. It also has a Compare All
Models view with a dashboard built from the generated evaluation results.

> **Simulation only.** The demo classifies commands and simulates navigation. It never executes a transaction, never
> connects to a wallet, never asks for a PIN, and never stores or logs what you type.

## Quick start (Windows)

From the repository root, with the project environment installed (see the main README, *Installation*):

```powershell
.\demo\start-demo.cmd                 # builds the frontend once, starts http://127.0.0.1:8000 and opens the browser
# options: powershell -File demo\start-demo.ps1 -Port 8001 -Rebuild -NoBrowser
# developer mode (hot reload): .\demo\start-dev.ps1   (API :8000 + Vite :5173)
```

Stop the server with **Ctrl+C** in its window. If a browser tab says it cannot reach the model server, refresh it after
the server has started.

The first request to the Custom Transformer or DistilBERT takes a few seconds while that model loads (lazy loading);
later requests take milliseconds. DistilBERT uses the GPU when one is available and runs on CPU otherwise.

## What you can do

| Feature | Details |
|---|---|
| Model selection | Bag of Words, TF-IDF, Custom Transformer, DistilBERT, **Compare All Models** (current Dataset V5 models only; the historical Bag of Words runs are kept in `artifacts/history/` but not offered in the demo). A model that is not trained yet is listed but disabled, with the command that creates it. |
| Command input | Type any command (max 500 characters) or click a sample (all nine intents, out-of-scope requests and ambiguous requests such as "I owe Mary $50", "My transfer didn't go through" or "What's the weather tomorrow?"). |
| Threshold slider | 0–100%. Starts at the selected model's validation-calibrated default. Moving it re-applies the threshold to the stored probabilities instantly, without re-running the model, and the backend confirms the decision. |
| Prediction display | Predicted intent, confidence, all nine probabilities (top 3 in Compare mode), pass/fail against the threshold, whether clarification is needed, and the simulated workflow. |
| Explanation | Bag of Words: exact per-word contributions to the score. Other models: their own tokenization (TF-IDF words, Transformer vocabulary / `[UNK]`, DistilBERT WordPiece) without invented attributions. |
| Compare All | The same command through every model side by side, using each model's own default threshold or one shared threshold. |
| Dashboard | Raw test accuracy, macro F1, unknown-intent handling, latency, model size and typo robustness, read from `results/comparison/model_comparison.json`. |

## Safety rules in the simulated workflow (`backend/workflow.py`, mirrored in `frontend/src/lib/workflow.ts`)

- **Below the threshold:** the app does not act. It asks "Did you mean one of these?" and offers the top intents plus
  Help & Support.
- **Money-moving intents (transfer, bill payment, withdrawal):** these always end in an explicit confirmation step.
  - A request without an explicit action verb becomes a question. "I owe Mary $50" becomes "Would you like to send $50
    to Mary?"
  - If an amount or recipient is missing, the app asks for it.
  - The "Confirm (simulated)" button only shows that nothing was executed.
- **No credentials:** no PIN, password or account number is ever requested. Real authentication would happen inside
  the official Sole app.

## API (FastAPI, `backend/main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | default model status |
| `GET /api/models` | registry: id, name, availability, group (`primary` / `history`) |
| `GET /api/model-info?model_id=` | model details and evaluation metrics read from `results/<model>/` |
| `POST /api/predict` `{text, threshold?, model_id}` | probabilities, decision, explanation, simulated workflow |
| `POST /api/compare` `{text, threshold?}` | the same command through every primary model |
| `POST /api/decide` `{probabilities, threshold, model_id}` | re-apply the threshold without inference |
| `GET /api/comparison` | dashboard data (`results/comparison/model_comparison.json`) |

Models are loaded from `artifacts/<model>/` through the project's common interface
(`sole_smart_actions.training.load_trained`). Thresholds and metrics come from `results/<model>/experiment_metadata.json`.
Nothing is hard-coded, and the demo never writes to `artifacts/` or `results/` (a test checks the file hashes).

## Tests

```powershell
.\demo\run-tests.ps1          # backend (pytest) + frontend (vitest)
```

- **Backend:** checks that demo predictions equal the saved evaluation predictions for every model, the threshold and
  fallback rules, all API responses including untrained-model errors, the workflow safety rules, and that no artifact
  is modified.
- **Frontend:** checks the UI behaviour and the TypeScript threshold and workflow logic against fixtures generated from
  the Python implementation (`python demo/backend/generate_threshold_cases.py`).

## Limitations

The models were trained on a synthetic dataset. Entity extraction in the simulated workflow is a few regular
expressions, for illustration only. Latency shown in the dashboard depends on the evaluation machine (the neural
models were measured on a GPU).
