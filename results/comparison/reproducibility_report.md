# Sole Smart Actions - reproducibility report

Mode: **full**. Environment: Python 3.13.15, Windows-11-10.0.26200-SP0, CPU Intel(R) Core(TM) i9-14900HX, GPU NVIDIA GeForce RTX 4080 Laptop GPU (CUDA 12.6), torch 2.14.1+cu126, scikit-learn 1.9.1.

## Checks

| Check | Result | Detail |
|---|---|---|
| Dataset validation (14 checks) | PASS | SHA-256 `8dc9c89cbb8af41f...` |
| Environment matches `requirements.lock` | PASS | 0 package mismatches; Python 3.13.15 (expected 3.13.15) |
| Artifact hashes `artifacts/bow` | PASS | 2 files; mismatched: none |
| Artifact hashes `artifacts/tfidf` | PASS | 2 files; mismatched: none |
| Artifact hashes `artifacts/transformer` | PASS | 5 files; mismatched: none |
| Artifact hashes `artifacts/distilbert` | PASS | 7 files; mismatched: none |
| Result schema `reproduction/full/results/bow` | PASS | all standard files and fields present |
| Result schema `reproduction/full/results/tfidf` | PASS | all standard files and fields present |
| Result schema `reproduction/full/results/transformer` | PASS | all standard files and fields present |
| Result schema `reproduction/full/results/distilbert` | PASS | all standard files and fields present |

## Reproduced vs published

| Model | Mode | Raw test accuracy (published -> reproduced) | Macro F1 | Same test predictions | Max prob. difference | Result |
|---|---|---|---|---|---|---|
| bow | exact | 0.9370 -> 0.9370 | 0.9366 -> 0.9366 | 100.0% | 0.00e+00 | PASS |
| tfidf | exact | 0.9741 -> 0.9741 | 0.9744 -> 0.9744 | 100.0% | 0.00e+00 | PASS |
| transformer | tolerance | 0.9741 -> 0.9741 | 0.9738 -> 0.9738 | 100.0% | 0.00e+00 | PASS |
| distilbert | tolerance | 0.9778 -> 0.9778 | 0.9777 -> 0.9777 | 100.0% | 0.00e+00 | PASS |

Criteria: exact: identical predictions, probabilities within 1e-9, same selected hyperparameters; tolerance: raw test accuracy within 0.0111 (3 commands), macro F1 within 0.012, >= 97% identical test predictions, default threshold within 0.05

## Notes

- Run started from `python -m scripts.reproduce_all`; took 34.5 min.

Neural networks: deterministic PyTorch kernels are requested (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, fixed seeds), but bit-identical weights across GPUs, drivers, CUDA/cuDNN or library versions are not guaranteed, so they are verified with tolerances.
