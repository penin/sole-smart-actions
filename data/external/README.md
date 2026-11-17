# Pretrained resources (downloaded, not committed)

Only needed to **train** the neural models. The trained models in `artifacts/` are self-contained, so evaluation and
the demo work without these downloads.

```bash
python -m scripts.download_pretrained
```

| Resource | Used by | Source (pinned) | Verification |
|---|---|---|---|
| GloVe 6B 100d (`glove/glove.6B.100d.txt.gz`, ~130 MB) | Custom Transformer | Stanford NLP GloVe (Pennington, Socher & Manning 2014, PDDL) via Hugging Face mirror `SLU-CSCI4750/glove.6B.100d.txt` at revision `1947a5a81ebbd83f2c831b58f308d1f11d6c6954` | SHA-256 `22d27d5f0080da2a57f48e6691af7b18aa39e96a54ef6fad02d9456a117239e1` (`config/transformer.yaml`, `CHECKSUMS.json`); the download is refused if it differs |
| `distilbert/distilbert-base-uncased` (~270 MB) | DistilBERT | Hugging Face Hub, revision `12040accade4e8a0f71eabdb258fecc2e7e948be` (Apache-2.0) | pinned commit hash; stored in the standard Hugging Face cache |

The training code also caches a GloVe subset (`glove/glove_100d_subset_<hash>.npz`): the 40,000 most frequent words plus
every word in the V5 workbook, as in the original notebook.
