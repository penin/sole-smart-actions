"""Download and verify the pinned pretrained resources (needed only to TRAIN the neural models).

    python -m scripts.download_pretrained

* GloVe 6B 100d (Custom Transformer): pinned Hugging Face mirror revision, SHA-256 checked,
  stored in data/external/glove/ (git-ignored, ~130 MB).
* distilbert/distilbert-base-uncased (DistilBERT): pinned revision, cached by huggingface_hub
  in the standard Hugging Face cache (~270 MB).
The trained models in artifacts/ are self-contained: the demo and evaluation never need these downloads.
"""

from __future__ import annotations

from sole_smart_actions.config import load_config, rel
from sole_smart_actions.models.transformer.glove import ensure_glove


def main(argv=None) -> int:
    t = load_config("transformer")
    path = ensure_glove(t)
    print(f"GloVe OK: {rel(path)} (revision {t['embeddings']['revision'][:7]}, SHA-256 verified)")
    d = load_config("distilbert")["pretrained"]
    from huggingface_hub import snapshot_download
    local = snapshot_download(d["name"], revision=d["revision"],
                              allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                              "tokenizer_config.json", "vocab.txt"])
    print(f"DistilBERT OK: {d['name']}@{d['revision'][:7]} cached ({local.split('snapshots')[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
