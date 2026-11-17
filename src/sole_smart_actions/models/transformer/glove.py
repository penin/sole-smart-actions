"""Pinned GloVe 6B 100d embeddings: download, checksum verification and a cached subset.

Source: Pennington, Socher & Manning (2014), https://nlp.stanford.edu/projects/glove/ (PDDL licence),
via the Hugging Face mirror pinned to an exact revision in ``config/transformer.yaml``.
The 40,000 most frequent words plus every word in the V5 workbook are cached (as in the original
notebook); only the training vocabulary is ever looked up when building an embedding matrix.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from ...config import rel, resolve
from ...utils import sha256_file, sha256_text

CHECKSUMS = resolve("data/external/CHECKSUMS.json")


class PretrainedResourceError(RuntimeError):
    pass


def expected_sha256(cfg: dict) -> str | None:
    """Pinned checksum: config value, else the one recorded in data/external/CHECKSUMS.json."""
    if cfg["embeddings"].get("sha256"):
        return cfg["embeddings"]["sha256"].lower()
    if CHECKSUMS.exists():
        from ...utils import load_json
        return load_json(CHECKSUMS).get(Path(cfg["embeddings"]["file"]).name, {}).get("sha256")
    return None


def ensure_glove(cfg: dict, *, download: bool = True) -> Path:
    e = cfg["embeddings"]
    path = resolve(e["file"])
    want = expected_sha256(cfg)
    if not path.exists():
        if not download:
            raise PretrainedResourceError(f"GloVe file missing: {rel(path)}. Run python -m scripts.download_pretrained")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"Downloading GloVe 6B 100d (~130 MB) from the pinned mirror revision {e['revision'][:7]} ...")
        with urllib.request.urlopen(e["url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(path)
    got = sha256_file(path)
    if want and got != want:
        raise PretrainedResourceError(f"GloVe checksum mismatch for {rel(path)}: expected {want}, got {got}")
    if not want:
        from ...utils import load_json, save_json
        data = load_json(CHECKSUMS) if CHECKSUMS.exists() else {}
        data[path.name] = {"sha256": got, "source": e["url"], "revision": e["revision"]}
        save_json(data, CHECKSUMS)
    return path


def glove_subset(cfg: dict, words_needed: set[str]) -> tuple[dict[str, int], np.ndarray]:
    """Return (word -> row, vectors) for the top-N words plus ``words_needed``; cached as .npz."""
    e = cfg["embeddings"]
    key = sha256_text("\n".join(sorted(words_needed)) + f"|{e['top_words']}|{e['revision']}")[:12]
    cache = resolve(e["file"]).parent / f"glove_100d_subset_{key}.npz"
    if not cache.exists():
        src = ensure_glove(cfg)
        words, vectors = [], []
        with gzip.open(src, "rt", encoding="utf-8") as f:
            for rank, line in enumerate(f):
                parts = line.rstrip().split(" ")
                if rank < e["top_words"] or parts[0] in words_needed:
                    words.append(parts[0])
                    vectors.append(np.asarray(parts[1:], dtype=np.float32))
        np.savez_compressed(cache, words=np.array(words), vectors=np.vstack(vectors))
    data = np.load(cache, allow_pickle=False)
    return {w: i for i, w in enumerate(data["words"])}, data["vectors"]


def embedding_matrix(vocab: list[str], index: dict[str, int], vectors: np.ndarray, dim: int, std: float,
                     seed: int) -> tuple[np.ndarray, int]:
    """The notebook's ``build_embedding_matrix``: N(0, std) rows, padding row zero, GloVe rows where known."""
    rng = np.random.RandomState(seed)
    matrix = rng.normal(0, std, (len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0
    hits = 0
    for i, tok in enumerate(vocab):
        if tok in index:
            matrix[i] = vectors[index[tok]]
            hits += 1
    return matrix, hits
