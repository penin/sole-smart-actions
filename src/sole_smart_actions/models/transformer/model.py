"""Custom Transformer trained from scratch, ported from Felipe's notebook ``01_custom_transformer_v5.ipynb``.

Preserved verbatim (apart from TensorFlow-only calls): ``PositionalEmbedding``, ``TransformerBlock``,
``build_transformer`` (token + positional embedding, Transformer blocks, mean over real tokens, dense head,
softmax), GloVe initialisation, the 36-configuration grid, Adam, sparse categorical cross-entropy, batch 32,
up to 40 epochs with patience 10 and best-epoch selection on validation accuracy (ties: log loss).

Changes for the shared protocol: nine classes; the primary model is the best validation epoch of the
train-only run (the notebook's train+validation refit is available as a secondary protocol);
Keras 3 runs on the PyTorch backend; TextVectorization is replaced by :mod:`.vectorizer`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils import load_json, save_json, set_seed
from ..base import IntentModel
from .glove import embedding_matrix, glove_subset
from .vectorizer import TextVectorizer, tokenize

WEIGHTS, VOCAB, SPEC = "model.weights.h5", "vocabulary.json", "architecture.json"


def _keras():
    import keras
    if keras.backend.backend() != "torch":
        raise RuntimeError("Set KERAS_BACKEND=torch before importing keras (the project standardises on PyTorch)")
    return keras


def _layers():
    keras = _keras()
    from keras import layers, ops

    class PositionalEmbedding(layers.Layer):
        def __init__(self, max_length, vocabulary_size, embedding_dim, pretrained=None, **kwargs):
            super().__init__(**kwargs)
            initializer = keras.initializers.Constant(pretrained) if pretrained is not None else "uniform"
            self.token_embedding = layers.Embedding(vocabulary_size, embedding_dim,
                                                    embeddings_initializer=initializer, mask_zero=True)
            self.position_embedding = layers.Embedding(max_length, embedding_dim)

        def call(self, inputs):
            positions = ops.arange(ops.shape(inputs)[-1])
            return self.token_embedding(inputs) + self.position_embedding(positions)

        def compute_mask(self, inputs, mask=None):
            return ops.not_equal(inputs, 0)

    class TransformerBlock(layers.Layer):
        def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(embedding_dim // num_heads, 8),
                                                       dropout=dropout)
            self.feed_forward = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embedding_dim)])
            self.norm_attention = layers.LayerNormalization()
            self.norm_feed_forward = layers.LayerNormalization()
            self.dropout_attention = layers.Dropout(dropout)
            self.dropout_feed_forward = layers.Dropout(dropout)

        def call(self, inputs, mask=None, training=False):
            attention_mask = ops.cast(mask[:, None, :], "int32") if mask is not None else None
            attended = self.attention(query=inputs, key=inputs, value=inputs,
                                      attention_mask=attention_mask, training=training)
            attended = self.dropout_attention(attended, training=training)
            attention_output = self.norm_attention(inputs + attended)
            transformed = self.dropout_feed_forward(self.feed_forward(attention_output), training=training)
            return self.norm_feed_forward(attention_output + transformed)

    return keras, layers, PositionalEmbedding, TransformerBlock


def build_transformer(embeddings: np.ndarray, n_classes: int, max_length: int, embedding_dim: int, seed: int,
                      num_heads=2, num_blocks=1, ff_dim=256, dropout=0.2, head_dim=64, learning_rate=3e-4,
                      pretrained=True):
    keras, layers, PositionalEmbedding, TransformerBlock = _layers()
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    token_ids = layers.Input(shape=(max_length,), dtype="int32")
    embedded = PositionalEmbedding(max_length, embeddings.shape[0], embedding_dim,
                                   pretrained=embeddings if pretrained else None)(token_ids)
    contextual = layers.Dropout(dropout)(embedded)
    for _ in range(num_blocks):
        contextual = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)(contextual)
    pooled = layers.Dropout(dropout)(layers.GlobalAveragePooling1D()(contextual))
    hidden = layers.Dropout(dropout)(layers.Dense(head_dim, activation="relu")(pooled))
    probabilities = layers.Dense(n_classes, activation="softmax")(hidden)
    model = keras.Model(token_ids, probabilities)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))))


class CustomTransformerModel(IntentModel):
    name = "transformer"
    framework = "keras-3 (torch backend)"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.vectorizer: TextVectorizer | None = None
        self.keras_model = None
        self.arch: dict = {}
        self.history: pd.DataFrame | None = None
        self.glove_hits: int | None = None

    # ------------------------------------------------------------------ helpers
    def _seed(self) -> int:
        return int(self.cfg["model"]["seed"])

    def _labels(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([self.class_order.index(i) for i in frame["intent"]], dtype=np.int64)

    def _prepare(self, texts) -> tuple[TextVectorizer, np.ndarray]:
        t = self.cfg["text"]
        vec = TextVectorizer(t["max_tokens"], t["max_length"]).adapt(texts)
        emb = self.cfg["embeddings"]
        words = self._dataset_words()
        index, vectors = glove_subset(self.cfg, words)
        matrix, self.glove_hits = embedding_matrix(vec.vocabulary, index, vectors, emb["dim"], emb["init_std"],
                                                   self._seed())
        return vec, matrix

    def _dataset_words(self) -> set[str]:
        from ...data import load_dataset
        return {w for t in load_dataset(self.cfg)["text"] for w in tokenize(t)}

    def _build(self, matrix: np.ndarray, config: dict):
        a = self.cfg["architecture"]
        return build_transformer(matrix, len(self.class_order), self.cfg["text"]["max_length"], a["embedding_dim"],
                                 self._seed(), ff_dim=a["ff_dim"], head_dim=a["head_dim"], **config)

    def _train_one(self, config: dict, X, y, Xv, yv, matrix, epochs: int | None = None):
        """One run with the notebook's best-validation-epoch callback (or a fixed number of epochs)."""
        keras = _keras()
        tr = self.cfg["training"]
        set_seed(self._seed())
        model = self._build(matrix, config)
        cfg_epochs = epochs or tr["max_epochs"]

        class BestValidationEpoch(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best, self.history, self.wait = None, [], 0

            def on_epoch_end(self, epoch, logs=None):
                p = self.model.predict(Xv, verbose=0, batch_size=256)
                acc = float((p.argmax(1) == yv).mean())
                ll = _log_loss(p, yv)
                self.history.append({"epoch": epoch + 1, "validation_accuracy": acc, "validation_log_loss": ll,
                                     "train_loss": float(logs.get("loss", np.nan)) if logs else None})
                if self.best is None or (acc, -ll) > (self.best["accuracy"], -self.best["log_loss"]):
                    self.best = {"accuracy": acc, "log_loss": ll, "epoch": epoch + 1,
                                 "weights": [w.copy() for w in self.model.get_weights()]}
                    self.wait = 0
                else:
                    self.wait += 1
                    if epochs is None and self.wait >= tr["patience"]:
                        self.model.stop_training = True

        tracker = BestValidationEpoch()
        start = time.perf_counter()
        model.fit(X, y, epochs=cfg_epochs, batch_size=tr["batch_size"], verbose=0, callbacks=[tracker], shuffle=True)
        seconds = time.perf_counter() - start
        if epochs is None:
            model.set_weights(tracker.best["weights"])
        return model, tracker, seconds

    # ------------------------------------------------------------------ protocol
    def fit(self, train: pd.DataFrame, validation: pd.DataFrame, *, override: dict | None = None,
            fixed_config: dict | None = None) -> "CustomTransformerModel":
        if self.cfg.get("train_subset"):
            n = int(self.cfg["train_subset"]) // len(self.class_order)
            train = train.groupby("intent", sort=False).head(n).reset_index(drop=True)
        self.vectorizer, matrix = self._prepare(train["text"])
        X, y = self.vectorizer(list(train["text"])), self._labels(train)
        Xv, yv = self.vectorizer(list(validation["text"])), self._labels(validation)
        g = self.cfg["grid"]
        grid = ([fixed_config] if fixed_config else
                [dict(num_heads=h, num_blocks=b, dropout=d, learning_rate=lr) for h, b, d, lr in
                 itertools.product(g["num_heads"], g["num_blocks"], g["dropout"], g["learning_rate"])])
        rows, best = [], None
        total = time.perf_counter()
        for i, config in enumerate(grid, 1):
            config = {**config, **(override or {})}
            model, tracker, seconds = self._train_one(config, X, y, Xv, yv, matrix)
            row = {"key": f"{config['num_blocks']}b/{config['num_heads']}h/d{config['dropout']}/lr{config['learning_rate']:g}",
                   **config, "best_epoch": tracker.best["epoch"], "epochs_run": len(tracker.history),
                   "validation_accuracy": tracker.best["accuracy"], "validation_log_loss": tracker.best["log_loss"],
                   "train_accuracy": float((model.predict(X, verbose=0, batch_size=256).argmax(1) == y).mean()),
                   "parameters": int(model.count_params()), "training_seconds": seconds}
            rows.append(row)
            print(f"  [{i:2d}/{len(grid)}] {row['key']:22s} val acc {row['validation_accuracy']:.3f} | "
                  f"log loss {row['validation_log_loss']:.3f} | best epoch {row['best_epoch']}", flush=True)
            if best is None or (row["validation_accuracy"], -row["validation_log_loss"]) > \
                    (best[0]["validation_accuracy"], -best[0]["validation_log_loss"]):
                best = (row, model, tracker, config)
        table = pd.DataFrame(rows).sort_values(["validation_accuracy", "validation_log_loss"],
                                               ascending=[False, True]).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        row, self.keras_model, tracker, config = best
        self.arch = {**config, "epochs": row["best_epoch"]}
        self.history = pd.DataFrame(tracker.history)
        self.search_table = table
        self.training_seconds = float(row["training_seconds"])
        self.selection = {"params": self.arch, "rule": "highest validation accuracy; ties -> lowest validation "
                                                       "log loss; best epoch kept (patience 10)",
                          "validation_accuracy": row["validation_accuracy"],
                          "validation_log_loss": row["validation_log_loss"], "n_configurations": len(table),
                          "grid_seconds": time.perf_counter() - total, "glove_tokens": self.glove_hits,
                          "vocabulary_size": len(self.vectorizer.vocabulary)}
        self.trained_on = ["train"]
        return self

    def refit(self, data: pd.DataFrame) -> "CustomTransformerModel":
        """Notebook's refit: vocabulary + GloVe matrix re-learned on ``data``, same config and epoch count."""
        self.vectorizer, matrix = self._prepare(data["text"])
        X, y = self.vectorizer(list(data["text"])), self._labels(data)
        config = {k: v for k, v in self.arch.items() if k != "epochs"}
        self.keras_model, _, self.training_seconds = self._train_one(config, X, y, X[:1], y[:1], matrix,
                                                                      epochs=self.arch["epochs"])
        self.trained_on = sorted(set(data["split"]))
        return self

    # ------------------------------------------------------------------ inference / io
    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer(texts)
        # A command with no tokens at all (e.g. "" or "$!?" once punctuation is stripped) would be fully masked and the
        # masked mean pooling would return NaN. Represent it as a single [UNK] token instead ("nothing recognisable").
        X[(X == 0).all(axis=1), 0] = 1
        if len(texts) == 1:  # single command: direct call avoids predict() overhead (latency)
            import torch
            with torch.no_grad():
                return np.asarray(self.keras_model(X, training=False).cpu().numpy(), dtype=np.float64)
        return self.keras_model.predict(X, verbose=0, batch_size=256).astype(np.float64)

    def save(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        self.keras_model.save_weights(directory / WEIGHTS)
        save_json({"vocabulary": self.vectorizer.vocabulary}, directory / VOCAB)
        save_json({"architecture": self.arch, "class_order": self.class_order, "text": self.cfg["text"],
                   "embedding_dim": self.cfg["architecture"]["embedding_dim"],
                   "ff_dim": self.cfg["architecture"]["ff_dim"], "head_dim": self.cfg["architecture"]["head_dim"],
                   "selection": self.selection}, directory / SPEC)
        if self.history is not None:
            self.history.to_csv(directory / "training_history.csv", index=False)
        return [directory / f for f in (WEIGHTS, VOCAB, SPEC)]

    @classmethod
    def load(cls, directory: Path, cfg: dict) -> "CustomTransformerModel":
        directory = Path(directory)
        m = cls(cfg)
        spec = load_json(directory / SPEC)
        if spec["class_order"] != m.class_order:
            raise ValueError("Transformer artifact class order differs from config")
        t = spec["text"]
        m.vectorizer = TextVectorizer(t["max_tokens"], t["max_length"], load_json(directory / VOCAB)["vocabulary"])
        m.arch, m.selection = spec["architecture"], spec.get("selection", {})
        config = {k: v for k, v in m.arch.items() if k != "epochs"}
        placeholder = np.zeros((len(m.vectorizer.vocabulary), spec["embedding_dim"]), dtype=np.float32)
        m.keras_model = m._build(placeholder, config)
        m.keras_model.load_weights(directory / WEIGHTS)
        return m

    @property
    def device(self):
        """Keras 3's torch backend runs on CUDA when available (used to label latency measurements)."""
        from keras.src.backend.torch.core import get_device
        import torch
        return torch.device(get_device())

    def n_parameters(self) -> int | None:
        return int(self.keras_model.count_params())

    def get_model_metadata(self) -> dict:
        meta = super().get_model_metadata()
        meta["vocabulary_size"] = len(self.vectorizer.vocabulary)
        return meta
