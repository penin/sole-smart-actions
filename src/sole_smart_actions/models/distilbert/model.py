"""Fine-tuned DistilBERT, ported from Ricardo's notebook ``01_sole_intent_classifier_v5.ipynb``.

Preserved: ``distilbert-base-uncased`` + its WordPiece tokenizer (now pinned to an exact Hugging Face
revision), ``AutoModelForSequenceClassification`` with eager attention, the three unfreezing strategies x
three learning rates, AdamW with 10% linear warm-up then linear decay, batch 16, 24 tokens, per-epoch
validation scoring and best-epoch selection (validation accuracy, ties: log loss).

Changes for the shared protocol: nine classes; the primary model is the best validation epoch of the
train-only run (the notebook's train+validation refit is available as a secondary protocol); weights are
stored in float16 like the original and always evaluated in float32 after reloading.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils import load_json, save_json, set_seed
from ..base import IntentModel

SPEC = "sole_model.json"


def _device(pref: str):
    import torch
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))))


class DistilBertModel(IntentModel):
    name = "distilbert"
    framework = "pytorch + transformers"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = None
        self.tokenizer = None
        self.strategy: dict = {}
        self.history: pd.DataFrame | None = None
        self.device = None

    # ------------------------------------------------------------------ building blocks
    def _pretrained(self) -> tuple[str, str]:
        p = self.cfg["pretrained"]
        return p["name"], p["revision"]

    def _tokenizer(self):
        from transformers import AutoTokenizer
        name, rev = self._pretrained()
        return AutoTokenizer.from_pretrained(name, revision=rev)

    def _build(self, unfrozen_blocks):
        """The notebook's ``build_classifier``: fresh 9-intent head; freeze per strategy."""
        import torch
        from transformers import AutoModelForSequenceClassification
        name, rev = self._pretrained()
        torch.manual_seed(int(self.cfg["model"]["seed"]))
        model = AutoModelForSequenceClassification.from_pretrained(
            name, revision=rev, num_labels=len(self.class_order),
            id2label=dict(enumerate(self.class_order)), label2id={c: i for i, c in enumerate(self.class_order)},
            attn_implementation="eager")
        if unfrozen_blocks is not None:
            for p in model.base_model.embeddings.parameters():
                p.requires_grad = False
            blocks = model.base_model.transformer.layer
            for block in blocks[:len(blocks) - unfrozen_blocks]:
                for p in block.parameters():
                    p.requires_grad = False
        return model.to(self.device)

    def _encode(self, texts):
        enc = self.tokenizer(list(texts), padding="max_length", truncation=True,
                             max_length=self.cfg["text"]["max_tokens"], return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def _proba(self, model, ids, mask, batch_size=128) -> np.ndarray:
        import torch
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(ids), batch_size):
                logits = model(input_ids=ids[s:s + batch_size].to(self.device),
                               attention_mask=mask[s:s + batch_size].to(self.device)).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.vstack(out)

    def _run(self, unfrozen_blocks, lr: float, epochs: int, ids, mask, y, val=None, stop_epoch=None):
        """The notebook's ``fine_tune`` (with ``val``) / ``refit_finetuned`` (with ``stop_epoch``)."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup
        seed = int(self.cfg["model"]["seed"])
        set_seed(seed)
        model = self._build(unfrozen_blocks)
        tr = self.cfg["training"]
        loader = DataLoader(TensorDataset(ids, mask, torch.tensor(y)), batch_size=tr["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        total = epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(tr["warmup_fraction"] * total), total)
        history, best = [], None
        start = time.perf_counter()
        for epoch in range(1, (stop_epoch or epochs) + 1):
            model.train()
            for b_ids, b_mask, b_y in loader:
                optimizer.zero_grad()
                loss = model(input_ids=b_ids.to(self.device), attention_mask=b_mask.to(self.device),
                             labels=b_y.to(self.device)).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
            if val is not None:
                vp = self._proba(model, val[0], val[1])
                acc, ll = float((vp.argmax(1) == val[2]).mean()), _log_loss(vp, val[2])
                history.append({"epoch": epoch, "validation_accuracy": acc, "validation_log_loss": ll})
                if best is None or (acc, -ll) > (best["accuracy"], -best["log_loss"]):
                    best = {"accuracy": acc, "log_loss": ll, "epoch": epoch,
                            "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
        seconds = time.perf_counter() - start
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, best, history, trainable, seconds

    # ------------------------------------------------------------------ protocol
    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "DistilBertModel":
        import torch
        self.device = _device(self.cfg["training"]["device"])
        if self.cfg.get("train_subset"):
            n = int(self.cfg["train_subset"]) // len(self.class_order)
            train = train.groupby("intent", sort=False).head(n).reset_index(drop=True)
        self.tokenizer = self._tokenizer()
        ids, mask = self._encode(train["text"])
        y = np.array([self.class_order.index(i) for i in train["intent"]])
        vids, vmask = self._encode(validation["text"])
        vy = np.array([self.class_order.index(i) for i in validation["intent"]])
        rows, histories, winner = [], [], None
        grid_start = time.perf_counter()
        for strategy, s in self.cfg["strategies"].items():
            for lr in s["learning_rates"]:
                model, best, history, trainable, seconds = self._run(s["unfrozen_blocks"], float(lr), s["epochs"],
                                                                     ids, mask, y, val=(vids, vmask, vy))
                row = {"strategy": strategy, "learning_rate": float(lr), "epochs_tried": s["epochs"],
                       "best_epoch": best["epoch"], "parameters": sum(p.numel() for p in model.parameters()),
                       "trainable_parameters": trainable, "training_seconds": seconds,
                       "validation_accuracy": best["accuracy"], "validation_log_loss": best["log_loss"]}
                rows.append(row)
                histories += [{"strategy": strategy, "learning_rate": float(lr), **h} for h in history]
                print(f"  {strategy:16s} lr={lr:<8g} val acc {best['accuracy']:.3f} | log loss "
                      f"{best['log_loss']:.3f} | best epoch {best['epoch']} | {seconds:.0f}s", flush=True)
                if winner is None or (best["accuracy"], -best["log_loss"]) > (winner[0]["validation_accuracy"],
                                                                               -winner[0]["validation_log_loss"]):
                    winner = (row, s["unfrozen_blocks"], best["state"])
                del model, best
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        table = pd.DataFrame(rows).sort_values(["validation_accuracy", "validation_log_loss"],
                                               ascending=[False, True]).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        row, unfrozen, state = winner
        self.model = self._build(unfrozen)
        self.model.load_state_dict(state)
        self.model.eval()
        self.strategy = {"strategy": row["strategy"], "unfrozen_blocks": unfrozen,
                         "learning_rate": row["learning_rate"], "epochs": int(row["best_epoch"]),
                         "schedule_epochs": int(row["epochs_tried"])}
        self.search_table = table
        self.history = pd.DataFrame(histories)
        self.training_seconds = float(row["training_seconds"])
        self.selection = {"params": self.strategy, "rule": "highest validation accuracy; ties -> lowest validation "
                                                           "log loss; best epoch kept",
                          "validation_accuracy": row["validation_accuracy"],
                          "validation_log_loss": row["validation_log_loss"], "n_configurations": len(table),
                          "grid_seconds": time.perf_counter() - grid_start,
                          "pretrained": self.cfg["pretrained"]}
        self.trained_on = ["train"]
        return self

    def refit(self, data: pd.DataFrame) -> "DistilBertModel":
        """Notebook's refit: same strategy and learning-rate schedule, stop at the chosen epoch."""
        self.device = self.device or _device(self.cfg["training"]["device"])
        self.tokenizer = self.tokenizer or self._tokenizer()
        ids, mask = self._encode(data["text"])
        y = np.array([self.class_order.index(i) for i in data["intent"]])
        s = self.strategy
        self.model, _, _, _, self.training_seconds = self._run(s["unfrozen_blocks"], s["learning_rate"],
                                                               s["schedule_epochs"], ids, mask, y,
                                                               stop_epoch=s["epochs"])
        self.model.eval()
        self.trained_on = sorted(set(data["split"]))
        return self

    # ------------------------------------------------------------------ inference / io
    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 1:  # the notebook's torch_predict_one: no padding to max_length
            import torch
            enc = self.tokenizer(texts, truncation=True, max_length=self.cfg["text"]["max_tokens"],
                                 return_tensors="pt")
            with torch.no_grad():
                logits = self.model(input_ids=enc["input_ids"].to(self.device),
                                    attention_mask=enc["attention_mask"].to(self.device)).logits
            return torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
        ids, mask = self._encode(texts)
        return self._proba(self.model, ids, mask).astype(np.float64)

    def save(self, directory: Path) -> list[Path]:
        import torch
        directory.mkdir(parents=True, exist_ok=True)
        dtype = {"float16": torch.float16, "float32": torch.float32}[self.cfg["training"]["save_dtype"]]
        self.model.to(dtype).save_pretrained(directory, safe_serialization=True)
        self.model.to(torch.float32)
        self.tokenizer.save_pretrained(directory)
        save_json({"class_order": self.class_order, "strategy": self.strategy, "pretrained": self.cfg["pretrained"],
                   "max_tokens": self.cfg["text"]["max_tokens"], "save_dtype": self.cfg["training"]["save_dtype"],
                   "selection": self.selection}, directory / SPEC)
        if self.history is not None:
            self.history.to_csv(directory / "training_history.csv", index=False)
        return sorted(p for p in directory.iterdir() if p.is_file())

    @classmethod
    def load(cls, directory: Path, cfg: dict, device: str | None = None) -> "DistilBertModel":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        directory = Path(directory)
        m = cls(cfg)
        spec = load_json(directory / SPEC)
        if spec["class_order"] != m.class_order:
            raise ValueError("DistilBERT artifact class order differs from config")
        m.device = _device(device or cfg["training"]["device"])
        m.model = AutoModelForSequenceClassification.from_pretrained(directory, dtype=torch.float32,
                                                                     attn_implementation="eager").to(m.device).eval()
        id2label = [m.model.config.id2label[i] for i in range(len(m.class_order))]
        if id2label != m.class_order:
            raise ValueError(f"DistilBERT label mapping {id2label} differs from the canonical order")
        m.tokenizer = AutoTokenizer.from_pretrained(directory)
        m.strategy, m.selection = spec["strategy"], spec.get("selection", {})
        return m

    def n_parameters(self) -> int | None:
        return int(sum(p.numel() for p in self.model.parameters()))
