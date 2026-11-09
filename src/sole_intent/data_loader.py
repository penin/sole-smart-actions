"""Load and validate the Sole Smart Actions workbook and expose the predefined splits.

The splits supplied in the workbook are preserved exactly. These helpers never drop,
re-label or re-split rows; they only select them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils import resolve_path, sha256_file, sha256_text

REQUIRED_COLUMNS = ["text", "intent", "split", "scope"]
VALID_SPLITS = ["train", "validation", "test", "evaluation_only"]


class DatasetError(RuntimeError):
    """Raised when the workbook is missing or violates the expected schema."""


@dataclass
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    unknown: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation,
                "test": self.test, "unknown": self.unknown}


def dataset_path(cfg: dict) -> Path:
    return resolve_path(cfg["data"]["path"])


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Read the 'Training Data' sheet. Adds ``row_id`` (0-based) and ``excel_row`` (1-based,
    including the header) so every prediction can be traced back to the workbook."""
    path = dataset_path(cfg)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    # Only genuinely empty cells become NaN; strings such as "NA" or "None" stay text.
    df = pd.read_excel(path, sheet_name=cfg["data"]["sheet"], dtype=str,
                       keep_default_na=False, na_values=[""], engine="openpyxl")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df.insert(0, "row_id", range(len(df)))
    df.insert(1, "excel_row", df["row_id"] + 2)
    return df


def validate_schema(df: pd.DataFrame, cfg: dict) -> None:
    """Hard validation: raise rather than silently fixing anything."""
    labels = set(cfg["labels"])
    unknown = cfg["unknown_label"]
    errors = []
    if df[REQUIRED_COLUMNS].isna().any().any():
        errors.append(f"missing values: {df[REQUIRED_COLUMNS].isna().sum().to_dict()}")
    bad_intents = sorted(set(df["intent"].dropna()) - labels - {unknown})
    if bad_intents:
        errors.append(f"invalid intents: {bad_intents}")
    bad_splits = sorted(set(df["split"].dropna()) - set(VALID_SPLITS))
    if bad_splits:
        errors.append(f"invalid splits: {bad_splits}")
    if ((df["intent"] == unknown) & (df["split"] != "evaluation_only")).any():
        errors.append("UNKNOWN_INTENT rows found outside evaluation_only")
    if ((df["intent"] != unknown) & (df["split"] == "evaluation_only")).any():
        errors.append("supported-intent rows found in evaluation_only")
    if errors:
        raise DatasetError("; ".join(errors))


def get_splits(df: pd.DataFrame, cfg: dict) -> DataSplits:
    """Select the predefined partitions. Classifier data never contains UNKNOWN_INTENT."""
    unknown = cfg["unknown_label"]
    supported = df[df["intent"] != unknown]

    def pick(frame, split):
        return frame[frame["split"] == split].reset_index(drop=True)

    return DataSplits(
        train=pick(supported, "train"),
        validation=pick(supported, "validation"),
        test=pick(supported, "test"),
        unknown=df[(df["intent"] == unknown) & (df["split"] == "evaluation_only")].reset_index(drop=True),
    )


def dataset_fingerprint(df: pd.DataFrame, cfg: dict) -> dict:
    """File hash plus a content hash of the training sheet (robust to workbook metadata changes)."""
    content = df[REQUIRED_COLUMNS].to_csv(index=False, lineterminator="\n")
    return {
        "dataset_path": str(dataset_path(cfg)),
        "dataset_version": cfg["data"]["dataset_version"],
        "file_sha256": sha256_file(dataset_path(cfg)),
        "content_sha256": sha256_text(content),
        "n_rows": int(len(df)),
    }
