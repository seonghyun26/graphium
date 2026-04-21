"""Data loader for the cell-bioactivity benchmark.

Reads the CSV produced by ``scripts/data/bioactivity/01_prepare_bioactivity.py``:
columns are ``smiles, assay_<id>..., fold`` with 0/1/NaN labels (0=inactive,
1=active, NaN=unknown). The loader remaps to the paper's -1/0/+1 encoding
(0=unknown) which the focal-BCE masked loss in ``head.py`` consumes directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ASSAY_COLS, N_FOLDS


@dataclass
class BioactivityArrays:
    smiles: list[str]
    labels: np.ndarray     # (N, n_assays) in {-1, 0, +1}
    folds:  np.ndarray     # (N,) int8 in [0, N_FOLDS)
    assay_cols: list[str]


def load_dataset(data_dir: str | Path, csv_name: str = "cell_bioactivity.csv") -> BioactivityArrays:
    path = Path(data_dir) / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run scripts/bioactivity/prepare_data.sh first."
        )
    df = pd.read_csv(path)
    if "fold" not in df.columns:
        raise ValueError(f"{path} lacks the ``fold`` column; re-run the prep script.")
    if "smiles" not in df.columns:
        raise ValueError(f"{path} lacks the ``smiles`` column.")

    # Enforce the paper's assay ordering; missing assays become all-NaN columns.
    missing = [c for c in ASSAY_COLS if c not in df.columns]
    if missing:
        print(f"  WARN: {len(missing)} expected assay column(s) absent; treating as all-NaN: {missing[:3]}...")
        for c in missing:
            df[c] = np.nan

    raw = df[ASSAY_COLS].values.astype(np.float32)
    labels = np.where(
        np.isnan(raw), 0.0, np.where(raw > 0.5, 1.0, -1.0),
    ).astype(np.float32)

    folds = df["fold"].values.astype(np.int8)
    if folds.min() < 0 or folds.max() >= N_FOLDS:
        raise ValueError(f"folds outside [0, {N_FOLDS}): got range [{folds.min()}, {folds.max()}]")

    return BioactivityArrays(
        smiles=df["smiles"].tolist(),
        labels=labels,
        folds=folds,
        assay_cols=list(ASSAY_COLS),
    )


def split_indices(
    folds: np.ndarray, valid: np.ndarray, test_fold: int, n_folds: int = N_FOLDS,
) -> dict[str, np.ndarray]:
    """For the rotating 6-fold CV: test={test_fold}, val={(test_fold+1)%n}, train=rest.

    ``valid`` masks out compounds the chosen encoder couldn't embed (e.g. SMILES
    that fail upstream featurization). Rows with ``valid=False`` are excluded
    from all three splits so x and y stay aligned.
    """
    val_fold = (test_fold + 1) % n_folds
    idx = np.arange(len(folds))
    return {
        "train": idx[(~np.isin(folds, [test_fold, val_fold])) & valid],
        "val":   idx[(folds == val_fold)  & valid],
        "test":  idx[(folds == test_fold) & valid],
    }
