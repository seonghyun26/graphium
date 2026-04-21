"""TDC DTI regression data loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PROT_EMB_COLS, SUBSET_LABEL_COL


@dataclass
class SubsetArrays:
    smiles:    list[str]
    label:     np.ndarray      # (N,) float32 — z-scored at prep time
    prot_emb:  np.ndarray      # (N, 1280) float32
    label_col: str             # pY or kiba_score


def load_subset(data_dir: str | Path, subset: str) -> SubsetArrays:
    if subset not in SUBSET_LABEL_COL:
        raise ValueError(f"Unknown subset {subset!r}. Known: {list(SUBSET_LABEL_COL)}")
    parquet_path = Path(data_dir) / f"{subset}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} missing. Run scripts/tdc_dti_regression/prepare_data.sh first."
        )
    label_col = SUBSET_LABEL_COL[subset]
    df = pd.read_parquet(
        parquet_path,
        columns=["SMILES_nometa", label_col, *PROT_EMB_COLS],
    )
    return SubsetArrays(
        smiles=df["SMILES_nometa"].tolist(),
        label=df[label_col].values.astype(np.float32),
        prot_emb=df[PROT_EMB_COLS].values.astype(np.float32),
        label_col=label_col,
    )


def load_split(data_dir: str | Path, subset: str, method: str, seed: int) -> dict[str, list[int]]:
    path = Path(data_dir) / "splits" / f"{subset}_{method}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing.")
    splits = torch.load(path, weights_only=False)
    for key in ("train", "val", "test"):
        if key not in splits:
            raise ValueError(f"{path} missing key {key!r}")
    return splits
