"""Data loaders for the GRAM-DTI benchmark.

Expects the per-subset parquets + split ``.pt`` files produced by
``scripts/data/gram_dti/02_prepare_gram_dti.py`` (or the legacy
``scripts/data/dti_classif_eval/02_prepare_dti_classif.py`` with its output
relocated to ``data/downstream/gram_dti/``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PROT_EMB_COLS


@dataclass
class SubsetArrays:
    """Dense arrays for one benchmark subset — loaded once, reused across folds."""
    smiles:    list[str]
    label:     np.ndarray   # (N,) int8 {0, 1}
    prot_emb:  np.ndarray   # (N, 1280) float32 — z-scored at prep time
    drug_id:   np.ndarray   # (N,) object
    target_id: np.ndarray   # (N,) object


def load_subset(data_dir: str | Path, subset: str) -> SubsetArrays:
    """Load a per-subset parquet; columns = SMILES_nometa, label, drug_id,
    target_id, prot_emb_0..prot_emb_(ESM2_DIM-1)."""
    parquet_path = Path(data_dir) / f"{subset}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} missing. Run scripts/gram_dti/prepare_data.sh first."
        )
    df = pd.read_parquet(parquet_path)
    missing = set(PROT_EMB_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{parquet_path} is missing {len(missing)} protein-embedding columns "
            f"(expected ESM-2 1280-d). First few: {sorted(missing)[:3]}. "
            "Rebuild the parquet with ESM-2 embeddings."
        )
    return SubsetArrays(
        smiles=df["SMILES_nometa"].tolist(),
        label=df["label"].values.astype(np.int8),
        prot_emb=df[PROT_EMB_COLS].values.astype(np.float32),
        drug_id=df["drug_id"].values,
        target_id=df["target_id"].values,
    )


def load_split(data_dir: str | Path, subset: str, method: str, fold: int) -> dict[str, list[int]]:
    """Return {'train', 'val', 'test'} index lists for one (subset, method, fold)."""
    path = Path(data_dir) / "splits" / f"{subset}_{method}_fold{fold}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing.")
    splits = torch.load(path, weights_only=False)
    for key in ("train", "val", "test"):
        if key not in splits:
            raise ValueError(f"{path} missing key {key!r}")
    return splits
