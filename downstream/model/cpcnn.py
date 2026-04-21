"""CPCNN cell-painting embedding baseline (not a molecule encoder per se).

The Fredinh et al. 2024 bioactivity paper compares chemistry-only baselines
(ECFP) against cell-painting baselines (CPCNN 672-d). Only compounds with at
least one Cell Painting well get a CPCNN profile, so this encoder returns
``mask=False`` for everything else and the task layer drops those rows.

Embeddings are mean-aggregated per SMILES from an external CSV (the one
prepped by ``scripts/data/jump_cpcnn/`` / Cell Painting CPCNN pipeline).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .base import MoleculeEncoder


class CPCNNEncoder(MoleculeEncoder):
    def __init__(self, *, cpcnn_csv: str | Path):
        self._cpcnn_csv = Path(cpcnn_csv)
        if not self._cpcnn_csv.exists():
            raise FileNotFoundError(
                f"CPCNN CSV not found: {self._cpcnn_csv}. Expected a per-well CSV with "
                "``SMILES_nometa`` + ``feature_*`` columns."
            )
        self._lookup: dict[str, np.ndarray] | None = None
        self.out_dim: int = -1
        self.encoder_tag: str = "cpcnn_jump"

    def _ensure_lookup(self) -> None:
        if self._lookup is not None:
            return
        df = pd.read_csv(self._cpcnn_csv)
        if "SMILES_nometa" not in df.columns:
            raise ValueError(
                f"{self._cpcnn_csv} missing SMILES_nometa; got {list(df.columns)[:8]}..."
            )
        feat_cols = [c for c in df.columns if c.startswith("feature_")]
        if not feat_cols:
            raise ValueError(f"{self._cpcnn_csv} has no ``feature_*`` columns.")
        # Average replicate wells per compound; paper does the same.
        agg = df.groupby("SMILES_nometa", sort=False)[feat_cols].mean()
        self._lookup = {
            s: row.to_numpy(dtype=np.float32) for s, row in agg.iterrows()
        }
        self.out_dim = len(feat_cols)

    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_lookup()
        mask = np.array([s in self._lookup for s in smiles], dtype=bool)
        if not mask.any():
            return np.zeros((0, self.out_dim), dtype=np.float32), mask
        feats = np.stack(
            [self._lookup[s] for s, m in zip(smiles, mask) if m], axis=0,
        ).astype(np.float32)
        return feats, mask
