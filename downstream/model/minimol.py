"""Minimol 512-d embedding adapter.

Wraps the ``minimol`` pip package. Featurizer can fail on SMILES with
kekulize errors, weird valences, or organometallic bonds — a single bad
SMILES poisons the whole ``model(batch)`` call — so we bisect failing chunks
down to singletons and drop the ones that still fail.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import MoleculeEncoder, bisect_embed


class MinimolEncoder(MoleculeEncoder):
    encoder_tag: str = "minimol_v1"
    out_dim: int = 512

    def __init__(self, *, chunk_size: int = 128):
        self._chunk_size = chunk_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            # minimol chdirs into its own datacache at import time, so import
            # lazily and BEFORE any TDC-group download logic (which also chdirs).
            from minimol import Minimol
            self._model = Minimol()

    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_model()

        def _embed_batch(batch: List[str]) -> List:
            return list(self._model(batch))

        results, mask = bisect_embed(_embed_batch, smiles, self._chunk_size, desc="minimol")
        survivors = [r for r, ok in zip(results, mask) if ok]
        if not survivors:
            return np.zeros((0, self.out_dim), dtype=np.float32), mask

        feats = np.stack(
            [v.detach().cpu().float().numpy() for v in survivors], axis=0,
        ).astype(np.float32)
        return feats, mask
