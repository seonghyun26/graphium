"""ECFP4 (Morgan r=2) fingerprint baseline.

Not really a learned encoder, but fits the same ``MoleculeEncoder`` interface
so every downstream task can treat ECFP as just another encoder choice.
Default bit count = 1024 to match the paper baselines in Fredinh et al. 2024
(``scripts/dep_baseline_mlp_cell_bioactivity.py``) and GRAM-DTI ablations.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import MoleculeEncoder


class ECFPEncoder(MoleculeEncoder):
    def __init__(self, *, n_bits: int = 1024, radius: int = 2):
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        self.out_dim = self.n_bits
        self.encoder_tag = f"ecfp{2 * self.radius}_r{self.radius}_{self.n_bits}"

    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        # Lazy import — rdkit is heavy and not every downstream task needs it.
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        feats = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        mask = np.zeros(len(smiles), dtype=bool)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((self.n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            feats[i] = arr
            mask[i] = True
        # Return only survivors to match the base-class contract.
        return feats[mask], mask
