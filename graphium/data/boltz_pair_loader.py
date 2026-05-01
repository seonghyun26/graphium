"""Lookup cache for precomputed Boltz-2 pair embeddings keyed by canonical SMILES.

Output of `scripts/boltz/02_extract_boltz_pair.py`: a HDF5 file with one group
per molecule (key = sha1_id of canonical SMILES). Each group contains:
  - `z`:    (N, N, 128) fp16 — pair representation, atoms in graphium canonical order
  - `perm`: (N,)        int32 — Boltz->Graphium atom permutation (saved for debugging)
  - attrs: `smiles`, `n_atoms`

This loader is worker-safe: the H5 file handle is opened lazily per process so
that DataLoader workers (which fork after init) each get their own handle.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def sha1_id(canonical_smiles: str) -> str:
    return hashlib.sha1(canonical_smiles.encode("utf-8")).hexdigest()[:16]


class BoltzPairCache:
    """Per-process H5 cache. Multi-shard layout supported via `h5_paths` list."""

    def __init__(self, h5_paths: Union[str, Path, Sequence[Union[str, Path]]]):
        if isinstance(h5_paths, (str, Path)):
            h5_paths = [h5_paths]
        self.h5_paths: list[Path] = [Path(p) for p in h5_paths]
        for p in self.h5_paths:
            if not p.exists():
                raise FileNotFoundError(f"BoltzPairCache: missing {p}")
        self._handles: list[h5py.File] | None = None
        self._key_to_shard: dict[str, int] | None = None
        self._owner_pid: int | None = None

    def _ensure_open(self) -> None:
        """Open per-process: handles are not fork-safe."""
        pid = os.getpid()
        if self._handles is None or self._owner_pid != pid:
            self._handles = [h5py.File(p, "r", swmr=True) for p in self.h5_paths]
            self._key_to_shard = {}
            for shard_idx, h in enumerate(self._handles):
                for k in h.keys():
                    self._key_to_shard.setdefault(k, shard_idx)
            self._owner_pid = pid

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._key_to_shard)

    def has(self, canonical_smiles: str) -> bool:
        self._ensure_open()
        return sha1_id(canonical_smiles) in self._key_to_shard

    def get(self, canonical_smiles: str) -> Optional[torch.Tensor]:
        """Return (N, N, 128) fp32 tensor in graphium canonical atom order, or None."""
        self._ensure_open()
        key = sha1_id(canonical_smiles)
        shard = self._key_to_shard.get(key)
        if shard is None:
            return None
        z = self._handles[shard][key]["z"][:]  # (N, N, 128) fp16
        return torch.from_numpy(z.astype(np.float32))

    def get_by_raw_smiles(self, raw_smiles: str) -> Optional[torch.Tensor]:
        canon = canonicalize_smiles(raw_smiles)
        if canon is None:
            return None
        return self.get(canon)

    def close(self) -> None:
        if self._handles is not None:
            for h in self._handles:
                try:
                    h.close()
                except Exception:
                    pass
            self._handles = None

    def __del__(self) -> None:
        self.close()
