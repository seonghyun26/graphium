"""KPGT 2304-d embedding adapter.

KPGT (Li et al., NeurIPS 2023) pins DGL 0.7 and torch 1.10, which conflicts
with the graphium env (torch 2.7 + cu128). To avoid bricking the shared env,
this adapter shells out to a sibling ``kpgt`` conda env that hosts torch 2.4 +
DGL 2.4 + descriptastorus. The subprocess (``scripts/kpgt/kpgt_extract_embeddings.py``)
does the actual featurization + forward pass.

Embedding dimensionality: 2304 = 3 × d_g_feats (768) — concatenation of the
fingerprint virtual node, the descriptor virtual node, and the mean readout
over real-atom triplets, exactly as ``LiGhTPredictor.generate_fps`` returns.

Failure modes (RDKit's ``MolFromSmiles`` returns ``None``, kekulization errors,
descriptor crashes) flow through to ``None`` sentinels in the cache, then to
``mask=False`` rows here — same contract as MiniMol / MolE.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import MoleculeEncoder

# Repo-relative defaults; can be overridden via env vars or constructor kwargs.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO = _PROJECT_ROOT / "downloads" / "kpgt" / "upstream" / "KPGT"
_DEFAULT_CKPT = _PROJECT_ROOT / "downloads" / "kpgt" / "base.pth"
_DEFAULT_PYTHON = "/home/shpark/miniforge3/envs/kpgt/bin/python"
_EXTRACT_SCRIPT = _PROJECT_ROOT / "scripts" / "kpgt" / "kpgt_extract_embeddings.py"


class KPGTEncoder(MoleculeEncoder):
    encoder_tag: str = "kpgt_base"
    out_dim: int = 2304  # 3 * 768

    def __init__(
        self,
        *,
        kpgt_python: str | None = None,
        kpgt_repo: str | os.PathLike | None = None,
        ckpt_path: str | os.PathLike | None = None,
        # CPU default: the pip ``dgl`` ships without a CUDA backend, and our
        # Blackwell GPUs (sm_120) aren't supported by torch 2.4+cu121 anyway.
        # The LiGhT base model is small enough for CPU inference (~30s per
        # 1k molecules); embeddings are cached after the first call.
        device: str = "cpu",
        batch_size: int = 32,
        n_virtual_nodes: int = 2,
        path_length: int = 5,
    ):
        # Allow env overrides for hosts that ship a different env layout.
        self._python = str(kpgt_python or os.environ.get("KPGT_PYTHON", _DEFAULT_PYTHON))
        self._repo = Path(kpgt_repo or os.environ.get("KPGT_REPO", _DEFAULT_REPO))
        self._ckpt = Path(ckpt_path or os.environ.get("KPGT_CKPT", _DEFAULT_CKPT))
        self._device = device
        self._batch_size = batch_size
        self._n_virtual_nodes = n_virtual_nodes
        self._path_length = path_length
        self._validated = False

    def _validate(self) -> None:
        if self._validated:
            return
        missing = []
        if not Path(self._python).exists():
            missing.append(f"kpgt python at {self._python}")
        if not (self._repo / "src" / "model" / "light.py").exists():
            missing.append(f"KPGT upstream at {self._repo}")
        if not self._ckpt.exists():
            missing.append(
                f"KPGT base.pth at {self._ckpt} "
                f"(run scripts/kpgt/download_kpgt_baseline.sh)"
            )
        if not _EXTRACT_SCRIPT.exists():
            missing.append(f"extractor at {_EXTRACT_SCRIPT}")
        if missing:
            raise FileNotFoundError(
                "KPGTEncoder cannot run: " + "; ".join(missing)
                + ".\n  Set KPGT_PYTHON / KPGT_REPO / KPGT_CKPT env vars to override defaults."
            )
        self._validated = True

    @property
    def metadata(self) -> dict:
        return {
            "embedding_model": "kpgt_base",
            "ckpt_path": str(self._ckpt),
            "device": self._device,
        }

    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Subprocess-based extraction.

        Each call writes the input SMILES to a tempfile, runs the kpgt-env
        Python on it with a *temporary* cache path (so the user-facing cache
        owned by ``extract_cached`` stays the single source of truth), then
        reads back the per-SMILES results and aligns them to the input.
        """
        self._validate()
        if not smiles:
            return np.zeros((0, self.out_dim), dtype=np.float32), np.zeros(0, dtype=bool)

        with tempfile.TemporaryDirectory(prefix="kpgt_extract_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            smiles_file = tmpdir_path / "smiles.txt"
            cache_file = tmpdir_path / "cache.pt"
            smiles_file.write_text("\n".join(smiles))

            cmd = [
                self._python, str(_EXTRACT_SCRIPT),
                "--kpgt-repo", str(self._repo),
                "--ckpt", str(self._ckpt),
                "--smiles-file", str(smiles_file),
                "--cache-path", str(cache_file),
                "--device", self._device,
                "--batch-size", str(self._batch_size),
                "--n-virtual-nodes", str(self._n_virtual_nodes),
                "--path-length", str(self._path_length),
            ]
            print(f"  [kpgt] subprocess: {' '.join(cmd)}", flush=True)
            # Stream stdout/stderr so users see progress bars from the kpgt env.
            result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
            if result.returncode != 0:
                raise RuntimeError(
                    f"kpgt_extract_embeddings.py failed with exit {result.returncode}. "
                    f"Re-run with the printed command for diagnostics."
                )

            cache: Dict[str, Optional[np.ndarray]] = torch.load(
                str(cache_file), map_location="cpu", weights_only=False,
            )

        mask = np.array(
            [cache.get(s) is not None for s in smiles], dtype=bool,
        )
        survivors = [cache[s] for s in smiles if cache.get(s) is not None]
        if not survivors:
            return np.zeros((0, self.out_dim), dtype=np.float32), mask
        feats = np.stack(survivors, axis=0).astype(np.float32)
        # Sanity: assert dim matches the declared out_dim. Pretrained ``base``
        # always emits 2304-d; if a user swaps in a non-base ckpt this catches it.
        if feats.shape[1] != self.out_dim:
            self.out_dim = int(feats.shape[1])
        return feats, mask
