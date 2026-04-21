"""MolE (gin_concat) 256-d embedding adapter.

Reuses the upstream-repo checkout + pretrained-checkpoint helpers from
``scripts/mole/mole_eval.py`` (ADMET path), so cloning / unpacking only
happens once per host. Bisects on failure because MolE's upstream
``BOND_LIST`` hardcodes SINGLE/DOUBLE/TRIPLE/AROMATIC only — DATIVE,
HYDROGEN, ZERO bonds (common in patent organometallics) crash the whole
chunk with ``ValueError: ... is not in list``.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .base import MoleculeEncoder, bisect_embed


class MolEEncoder(MoleculeEncoder):
    encoder_tag: str = "mole_gin_concat"
    # Lazy — depends on the upstream checkpoint. The current default
    # (gin_concat_R1000_E8000_lambda0.0001) produces 1000-d embeddings; older
    # R256 variants emit 256-d. Inferred on the first ``extract()`` call.
    out_dim: int = -1

    def __init__(self, *, device: str = "cpu", batch_size: int = 2048):
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._batch_representation = None
        self._model_name = None
        self._upstream_commit = None

    def _ensure_model(self):
        if self._model is not None:
            return

        # Delegate asset management to the existing ADMET MolE script so we
        # don't duplicate the git-clone / checkpoint-symlink logic.
        scripts_mole = Path(__file__).resolve().parents[2] / "scripts" / "mole"
        sys.path.insert(0, str(scripts_mole))
        from mole_eval import (  # type: ignore  # noqa: E402
            MOLE_MODEL_NAME,
            UPSTREAM_COMMIT,
            ensure_upstream_checkpoint_layout,
        )
        self._model_name = MOLE_MODEL_NAME
        self._upstream_commit = UPSTREAM_COMMIT

        repo_dir = ensure_upstream_checkpoint_layout()
        sys.path.insert(0, str(repo_dir))
        from dataset.dataset_representation import (  # type: ignore  # noqa: E402
            batch_representation,
            load_pretrained_model,
        )
        self._batch_representation = batch_representation

        with contextlib.redirect_stdout(io.StringIO()):
            self._model = load_pretrained_model(
                pretrain_architecture="gin_concat",
                pretrained_model=MOLE_MODEL_NAME,
                pretrained_dir=str(repo_dir / "ckpt"),
                device=self._device,
            )

    @property
    def metadata(self) -> dict:
        """Model-provenance fields suitable for the results CSV."""
        return {
            "embedding_model": self._model_name or "",
            "upstream_commit": self._upstream_commit or "",
            "device":          self._device,
        }

    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_model()

        def _embed_batch(batch: List[str]) -> List[np.ndarray]:
            smile_df = pd.DataFrame({"chem_id": batch, "smiles": batch})
            embs_df = self._batch_representation(
                smile_df, self._model, batch_size=self._batch_size, device=self._device,
            )
            # Upstream returns a DataFrame indexed by chem_id; preserve input order.
            return [embs_df.loc[s].to_numpy(dtype=np.float32) for s in batch]

        # MolE can handle fairly large chunks; bisection kicks in only on
        # pathological SMILES so cap chunk size at batch_size.
        results, mask = bisect_embed(
            _embed_batch, smiles, self._batch_size, desc="mole",
        )
        survivors = [r for r, ok in zip(results, mask) if ok]
        if not survivors:
            if self.out_dim < 0:
                raise RuntimeError(
                    "MolEEncoder: all SMILES failed on the first call; cannot infer out_dim. "
                    "Pass at least one valid SMILES first."
                )
            return np.zeros((0, self.out_dim), dtype=np.float32), mask
        feats = np.stack(survivors, axis=0).astype(np.float32)
        if self.out_dim < 0:
            self.out_dim = int(feats.shape[1])
        return feats, mask
