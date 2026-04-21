"""Abstract molecule-encoder interface for downstream task eval.

Uniform contract:
    extract(smiles) -> (features, mask)

    - ``mask`` is length-N bool; True = encoder produced an embedding.
    - ``features`` has shape ``(mask.sum(), out_dim)`` in input order over
      survivors.
    - Callers drop rows where ``mask`` is False from labels / aux features
      before stacking feature matrices.

Failures are expected — real-world benchmarks include SMILES that graphium /
minimol / mole can't featurize (organometallics, kekulize errors, etc.). The
two-return shape forces the task layer to handle that rather than silently
crashing mid-sweep.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


class MoleculeEncoder(abc.ABC):
    """Base class: implement ``extract`` and set ``encoder_tag`` + ``out_dim``."""

    encoder_tag: str
    out_dim: int

    @abc.abstractmethod
    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Embed SMILES; return (features_of_survivors, mask). See module docstring."""

    def extract_cached(
        self, smiles: List[str], cache_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``extract`` + per-SMILES disk cache.

        Cache stores ``None`` sentinels for SMILES the encoder has previously
        failed on — avoids retrying known-bad molecules on every sweep. The
        cache persists across subsets so a drug appearing in multiple DTI/MoA
        datasets only pays the cost once.
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache: Dict[str, Optional[np.ndarray]] = (
            torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
        )

        missing = [s for s in dict.fromkeys(smiles) if s not in cache]
        if missing:
            print(
                f"  [{self.encoder_tag}] embedding {len(missing):,} new SMILES...",
                flush=True,
            )
            feats, mask = self.extract(missing)
            if feats.size and feats.shape[1] != self.out_dim:
                raise ValueError(
                    f"{self.encoder_tag}.extract feature dim {feats.shape[1]} "
                    f"!= declared out_dim {self.out_dim}"
                )
            survivor_iter = iter(feats)
            for s, alive in zip(missing, mask):
                cache[s] = np.asarray(next(survivor_iter), dtype=np.float32) if alive else None
            torch.save(cache, cache_path)

        mask = np.array([cache.get(s) is not None for s in smiles], dtype=bool)
        if mask.any():
            out = np.stack(
                [cache[s] for s in smiles if cache.get(s) is not None], axis=0,
            ).astype(np.float32)
        else:
            out = np.zeros((0, self.out_dim), dtype=np.float32)
        dropped = int((~mask).sum())
        if dropped:
            print(f"  [{self.encoder_tag}] WARN: {dropped:,} SMILES un-embeddable; rows dropped.")
        return out, mask


def bisect_embed(
    embed_fn: Callable[[List[str]], List[object]],
    smiles: List[str],
    chunk_size: int,
    desc: str = "embedding",
) -> Tuple[List[Optional[object]], np.ndarray]:
    """Embed via recursive bisection.

    Several encoders crash the whole batch when a single SMILES is invalid
    (minimol: kekulize errors; mole: DATIVE bonds not in BOND_LIST). Bisecting
    failing chunks down to singletons isolates bad molecules so we only drop
    the ones that actually fail.

    Returns ``(per_smiles_result, mask)`` where each entry is the encoder's raw
    return value (or ``None`` on failure) and ``mask`` aligns to input order.
    """
    from tqdm import tqdm

    output: List[Optional[object]] = [None] * len(smiles)
    stack: List[List[int]] = []
    for i in range(0, len(smiles), chunk_size):
        stack.append(list(range(i, min(i + chunk_size, len(smiles)))))

    pbar = tqdm(total=len(smiles), desc=desc, unit="mol", smoothing=0.05)
    while stack:
        chunk_idx = stack.pop()
        if not chunk_idx:
            continue
        batch = [smiles[i] for i in chunk_idx]
        try:
            embs = embed_fn(batch)
            for i, e in zip(chunk_idx, embs):
                output[i] = e
            pbar.update(len(chunk_idx))
        except Exception:
            if len(chunk_idx) == 1:
                pbar.update(1)  # leave output[idx] = None
                continue
            mid = len(chunk_idx) // 2
            stack.append(chunk_idx[mid:])
            stack.append(chunk_idx[:mid])
    pbar.close()

    mask = np.array([v is not None for v in output], dtype=bool)
    return output, mask
