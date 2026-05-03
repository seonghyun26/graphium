"""REPA-style intermediate-layer alignment for multi-modal pretraining.

Following Yu et al. 2024 ("Representation Alignment for Generation: Training
Diffusion Transformers Is Easier Than You Think"), this loss aligns a
configurable mid-layer trunk representation against precomputed modality
embeddings via a small learnable MLP projector and negative cosine similarity:

    L_REPA = - lambda * E[ cos( h_phi(z_mid), y_target ) ]

For PairMixer, "trunk representation" is the pair tensor (B, N, N, D_z) — node
features are not updated layer-by-layer (s_backbone = s_init). We pool the
mid-layer pair tensor to graph level via masked mean, then apply a per-task
projector and cosine.

The aux/supervised split is intentional: ToyMix supervised tasks (qm9, tox21,
zinc) pre-train at the END (final-layer task heads), while the dense modality
embeddings (DTI ESM-C, LPM-24 OpenAI, BBBC047 Cell Painting) align at a MIDDLE
layer. This frees the deep layers to specialize for the supervised objective
while the prefix carries modality-aligned features (REPA paper §5.2 ablation).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RepaProjector(nn.Module):
    """Small MLP projector matching REPA paper convention (Linear -> SiLU -> Linear)."""

    def __init__(self, d_in: int, d_target: int, hidden: Optional[int] = None):
        super().__init__()
        h = int(hidden) if hidden is not None else max(d_in, d_target)
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.SiLU(),
            nn.Linear(h, d_target),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepaIntermediateAlignment(nn.Module):
    """Multi-task REPA alignment at a configured mid-layer pair representation.

    Args:
        d_pair: trunk pair-feature dim (D_z), e.g. 224 for PairMixer 12M.
        tasks:  dict mapping task_name -> {d_target, hidden, weight, label_key}.
                label_key is the key under which the precomputed modality target
                lives in the batch's labels dict (e.g. "graph_dti", "graph_lpm24").
    """

    def __init__(self, d_pair: int, tasks: Dict[str, Dict]):
        super().__init__()
        self.d_pair = int(d_pair)
        # Store a normalized copy so config order/keys are deterministic.
        self.tasks: Dict[str, Dict] = {
            str(name): {
                "d_target": int(cfg["d_target"]),
                "hidden": int(cfg["hidden"]) if cfg.get("hidden") is not None else max(d_pair, int(cfg["d_target"])),
                "weight": float(cfg.get("weight", 1.0)),
                "label_key": str(cfg["label_key"]),
            }
            for name, cfg in tasks.items()
        }
        self.projectors = nn.ModuleDict()
        for name, cfg in self.tasks.items():
            self.projectors[name] = _RepaProjector(
                d_in=d_pair,
                d_target=cfg["d_target"],
                hidden=cfg["hidden"],
            )

    @staticmethod
    def _pool_pair_mean(z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of (B, N, N, D_z) over both spatial dims -> (B, D_z)."""
        m = pair_mask.to(z.dtype).unsqueeze(-1)            # (B, N, N, 1)
        denom = m.sum(dim=(1, 2)).clamp(min=1.0)           # (B, 1)
        return (z * m).sum(dim=(1, 2)) / denom             # (B, D_z)

    def forward(
        self,
        pair_feat_mid: torch.Tensor,                       # (B, N, N, D_z)
        pair_mask: torch.Tensor,                           # (B, N, N) bool/0-1
        targets: Dict[str, torch.Tensor],                  # batch.labels dict
    ) -> torch.Tensor:
        """Sum of weighted per-task cosine alignment losses."""
        if pair_mask.dtype != pair_feat_mid.dtype:
            pair_mask_t = pair_mask.to(pair_feat_mid.dtype)
        else:
            pair_mask_t = pair_mask
        pooled = self._pool_pair_mean(pair_feat_mid, pair_mask_t)  # (B, D_z)

        total = pair_feat_mid.new_zeros(())
        for name, cfg in self.tasks.items():
            tgt = targets.get(cfg["label_key"], None)
            if tgt is None:
                continue
            tgt = tgt.to(pooled.dtype)
            # Reshape if the dataloader collated to (B*D_target,) or (N, 1).
            tgt = tgt.reshape(-1, cfg["d_target"])
            row_valid = ~tgt.isnan().any(dim=-1)
            if row_valid.sum() == 0:
                continue
            proj = self.projectors[name](pooled)               # (B, D_target)
            cos = F.cosine_similarity(
                proj[row_valid], tgt[row_valid], dim=-1
            )
            total = total + cfg["weight"] * (1.0 - cos.mean())
        return total

    def extra_repr(self) -> str:
        parts = [f"d_pair={self.d_pair}"]
        for name, cfg in self.tasks.items():
            parts.append(
                f"{name}(d_target={cfg['d_target']}, hidden={cfg['hidden']}, "
                f"weight={cfg['weight']}, label_key={cfg['label_key']!r})"
            )
        return "\n  ".join(parts)
