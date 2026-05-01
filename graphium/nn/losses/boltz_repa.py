"""REPA-style alignment loss between PairMixer's final-layer pair representation
and a precomputed Boltz-2 pair embedding (`(N, N, D_b)`, D_b=128).

Trained representation: `pair_feat_g: (B, N_max, N_max, D_g)` from PairMixer's
last layer.

Frozen target: `pair_target_b: (B, N_max, N_max, D_b)` loaded from H5 cache,
already permuted to graphium canonical atom order in P2.

Loss: cosine similarity per atom-pair, masked by valid pairs and by per-mol
presence (a mol may not have a Boltz target if precompute failed for it).

  L = 1 - mean_over_valid_pairs(cos(MLP(pair_feat_g), pair_target_b))

Following the REPA paper (Yu et al. 2024), the projector is a small learnable
MLP: D_g -> hidden -> D_b, applied per atom-pair.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoltzRepaLoss(nn.Module):
    def __init__(
        self,
        d_pair_g: int,
        d_pair_b: int = 128,
        hidden: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        h = hidden if hidden is not None else max(d_pair_g, d_pair_b)
        self.projector = nn.Sequential(
            nn.Linear(d_pair_g, h),
            nn.SiLU(),
            nn.Linear(h, d_pair_b),
        )
        self.loss_weight = loss_weight
        self.d_pair_g = d_pair_g
        self.d_pair_b = d_pair_b

    def forward(
        self,
        pair_feat_g: torch.Tensor,        # (B, N_max, N_max, D_g)
        pair_target_b: torch.Tensor,       # (B, N_max, N_max, D_b)
        pair_mask: torch.Tensor,           # (B, N_max, N_max), True = valid pair
        present_mask: torch.Tensor,        # (B,), True = mol has Boltz target
    ) -> torch.Tensor:
        if present_mask.dtype != torch.bool:
            present_mask = present_mask.bool()
        if pair_mask.dtype != torch.bool:
            pair_mask = pair_mask.bool()

        # Effective pair mask: valid pair AND mol has target
        eff_mask = pair_mask & present_mask.view(-1, 1, 1)
        n_valid = eff_mask.sum()
        if n_valid == 0:
            return pair_feat_g.new_zeros(())

        proj_g = self.projector(pair_feat_g)  # (B, N_max, N_max, D_b)

        # Cosine sim per pair
        cos = F.cosine_similarity(proj_g, pair_target_b, dim=-1)  # (B, N_max, N_max)
        cos_masked = cos * eff_mask
        mean_cos = cos_masked.sum() / n_valid.clamp(min=1).to(cos_masked.dtype)

        return self.loss_weight * (1.0 - mean_cos)

    def extra_repr(self) -> str:
        return f"d_pair_g={self.d_pair_g}, d_pair_b={self.d_pair_b}, loss_weight={self.loss_weight}"
