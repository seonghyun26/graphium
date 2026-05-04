#!/usr/bin/env python
"""One-shot variant: skip graph_output_nn, pool the PAIR representation (not
the node track) at layers 4 and 8 (1-indexed → 0-indexed cache keys 3 and 7).

PairMixer's pair feat is dense (B, N, N, D_z=224 for pairmixer_12M). For each
captured layer we apply (max + mean + std) over valid pairs (via ``pair_mask``)
to get a per-graph (3 * D_z) vector, then concat both layers
→ fingerprint dim = 2 * 3 * 224 = 1344.

This is the architecture-aware probe — pair features are PairMixer's
distinguishing track and were never pooled in any prior arm.

Output: results/admet_pair_layers48_mms_oneshot.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from downstream.model import _REGISTRY
from downstream.model.pairmixer import PairMixerEncoder


CKPT = (
    "models_checkpoints/toymix_dti_esmc_v2_lpm24_litopenai_bbbc047/"
    "pairmixer_12M/2026-05-01_01-59-58_20260501_015958/"
    "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_"
    "epochepoch=099_20260501_015958.ckpt"
)

REPRESENTATIVE_TASKS = [
    "caco2_wang", "bbb_martins", "cyp2d6_veith",
    "clearance_microsome_az", "ld50_zhu",
]

LAYER_IDS = (3, 7)  # 1-indexed "4th, 8th"


def _pair_pool_max_mean_std(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool dense pair features (B, N, N, D_z) over valid pairs into (B, 3*D_z).

    Stats: max + mean + std (population variance, clamped for numerics).
    Invalid pairs are zero-weighted in mean/std and -inf-substituted before max.
    """
    m4 = mask.unsqueeze(-1).to(z.dtype)                     # (B, N, N, 1)
    count = mask.sum(dim=(1, 2)).clamp_min(1).to(z.dtype).unsqueeze(-1)  # (B, 1)

    z_sum   = (z * m4).sum(dim=(1, 2))                      # (B, D_z)
    z_mean  = z_sum / count
    z2_sum  = (z * z * m4).sum(dim=(1, 2))
    z2_mean = z2_sum / count
    z_var   = (z2_mean - z_mean * z_mean).clamp_min(0.0)
    z_std   = torch.sqrt(z_var + 1e-8)

    # Mask invalid entries with the dtype's min before reducing for max.
    neg_large = torch.finfo(z.dtype).min
    z_for_max = torch.where(mask.unsqueeze(-1).bool(), z, torch.full_like(z, neg_large))
    z_max     = z_for_max.amax(dim=(1, 2))                  # (B, D_z)

    return torch.cat([z_max, z_mean, z_std], dim=-1)        # (B, 3*D_z)


class PairLayers48MMSEncoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = (
            self.encoder_tag + f"_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_mms"
        )

    def _ensure_backbone(self):
        super()._ensure_backbone()
        # Enable pair-feature capture at the requested layers BEFORE forward.
        self._backbone.gnn._pair_capture_layers = set(LAYER_IDS)

    def _forward_graphs(self, graphs):
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader

        backbone = self._backbone
        device = self._torch_device
        target_dtype = next(backbone.parameters()).dtype
        depth = len(backbone.gnn.layers)
        for li in LAYER_IDS:
            if not (0 <= li < depth):
                raise ValueError(f"layer id {li} out of range for depth {depth}")

        loader = DataLoader(
            graphs,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(graphium_collate_fn, mask_nan=0),
            num_workers=0,
        )
        chunks = []
        backbone.eval()
        n_batches = (len(graphs) + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for batch in tqdm(
                loader, total=n_batches,
                desc=f"GNN forward (PAIR concat[{LAYER_IDS[0]},{LAYER_IDS[1]}]+max,mean,std)",
                unit="batch", smoothing=0.05,
            ):
                batch = batch.to(device)
                keys = batch.keys() if callable(getattr(batch, "keys", None)) else list(batch.keys)
                for k in list(keys):
                    v = batch[k]
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        batch[k] = v.to(target_dtype)
                g = backbone.encoder_manager(batch)
                if backbone.pre_nn is not None:
                    g["feat"] = backbone.pre_nn.forward(g["feat"])
                if backbone.pre_nn_edges is not None:
                    e = g["edge_feat"]
                    if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                        e = torch.zeros(
                            list(e.shape[:-1]) + [backbone.pre_nn_edges.out_dim],
                            device=e.device, dtype=e.dtype,
                        )
                    else:
                        e = backbone.pre_nn_edges.forward(e)
                    g["edge_feat"] = e
                _ = backbone.gnn.forward(g)

                # Pull pair tensors from the per-layer capture stack. The last
                # layer is also exposed as g.pair_feat — fall back to it for
                # the index that equals depth - 1.
                stack = getattr(g, "pair_feat_stack", None) or {}
                pair_mask = g.pair_mask                      # (B, N, N)
                per_layer = []
                for li in LAYER_IDS:
                    z_layer = stack.get(li, g.pair_feat)     # (B, N, N, D_z)
                    per_layer.append(_pair_pool_max_mean_std(z_layer, pair_mask))
                z = torch.cat(per_layer, dim=-1)             # (B, 2 * 3 * D_z)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "PairLayers48MMSEncoder")

    out_csv = ROOT / "results" / "admet_pair_layers48_mms_oneshot.csv"
    cache_dir = (
        ROOT / "datacache"
        / f"pairmixer_probe_fps_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_mms"
    )

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", f"4ds_ep099_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_mms",
        "--pretrain-label",
        f"4ds-ep099-pair-layers{LAYER_IDS[0]}_{LAYER_IDS[1]}-mms-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", "cuda:1",
        # Pair pool memory is O(B * N² * D_z) — keep batch small for big mols.
        "--embed-batch-size", "16",
        "--featurize-n-jobs", "8",
        "--mol-cache-dir", str(cache_dir),
        "--output-csv", str(out_csv),
        "--sweep-results", "/dev/null",
    ]

    import importlib.util
    pmp_path = Path(__file__).parent / "pairmixer_minimol_probe.py"
    spec = importlib.util.spec_from_file_location("pairmixer_minimol_probe", str(pmp_path))
    pmp = importlib.util.module_from_spec(spec)
    sys.modules["pairmixer_minimol_probe"] = pmp
    spec.loader.exec_module(pmp)
    pmp.main()


if __name__ == "__main__":
    main()
