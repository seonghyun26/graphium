#!/usr/bin/env python
"""One-shot variant: node MMS + pair 1-hop-only stats (drop diag and global
strata from the pair pool, keeping only mean+std over bonded pairs).

Motivation:
  - diag pair entries (i==i) overlap with node features → redundant
  - global pair entries (any valid i,j) average over far pairs → noisy
  - 1-hop pair entries (bonded i↔j) carry the chemically meaningful pair signal

Per layer:
    node MMS         = 3 stats * 96    = 288
    pair 1-hop stats = 2 stats * 224   = 448
    per layer        = 288 + 448       = 736
Two layers (4, 8):
    fingerprint      = 2 * 736         = 1472-d

Output: results/admet_node_mms_pair_1hop_oneshot.csv
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


def _node_mms(node_feats, batch_idx):
    """Pool node features per graph with max + mean + std → (B, 3*D_node)."""
    from torch_geometric.nn import global_mean_pool, global_max_pool

    z_max  = global_max_pool(node_feats, batch_idx)
    z_mean = global_mean_pool(node_feats, batch_idx)
    z_meansq = global_mean_pool(node_feats * node_feats, batch_idx)
    z_var = (z_meansq - z_mean * z_mean).clamp_min(0.0)
    z_std = torch.sqrt(z_var + 1e-8)
    return torch.cat([z_max, z_mean, z_std], dim=-1)


def _pair_1hop_stats(z, pair_mask, edge_index, batch_idx):
    """Stats (mean + std) over BONDED pairs only → (B, 2*D_z).

    Drops the diag and global strata from the standard `_pool_pair_stats`,
    keeping only the chemically meaningful 1-hop entries.
    """
    from torch_geometric.utils import to_dense_adj

    B, N, _ = pair_mask.shape
    device = pair_mask.device
    dtype = pair_mask.dtype

    # Bonded mask: dense adj > 0, ANDed with valid pair mask.
    adj = to_dense_adj(edge_index, batch=batch_idx, max_num_nodes=N).to(dtype=dtype)
    mask_1hop = (adj > 0).to(dtype=dtype) * pair_mask                    # (B, N, N)

    m4 = mask_1hop.unsqueeze(-1)                                          # (B, N, N, 1)
    raw_count = mask_1hop.sum(dim=(1, 2))                                 # (B,)
    count = raw_count.clamp(min=1.0).unsqueeze(-1)                        # (B, 1)
    sum_z  = (z * m4).sum(dim=(1, 2))                                     # (B, D_z)
    sum_z2 = (z * z * m4).sum(dim=(1, 2))
    mean = sum_z / count
    mean_sq = sum_z2 / count
    var = (mean_sq - mean * mean).clamp(min=1e-6)
    std = var.sqrt()
    has_any = (raw_count > 0).to(z.dtype).unsqueeze(-1)                   # (B, 1)

    return torch.cat([mean * has_any, std * has_any], dim=-1)             # (B, 2*D_z)


class NodeMMSPair1hopEncoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = (
            self.encoder_tag
            + f"_node_mms_pair_1hop_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}"
        )

    def _ensure_backbone(self):
        super()._ensure_backbone()
        self._backbone.gnn._enable_readout_cache()
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
                desc=f"GNN forward (NODE-MMS + PAIR-1HOP[{LAYER_IDS[0]},{LAYER_IDS[1]}])",
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

                node_cache = backbone.gnn._readout_cache
                pair_stack = getattr(g, "pair_feat_stack", None) or {}
                pair_mask = g.pair_mask
                edge_index = g.edge_index
                batch_idx = g.batch

                per_layer = []
                for li in LAYER_IDS:
                    z_node = _node_mms(node_cache[li], batch_idx)         # (B, 288)
                    z_pair = _pair_1hop_stats(
                        pair_stack.get(li, g.pair_feat),
                        pair_mask, edge_index, batch_idx,
                    )                                                     # (B, 448)
                    per_layer.append(torch.cat([z_node, z_pair], dim=-1)) # (B, 736)
                z = torch.cat(per_layer, dim=-1)                          # (B, 1472)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "NodeMMSPair1hopEncoder")

    out_csv = ROOT / "results" / "admet_node_mms_pair_1hop_oneshot.csv"
    cache_dir = (
        ROOT / "datacache"
        / f"pairmixer_probe_fps_node_mms_pair_1hop_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}"
    )

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag",
        f"4ds_ep099_node_mms_pair_1hop_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}",
        "--pretrain-label",
        f"4ds-ep099-node-mms-pair-1hop-layers{LAYER_IDS[0]}_{LAYER_IDS[1]}-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", "cuda:2",
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
