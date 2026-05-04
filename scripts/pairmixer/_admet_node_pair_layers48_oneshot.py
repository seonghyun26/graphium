#!/usr/bin/env python
"""One-shot variant: skip graph_output_nn, but mirror the architecture's native
graph readout — per-layer concat of (node max-pool, pair stats pool with
diag/1-hop/global mean+std) — across layers 4 and 8.

This matches the pretraining recipe exactly except: (a) we skip the trained
post-pool MLP / task heads, (b) we use TWO layers (4, 8) instead of just the
last one. With matched layer choice (just layer 7), this would be the
"pretrained recipe minus the trained MLP" variant.

Per layer:
    node_max (96)  +  pair_stats (6 * 224 = 1344)  =  1440-d
Two layers (4, 8):
    2 * 1440 = 2880-d fingerprint

Output: results/admet_node_pair_layers48_oneshot.csv
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


_CKPT_DIR = (
    "models_checkpoints/toymix_dti_esmc_v2_lpm24_litopenai_bbbc047/"
    "pairmixer_12M/2026-05-01_01-59-58_20260501_015958/"
)
CKPT_NAME = os.environ.get("PAIRMIXER_CKPT_NAME", "ep099")
assert CKPT_NAME in ("ep019", "ep099"), CKPT_NAME
CKPT = _CKPT_DIR + (
    f"toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_epoch"
    f"epoch={'019' if CKPT_NAME == 'ep019' else '099'}_20260501_015958.ckpt"
)
DEVICE = os.environ.get("PAIRMIXER_DEVICE", "cuda:2")

REPRESENTATIVE_TASKS = [
    "caco2_wang", "bbb_martins", "cyp2d6_veith",
    "clearance_microsome_az", "ld50_zhu",
]

LAYER_IDS = (3, 7)  # 1-indexed "4th, 8th"


def _stratified_pair_stats(z, pair_mask, edge_index, batch_idx):
    """Mirrors graphium's `_pool_pair_stats` — diag/1-hop/global mean+std → (B, 6*D_z)."""
    from torch_geometric.utils import to_dense_adj

    B, N, _ = pair_mask.shape
    device = pair_mask.device
    dtype = pair_mask.dtype

    eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
    mask_diag = eye * pair_mask
    adj = to_dense_adj(edge_index, batch=batch_idx, max_num_nodes=N).to(dtype=dtype)
    mask_1hop = (adj > 0).to(dtype=dtype) * pair_mask
    mask_global = pair_mask

    z2 = z * z

    def stratum_stats(mask):
        m4 = mask.unsqueeze(-1)
        raw_count = mask.sum(dim=(1, 2))
        count = raw_count.clamp(min=1.0).unsqueeze(-1)
        sum_z  = (z  * m4).sum(dim=(1, 2))
        sum_z2 = (z2 * m4).sum(dim=(1, 2))
        mean = sum_z / count
        mean_sq = sum_z2 / count
        var = (mean_sq - mean * mean).clamp(min=1e-6)
        std = var.sqrt()
        has_any = (raw_count > 0).to(z.dtype).unsqueeze(-1)
        return mean * has_any, std * has_any

    mean_d, std_d = stratum_stats(mask_diag)
    mean_h, std_h = stratum_stats(mask_1hop)
    mean_g, std_g = stratum_stats(mask_global)

    return torch.cat([mean_d, mean_h, mean_g, std_d, std_h, std_g], dim=-1)  # (B, 6*D_z)


class NodePairLayers48Encoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = (
            self.encoder_tag + f"_node_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}"
        )

    def _ensure_backbone(self):
        super()._ensure_backbone()
        self._backbone.gnn._enable_readout_cache()
        self._backbone.gnn._pair_capture_layers = set(LAYER_IDS)

    def _forward_graphs(self, graphs):
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader
        from torch_geometric.nn import global_max_pool

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
                desc=f"GNN forward (NODE-MAX + PAIR-STRATA[{LAYER_IDS[0]},{LAYER_IDS[1]}])",
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
                    # Node max-pool from this layer's node features.
                    node_feats = node_cache[li]                          # (N_total, 96)
                    z_node = global_max_pool(node_feats, batch_idx)      # (B, 96)
                    # Stratified pair stats pool from this layer's pair features.
                    z_pair_layer = pair_stack.get(li, g.pair_feat)
                    z_pair = _stratified_pair_stats(
                        z_pair_layer, pair_mask, edge_index, batch_idx,
                    )                                                    # (B, 1344)
                    per_layer.append(torch.cat([z_node, z_pair], dim=-1))  # (B, 1440)
                z = torch.cat(per_layer, dim=-1)                         # (B, 2880)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "NodePairLayers48Encoder")

    out_csv = ROOT / "results" / f"admet_node_pair_layers48_oneshot_{CKPT_NAME}.csv"
    cache_dir = (
        ROOT / "datacache"
        / f"pairmixer_probe_fps_node_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_{CKPT_NAME}"
    )

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", f"4ds_{CKPT_NAME}_node_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}",
        "--pretrain-label",
        f"4ds-{CKPT_NAME}-node-pair-layers{LAYER_IDS[0]}_{LAYER_IDS[1]}-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", DEVICE,
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
