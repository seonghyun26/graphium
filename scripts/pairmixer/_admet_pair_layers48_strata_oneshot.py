#!/usr/bin/env python
"""One-shot variant: skip graph_output_nn, pool the PAIR representation at
layers 4 and 8 using the architecture's NATIVE STRATIFIED stats pool — diag /
1-hop (bonded) / global, mean+std per stratum. Mirrors graphium's
``_pool_pair_stats`` (the recipe the model was *trained* against; see
``pairmixer_12M.yaml: pair_pool: stats``).

Per layer: 6 stratum-stats × D_z = 6 * 224 = 1344-d.
Concat 2 layers (4, 8) → fingerprint dim = 2688-d.

Output: results/admet_pair_layers48_strata_oneshot.csv
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


def _stratified_pair_stats(z, pair_mask, edge_index, batch_idx):
    """Replicates `GraphOutputNN._pool_pair_stats` for one layer's pair tensor.

    Args:
        z: (B, N, N, D_z) pair representation.
        pair_mask: (B, N, N) validity mask.
        edge_index: (2, E) sparse PyG edge indices.
        batch_idx: (total_nodes,) node-to-graph index.

    Returns:
        (B, 6*D_z) — concat of [mean_diag, mean_1hop, mean_global,
                                std_diag, std_1hop, std_global].
    """
    from torch_geometric.utils import to_dense_adj

    B, N, _ = pair_mask.shape
    device = pair_mask.device
    dtype = pair_mask.dtype

    eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)        # (1, N, N)
    mask_diag = eye * pair_mask                                         # (B, N, N)

    adj = to_dense_adj(
        edge_index, batch=batch_idx, max_num_nodes=N
    ).to(dtype=dtype)                                                   # (B, N, N)
    mask_1hop = (adj > 0).to(dtype=dtype) * pair_mask
    mask_global = pair_mask

    z2 = z * z

    def stratum_stats(mask):
        m4 = mask.unsqueeze(-1)                                         # (B, N, N, 1)
        raw_count = mask.sum(dim=(1, 2))                                # (B,)
        count = raw_count.clamp(min=1.0).unsqueeze(-1)                  # (B, 1)
        sum_z  = (z  * m4).sum(dim=(1, 2))                              # (B, D_z)
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

    return torch.cat([mean_d, mean_h, mean_g, std_d, std_h, std_g], dim=-1)


class PairLayers48StrataEncoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = (
            self.encoder_tag + f"_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_strata"
        )

    def _ensure_backbone(self):
        super()._ensure_backbone()
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
                desc=f"GNN forward (PAIR strata stats[{LAYER_IDS[0]},{LAYER_IDS[1]}])",
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

                stack = getattr(g, "pair_feat_stack", None) or {}
                pair_mask = g.pair_mask
                edge_index = g.edge_index
                batch_idx = g.batch

                per_layer = []
                for li in LAYER_IDS:
                    z_layer = stack.get(li, g.pair_feat)
                    per_layer.append(_stratified_pair_stats(
                        z_layer, pair_mask, edge_index, batch_idx,
                    ))
                z = torch.cat(per_layer, dim=-1)                        # (B, 12 * D_z)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "PairLayers48StrataEncoder")

    out_csv = ROOT / "results" / "admet_pair_layers48_strata_oneshot.csv"
    cache_dir = (
        ROOT / "datacache"
        / f"pairmixer_probe_fps_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_strata"
    )

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", f"4ds_ep099_pair_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_strata",
        "--pretrain-label",
        f"4ds-ep099-pair-layers{LAYER_IDS[0]}_{LAYER_IDS[1]}-strata-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", "cuda:3",  # GPU 3 to run parallel with naive pair-pool on GPU 1
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
