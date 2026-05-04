#!/usr/bin/env python
"""One-shot variant: skip graph_output_nn, concat layers 4 and 8 (1-indexed →
``_readout_cache[3]`` and ``[7]``) per node, then STATS pooling per graph
(mean + max + sum + std concatenated) instead of max-pool alone.

Std is computed from the variance formula (E[X²] - E[X]²) since PyG doesn't
ship a ``global_std_pool``.

Fingerprint dim = 2 layers * 96 hidden * 4 stats = 768-d.
Verbatim MiniMol HPs (no hidden_dim override).

Output: results/admet_layers48_stats_oneshot.csv
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

LAYER_IDS = (3, 7)  # 1-indexed "4th, 8th" → 0-indexed cache keys


class Layers48StatsPairMixerEncoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = (
            self.encoder_tag + f"_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_stats"
        )

    def _ensure_backbone(self):
        super()._ensure_backbone()
        self._backbone.gnn._enable_readout_cache()

    def _forward_graphs(self, graphs):
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader
        from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

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
                desc=f"GNN forward (concat[{LAYER_IDS[0]},{LAYER_IDS[1]}]+stats)",
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
                node_concat = torch.cat(
                    [backbone.gnn._readout_cache[li] for li in LAYER_IDS], dim=-1,
                )                                                     # (N, 192)
                z_mean = global_mean_pool(node_concat, batch.batch)   # (G, 192)
                z_max  = global_max_pool(node_concat, batch.batch)    # (G, 192)
                z_sum  = global_add_pool(node_concat, batch.batch)    # (G, 192)
                # Population std via Var = E[X²] - E[X]², clamped for numerics.
                z_meansq = global_mean_pool(node_concat * node_concat, batch.batch)
                z_var = (z_meansq - z_mean * z_mean).clamp_min(0.0)
                z_std = torch.sqrt(z_var + 1e-8)                      # (G, 192)
                z = torch.cat([z_mean, z_max, z_sum, z_std], dim=-1)  # (G, 768)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "Layers48StatsPairMixerEncoder")

    out_csv = ROOT / "results" / "admet_layers48_stats_oneshot.csv"
    cache_dir = (
        ROOT / "datacache"
        / f"pairmixer_probe_fps_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_stats"
    )

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", f"4ds_ep099_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}_stats",
        "--pretrain-label",
        f"4ds-ep099-layers{LAYER_IDS[0]}_{LAYER_IDS[1]}-stats-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", "cuda:1",
        "--embed-batch-size", "32",
        "--featurize-n-jobs", "8",
        "--mol-cache-dir", str(cache_dir),
        "--output-csv", str(out_csv),
        "--sweep-results", "/dev/null",  # verbatim MiniMol HPs
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
