#!/usr/bin/env python
"""One-shot variant: concatenate all GNN layer outputs (MolE-style multi-scale
fingerprint), then external max-pool. PairMixer 12M depth=8, hidden 96 →
fingerprint dim 8*96 = 768.

Compares against the same dashboard 4ds-ep099 graph_output_nn baseline as the
gnn_last and layer-5 variants, on the same 5 representative ADMET tasks, with
verbatim MiniMol HPs.

Output: results/admet_multilayer_oneshot.csv
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

# Remaining 17 of TDC's 22 ADMET tasks (the 5 above are already in the CSV).
REMAINING_TASKS = [
    "ames", "bioavailability_ma", "clearance_hepatocyte_az",
    "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith", "dili", "half_life_obach", "herg", "hia_hou",
    "lipophilicity_astrazeneca", "pgp_broccatelli", "ppbr_az",
    "solubility_aqsoldb", "vdss_lombardo",
]

# Tasks that crashed in the prior 17-task run (CUDA OOM in pair-track ops on
# big molecules with batch=64). Rerun with smaller embed batch.
MISSING_TASKS = ["solubility_aqsoldb", "vdss_lombardo"]

# Switch to the full 22-task set (the 5 representative + 17 remaining) to
# match the dashboard. To rerun only the 5 representative tasks, set
# TASKS_TO_RUN = REPRESENTATIVE_TASKS.
TASKS_TO_RUN = MISSING_TASKS

# Concat ALL 0..depth-1 layer outputs (MolE-style). Per-layer dim = 96 →
# fingerprint dim = depth * 96 = 768 for PairMixer 12M (depth=8).
LAYERS_SPEC = "all"  # "all" or list[int] of 0-indexed layer ids


class MultiLayerPairMixerEncoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = self.encoder_tag + "_multilayer"

    def _ensure_backbone(self):
        super()._ensure_backbone()
        self._backbone.gnn._enable_readout_cache()

    def _forward_graphs(self, graphs):
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader
        from torch_geometric.nn import global_max_pool

        backbone = self._backbone
        device = self._torch_device
        target_dtype = next(backbone.parameters()).dtype
        depth = len(backbone.gnn.layers)
        layer_ids = list(range(depth)) if LAYERS_SPEC == "all" else list(LAYERS_SPEC)
        for li in layer_ids:
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
                desc=f"GNN forward (concat[{layer_ids[0]}..{layer_ids[-1]}]+max)",
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
                # Stack per-layer node features along feat dim (N, depth*hidden).
                node_concat = torch.cat(
                    [backbone.gnn._readout_cache[li] for li in layer_ids], dim=-1,
                )
                z = global_max_pool(node_concat, batch.batch)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "MultiLayerPairMixerEncoder")

    out_csv = ROOT / "results" / "admet_multilayer_oneshot.csv"
    cache_dir = ROOT / "datacache" / "pairmixer_probe_fps_multilayer"

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", "4ds_ep099_multilayer_maxpool",
        "--pretrain-label", "4ds-ep099-multilayer-maxpool-mmhp",
        "--tasks", *TASKS_TO_RUN,
        "--device", "cuda:1",
        "--embed-batch-size", "8",
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
