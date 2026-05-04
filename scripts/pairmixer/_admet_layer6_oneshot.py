#!/usr/bin/env python
"""One-shot variant: pull embeddings from a MIDDLE GNN layer instead of the
last one, then external max-pool. PairMixer 12M has gnn.depth=8, so layer
index 5 is the 6th-of-8 (human-counted "6-th layer", roughly mid-stack).

Compares against the same dashboard 4ds-ep099 graph_output_nn baseline as
the gnn_last variant, on the same 5 representative ADMET tasks, with verbatim
MiniMol HPs.

Output: results/admet_layer6_oneshot.csv
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


# Same 4ds ep099 ckpt used by _admet_gnnlast_oneshot.py.
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

# 0-indexed layer index. PairMixer 12M depth=8, so layers are 0..7.
# Index 5 == "6th-of-8 layer" in human counting (1-indexed).
LAYER_IDX = 5


class Layer6PairMixerEncoder(PairMixerEncoder):
    """Read mid-stack layer features from the GNN readout cache + external max-pool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = self.encoder_tag + f"_layer{LAYER_IDX}"

    def _ensure_backbone(self):
        super()._ensure_backbone()
        # Force the GNN to populate _readout_cache[ii] = h on every layer.
        # Without this the dict stays None and indexing fails.
        self._backbone.gnn._enable_readout_cache()

    def _forward_graphs(self, graphs):
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader
        from torch_geometric.nn import global_max_pool

        backbone = self._backbone
        device = self._torch_device
        target_dtype = next(backbone.parameters()).dtype
        depth = len(backbone.gnn.layers)
        if not (0 <= LAYER_IDX < depth):
            raise ValueError(
                f"LAYER_IDX={LAYER_IDX} out of range for GNN depth {depth}"
            )

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
                desc=f"GNN forward (layer{LAYER_IDX}+max)", unit="batch", smoothing=0.05,
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
                # Forward populates _readout_cache[i] for i = 0..depth-1.
                _ = backbone.gnn.forward(g)
                node_feats = backbone.gnn._readout_cache[LAYER_IDX]
                z = global_max_pool(node_feats, batch.batch)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "Layer6PairMixerEncoder")

    out_csv = ROOT / "results" / "admet_layer6_oneshot.csv"
    cache_dir = ROOT / "datacache" / f"pairmixer_probe_fps_layer{LAYER_IDX}"

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", f"4ds_ep099_layer{LAYER_IDX}_maxpool",
        "--pretrain-label", f"4ds-ep099-layer{LAYER_IDX}-maxpool-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", "cuda:3",
        "--embed-batch-size", "64",
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
