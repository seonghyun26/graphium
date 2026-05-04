#!/usr/bin/env python
"""One-shot variant: gnn_last + per-node L2 normalization + external max-pool.

L2-normalizing each node feature vector to unit length BEFORE max-pool removes
magnitude info but preserves direction — every node gets equal "voting weight"
in the max-pool, so the fingerprint becomes a max-direction summary on the
unit sphere rather than a max-magnitude summary.

Compares against the same dashboard 4ds-ep099 graph_output_nn baseline as the
gnn_last / layer-5 / multi-layer variants, on the same 5 representative ADMET
tasks, with verbatim MiniMol HPs.

Output: results/admet_l2norm_oneshot.csv
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
import torch.nn.functional as F
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


class L2NormPairMixerEncoder(PairMixerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = self.encoder_tag + "_l2norm"

    def _forward_graphs(self, graphs):
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader
        from torch_geometric.nn import global_max_pool

        backbone = self._backbone
        device = self._torch_device
        target_dtype = next(backbone.parameters()).dtype

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
                desc="GNN forward (gnn_last+L2+max)", unit="batch", smoothing=0.05,
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
                g = backbone.gnn.forward(g)
                # Per-node L2 norm BEFORE max-pool. eps avoids divide-by-zero
                # for any all-zero rows (shouldn't happen but cheap insurance).
                node_feats = F.normalize(g["feat"], p=2.0, dim=-1, eps=1e-12)
                z = global_max_pool(node_feats, batch.batch)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "L2NormPairMixerEncoder")

    out_csv = ROOT / "results" / "admet_l2norm_oneshot.csv"
    cache_dir = ROOT / "datacache" / "pairmixer_probe_fps_l2norm"

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", "4ds_ep099_l2norm_maxpool",
        "--pretrain-label", "4ds-ep099-l2norm-maxpool-mmhp",
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
