#!/usr/bin/env python
"""One-shot: re-evaluate the 4ds ep099 PairMixer 12M ckpt on a representative
ADMET subset, but pull embeddings from the LAST GNN layer + external max-pool
(MiniMol-faithful) instead of the trained ``graph_output_nn``.

Compares against the dashboard's existing 4ds-ep099 rows (which used
``graph_output_nn`` embeddings + a sweep JSON of HPs tuned on those
embeddings). This arm uses verbatim MiniMol HPs because the sweep JSON
doesn't apply cleanly to a different embedding distribution.

The mechanism is a thin ``PairMixerEncoder`` subclass + an in-memory swap of
the ``downstream.model._REGISTRY["pairmixer"]`` entry. No shipped code is
modified — the existing ``pairmixer_minimol_probe.main()`` runs unchanged.

Output: results/admet_gnnlast_oneshot.csv (kept separate from the dashboard).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable for `downstream.*` and the script package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm
from functools import partial

from downstream.model import _REGISTRY
from downstream.model.pairmixer import PairMixerEncoder


CKPT = (
    "models_checkpoints/toymix_dti_esmc_v2_lpm24_litopenai_bbbc047/"
    "pairmixer_12M/2026-05-01_01-59-58_20260501_015958/"
    "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_"
    "epochepoch=099_20260501_015958.ckpt"
)

REPRESENTATIVE_TASKS = [
    "caco2_wang",             # A (regression)
    "bbb_martins",            # D (classification)
    "cyp2d6_veith",           # M (classification)
    "clearance_microsome_az", # E (regression)
    "ld50_zhu",               # T (regression)
]


class GnnLastPairMixerEncoder(PairMixerEncoder):
    """PairMixer encoder using last-GNN-layer features + external max-pool.

    Overrides only ``_forward_graphs`` and ``_ensure_backbone`` (to skip the
    ``graph_output_nn['graph']`` presence check, which isn't required when we
    don't route through that head). Encoder tag is suffixed so per-SMILES
    fingerprint caches don't collide with the default arm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = self.encoder_tag + "_gnn_last"

    # Parent's _ensure_backbone is fine — the 4ds ckpt does have
    # graph_output_nn['graph']; we just don't use it.

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
                desc="GNN forward (gnn_last+max)", unit="batch", smoothing=0.05,
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
                # MiniMol-faithful: skip graph_output_nn, max-pool last-layer
                # node features against the input PyG Batch's node->graph index.
                z = global_max_pool(g["feat"], batch.batch)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    # Patch the encoder registry in-process. pmp.main() will then build a
    # GnnLastPairMixerEncoder instead of the stock PairMixerEncoder when it
    # calls load_encoder("pairmixer", ...).
    _REGISTRY["pairmixer"] = (__name__, "GnnLastPairMixerEncoder")

    out_csv = ROOT / "results" / "admet_gnnlast_oneshot.csv"
    cache_dir = ROOT / "datacache" / "pairmixer_probe_fps_gnnlast"

    # Override sys.argv and dispatch to pmp.main(). Sentinel /dev/null forces
    # verbatim MiniMol HPs (no sweep JSON auto-load); see
    # pairmixer_minimol_probe._resolve_hp_table.
    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag", "4ds_ep099_gnnlast_maxpool",
        "--pretrain-label", "4ds-ep099-gnnlast-maxpool-mmhp",
        "--tasks", *REPRESENTATIVE_TASKS,
        "--device", "cuda:1",
        "--embed-batch-size", "64",
        "--featurize-n-jobs", "8",
        "--mol-cache-dir", str(cache_dir),
        "--output-csv", str(out_csv),
        "--sweep-results", "/dev/null",  # force verbatim MiniMol HPs
    ]

    # Lazy load pmp by file path so we don't depend on scripts.* being a
    # package (no __init__.py exists under scripts/).
    import importlib.util
    pmp_path = Path(__file__).parent / "pairmixer_minimol_probe.py"
    spec = importlib.util.spec_from_file_location("pairmixer_minimol_probe", str(pmp_path))
    pmp = importlib.util.module_from_spec(spec)
    sys.modules["pairmixer_minimol_probe"] = pmp
    spec.loader.exec_module(pmp)
    pmp.main()


if __name__ == "__main__":
    main()
