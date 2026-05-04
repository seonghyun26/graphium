#!/usr/bin/env python
"""One-shot variant: same fingerprint as ``_admet_node_mms_pair_strata_oneshot``
(node MMS + pair strata, layers 4+8) but laid out as ``[all_node | all_pair]``,
and the probe head is replaced by a learnable-gated head:

    single_norm = LayerNorm(node_dim)
    pair_norm   = LayerNorm(pair_dim)
    alpha       = sigmoid(pair_gate)              # init pair_gate=-3 → α≈0.047
    h           = concat(single_norm(single), alpha * pair_norm(pair))
    pred        = MiniMolTaskHead(h)              # the existing BN+ReLU MLP

Motivation: pair info helps some tasks (cyp2d6_veith) but hurts others
(caco2_wang). A learned per-task scalar gate lets each task's probe weight
the pair contribution to its needs.

Fingerprint layout:
    [layer4_node_mms (288) | layer8_node_mms (288)]   ← single block, dim 576
    [layer4_pair_strata (1344) | layer8_pair_strata (1344)]  ← pair block, dim 2688
    Total: 3264-d, with the first 576 = node, last 2688 = pair.

Output: results/admet_node_mms_pair_strata_gated_oneshot.csv
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
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
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

LAYER_IDS = (3, 7)
NODE_DIM = 576    # 2 layers * 3 stats (max+mean+std) * 96
PAIR_DIM = 2688   # 2 layers * 6 stats (3 strata * mean+std) * 224
FP_DIM   = NODE_DIM + PAIR_DIM


def _node_mms(node_feats, batch_idx):
    from torch_geometric.nn import global_mean_pool, global_max_pool

    z_max  = global_max_pool(node_feats, batch_idx)
    z_mean = global_mean_pool(node_feats, batch_idx)
    z_meansq = global_mean_pool(node_feats * node_feats, batch_idx)
    z_var = (z_meansq - z_mean * z_mean).clamp_min(0.0)
    z_std = torch.sqrt(z_var + 1e-8)
    return torch.cat([z_max, z_mean, z_std], dim=-1)


def _stratified_pair_stats(z, pair_mask, edge_index, batch_idx):
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

    return torch.cat([mean_d, mean_h, mean_g, std_d, std_h, std_g], dim=-1)


class NodeMMSPairStrataGatedEncoder(PairMixerEncoder):
    """Same content as the strata combined arm, but with [all_node | all_pair] layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_tag = (
            self.encoder_tag
            + f"_node_mms_pair_strata_gated_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}"
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
                desc=f"GNN forward (NODE-MMS|PAIR-STRATA gated [{LAYER_IDS[0]},{LAYER_IDS[1]}])",
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

                node_parts = []
                pair_parts = []
                for li in LAYER_IDS:
                    node_parts.append(_node_mms(node_cache[li], batch_idx))      # (B, 288)
                    pair_parts.append(_stratified_pair_stats(
                        pair_stack.get(li, g.pair_feat),
                        pair_mask, edge_index, batch_idx,
                    ))                                                            # (B, 1344)
                # Layout: [all_node | all_pair] → (B, NODE_DIM + PAIR_DIM)
                z_node = torch.cat(node_parts, dim=-1)                           # (B, 576)
                z_pair = torch.cat(pair_parts, dim=-1)                           # (B, 2688)
                z = torch.cat([z_node, z_pair], dim=-1)                          # (B, 3264)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


def _make_gated_model_factory(pmp, node_dim: int, pair_dim: int, alpha_log: list):
    """Return a drop-in replacement for ``pmp.model_factory`` that builds a
    GatedTaskHead wrapping MiniMol's TaskHead. Each created head registers
    itself with ``alpha_log`` so the script can read the final alphas.
    """
    TaskHead = pmp.TaskHead

    class GatedTaskHead(nn.Module):
        def __init__(self, hidden_dim: int, depth: int, combine: bool,
                     dropout: float = 0.1, pair_gate_init: float = -3.0):
            super().__init__()
            self.node_dim = node_dim
            self.pair_dim = pair_dim
            self.single_norm = nn.LayerNorm(node_dim)
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.pair_gate = nn.Parameter(torch.tensor(pair_gate_init))
            self.head = TaskHead(
                hidden_dim=hidden_dim,
                input_dim=node_dim + pair_dim,
                depth=depth, combine=combine, dropout=dropout,
            )
            # Append a closure ref to alpha_log so the caller can poll the
            # learned gate after training completes.
            alpha_log.append(self)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            single = x[:, : self.node_dim]
            pair   = x[:, self.node_dim :]
            single = self.single_norm(single)
            pair = self.pair_norm(pair)
            alpha = torch.sigmoid(self.pair_gate)
            h = torch.cat([single, alpha * pair], dim=-1)
            return self.head(h)

    def factory(hidden_dim, depth, combine, task, lr, *,
                input_dim, epochs, warmup=5, weight_decay=1e-4):
        if input_dim != node_dim + pair_dim:
            raise ValueError(
                f"GatedTaskHead expects input_dim={node_dim + pair_dim} "
                f"({node_dim} node + {pair_dim} pair); got {input_dim}"
            )
        model = GatedTaskHead(hidden_dim=hidden_dim, depth=depth, combine=combine)
        optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCELoss() if task == "classification" else nn.MSELoss()

        def lr_fn(epoch: int) -> float:
            import math
            if epoch < warmup:
                return epoch / warmup
            return (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2

        scheduler = LambdaLR(optimiser, lr_lambda=lr_fn)
        return model, optimiser, scheduler, loss_fn

    return factory


def main() -> None:
    if not Path(CKPT).exists():
        sys.exit(f"ERROR: ckpt not found: {CKPT}")

    _REGISTRY["pairmixer"] = (__name__, "NodeMMSPairStrataGatedEncoder")

    out_csv = ROOT / "results" / "admet_node_mms_pair_strata_gated_oneshot.csv"
    cache_dir = (
        ROOT / "datacache"
        / f"pairmixer_probe_fps_node_mms_pair_strata_gated_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}"
    )

    sys.argv = [
        "pairmixer_minimol_probe",
        "--ckpt", CKPT,
        "--model-name", "pairmixer_12M",
        "--ckpt-tag",
        f"4ds_ep099_node_mms_pair_strata_gated_layers{LAYER_IDS[0]}_{LAYER_IDS[1]}",
        "--pretrain-label",
        f"4ds-ep099-node-mms-pair-strata-gated-layers{LAYER_IDS[0]}_{LAYER_IDS[1]}-mmhp",
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

    # Replace pmp.model_factory with the gated version that splits the
    # fingerprint at NODE_DIM and applies LN + sigmoid-gate before the MLP.
    # Capture all created heads in a per-task list so we can read the learned
    # alphas after each task completes.
    import collections, numpy as np
    from copy import copy
    alpha_log: list = []
    pmp.model_factory = _make_gated_model_factory(pmp, NODE_DIM, PAIR_DIM, alpha_log)
    print(
        f"[oneshot] Replaced pmp.model_factory with GatedTaskHead "
        f"(node_dim={NODE_DIM}, pair_dim={PAIR_DIM}, init pair_gate=-3.0 → α≈0.047)",
        flush=True,
    )

    # Hook run_one_task to print the learned alpha summary after each task.
    _orig_run_one_task = pmp.run_one_task
    _task_alphas: dict = {}

    def _run_with_alpha(task, *args, **kwargs):
        before = list(alpha_log)
        try:
            return _orig_run_one_task(task, *args, **kwargs)
        finally:
            after = alpha_log[len(before):]   # heads created during this task
            alphas = [
                float(torch.sigmoid(m.pair_gate).detach().cpu().item()) for m in after
            ]
            if alphas:
                arr = np.array(alphas)
                _task_alphas[task] = (float(arr.mean()), float(arr.std()), len(arr))
                print(
                    f"  [{task}] learned α (sigmoid(pair_gate)): "
                    f"mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}  "
                    f"min={arr.min():.4f}  max={arr.max():.4f}",
                    flush=True,
                )

    pmp.run_one_task = _run_with_alpha

    pmp.main()

    # Final summary across tasks.
    if _task_alphas:
        print("\n[oneshot] Per-task learned α summary:", flush=True)
        for task, (m, s, n) in _task_alphas.items():
            print(f"  {task:<32s}  α = {m:.4f} ± {s:.4f}  (n={n})", flush=True)


if __name__ == "__main__":
    main()
