"""One-shot probe with multi-layer GNN fingerprints (concat-then-pool).

Patches ``PairMixerEncoder._forward_graphs`` to enable the GNN's per-layer
readout cache, concat the features from every layer at the node level, then
mean-pool to a graph-level fingerprint of size ``sum(layer_dims)``. After the
patch, the regular ``pairmixer_minimol_probe.py`` machinery is reused.

Usage:
    python scripts/pairmixer/probe_multilayer.py \\
        --ckpt <path.ckpt> \\
        --pretrain-label toymix-4ds-may3-ep39-multilayer \\
        --device cuda:0 \\
        --output-csv /tmp/probe_multilayer.csv \\
        --sweep-results results/pairmixer_minimol_probe_sweep_toymix-3aux-combined.json
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "pairmixer"))

import torch
from torch.utils.data import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm.auto import tqdm

try:
    from torch_scatter import scatter_std as _scatter_std
except ImportError:  # fallback: compute std manually via scatter_mean
    from torch_scatter import scatter_mean

    def _scatter_std(x, batch, dim=0):
        m = scatter_mean(x, batch, dim=dim)[batch]
        sq = scatter_mean((x - m) ** 2, batch, dim=dim)
        return torch.sqrt(sq.clamp(min=1e-12))

import os
from downstream.model.pairmixer import PairMixerEncoder

# Which GNN layer indices to concat. Default = every layer.
# Override with env var, e.g. PROBE_LAYERS=4,7 for middle + last only.
_LAYERS_ENV = os.environ.get("PROBE_LAYERS", "").strip()
SELECT_LAYERS = (
    [int(x) for x in _LAYERS_ENV.split(",")] if _LAYERS_ENV else None
)

# Which graph-level poolings to concat. Default = mean only (matches original).
# Override e.g. PROBE_POOLINGS=mean,max,std for a richer set summary.
_POOL_ENV = os.environ.get("PROBE_POOLINGS", "mean").strip()
SELECT_POOLINGS = [p.strip() for p in _POOL_ENV.split(",") if p.strip()]
_VALID_POOLS = {"mean", "max", "std"}
_unknown = [p for p in SELECT_POOLINGS if p not in _VALID_POOLS]
if _unknown:
    raise ValueError(f"Unknown poolings {_unknown}; valid: {sorted(_VALID_POOLS)}")


def _multilayer_forward_graphs(self: PairMixerEncoder, graphs: List[Any]) -> torch.Tensor:
    """Replaces ``_forward_graphs``: returns concat(selected GNN layers) → mean-pool."""
    from graphium.data.collate import graphium_collate_fn

    backbone = self._backbone
    device = self._torch_device
    target_dtype = next(backbone.parameters()).dtype

    loader = DataLoader(
        graphs, batch_size=self.batch_size, shuffle=False,
        collate_fn=partial(graphium_collate_fn, mask_nan=0), num_workers=0,
    )
    chunks: List[torch.Tensor] = []
    backbone.eval()
    backbone.gnn._enable_readout_cache()
    n_batches = (len(graphs) + self.batch_size - 1) // self.batch_size
    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc="GNN forward (multi-layer)",
                          unit="batch", smoothing=0.05):
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
            backbone.gnn.forward(g)

            cache = backbone.gnn._readout_cache  # {layer_idx: (n_nodes, D_layer)}
            available = sorted(cache.keys())
            if SELECT_LAYERS is None:
                layer_keys = available
            else:
                missing = [i for i in SELECT_LAYERS if i not in cache]
                if missing:
                    raise KeyError(
                        f"PROBE_LAYERS requested layers {SELECT_LAYERS} but readout "
                        f"cache only has {available} (missing {missing})."
                    )
                layer_keys = list(SELECT_LAYERS)
            node_concat = torch.cat([cache[k].float() for k in layer_keys], dim=-1)
            pooled = []
            for pool_name in SELECT_POOLINGS:
                if pool_name == "mean":
                    pooled.append(global_mean_pool(node_concat, g.batch))
                elif pool_name == "max":
                    pooled.append(global_max_pool(node_concat, g.batch))
                elif pool_name == "std":
                    pooled.append(_scatter_std(node_concat, g.batch, dim=0))
            graph_emb = torch.cat(pooled, dim=-1).cpu()
            chunks.append(graph_emb)

    backbone.gnn._disable_readout_cache()
    out = torch.cat(chunks, dim=0)
    if self.out_dim < 0 or self.out_dim != out.shape[1]:
        self.out_dim = int(out.shape[1])
    return out


# Apply the monkey-patch BEFORE the standard probe runs.
PairMixerEncoder._forward_graphs = _multilayer_forward_graphs

# Hand off to the standard probe entry point.
import pairmixer_minimol_probe as probe  # noqa: E402

if __name__ == "__main__":
    probe.main()
