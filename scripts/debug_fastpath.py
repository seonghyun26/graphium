"""Debug: compare fast vs slow path layer-by-layer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg


def build_batch():
    rng = torch.Generator().manual_seed(42)
    graphs = []
    for _ in range(4):
        n = torch.randint(5, 10, (1,), generator=rng).item()
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, 64, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], 32, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    return Batch.from_data_list(graphs)


def main():
    common = dict(
        in_dim=64, out_dim=64, pair_dim=32, in_dim_edges=32,
        hidden_dim_scaling=2.0, opm_hidden=16,
        pair_dropout=0.0, normalization="none",
        use_checkpoint=False, force_float32_einsums=True,
        tri_mul_mode="einsum", parallel_pair_ops=False,
        pair_init={
            "mode": "additive", "use_opm": True, "opm_init": "lecun",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance", "adjacency", "edge_feat"],
                "graph_distance": {"max_dist": 8, "embedding_dim": 8},
                "adjacency": {"embedding_dim": 4},
                "edge_feat": {"embedding_dim": 8},
            },
        },
    )
    torch.manual_seed(0)
    layers = nn.ModuleList([
        PairMixerLayerPyg(layer_idx=i, **common) for i in range(3)
    ]).cuda()
    for l in layers:
        l.eval()

    batch = build_batch().cuda()

    def run(clear_pair_mask_between):
        b = batch.clone()
        b.pair_feat = None
        b.pair_mask = None
        snapshots = []
        for i, l in enumerate(layers):
            b = l(b)
            snapshots.append({
                "layer": i,
                "pair_feat": b.pair_feat.detach().clone(),
                "pair_mask": b.pair_mask.detach().clone() if b.pair_mask is not None else None,
                "feat": b.feat.detach().clone(),
            })
            if clear_pair_mask_between and i < len(layers) - 1:
                b.pair_mask = None
        return snapshots

    fast = run(clear_pair_mask_between=False)
    slow = run(clear_pair_mask_between=True)

    for i in range(len(layers)):
        pf_diff = (fast[i]["pair_feat"] - slow[i]["pair_feat"]).abs().max().item()
        pm_diff = (fast[i]["pair_mask"] - slow[i]["pair_mask"]).abs().max().item() if slow[i]["pair_mask"] is not None else "N/A"
        ft_diff = (fast[i]["feat"] - slow[i]["feat"]).abs().max().item()
        print(f"Layer {i}: pair_feat diff={pf_diff:.3e}, pair_mask diff={pm_diff}, feat diff={ft_diff:.3e}")

    # Also check: is batch.pair_mask in fast path exactly equal to recomputed pair_mask?
    b = batch.clone()
    b.pair_feat = None
    b.pair_mask = None
    _ = layers[0](b)
    pair_mask_layer0 = b.pair_mask.clone()

    # Manually recompute pair_mask via to_dense_batch
    from graphium.ipu.to_dense_batch import to_dense_batch
    s_dense, kpm, idx = to_dense_batch(batch.feat, batch=batch.batch)
    node_mask = kpm.float()
    pair_mask_manual = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
    print(f"\npair_mask from layer 0: shape={pair_mask_layer0.shape}, dtype={pair_mask_layer0.dtype}")
    print(f"pair_mask recomputed:   shape={pair_mask_manual.shape}, dtype={pair_mask_manual.dtype}")
    print(f"pair_mask diff: {(pair_mask_layer0 - pair_mask_manual).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
