"""Correctness test: fast-path vs slow-path output must be numerically equal.

The fast-path kicks in for middle layers when:
  - layer_idx > 0 (so pair_init is None)
  - out_proj is Identity (in_dim == out_dim)
  - batch carries cached pair_feat + pair_mask

To verify equivalence we construct two identical 4-layer stacks, run one
through the normal forward() which uses the fast path for layers 1-3, and
compare with a manually-constructed slow-path forward that re-derives
pair_mask per layer via to_dense_batch.
"""
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_batch(B=6, N_min=5, N_max=12, D_s=64, d_edge=32):
    """Chain-only graphs with UNIQUE edges (no duplicates → deterministic).

    Random extra edges can produce duplicate (i, j) pairs which trigger
    CUDA's non-deterministic index_put_ inside _build_dense_edge_feat,
    breaking bit-exact comparisons unrelated to the fast path.
    """
    rng = torch.Generator().manual_seed(42)
    graphs = []
    for _ in range(B):
        n = torch.randint(N_min, N_max + 1, (1,), generator=rng).item()
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, D_s, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], d_edge, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    return Batch.from_data_list(graphs)


def run_with_layers(layers, batch, seed=0):
    b = batch.clone()
    if hasattr(b, "pair_feat"):
        b.pair_feat = None
    if hasattr(b, "pair_mask"):
        b.pair_mask = None
    torch.manual_seed(seed)
    for l in layers:
        l.eval()  # disable dropout
        b = l(b)
    return b


def main():
    from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg
    from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch

    common = dict(
        in_dim=64, out_dim=64, pair_dim=32, in_dim_edges=32,
        hidden_dim_scaling=4.0, opm_hidden=16,
        pair_dropout=0.0,  # deterministic
        normalization="none",
        use_checkpoint=False,  # eval-mode skips checkpoint anyway
        force_float32_einsums=True,
        tri_mul_mode="einsum",
        parallel_pair_ops=False,
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
        PairMixerLayerPyg(layer_idx=i, **common) for i in range(4)
    ]).cuda()

    batch = build_batch().cuda()

    # Run 1: current forward (uses fast path for layers 1+)
    b1 = batch.clone()
    b1.pair_feat = None
    b1.pair_mask = None
    torch.manual_seed(0)
    snaps_fast = []
    for l in layers:
        l.eval()
        b1 = l(b1)
        snaps_fast.append((b1.pair_feat.detach().clone(),
                           b1.pair_mask.detach().clone()))
    pair_feat_fast = b1.pair_feat.clone()
    feat_fast = b1.feat.clone()

    # Run 2: clear pair_mask between layers → force slow path
    b2 = batch.clone()
    b2.pair_feat = None
    b2.pair_mask = None
    torch.manual_seed(0)
    snaps_slow = []
    for i, l in enumerate(layers):
        l.eval()
        b2 = l(b2)
        snaps_slow.append((b2.pair_feat.detach().clone(),
                           b2.pair_mask.detach().clone()))
        if i < len(layers) - 1:
            b2.pair_mask = None

    for i in range(len(layers)):
        pf_diff = (snaps_fast[i][0] - snaps_slow[i][0]).abs().max().item()
        pm_diff = (snaps_fast[i][1] - snaps_slow[i][1]).abs().max().item()
        print(f"Layer {i}: pair_feat diff={pf_diff:.3e}, pair_mask diff={pm_diff:.3e}")

    pair_feat_slow = b2.pair_feat.clone()
    feat_slow = b2.feat.clone()

    max_err_pair = (pair_feat_fast - pair_feat_slow).abs().max().item()
    max_err_feat = (feat_fast - feat_slow).abs().max().item()
    print(f"pair_feat max abs diff: {max_err_pair:.3e}")
    print(f"feat       max abs diff: {max_err_feat:.3e}")
    # Identity out_proj → feat should be *exactly* equal; pair_feat too
    # because we disabled dropout and used deterministic seeds.
    assert max_err_pair < 1e-5, f"pair_feat mismatch: {max_err_pair}"
    assert max_err_feat < 1e-5, f"feat mismatch: {max_err_feat}"
    print("OK: fast path output matches slow path output")

    # --- Additional: with non-identity out_proj, fast path should not fire.
    common_ndim = dict(common)
    common_ndim.update(in_dim=64, out_dim=128)  # in != out → slow path forced
    torch.manual_seed(0)
    layers_ndim = nn.ModuleList([
        PairMixerLayerPyg(layer_idx=i,
                          **{**common_ndim,
                             "in_dim": 128 if i > 0 else 64,
                             "out_dim": 128}) for i in range(3)
    ]).cuda()
    b2 = batch.clone()
    if hasattr(b2, "pair_feat"): b2.pair_feat = None
    if hasattr(b2, "pair_mask"): b2.pair_mask = None
    for l in layers_ndim:
        l.eval()
        b2 = l(b2)
    print(f"non-identity out_proj stack ran OK; feat.shape={tuple(b2.feat.shape)}")

if __name__ == "__main__":
    main()
