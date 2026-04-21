"""Verify torch.compile on PairMixer gives same output as eager.

Forward-only check: compiled _pair_track_ops must match eager mode to
within bf16 precision.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


def build_batch():
    rng = torch.Generator().manual_seed(42)
    graphs = []
    for _ in range(8):
        n = torch.randint(10, 30, (1,), generator=rng).item()
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, 128, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], 64, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    return Batch.from_data_list(graphs)


def main():
    from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg

    common = dict(
        in_dim=128, out_dim=128, pair_dim=64, in_dim_edges=64,
        hidden_dim_scaling=4.0, opm_hidden=32,
        pair_dropout=0.0,  # deterministic
        normalization="none",
        use_checkpoint=False,
        force_float32_einsums=True,
        tri_mul_mode="einsum",
        parallel_pair_ops=False,
        pair_init={"mode": "opm_only", "use_opm": True, "opm_init": "lecun"},
    )

    # Build eager stack
    torch.manual_seed(0)
    eager = nn.ModuleList([
        PairMixerLayerPyg(layer_idx=i, **common) for i in range(4)
    ]).cuda()

    # Deep-copy to compiled version and wrap _pair_track_ops
    compiled = copy.deepcopy(eager)
    for layer in compiled:
        layer._pair_track_ops = torch.compile(
            layer._pair_track_ops, mode="default", dynamic=True,
        )

    batch = build_batch().cuda()

    def run(stack):
        b = batch.clone()
        b.pair_feat = None
        b.pair_mask = None
        for l in stack:
            l.eval()
            b = l(b)
        return b

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_e = run(eager)
        out_c = run(compiled)

    # Compare
    pf_diff = (out_e.pair_feat.float() - out_c.pair_feat.float()).abs().max().item()
    ft_diff = (out_e.feat.float() - out_c.feat.float()).abs().max().item()
    print(f"Compile vs eager (bf16 autocast):")
    print(f"  pair_feat max diff: {pf_diff:.3e}")
    print(f"  feat       max diff: {ft_diff:.3e}")

    # bf16 has ~3e-3 epsilon, so allow some slack for compile's fused kernels
    assert pf_diff < 5e-2, f"pair_feat diverged: {pf_diff}"
    assert ft_diff < 5e-2, f"feat diverged: {ft_diff}"
    print("OK: compile output matches eager within bf16 tolerance")


if __name__ == "__main__":
    main()
