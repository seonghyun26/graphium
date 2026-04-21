"""Per-op profiling of PairMixer to identify true bottlenecks.

Uses torch.profiler to rank CUDA kernel time by operator.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


def build_synthetic_batch(B=128, N_min=30, N_max=48, D_s=256, d_edge=128, device="cuda"):
    rng = torch.Generator().manual_seed(0)
    graphs = []
    for _ in range(B):
        n = torch.randint(N_min, N_max + 1, (1,), generator=rng).item()
        src = torch.cat([torch.arange(n - 1), torch.randint(0, n, (n // 3,), generator=rng)])
        dst = torch.cat([torch.arange(1, n), torch.randint(0, n, (n // 3,), generator=rng)])
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, D_s, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], d_edge, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    batch = Batch.from_data_list(graphs)
    return batch.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--force_f32", action="store_true", default=True)
    p.add_argument("--no_force_f32", dest="force_f32", action="store_false")
    p.add_argument("--no_ckpt", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="apply compile_mode=pair_track with default+dynamic")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg

    batch = build_synthetic_batch(B=args.B)
    print(f"# Profiling PairMixer depth={args.depth} B={args.B} "
          f"force_f32={args.force_f32} ckpt={not args.no_ckpt}")

    common = dict(
        in_dim=256, out_dim=256, pair_dim=128, in_dim_edges=128,
        hidden_dim_scaling=4.0, opm_hidden=32, pair_dropout=0.25,
        normalization="none",
        force_float32_einsums=args.force_f32,
        use_checkpoint=(not args.no_ckpt),
        tri_mul_mode="einsum",
        parallel_pair_ops=False,
        pair_init={
            "mode": "additive", "use_opm": True, "opm_init": "lecun",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance", "adjacency", "edge_feat"],
                "graph_distance": {"max_dist": 8, "embedding_dim": 32},
                "adjacency": {"embedding_dim": 8},
                "edge_feat": {"embedding_dim": 16},
            },
        },
    )

    torch.manual_seed(0)
    layers = nn.ModuleList([
        PairMixerLayerPyg(layer_idx=i, **common) for i in range(args.depth)
    ]).cuda()

    if args.compile:
        for l in layers:
            l._pair_track_ops = torch.compile(
                l._pair_track_ops, mode="default", dynamic=True,
            )

    def run(train=True):
        b = batch.clone()
        if hasattr(b, "pair_feat"):
            b.pair_feat = None
        b.feat = b.feat.detach().requires_grad_(True)
        for l in layers:
            l.train(train)
        for l in layers:
            b = l(b)
        loss = b.pair_feat.pow(2).mean()
        if train:
            loss.backward()
        return loss

    # Warmup
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):
            run(train=True)
    torch.cuda.synchronize()

    # Profile
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                run(train=True)

    print("\n=== Top ops by CUDA time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25,
        max_name_column_width=55, max_src_column_width=60,
    ))


if __name__ == "__main__":
    main()
