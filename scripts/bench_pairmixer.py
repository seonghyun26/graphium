"""Microbenchmark for PairMixer layer speed on GPU.

Measures forward + forward/backward wall time for:
  * Baseline config (force_float32=True, use_checkpoint=True, tri_mul_mode=einsum)
  * force_float32_einsums=False   (bf16 tensor cores on Blackwell → ~2x)
  * tri_mul_mode="bmm"            (explicit bmm vs einsum)
  * parallel_pair_ops=True        (AlphaFold-style parallel residuals)
  * use_checkpoint=False          (no recompute, but needs more memory)

Uses a realistic-size synthetic batch (B=128, N_max=48, D_s=256, D_z=128, depth=18).
"""
import argparse
import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Make sure CUDA device is set before importing anything that touches CUDA
def _pin_device(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def build_synthetic_batch(B=128, N_min=30, N_max=48, D_s=256, d_edge=128, device="cuda"):
    """Create a realistic-size molecular-graph-like PyG batch."""
    rng = torch.Generator().manual_seed(0)
    graphs = []
    for _ in range(B):
        n = torch.randint(N_min, N_max + 1, (1,), generator=rng).item()
        # Random chain-ish graph with some branching
        src = torch.cat([torch.arange(n - 1), torch.randint(0, n, (n // 3,), generator=rng)])
        dst = torch.cat([torch.arange(1, n), torch.randint(0, n, (n // 3,), generator=rng)])
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, D_s, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], d_edge, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    batch = Batch.from_data_list(graphs)
    return batch.to(device)


def build_layer_stack(depth, **kwargs):
    """Instantiate a depth-layer PairMixer stack (no pre_nn / graph_output)."""
    from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg
    compile_mode = kwargs.pop("_bench_compile_mode", None)
    layers = nn.ModuleList()
    for i in range(depth):
        lk = dict(kwargs)
        lk["layer_idx"] = i
        lk["compile_mode"] = "none"  # apply ourselves below to pick a safer mode
        layer = PairMixerLayerPyg(**lk)
        if compile_mode == "pair_track":
            # "default" mode avoids the CUDA-graph replay issues that
            # "reduce-overhead" hits when inputs change between calls.
            layer._pair_track_ops = torch.compile(
                layer._pair_track_ops, mode="default", dynamic=True,
            )
        elif compile_mode == "full":
            layer.forward = torch.compile(
                layer.forward, mode="default", dynamic=True,
            )
        layers.append(layer)
    return layers


@torch.no_grad()
def _clone_batch(batch):
    out = batch.clone()
    # Clear any pair_feat so layer 0 rebuilds it
    if hasattr(out, "pair_feat"):
        out.pair_feat = None
    return out


def bench_once(layers, batch, train=True, loss_scale=1.0):
    """One fwd+bwd pass. Returns elapsed seconds."""
    if train:
        for l in layers:
            l.train()
    else:
        for l in layers:
            l.eval()
    batch = _clone_batch(batch)
    # Make node features require grad so backward has something to do
    batch.feat = batch.feat.detach().requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for l in layers:
        batch = l(batch)
    loss = batch.pair_feat.pow(2).mean() * loss_scale
    if train:
        loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@dataclass
class Result:
    name: str
    fwd_ms: float = 0.0
    fwdbwd_ms: float = 0.0
    peak_mb: float = 0.0
    notes: str = ""


def run_config(name, depth, batch, **layer_kwargs) -> Result:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Build fresh stack for each config so random init is controlled
    torch.manual_seed(0)
    layers = build_layer_stack(depth, **layer_kwargs).cuda()

    # Warmup (triggers autotune, compile, etc.)
    for _ in range(2):
        _ = bench_once(layers, batch, train=True)

    # Time forward+backward
    torch.cuda.synchronize()
    times_fb = []
    for _ in range(5):
        times_fb.append(bench_once(layers, batch, train=True))

    # Time forward-only (eval)
    times_f = []
    for _ in range(5):
        times_f.append(bench_once(layers, batch, train=False))

    peak = torch.cuda.max_memory_allocated() / 1024**2
    return Result(
        name=name,
        fwd_ms=1e3 * (sum(times_f) / len(times_f)),
        fwdbwd_ms=1e3 * (sum(times_fb) / len(times_fb)),
        peak_mb=peak,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--out", type=str, default="/tmp/pairmixer_bench.json")
    p.add_argument("--quick", action="store_true", help="depth=6 B=64")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16",
                   help="autocast dtype for model fwd/bwd")
    args = p.parse_args()

    if args.quick:
        args.depth = 6
        args.B = 64

    _pin_device(args.gpu)

    # These imports after device pinning
    batch = build_synthetic_batch(B=args.B)
    print(f"# batch: B={args.B}, depth={args.depth}, device=cuda:{args.gpu} "
          f"dtype={args.dtype}")

    common = dict(
        in_dim=256, out_dim=256, pair_dim=128, in_dim_edges=128,
        hidden_dim_scaling=4.0, opm_hidden=32, pair_dropout=0.25,
        normalization="none",
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

    autocast_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    def run(name, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        # Wrap every config in autocast for realistic training conditions
        # We approximate by setting fp32 einsums flag + running in the requested dtype
        with torch.cuda.amp.autocast(enabled=(args.dtype != "fp32"),
                                     dtype=autocast_dtype):
            r = run_config(name, args.depth, batch, **cfg)
        return r

    results: List[Result] = []

    # Baseline (current default)
    results.append(run("baseline (f32ein=T, ckpt=T, einsum)",
                       force_float32_einsums=True, use_checkpoint=True,
                       tri_mul_mode="einsum", parallel_pair_ops=False))

    # No float32 upcast in tri_mul / OPM
    results.append(run("f32ein=F",
                       force_float32_einsums=False, use_checkpoint=True,
                       tri_mul_mode="einsum", parallel_pair_ops=False))

    # Explicit bmm for tri_mul
    results.append(run("tri_mul=bmm, f32ein=F",
                       force_float32_einsums=False, use_checkpoint=True,
                       tri_mul_mode="bmm", parallel_pair_ops=False))

    # No gradient checkpointing
    results.append(run("ckpt=F, f32ein=T",
                       force_float32_einsums=True, use_checkpoint=False,
                       tri_mul_mode="einsum", parallel_pair_ops=False))

    results.append(run("ckpt=F, f32ein=F",
                       force_float32_einsums=False, use_checkpoint=False,
                       tri_mul_mode="einsum", parallel_pair_ops=False))

    # Parallel pair ops
    results.append(run("parallel_ops, f32ein=F",
                       force_float32_einsums=False, use_checkpoint=True,
                       tri_mul_mode="einsum", parallel_pair_ops=True))

    # Best combo
    results.append(run("all-opts: bmm+parallel+f32ein=F+ckpt=F",
                       force_float32_einsums=False, use_checkpoint=False,
                       tri_mul_mode="bmm", parallel_pair_ops=True))

    # torch.compile on the pair-track block (dummy bench showed 1.57-1.59x)
    # Use _bench_compile_mode (monkey-patched in build_layer_stack) to pick
    # "default" compile mode instead of the shipped "reduce-overhead"
    # which fails on dynamic PyG batch shapes.
    results.append(run("compile=pair_track, f32ein=T, ckpt=T",
                       force_float32_einsums=True, use_checkpoint=True,
                       tri_mul_mode="einsum", parallel_pair_ops=False,
                       _bench_compile_mode="pair_track"))
    results.append(run("compile=pair_track, ckpt=F",
                       force_float32_einsums=True, use_checkpoint=False,
                       tri_mul_mode="einsum", parallel_pair_ops=False,
                       _bench_compile_mode="pair_track"))

    # Print table
    print("\n{:<48} {:>10} {:>12} {:>10}".format(
        "config", "fwd (ms)", "fwd+bwd (ms)", "peak (MB)"))
    print("-" * 84)
    baseline_fb = results[0].fwdbwd_ms
    for r in results:
        sp = baseline_fb / r.fwdbwd_ms if r.fwdbwd_ms > 0 else 0.0
        print("{:<48} {:>10.1f} {:>12.1f} {:>10.0f}  [{:.2f}x]".format(
            r.name, r.fwd_ms, r.fwdbwd_ms, r.peak_mb, sp))

    with open(args.out, "w") as f:
        json.dump({
            "meta": {"gpu": args.gpu, "depth": args.depth, "B": args.B,
                     "dtype": args.dtype},
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\n# Saved to {args.out}")


if __name__ == "__main__":
    main()
