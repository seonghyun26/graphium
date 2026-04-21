#!/usr/bin/env python3
"""
Pairformer layer profiling script.

Measures per-component and end-to-end forward/backward timing using CUDA events.
Uses ADMET molecules for realistic molecule size distributions.

Usage:
    python profiling/profile_pairformer.py
    python profiling/profile_pairformer.py --d_s 384 --d_z 128 --depth 2 --batch_size 4
    python profiling/profile_pairformer.py --compile_mode pair_track --use_sdpa
    python profiling/profile_pairformer.py --csv results.csv
"""

import argparse
import sys
import os
import time
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from graphium.nn.pyg_layers.pairformer_pyg import PairformerLayerPyg


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------

def make_batch_synthetic(num_graphs, min_nodes, max_nodes, in_dim, seed=42):
    """Create a PyG Batch with random graphs of varying sizes."""
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(num_graphs):
        n = torch.randint(min_nodes, max_nodes + 1, (1,), generator=rng).item()
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, in_dim, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index))
    return Batch.from_data_list(graphs)


def make_batch_admet(batch_size, in_dim, seed=42):
    """Create a batch from real ADMET molecules (caco2_wang)."""
    try:
        from tdc.benchmark_group import admet_group
        import tempfile

        group = admet_group(path=os.path.join(tempfile.gettempdir(), "tdc_profile_cache"))
        benchmark = group.get("caco2_wang")
        train_df = benchmark["train"]
        smiles_list = train_df["Drug"].tolist()

        from rdkit import Chem
        # Get atom counts for realistic size distribution
        atom_counts = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                atom_counts.append(mol.GetNumAtoms())
        atom_counts = np.array(atom_counts)

        rng = np.random.RandomState(seed)
        selected = rng.choice(atom_counts, size=batch_size, replace=False)
        print(f"ADMET molecule sizes: min={selected.min()}, max={selected.max()}, "
              f"mean={selected.mean():.0f}, median={np.median(selected):.0f}")
        return make_batch_synthetic(
            batch_size, min_nodes=int(selected.min()), max_nodes=int(selected.max()),
            in_dim=in_dim, seed=seed
        )
    except ImportError:
        print("TDC not available, using synthetic data")
        return make_batch_synthetic(batch_size, 10, 50, in_dim, seed=seed)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class CudaTimer:
    """Accurate GPU timing using CUDA events."""

    def __init__(self):
        self.records = defaultdict(list)

    @contextmanager
    def time(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.records[name].append(start.elapsed_time(end))

    def summary(self, skip_first=2):
        """Return dict of name -> median ms (skipping warmup iterations)."""
        result = {}
        for name, times in self.records.items():
            trimmed = times[skip_first:] if len(times) > skip_first else times
            if trimmed:
                result[name] = {
                    "median_ms": np.median(trimmed),
                    "mean_ms": np.mean(trimmed),
                    "std_ms": np.std(trimmed),
                    "min_ms": np.min(trimmed),
                    "n": len(trimmed),
                }
        return result


def hook_submodule_timing(layer, timer, prefix=""):
    """Monkey-patch forward methods to add timing around each sub-module."""
    components = {
        "tri_mul_out": layer.tri_mul_out,
        "tri_mul_in": layer.tri_mul_in,
        "tri_att_start": layer.tri_att_start,
        "tri_att_end": layer.tri_att_end,
        "transition_z": layer.transition_z,
        "attn": layer.attn,
        "transition_s": layer.transition_s,
        "opm": layer.opm,
    }

    for name, module in components.items():
        orig_forward = module.forward
        tag = f"{prefix}{name}"

        def make_timed(fn, tag):
            def timed_forward(*args, **kwargs):
                with timer.time(tag):
                    return fn(*args, **kwargs)
            return timed_forward

        module.forward = make_timed(orig_forward, tag)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args):
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    print(f"\n{'='*70}")
    print(f"Pairformer Benchmark")
    print(f"{'='*70}")
    print(f"Config: d_s={args.d_s}, d_z={args.d_z}, depth={args.depth}")
    print(f"Batch: size={args.batch_size}, max_atoms={args.max_atoms}")
    print(f"Options: compile={args.compile_mode}, sdpa={args.use_sdpa}, "
          f"tri_mul={args.tri_mul_mode}, fp32_einsum={args.force_float32}")
    print(f"Device: {device}, Precision: bf16-mixed")
    print(f"Iterations: {args.n_iters} (skip first {args.warmup})")
    print(f"{'='*70}\n")

    # Build layer kwargs
    layer_kwargs = {
        "pair_dim": args.d_z,
        "num_heads": max(1, args.d_s // 48),  # ~48 head dim
        "pairwise_head_width": 32,
        "pairwise_num_heads": 4,
        "pair_dropout": 0.0,  # disable for deterministic timing
        "hidden_dim_scaling": 4.0,
        "opm_hidden": 32,
        "use_checkpoint": False,  # disable for accurate per-component timing
        "compile_mode": args.compile_mode,
        "use_sdpa_attn_pair_bias": args.use_sdpa,
        "tri_mul_mode": args.tri_mul_mode,
        "force_float32_einsums": args.force_float32,
        "parallel_pair_ops": args.parallel_pair_ops,
    }

    # Create layers
    layers = []
    for _ in range(args.depth):
        layer = PairformerLayerPyg(
            in_dim=args.d_s, out_dim=args.d_s, **layer_kwargs
        ).to(device).train()
        layers.append(layer)

    n_params = sum(p.numel() for p in layers[0].parameters())
    print(f"Params per layer: {n_params:,} | Total: {n_params * args.depth:,}")

    # Create batch
    if args.source == "admet":
        batch = make_batch_admet(args.batch_size, args.d_s, seed=42).to(device)
    else:
        batch = make_batch_synthetic(
            args.batch_size, args.max_atoms // 2, args.max_atoms,
            args.d_s, seed=42
        ).to(device)

    n_nodes = batch.num_nodes
    n_graphs = batch.num_graphs
    max_n = max(torch.bincount(batch.batch).tolist())
    print(f"Batch: {n_graphs} graphs, {n_nodes} total nodes, max_n={max_n}\n")

    # --- Per-component timing (single layer, no grad) ---
    timer = CudaTimer()
    layer0 = layers[0]
    hook_submodule_timing(layer0, timer)

    print("Phase 1: Per-component timing (single layer, no grad)...")
    for i in range(args.n_iters):
        b = batch.clone()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with timer.time("full_forward"):
                b = layer0(b)

    comp_summary = timer.summary(skip_first=args.warmup)

    # --- Full forward+backward timing ---
    timer2 = CudaTimer()
    print("Phase 2: Full forward+backward timing (all layers)...")
    for i in range(args.n_iters):
        b = batch.clone()
        for layer in layers:
            layer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with timer2.time("fwd"):
                for layer in layers:
                    b = layer(b)
            loss = b.feat.sum()

        with timer2.time("bwd"):
            loss.backward()

        with timer2.time("fwd+bwd"):
            pass  # just recording total

    # Manually compute fwd+bwd from individual measurements
    fwd_bwd_summary = timer2.summary(skip_first=args.warmup)

    # --- Print results ---
    print(f"\n{'='*70}")
    print(f"RESULTS (median ms, {args.n_iters - args.warmup} iterations)")
    print(f"{'='*70}")

    print(f"\n{'Component':<25} {'Median (ms)':>12} {'Mean (ms)':>12} {'Std':>8} {'%':>6}")
    print("-" * 65)

    total_fwd = comp_summary.get("full_forward", {}).get("median_ms", 0)
    for name in ["opm", "tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
                  "transition_z", "attn", "transition_s", "full_forward"]:
        if name in comp_summary:
            s = comp_summary[name]
            pct = (s["median_ms"] / total_fwd * 100) if total_fwd > 0 and name != "full_forward" else 0
            marker = " <<<" if name == "full_forward" else ""
            print(f"  {name:<23} {s['median_ms']:>10.2f}ms {s['mean_ms']:>10.2f}ms {s['std_ms']:>6.2f} {pct:>5.1f}%{marker}")

    if "fwd" in fwd_bwd_summary:
        print(f"\n{'Stage':<25} {'Median (ms)':>12}")
        print("-" * 40)
        for name in ["fwd", "bwd"]:
            if name in fwd_bwd_summary:
                s = fwd_bwd_summary[name]
                print(f"  {name} ({args.depth} layers)     {s['median_ms']:>10.2f}ms")

    throughput = n_graphs / (fwd_bwd_summary.get("fwd", {}).get("median_ms", 1) / 1000)
    print(f"\n  Throughput (fwd only): {throughput:.0f} graphs/s")

    # --- CSV output ---
    if args.csv:
        import csv
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "d_s", "d_z", "depth", "batch_size", "max_n", "compile_mode",
                    "use_sdpa", "tri_mul_mode", "force_fp32",
                    "fwd_ms", "bwd_ms", "tri_mul_out_ms", "tri_mul_in_ms",
                    "tri_att_start_ms", "tri_att_end_ms", "attn_ms",
                    "transition_z_ms", "transition_s_ms", "opm_ms",
                    "throughput_graphs_s"
                ])
            writer.writerow([
                args.d_s, args.d_z, args.depth, args.batch_size, max_n,
                args.compile_mode, args.use_sdpa, args.tri_mul_mode, args.force_float32,
                fwd_bwd_summary.get("fwd", {}).get("median_ms", 0),
                fwd_bwd_summary.get("bwd", {}).get("median_ms", 0),
                comp_summary.get("tri_mul_out", {}).get("median_ms", 0),
                comp_summary.get("tri_mul_in", {}).get("median_ms", 0),
                comp_summary.get("tri_att_start", {}).get("median_ms", 0),
                comp_summary.get("tri_att_end", {}).get("median_ms", 0),
                comp_summary.get("attn", {}).get("median_ms", 0),
                comp_summary.get("transition_z", {}).get("median_ms", 0),
                comp_summary.get("transition_s", {}).get("median_ms", 0),
                comp_summary.get("opm", {}).get("median_ms", 0),
                throughput,
            ])
        print(f"\nResults appended to {args.csv}")


def main():
    parser = argparse.ArgumentParser(description="Profile Pairformer layer performance")
    # Architecture
    parser.add_argument("--d_s", type=int, default=384, help="Node feature dim")
    parser.add_argument("--d_z", type=int, default=128, help="Pair feature dim")
    parser.add_argument("--depth", type=int, default=2, help="Number of layers for fwd+bwd timing")
    # Batch
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_atoms", type=int, default=50, help="Max atoms per molecule")
    parser.add_argument("--source", choices=["synthetic", "admet"], default="synthetic",
                        help="Batch source")
    # Optimizations
    parser.add_argument("--compile_mode", choices=["none", "pair_track", "full"], default="none")
    parser.add_argument("--use_sdpa", action="store_true", help="Use SDPA for AttentionPairBias")
    parser.add_argument("--tri_mul_mode", choices=["einsum", "bmm"], default="einsum")
    parser.add_argument("--force_float32", action="store_true", default=True,
                        help="Force float32 in einsums (default: True)")
    parser.add_argument("--no_force_float32", dest="force_float32", action="store_false")
    parser.add_argument("--parallel_pair_ops", action="store_true",
                        help="Parallel triangle mul/attn residuals (AlphaFold-style)")
    # Runtime
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--n_iters", type=int, default=20, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations to skip")
    parser.add_argument("--csv", type=str, default=None, help="Append results to CSV file")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
