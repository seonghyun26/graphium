"""Thorough speed + correctness verification for compile_mode=pair_track + ckpt=F.

1. Speed: measure fwd+bwd of a realistic 18-layer PairMixer stack.
2. Correctness:
   a. Forward output (pair_feat, feat) must match eager within bf16 tol.
   b. Backward gradient on every parameter must match eager within bf16 tol.
   c. Also test with training=True (dropout=0.25) using deterministic seed.
"""
import os
import time
import copy
import argparse

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


def build_batch(B=128, N_min=30, N_max=48, D_s=256, d_edge=128):
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


def build_stack(depth, compile_it=False, use_checkpoint=False):
    from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg
    common = dict(
        in_dim=256, out_dim=256, pair_dim=128, in_dim_edges=128,
        hidden_dim_scaling=4.0, opm_hidden=32,
        pair_dropout=0.25, normalization="none",
        use_checkpoint=use_checkpoint,
        force_float32_einsums=True,
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
    stack = nn.ModuleList([
        PairMixerLayerPyg(layer_idx=i, **common) for i in range(depth)
    ])
    if compile_it:
        for l in stack:
            l._pair_track_ops = torch.compile(
                l._pair_track_ops, mode="default", dynamic=True,
            )
    return stack.cuda()


def one_step(stack, batch, train=True, seed=0):
    """Run one fwd+bwd step with a fixed seed (for deterministic dropout)."""
    for l in stack:
        l.train(train)
    b = batch.clone()
    b.pair_feat = None
    b.pair_mask = None
    b.feat = b.feat.detach().requires_grad_(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for l in stack:
            b = l(b)
        loss = b.pair_feat.pow(2).mean()
    loss.backward()
    return b.pair_feat.detach(), b.feat.detach(), loss.detach()


def timed(stack, batch, iters=5):
    """Warm up then time iters iterations of fwd+bwd."""
    for _ in range(2):
        one_step(stack, batch, train=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one_step(stack, batch, train=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--B", type=int, default=128)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    batch = build_batch(B=args.B).cuda()

    # Build two stacks with identical weights (eager + compiled) and ckpt=F.
    torch.manual_seed(0)
    eager = build_stack(args.depth, compile_it=False, use_checkpoint=False)
    compiled = copy.deepcopy(eager)
    for l in compiled:
        l._pair_track_ops = torch.compile(
            l._pair_track_ops, mode="default", dynamic=True,
        )

    # === 1. CORRECTNESS: eval mode (no dropout noise) ===
    print("=" * 70)
    print("Correctness @ eval mode (dropout off) — expect bit-exact")
    print("=" * 70)
    pf_e, ft_e, _ = one_step(eager, batch, train=False, seed=0)
    pf_c, ft_c, _ = one_step(compiled, batch, train=False, seed=0)
    err_pf = (pf_e.float() - pf_c.float()).abs().max().item()
    err_ft = (ft_e.float() - ft_c.float()).abs().max().item()
    print(f"  pair_feat max diff: {err_pf:.3e}")
    print(f"  feat       max diff: {err_ft:.3e}")
    ok1 = err_pf < 1e-2 and err_ft < 1e-2

    # === 2. CORRECTNESS: training mode with fixed seed (dropout 0.25) ===
    print()
    print("=" * 70)
    print("Correctness @ train mode with fixed seed (dropout=0.25)")
    print("=" * 70)
    # Zero all grads on both stacks
    for p in eager.parameters(): p.grad = None
    for p in compiled.parameters(): p.grad = None
    pf_e, ft_e, loss_e = one_step(eager, batch, train=True, seed=1234)
    for p in eager.parameters():
        if p.grad is not None:
            p.eager_grad = p.grad.detach().clone()

    for p in compiled.parameters(): p.grad = None
    pf_c, ft_c, loss_c = one_step(compiled, batch, train=True, seed=1234)

    err_pf = (pf_e.float() - pf_c.float()).abs().max().item()
    err_ft = (ft_e.float() - ft_c.float()).abs().max().item()
    err_loss = (loss_e - loss_c).abs().item()
    print(f"  pair_feat max diff: {err_pf:.3e}")
    print(f"  feat       max diff: {err_ft:.3e}")
    print(f"  loss       diff:    {err_loss:.3e}  (eager={loss_e.item():.4f}, compiled={loss_c.item():.4f})")

    # Gradient check: iterate matching parameters
    max_grad_diff = 0.0
    mismatches = 0
    for (n_e, p_e), (n_c, p_c) in zip(eager.named_parameters(), compiled.named_parameters()):
        g_e = getattr(p_e, "eager_grad", None)
        g_c = p_c.grad
        if g_e is None or g_c is None:
            continue
        d = (g_e.float() - g_c.float()).abs().max().item()
        max_grad_diff = max(max_grad_diff, d)
        if d > 5e-2:
            mismatches += 1
            if mismatches <= 3:
                print(f"  >> grad mismatch on {n_e}: {d:.3e}")
    print(f"  grad       max diff: {max_grad_diff:.3e}  ({mismatches} params > 5e-2)")
    ok2 = (err_pf < 1e-1 and err_ft < 1e-1 and err_loss < 1e-2
           and max_grad_diff < 5e-2)

    # === 3. SPEED ===
    print()
    print("=" * 70)
    print(f"Speed (depth={args.depth}, B={args.B}, 5-iter avg)")
    print("=" * 70)
    # Baseline: eager + ckpt=T (current default config)
    torch.manual_seed(0)
    baseline = build_stack(args.depth, compile_it=False, use_checkpoint=True)
    t_baseline = timed(baseline, batch, iters=5)

    # No-ckpt (the 1.32x lever)
    torch.manual_seed(0)
    noc = build_stack(args.depth, compile_it=False, use_checkpoint=False)
    t_noc = timed(noc, batch, iters=5)

    # Compile + ckpt=T
    torch.manual_seed(0)
    cpt = build_stack(args.depth, compile_it=True, use_checkpoint=True)
    t_cpt = timed(cpt, batch, iters=5)

    # Compile + ckpt=F (the 2.59x proposal)
    torch.manual_seed(0)
    cnc = build_stack(args.depth, compile_it=True, use_checkpoint=False)
    t_cnc = timed(cnc, batch, iters=5)

    print(f"  baseline (eager, ckpt=T)         {t_baseline:8.1f} ms   [1.00x]")
    print(f"  ckpt=F                            {t_noc:8.1f} ms   [{t_baseline/t_noc:.2f}x]")
    print(f"  compile + ckpt=T                  {t_cpt:8.1f} ms   [{t_baseline/t_cpt:.2f}x]")
    print(f"  compile + ckpt=F                  {t_cnc:8.1f} ms   [{t_baseline/t_cnc:.2f}x]")

    # === Final verdict ===
    print()
    if ok1 and ok2:
        print("CORRECTNESS: PASS (compile output & grads match eager within bf16 tol)")
    else:
        print(f"CORRECTNESS: FAIL  (ok1={ok1}, ok2={ok2})")


if __name__ == "__main__":
    main()
