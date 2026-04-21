"""Isolated dummy benchmark for proposed fusion optimizations.

Tests four things on synthetic tensors without touching shipped code:

  1. Transition (SwiGLU):
       baseline:  fc1(x), fc2(x) — two matmuls on same input
       fused:     cat([fc1.w, fc2.w]) once, one big matmul, chunk output

  2. TriangleMultiplication p_in + g_in:
       same pattern (both take norm_in(x))

  3. Numerical equivalence between baseline and fused variants.

  4. Whether weight-cat-per-forward has material overhead vs pre-allocating
     a fused parameter (to decide whether to change the module structure).

At realistic shapes:  B=128, N=48, D_z=128, hidden_scale=4 (hidden=512).
"""
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standalone copies of Transition / TriMul projection blocks
# ---------------------------------------------------------------------------


class TransitionBase(nn.Module):
    """Matches graphium.nn.pyg_layers.pairformer_pyg.Transition."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        return self.fc3(self.silu(self.fc1(x)) * self.fc2(x))


class TransitionFusedCat(nn.Module):
    """fc1+fc2 fused via torch.cat on weights in forward (keeps state_dict)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        w12 = torch.cat([self.fc1.weight, self.fc2.weight], dim=0)
        h12 = F.linear(x, w12)
        h1, h2 = h12.chunk(2, dim=-1)
        return self.fc3(self.silu(h1) * h2)


class TransitionFusedParam(nn.Module):
    """fc1+fc2 merged into a single Linear (breaks state_dict compat)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.fc3 = nn.Linear(hidden, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        h12 = self.fc12(x)
        h1, h2 = h12.chunk(2, dim=-1)
        return self.fc3(self.silu(h1) * h2)


class TriMulProjBase(nn.Module):
    """Just the norm_in → (p_in * sigmoid(g_in)) block — the part we'd fuse."""

    def __init__(self, dim):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x):
        x = self.norm_in(x)
        return self.p_in(x) * self.g_in(x).sigmoid()


class TriMulProjFusedCat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x):
        x = self.norm_in(x)
        w_fused = torch.cat([self.p_in.weight, self.g_in.weight], dim=0)
        pg = F.linear(x, w_fused)
        p, g = pg.chunk(2, dim=-1)
        return p * g.sigmoid()


class TriMulProjFusedParam(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_g_in = nn.Linear(dim, 4 * dim, bias=False)

    def forward(self, x):
        x = self.norm_in(x)
        pg = self.p_g_in(x)
        p, g = pg.chunk(2, dim=-1)
        return p * g.sigmoid()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync(): torch.cuda.synchronize()


def bench(module, x, iters=100, warmup=20, use_bwd=True, autocast=True):
    module = module.cuda().train()
    x = x.cuda().detach().requires_grad_(use_bwd)

    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if autocast else _nullcontext()

    # Warmup
    for _ in range(warmup):
        with ctx:
            y = module(x)
        if use_bwd:
            y.pow(2).mean().backward()
            x.grad = None
            for p in module.parameters():
                p.grad = None
    _sync()

    # Measure
    t0 = time.perf_counter()
    for _ in range(iters):
        with ctx:
            y = module(x)
        if use_bwd:
            y.pow(2).mean().backward()
            x.grad = None
            for p in module.parameters():
                p.grad = None
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms per iter


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


def check_numerics(baseline_cls, fused_cls, dim, shape, tol=1e-4):
    """Verify fused variant gives same output as baseline when weights copied."""
    is_transition = "Transition" in baseline_cls.__name__
    build = (lambda cls: cls(dim, dim * 4)) if is_transition else (lambda cls: cls(dim))
    torch.manual_seed(0)
    b = build(baseline_cls).cuda().eval()
    torch.manual_seed(0)
    f = build(fused_cls).cuda().eval()
    # Copy matching weights
    b_sd, f_sd = b.state_dict(), f.state_dict()
    for k, v in b_sd.items():
        if k in f_sd:
            f_sd[k] = v.clone()
    # For "FusedParam" the weight layout differs — reconstruct it.
    if "FusedParam" in fused_cls.__name__:
        if "Transition" in baseline_cls.__name__:
            f_sd["fc12.weight"] = torch.cat([b_sd["fc1.weight"], b_sd["fc2.weight"]], dim=0)
        else:
            f_sd["p_g_in.weight"] = torch.cat([b_sd["p_in.weight"], b_sd["g_in.weight"]], dim=0)
    f.load_state_dict(f_sd)

    x = torch.randn(*shape).cuda()
    with torch.no_grad():
        y_b = b(x)
        y_f = f(x)
    err = (y_b - y_f).abs().max().item()
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    B, N, D = args.B, args.N, args.dim
    hidden = D * 4
    shape = (B, N, N, D)
    print(f"# shape={shape}, hidden={hidden}, bf16 autocast, fwd+bwd")

    # ---- Correctness checks ----
    err1 = check_numerics(TransitionBase, TransitionFusedCat, D, shape)
    err2 = check_numerics(TransitionBase, TransitionFusedParam, D, shape)
    err3 = check_numerics(TriMulProjBase, TriMulProjFusedCat, D, shape)
    err4 = check_numerics(TriMulProjBase, TriMulProjFusedParam, D, shape)
    print(f"# numerical equivalence (max abs diff):")
    print(f"   Transition   base vs fused_cat   : {err1:.2e}")
    print(f"   Transition   base vs fused_param : {err2:.2e}")
    print(f"   TriMulProj   base vs fused_cat   : {err3:.2e}")
    print(f"   TriMulProj   base vs fused_param : {err4:.2e}")

    x = torch.randn(*shape)

    results = []
    # torch.compile variants — "reduce-overhead" fuses small ops
    compiled_tbase = torch.compile(TransitionBase(D, hidden), mode="reduce-overhead")
    compiled_tmbase = torch.compile(TriMulProjBase(D), mode="reduce-overhead")
    for name, mod in [
        ("Transition        BASE          ", TransitionBase(D, hidden)),
        ("Transition        FUSED (cat)   ", TransitionFusedCat(D, hidden)),
        ("Transition        FUSED (param) ", TransitionFusedParam(D, hidden)),
        ("Transition        COMPILED      ", compiled_tbase),
        ("TriMulProj        BASE          ", TriMulProjBase(D)),
        ("TriMulProj        FUSED (cat)   ", TriMulProjFusedCat(D)),
        ("TriMulProj        FUSED (param) ", TriMulProjFusedParam(D)),
        ("TriMulProj        COMPILED      ", compiled_tmbase),
    ]:
        t = bench(mod, x, iters=args.iters)
        results.append((name, t))

    print("\n# Benchmarks (ms per fwd+bwd iter, {} iters)".format(args.iters))
    print("-" * 60)
    base_t = None
    for name, t in results:
        if "BASE" in name:
            base_t = t
            sp = 1.0
        else:
            sp = base_t / t
        print(f"  {name}  {t:7.3f} ms   [{sp:.2f}x]")

    # Multi-layer scenario (18 layers) — measure compound effect
    # Only Transition stacks cleanly (output shape == input shape);
    # TriMulProj produces (B,N,N,2D) so we can't stack those as-is.
    print("\n# 18-layer stack (Transition only — same shape in/out)")
    for name, factory in [
        ("18x Transition   BASE          ", lambda: TransitionBase(D, hidden)),
        ("18x Transition   FUSED (cat)   ", lambda: TransitionFusedCat(D, hidden)),
        ("18x Transition   FUSED (param) ", lambda: TransitionFusedParam(D, hidden)),
    ]:
        stack = nn.Sequential(*[factory() for _ in range(18)])
        t = bench(stack, x, iters=max(30, args.iters // 4))
        print(f"  {name}  {t:7.3f} ms")


if __name__ == "__main__":
    main()
