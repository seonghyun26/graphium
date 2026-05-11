#!/usr/bin/env python
"""Two-model fingerprint ensemble probe using the MiniMol recipe.

Extracts 512-d fingerprints from two frozen encoders (e.g. PairMixer-12M +
MPNN++-12M), concatenates them to 1024-d, then runs the MiniMol-faithful
probing head (TaskHead, Algorithm 1) on the concatenated features using
per-task HP from the MiniMol SWEEP_RESULTS table.

Usage:
    python scripts/pairmixer/ensemble_minimol_probe.py \\
        --ckpt-a models_checkpoints/.../pairmixer_12M/epoch=099.ckpt \\
        --ckpt-b models_checkpoints/.../mpnnpp_12M/epoch=049.ckpt \\
        --model-a pairmixer_12M --model-b mpnnpp_12M \\
        --pretrain-label pairmixerv3_ep099+mpnnpp_ep049 \\
        --tasks lipophilicity_astrazeneca bbb_martins cyp3a4_veith \\
                half_life_obach ld50_zhu \\
        --reps 1 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from downstream.model import load_encoder  # noqa: E402
from pairmixer_minimol_probe import (     # noqa: E402
    SWEEP_RESULTS,
    _append_rows,
    run_one_task,
)

DEFAULT_OUTPUT_CSV = ROOT / "results" / "ensemble_minimol_probe.csv"
DEFAULT_MOL_CACHE  = ROOT / "datacache" / "ensemble_minimol_fps"


def _concat_embeddings(
    enc_a, enc_b,
    all_smiles: List[str],
    cache_dir: Path,
    task: str,
) -> Dict[str, np.ndarray]:
    """Extract from both encoders, intersect survived SMILES, concatenate."""
    cache_a = cache_dir / f"encA_{enc_a.encoder_tag}__{task}.pt"
    cache_b = cache_dir / f"encB_{enc_b.encoder_tag}__{task}.pt"

    X_a, mask_a = enc_a.extract_cached(all_smiles, cache_a)
    X_b, mask_b = enc_b.extract_cached(all_smiles, cache_b)

    surv_a = {s: X_a[i] for i, (s, ok) in enumerate(zip(all_smiles, mask_a)) if ok}
    surv_b = {s: X_b[i] for i, (s, ok) in enumerate(zip(all_smiles, mask_b)) if ok}

    common = sorted(set(surv_a) & set(surv_b))
    n_drop = len(all_smiles) - len(common)
    print(
        f"  fp_a={X_a.shape[1]}d  fp_b={X_b.shape[1]}d  "
        f"concat={X_a.shape[1] + X_b.shape[1]}d  "
        f"survived={len(common):,}/{len(all_smiles):,}"
        + (f"  (dropped {n_drop})" if n_drop else ""),
        flush=True,
    )
    return {s: np.concatenate([surv_a[s], surv_b[s]], axis=0) for s in common}


def main() -> None:
    p = argparse.ArgumentParser(description="Two-model concat MiniMol probe")
    p.add_argument("--ckpt-a", required=True, help="Checkpoint for model A")
    p.add_argument("--ckpt-b", required=True, help="Checkpoint for model B")
    p.add_argument("--model-a", default="pairmixer_12M", help="Hydra model name for A")
    p.add_argument("--model-b", default="mpnnpp_12M",    help="Hydra model name for B")
    p.add_argument("--ckpt-tag-a", default=None, help="Short label for ckpt A")
    p.add_argument("--ckpt-tag-b", default=None, help="Short label for ckpt B")
    p.add_argument("--pretrain-label", required=True)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reps",           type=int, default=5)
    p.add_argument("--ensemble-size",  type=int, default=5)
    p.add_argument("--epochs",         type=int, default=25)
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--featurize-n-jobs", type=int, default=8)
    p.add_argument("--reg-loss", choices=("mse", "mae"), default="mse")
    p.add_argument("--admet-cache",  default=str(ROOT / "expts" / "data" / "admet"))
    p.add_argument("--mol-cache-dir", default=str(DEFAULT_MOL_CACHE))
    p.add_argument("--output-csv",    default=str(DEFAULT_OUTPUT_CSV))
    args = p.parse_args()

    for ckpt in (args.ckpt_a, args.ckpt_b):
        if not Path(ckpt).exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt}")

    Path(args.admet_cache).mkdir(parents=True, exist_ok=True)
    Path(args.mol_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    from tdc.benchmark_group import admet_group  # noqa: E402
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        group = admet_group(path=args.admet_cache)

    tasks = list(args.tasks) if args.tasks else sorted(SWEEP_RESULTS.keys())
    unknown = [t for t in tasks if t not in SWEEP_RESULTS]
    if unknown:
        sys.exit(f"ERROR: unknown tasks (no MiniMol HP): {unknown}")

    print(f"Loading encoder A: {args.model_a}  {Path(args.ckpt_a).name}", flush=True)
    enc_a = load_encoder(
        "pairmixer", ckpt_path=args.ckpt_a, model_name=args.model_a,
        device=args.device, batch_size=args.embed_batch_size,
        featurize_n_jobs=args.featurize_n_jobs, ckpt_tag=args.ckpt_tag_a,
    )
    print(f"Loading encoder B: {args.model_b}  {Path(args.ckpt_b).name}", flush=True)
    enc_b = load_encoder(
        "pairmixer", ckpt_path=args.ckpt_b, model_name=args.model_b,
        device=args.device, batch_size=args.embed_batch_size,
        featurize_n_jobs=args.featurize_n_jobs, ckpt_tag=args.ckpt_tag_b,
    )

    cache_dir = Path(args.mol_cache_dir)
    device    = torch.device(args.device)

    # Minimal namespace that run_one_task reads for metadata + training config.
    _probe_args = types.SimpleNamespace(
        reps=args.reps,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        reg_loss=args.reg_loss,
        model_name=f"{args.model_a}+{args.model_b}",
        pretrain_label=args.pretrain_label,
        ckpt=f"{args.ckpt_a}|{args.ckpt_b}",
        ckpt_tag=f"{enc_a.ckpt_tag}+{enc_b.ckpt_tag}",
    )

    for task in tasks:
        print(f"\n=== {task} ===", flush=True)
        bench = group.get(task)
        all_smiles = sorted(
            set(bench["train_val"]["Drug"].tolist()) | set(bench["test"]["Drug"].tolist())
        )

        smi_to_fp = _concat_embeddings(enc_a, enc_b, all_smiles, cache_dir, task)
        if not smi_to_fp:
            print(f"  [{task}] no SMILES survived both encoders; skipping.", flush=True)
            continue

        input_dim = next(iter(smi_to_fp.values())).shape[0]
        hp_override = SWEEP_RESULTS.get(task)

        try:
            row = run_one_task(
                task, group, enc_a, smi_to_fp, input_dim,
                args=_probe_args, device=device,
                hp_override=hp_override, hp_source="minimol",
            )
        except Exception as exc:
            print(f"  [{task}] FAIL: {type(exc).__name__}: {exc}", flush=True)
            row = None

        if row is not None:
            row["model"]    = _probe_args.model_name
            row["ckpt"]     = _probe_args.ckpt
            row["ckpt_tag"] = _probe_args.ckpt_tag
            _append_rows(Path(args.output_csv), [row])

    print(f"\nDone. Results appended to {args.output_csv}")


if __name__ == "__main__":
    main()
