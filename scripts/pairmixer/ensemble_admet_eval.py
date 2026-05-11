#!/usr/bin/env python
"""Evaluate a two-model fingerprint ensemble on TDC ADMET or Polaris ADME-Fang.

MolGPS-style: embed each molecule with two independent backbones, concatenate
the fingerprints, then fit a sklearn head on the concatenated features.

Pipeline:
  1. Embed all SMILES with model-A (e.g. pairmixer_12M) → z_A  (512-d)
  2. Embed all SMILES with model-B (e.g. mpnnpp_12M)   → z_B  (512-d)
  3. Concatenate  [z_A | z_B]                           → 1024-d features
  4. Fit sklearn head (Ridge / LogReg / MLP) on train+val, evaluate on test.
  5. Append metrics to results/ensemble_admet_results.csv.

Usage:
    python scripts/pairmixer/ensemble_admet_eval.py \\
        --ckpt-a models_checkpoints/pretender_v3/pairmixer_12M/.../last.ckpt \\
        --ckpt-b models_checkpoints/pretender_v3/mpnnpp_12M/.../last.ckpt \\
        --model-a pairmixer_12M --model-b mpnnpp_12M \\
        --benchmark tdc --task lipophilicity_astrazeneca --seed 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── Re-use the embedding machinery from pairmixer_admet_eval ─────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from pairmixer_admet_eval import (
    EMB_CACHE_DIR,
    POLARIS_BENCHMARKS,
    TDC_CLASSIFICATION,
    TDC_REGRESSION,
    embed_smiles_cached,
    fit_and_eval,
    load_polaris_task,
    load_tdc_task,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = ROOT / "results" / "ensemble_admet_results.csv"


# ── Embedding dict (smiles → vector or None) ──────────────────────────────────
def _build_embedding_dict(
    smiles_list: List[str],
    cache_path: Path,
    ckpt: str,
    model_name: str,
    device: str,
    batch_size: int,
    featurize_n_jobs: int,
) -> Dict[str, Optional[np.ndarray]]:
    """Return {smiles: embedding_array} for every unique SMILES in smiles_list."""
    unique = list(dict.fromkeys(smiles_list))
    z, mask = embed_smiles_cached(
        unique, cache_path, ckpt, model_name, device, batch_size, featurize_n_jobs,
    )
    result: Dict[str, Optional[np.ndarray]] = {}
    z_idx = 0
    for smi, valid in zip(unique, mask):
        if valid:
            result[smi] = z[z_idx]
            z_idx += 1
        else:
            result[smi] = None
    return result


def _build_features(
    df: pd.DataFrame,
    emb_a: Dict[str, Optional[np.ndarray]],
    emb_b: Dict[str, Optional[np.ndarray]],
    smiles_col: str = "Drug",
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate [z_A | z_B] for rows where both embeddings are valid."""
    rows_z, rows_y = [], []
    for _, row in df.iterrows():
        s = row[smiles_col]
        za = emb_a.get(s)
        zb = emb_b.get(s)
        if za is not None and zb is not None:
            rows_z.append(np.concatenate([za, zb]))
            rows_y.append(float(row["Y"]))
    if not rows_z:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0)
    return np.array(rows_z, dtype=np.float32), np.array(rows_y)


# ── Results CSV ───────────────────────────────────────────────────────────────
def append_results_row(row: Dict[str, object]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    fieldnames = list(row.keys())
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            fieldnames = list(dict.fromkeys(list(reader.fieldnames or []) + fieldnames))
    with open(RESULTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, help="Checkpoint for model A (e.g. pairmixer_12M)")
    parser.add_argument("--ckpt-b", required=True, help="Checkpoint for model B (e.g. mpnnpp_12M)")
    parser.add_argument("--model-a", default="pairmixer_12M", help="model config name for ckpt-a")
    parser.add_argument("--model-b", default="mpnnpp_12M",    help="model config name for ckpt-b")
    parser.add_argument("--ckpt-tag-a", default=None, help="Short label for model A in results CSV")
    parser.add_argument("--ckpt-tag-b", default=None, help="Short label for model B in results CSV")
    parser.add_argument("--benchmark", choices=["tdc", "polaris"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp", "xgboost"], default="mlp")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--featurize-n-jobs", type=int, default=8)
    args = parser.parse_args()

    for ckpt_arg in ("ckpt_a", "ckpt_b"):
        ckpt_val = getattr(args, ckpt_arg)
        if ckpt_val.lower() not in ("scratch", "random", "none") and not Path(ckpt_val).exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt_val}")

    t0 = time.time()
    if args.benchmark == "tdc":
        if args.task in TDC_REGRESSION:
            task_type = "regression"
        elif args.task in TDC_CLASSIFICATION:
            task_type = "classification"
        else:
            sys.exit(f"Unknown TDC task: {args.task}")
        train, val, test = load_tdc_task(args.task, args.seed)
    else:
        if args.task not in POLARIS_BENCHMARKS:
            sys.exit(f"Unknown Polaris task: {args.task}")
        task_type = "regression"
        train, val, test = load_polaris_task(args.task, args.seed)

    smiles_col = "Drug"
    print(
        f"[ensemble:{args.benchmark}/{args.task} seed={args.seed}] "
        f"train={len(train)} val={len(val)} test={len(test)}  task_type={task_type}"
    )

    all_smiles = pd.concat([train[smiles_col], val[smiles_col], test[smiles_col]]).unique().tolist()

    def _cache_path(ckpt: str, model: str) -> Path:
        ckpt_hash = hashlib.sha256(ckpt.encode()).hexdigest()[:12]
        return EMB_CACHE_DIR / f"{args.benchmark}_{args.task}_{model}_{ckpt_hash}.pt"

    print(f"  [A] {args.model_a}: embedding {len(all_smiles):,} SMILES...")
    emb_a = _build_embedding_dict(
        all_smiles, _cache_path(args.ckpt_a, args.model_a),
        args.ckpt_a, args.model_a, args.device, args.embed_batch_size, args.featurize_n_jobs,
    )
    print(f"  [B] {args.model_b}: embedding {len(all_smiles):,} SMILES...")
    emb_b = _build_embedding_dict(
        all_smiles, _cache_path(args.ckpt_b, args.model_b),
        args.ckpt_b, args.model_b, args.device, args.embed_batch_size, args.featurize_n_jobs,
    )

    x_train, y_train = _build_features(train, emb_a, emb_b, smiles_col)
    x_val,   y_val   = _build_features(val,   emb_a, emb_b, smiles_col)
    x_test,  y_test  = _build_features(test,  emb_a, emb_b, smiles_col)

    if x_train.shape[0] == 0:
        sys.exit("ERROR: no valid training samples after embedding intersection.")

    z_dim = x_train.shape[1]
    print(f"  concat feature dim: {z_dim}-d  ({z_dim//2} + {z_dim//2})")

    metrics = fit_and_eval(
        x_train, y_train, x_val, y_val, x_test, y_test, task_type, args.head, args.seed,
    )
    elapsed = time.time() - t0
    print(
        f"[ensemble:{args.benchmark}/{args.task} seed={args.seed}] "
        f"head={args.head} task_type={task_type} ({elapsed:.1f}s)"
    )
    for key, value in metrics.items():
        print(f"    {key:18s} = {value:.4f}")

    tag_a = args.ckpt_tag_a or Path(args.ckpt_a).stem
    tag_b = args.ckpt_tag_b or Path(args.ckpt_b).stem
    row = {
        "benchmark": args.benchmark,
        "task": args.task,
        "seed": args.seed,
        "head": args.head,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "ckpt_tag_a": tag_a,
        "ckpt_tag_b": tag_b,
        "ckpt_path_a": args.ckpt_a,
        "ckpt_path_b": args.ckpt_b,
        "task_type": task_type,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "z_dim": int(z_dim),
        "elapsed_sec": round(elapsed, 1),
        **metrics,
    }
    append_results_row(row)
    print(f"  -> appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
