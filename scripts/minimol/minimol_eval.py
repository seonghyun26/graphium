#!/usr/bin/env python
"""Evaluate Minimol embeddings as a feature extractor on TDC ADMET or Polaris ADME-Fang.

Usage:
    python minimol_eval.py --benchmark tdc --task caco2_wang --seed 0
    python minimol_eval.py --benchmark polaris --task adme_fang_hclint --seed 0

Writes results to results/minimol_results.csv (one row per (benchmark, task, seed, head)).
Caches per-SMILES embeddings at datacache/minimol_embeddings/{benchmark}_{task}.pt so
subsequent seeds reuse them.

CPU usage: sklearn MLP / LogReg and CPU-side torch default to one BLAS thread
per physical core, which on high-core hosts pushes load averages > 300 and
starves any concurrent GPU run of its dataloader cores. Cap with
``--cpu-threads N`` (or ``MINIMOL_CPU_THREADS=N``); default is 8.
"""

# ---- CPU thread caps (must come before numpy / sklearn / torch / minimol import) ----
import os
import sys as _sys


def _cap_cpu_threads() -> int:
    # CLI flag wins over env. Peek sys.argv early since argparse hasn't run.
    n = int(os.environ.get("MINIMOL_CPU_THREADS", "8"))
    argv = _sys.argv
    for i, tok in enumerate(argv):
        if tok == "--cpu-threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
            break
        if tok.startswith("--cpu-threads="):
            n = int(tok.split("=", 1)[1])
            break
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))
    return n


_CPU_THREADS = _cap_cpu_threads()

# ---- standard imports ----
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import numpy as np
import pandas as pd
import torch

# Belt-and-suspenders: cap torch's internal pools too (env vars don't always
# stick for libs that cache thread counts at import time).
try:
    torch.set_num_threads(_CPU_THREADS)
    torch.set_num_interop_threads(max(1, _CPU_THREADS // 2))
except RuntimeError:
    pass  # torch refuses to set interop threads twice; ignore on reload

from minimol import Minimol  # import early: pyTDC's admet_group chdir's and breaks editable graphium imports
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = ROOT / "results" / "minimol_results.csv"
EMB_CACHE_DIR = ROOT / "datacache" / "minimol_embeddings"
POLARIS_CACHE_DIR = ROOT / "datacache" / "polaris_adme_fang"
TDC_CACHE_DIR = ROOT / "expts" / "data" / "admet"

# ── Task type lookups ───────────────────────────────────────────────────────
TDC_REGRESSION = {
    "caco2_wang", "clearance_hepatocyte_az", "clearance_microsome_az",
    "half_life_obach", "ld50_zhu", "lipophilicity_astrazeneca", "ppbr_az",
    "solubility_aqsoldb", "vdss_lombardo",
}
TDC_CLASSIFICATION = {
    "ames", "bbb_martins", "bioavailability_ma",
    "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels", "cyp3a4_veith",
    "dili", "herg", "hia_hou", "pgp_broccatelli",
}

POLARIS_BENCHMARKS = {
    "adme_fang_hclint": ("adme-fang-hclint-reg-v1", "LOG_HLM_CLint"),
    "adme_fang_rclint": ("adme-fang-rclint-reg-v1", "LOG_RLM_CLint"),
    "adme_fang_perm":   ("adme-fang-perm-reg-v1",   "LOG_MDR1-MDCK_ER"),
    "adme_fang_hppb":   ("adme-fang-hppb-reg-v1",   "LOG_HPPB"),
    "adme_fang_rppb":   ("adme-fang-rppb-reg-v1",   "LOG_RPPB"),
    "adme_fang_solu":   ("adme-fang-solu-reg-v1",   "LOG_SOLUBILITY"),
}
POLARIS_DATASET_URL = "https://data.polarishub.io/dataset/biogen/adme-fang-v1/table.parquet"
POLARIS_BENCHMARK_API = "https://polarishub.io/api/v1/benchmark/biogen/{slug}"


# ── Data loading ────────────────────────────────────────────────────────────
def load_tdc_task(task: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TDC ADMET train/val/test splits. Uses the same split as graphium's datamodule."""
    from tdc.benchmark_group import admet_group
    TDC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(TDC_CACHE_DIR))
    benchmark = group.get(task)
    train_val, test = benchmark["train_val"], benchmark["test"]
    # TDC's official per-seed train/val split
    train, val = group.get_train_valid_split(benchmark=task, split_type="default", seed=seed)
    return train, val, test


def _load_polaris_parquet() -> pd.DataFrame:
    POLARIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = POLARIS_CACHE_DIR / "adme-fang-v1.parquet"
    if not parquet_path.exists():
        resp = httpx.get(POLARIS_DATASET_URL, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        parquet_path.write_bytes(resp.content)
    return pd.read_parquet(parquet_path)


def _load_polaris_benchmark_meta(slug: str) -> dict:
    meta_path = POLARIS_CACHE_DIR / f"{slug}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    resp = httpx.get(POLARIS_BENCHMARK_API.format(slug=slug), timeout=60.0)
    resp.raise_for_status()
    meta = resp.json()
    meta_path.write_text(json.dumps(meta))
    return meta


def load_polaris_task(task: str, seed: int, val_fraction: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slug, target_col = POLARIS_BENCHMARKS[task]
    raw_df = _load_polaris_parquet()
    meta = _load_polaris_benchmark_meta(slug)
    train_idx, test_idx = meta["split"]
    # drop NaN targets
    valid = raw_df[target_col].notna()
    train_idx = [i for i in train_idx if valid.iloc[i]]
    test_idx = [i for i in test_idx if valid.iloc[i]]
    # seed-deterministic val split
    train_idx, val_idx = train_test_split(train_idx, test_size=val_fraction, random_state=seed, shuffle=True)

    def subset(idx):
        d = raw_df.iloc[idx][["MOL_smiles", target_col]].reset_index(drop=True)
        d = d.rename(columns={"MOL_smiles": "Drug", target_col: "Y"})
        return d

    return subset(train_idx), subset(val_idx), subset(test_idx)


# ── Embedding computation with caching ──────────────────────────────────────
def embed_smiles(smiles: List[str], cache_path: Path) -> np.ndarray:
    """Compute minimol embeddings with a per-SMILES cache."""
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, torch.Tensor] = torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    missing = [s for s in smiles if s not in cache]
    if missing:
        print(f"  [minimol] embedding {len(missing)} / {len(smiles)} new SMILES...", flush=True)
        model = Minimol()
        # Minimol returns list of 512-dim tensors
        new_embs = model(missing)
        for s, e in zip(missing, new_embs):
            cache[s] = e.detach().cpu().float()
        torch.save(cache, cache_path)
    return np.stack([cache[s].numpy() for s in smiles], axis=0)


# ── Training + evaluation ────────────────────────────────────────────────────
def fit_and_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    task_type: str, head: str, seed: int,
) -> Dict[str, float]:
    """Fit sklearn head on train+val combined (standard practice for benchmarks) and eval on test."""
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    if task_type == "regression":
        if head == "linear":
            model = Ridge(alpha=1.0, random_state=seed)
        else:  # mlp
            model = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=500, random_state=seed, early_stopping=True)
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_test)
        pear = pearsonr(y_test, y_pred).statistic if len(y_test) > 1 else float("nan")
        spear = spearmanr(y_test, y_pred).statistic if len(y_test) > 1 else float("nan")
        return {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred)),
            "pearsonr": float(pear) if pear is not None else float("nan"),
            "spearmanr": float(spear) if spear is not None else float("nan"),
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
            "mean_target": float(np.mean(y_test)),
            "std_target": float(np.std(y_test)),
        }
    else:  # classification
        if head == "linear":
            model = LogisticRegression(max_iter=1000, random_state=seed)
        else:  # mlp
            model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=500, random_state=seed, early_stopping=True)
        model.fit(X_fit, y_fit)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            "auroc": float(roc_auc_score(y_test, y_prob)),
            "auprc": float(average_precision_score(y_test, y_prob)),
            "accuracy": float((model.predict(X_test) == y_test).mean()),
            "mean_pred": float(np.mean(y_prob)),
            "mean_target": float(np.mean(y_test)),
        }


# ── Results CSV ──────────────────────────────────────────────────────────────
def append_results_row(row: Dict[str, object]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_CSV.exists()
    existing_rows = []
    fieldnames = list(row.keys())
    if file_exists:
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(dict.fromkeys(list(reader.fieldnames or []) + fieldnames))
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(row)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["tdc", "polaris"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument(
        "--cpu-threads", type=int, default=_CPU_THREADS,
        help=f"BLAS/OpenMP thread cap (already applied; default {_CPU_THREADS}, "
             "override with --cpu-threads N or MINIMOL_CPU_THREADS=N).",
    )
    args = parser.parse_args()

    t0 = time.time()

    # 1. Load splits
    if args.benchmark == "tdc":
        if args.task in TDC_REGRESSION:
            task_type = "regression"
        elif args.task in TDC_CLASSIFICATION:
            task_type = "classification"
        else:
            sys.exit(f"Unknown TDC task: {args.task}")
        train, val, test = load_tdc_task(args.task, args.seed)
    else:  # polaris
        if args.task not in POLARIS_BENCHMARKS:
            sys.exit(f"Unknown Polaris task: {args.task}")
        task_type = "regression"
        train, val, test = load_polaris_task(args.task, args.seed)

    smiles_col = "Drug"
    print(f"[{args.benchmark}:{args.task} seed={args.seed}] train={len(train)} val={len(val)} test={len(test)}")

    # 2. Embed (seed-independent cache)
    cache_path = EMB_CACHE_DIR / f"{args.benchmark}_{args.task}.pt"
    all_smiles = pd.concat([train[smiles_col], val[smiles_col], test[smiles_col]]).unique().tolist()
    _ = embed_smiles(all_smiles, cache_path)

    X_train = embed_smiles(train[smiles_col].tolist(), cache_path)
    X_val = embed_smiles(val[smiles_col].tolist(), cache_path)
    X_test = embed_smiles(test[smiles_col].tolist(), cache_path)
    y_train = train["Y"].to_numpy(dtype=float)
    y_val = val["Y"].to_numpy(dtype=float)
    y_test = test["Y"].to_numpy(dtype=float)

    # 3. Fit + eval
    metrics = fit_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, task_type, args.head, args.seed)

    # 4. Log + record
    elapsed = time.time() - t0
    print(f"[{args.benchmark}:{args.task} seed={args.seed}] task_type={task_type} head={args.head} ({elapsed:.1f}s)")
    for k, v in metrics.items():
        print(f"    {k:12s} = {v:.4f}")

    row = {
        "benchmark": args.benchmark,
        "task": args.task,
        "seed": args.seed,
        "head": args.head,
        "task_type": task_type,
        "model": "minimol_v1",
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "elapsed_sec": round(elapsed, 1),
        **metrics,
    }
    append_results_row(row)
    print(f"  → appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
