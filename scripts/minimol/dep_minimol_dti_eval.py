#!/usr/bin/env python
"""Evaluate Minimol embeddings as a feature extractor on TDC DTI subsets.

Mirrors scripts/minimol/minimol_eval.py (ADMET/Polaris) but:
  - reads from per-subset CSV + splits.pt produced by scripts/data/dti_eval/01_prepare_dti_eval.py
  - concatenates frozen ESM-C 600M protein embeddings (1152-d) to the 512-d minimol
    molecule embedding, yielding a 1664-d feature vector per (drug, target) row
  - fits a sklearn Ridge or MLP head and reports regression metrics on the held-out
    test split defined by TDC's `random` or `cold_target` splits

Usage:
    python scripts/minimol/minimol_dti_eval.py --subset DAVIS --method random --seed 0
    python scripts/minimol/minimol_dti_eval.py --subset KIBA  --method cold_target --seed 2 --head linear

Writes one row per (subset, method, seed, head) to results/minimol_dti_results.csv.
Per-SMILES embeddings cached in datacache/minimol_dti_embeddings/<subset>.pt (shared across
method/seed since only the split differs).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from minimol import Minimol  # imported early: TDC's admet_group chdirs into datacache
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = ROOT / "results" / "minimol_dti_results.csv"
EMB_CACHE_DIR = ROOT / "datacache" / "minimol_dti_embeddings"
DTI_EVAL_DIR = ROOT / "data" / "dti-eval"

SUBSET_LABEL_COL = {
    "DAVIS": "pY",
    "BindingDB_Kd": "pY",
    "BindingDB_Ki": "pY",
    "BindingDB_IC50": "pY",
    "BindingDB_Patent_DG": "pY",   # DTI-DG temporal-split leaderboard
    "KIBA": "kiba_score",
}
ESMC_DIM = 1152
PROT_EMB_COLS = [f"prot_emb_{i}" for i in range(ESMC_DIM)]


def load_dti_split(
    subset: str, method: str, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load prepped DTI CSV + split indices, slice into train/val/test frames.

    Returns (train_df, val_df, test_df, label_col). Each df has columns
    [SMILES_nometa, <label_col>, prot_emb_0..prot_emb_1151].
    """
    parquet_path = DTI_EVAL_DIR / f"{subset}.parquet"
    splits_path = DTI_EVAL_DIR / "splits" / f"{subset}_{method}_seed{seed}.pt"
    if not parquet_path.exists():
        sys.exit(f"ERROR: {parquet_path} missing. Run scripts/data/dti_eval/01_prepare_dti_eval.py.")
    if not splits_path.exists():
        sys.exit(f"ERROR: {splits_path} missing. Re-run data prep with --methods {method}.")

    label_col = SUBSET_LABEL_COL[subset]
    df = pd.read_parquet(parquet_path, columns=["SMILES_nometa", label_col, *PROT_EMB_COLS])
    splits = torch.load(splits_path, weights_only=False)
    return (
        df.iloc[splits["train"]].reset_index(drop=True),
        df.iloc[splits["val"]].reset_index(drop=True),
        df.iloc[splits["test"]].reset_index(drop=True),
        label_col,
    )


def _safe_minimol_embed(model, smiles_batch: List[str], chunk_size: int = 128) -> Dict[str, torch.Tensor]:
    """Embed SMILES via Minimol, bisecting chunks that crash.

    Minimol's featurizer can fail on certain SMILES (kekulize errors, weird
    valences, organometallic bonds); a single bad SMILES poisons the whole
    `model(batch)` call with `'str' has no 'stores'`. We bisect failing chunks
    down to single SMILES and drop those that can't be embedded.
    """
    results: Dict[str, torch.Tensor] = {}
    stack: List[List[str]] = []
    for i in range(0, len(smiles_batch), chunk_size):
        stack.append(smiles_batch[i:i + chunk_size])
    pbar = tqdm(total=len(smiles_batch), desc="minimol embedding", unit="mol", smoothing=0.05)
    while stack:
        chunk = stack.pop()
        if not chunk:
            continue
        try:
            embs = model(chunk)
            for s, e in zip(chunk, embs):
                results[s] = e.detach().cpu().float()
            pbar.update(len(chunk))
        except Exception:
            if len(chunk) == 1:
                pbar.update(1)
                continue
            mid = len(chunk) // 2
            stack.append(chunk[mid:])
            stack.append(chunk[:mid])
    pbar.close()
    return results


def embed_smiles(smiles: List[str], cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Minimol embeddings with a per-SMILES cache.

    Returns (embeddings, mask) where mask is True for SMILES the encoder could
    handle. Caller must drop masked-out rows from y / prot_emb.
    """
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, torch.Tensor] = (
        torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    )
    unique_smiles = list(dict.fromkeys(smiles))
    missing = [s for s in unique_smiles if s not in cache]
    if missing:
        print(f"  [minimol] embedding {len(missing):,} / {len(unique_smiles):,} new SMILES...", flush=True)
        model = Minimol()
        new_embs = _safe_minimol_embed(model, missing)
        cache.update(new_embs)
        torch.save(cache, cache_path)
    mask = np.array([s in cache for s in smiles], dtype=bool)
    if mask.any():
        out = np.stack([cache[s].numpy() for s in smiles if s in cache], axis=0)
    else:
        out = np.zeros((0, 0), dtype=np.float32)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  [minimol] WARN: {dropped:,} SMILES couldn't be embedded; dropping those rows.")
    return out, mask


def fit_and_eval(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    head: str, seed: int,
) -> Dict[str, float]:
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    if head == "linear":
        model = Ridge(alpha=1.0, random_state=seed)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(256, 256),
            max_iter=500,
            random_state=seed,
            early_stopping=True,
        )
    model.fit(x_fit, y_fit)
    y_pred = model.predict(x_test)
    pear = pearsonr(y_test, y_pred).statistic if len(y_test) > 1 else float("nan")
    spear = spearmanr(y_test, y_pred).statistic if len(y_test) > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "r2_score": float(r2_score(y_test, y_pred)),
        "pearsonr": float(pear) if pear is not None else float("nan"),
        "spearmanr": float(spear) if spear is not None else float("nan"),
        "concordance_index": _concordance_index(y_test, y_pred),
        "mean_pred": float(np.mean(y_pred)),
        "std_pred": float(np.std(y_pred)),
        "mean_target": float(np.mean(y_test)),
        "std_target": float(np.std(y_test)),
    }


def _concordance_index(y: np.ndarray, p: np.ndarray) -> float:
    """Pairwise CI — the DTI-literature metric (DeepDTA, GraphDTA, MolTrans).

    CI = P(p_i > p_j | y_i > y_j)  (ties counted as 0.5). O(n^2) but DAVIS/KIBA
    test splits are small (<= ~25K), so this is fine on CPU.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) < 2:
        return float("nan")
    dy = y[:, None] - y[None, :]
    dp = p[:, None] - p[None, :]
    mask = dy > 0
    n_pairs = mask.sum()
    if n_pairs == 0:
        return float("nan")
    concord = (dp[mask] > 0).sum() + 0.5 * (dp[mask] == 0).sum()
    return float(concord / n_pairs)


def append_results_row(row: Dict[str, object]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    fieldnames = list(row.keys())
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(dict.fromkeys(list(reader.fieldnames or []) + fieldnames))
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=list(SUBSET_LABEL_COL.keys()))
    parser.add_argument("--method", required=True, choices=["random", "cold_target", "temporal"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    args = parser.parse_args()

    t0 = time.time()
    train, val, test, label_col = load_dti_split(args.subset, args.method, args.seed)
    print(
        f"[minimol_dti:{args.subset}/{args.method}/seed={args.seed}] "
        f"train={len(train):,} val={len(val):,} test={len(test):,}  label={label_col}"
    )

    # Embed SMILES (cache per subset; seed/method only affect the split indices).
    cache_path = EMB_CACHE_DIR / f"{args.subset}.pt"
    all_smiles = (
        pd.concat([train["SMILES_nometa"], val["SMILES_nometa"], test["SMILES_nometa"]])
        .unique().tolist()
    )
    embed_smiles(all_smiles, cache_path)   # pre-warm cache

    def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        z_mol, mask = embed_smiles(df["SMILES_nometa"].tolist(), cache_path)
        df_ok = df[mask].reset_index(drop=True)
        z_prot = df_ok[PROT_EMB_COLS].to_numpy(dtype=np.float32)
        x = np.concatenate([z_mol, z_prot], axis=1)
        y = df_ok[label_col].to_numpy(dtype=np.float32)
        return x, y

    x_train, y_train = build_features(train)
    x_val, y_val = build_features(val)
    x_test, y_test = build_features(test)
    print(f"  feature shape: {x_train.shape[1]}-d  (minimol 512 + prot_emb {ESMC_DIM})")

    metrics = fit_and_eval(x_train, y_train, x_val, y_val, x_test, y_test, args.head, args.seed)
    elapsed = time.time() - t0
    print(
        f"[minimol_dti:{args.subset}/{args.method}/seed={args.seed}] "
        f"head={args.head} ({elapsed:.1f}s)"
    )
    for key, value in metrics.items():
        print(f"    {key:18s} = {value:.4f}")

    row = {
        "model": "minimol_v1",
        "subset": args.subset,
        "method": args.method,
        "seed": args.seed,
        "head": args.head,
        "label_col": label_col,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "feature_dim": int(x_train.shape[1]),
        "elapsed_sec": round(elapsed, 1),
        **metrics,
    }
    append_results_row(row)
    print(f"  -> appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
