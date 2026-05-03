#!/usr/bin/env python
"""Evaluate KPGT embeddings on TDC ADMET (22 tasks) or Polaris ADME-Fang (6 tasks).

Mirrors ``scripts/mole/mole_eval.py`` exactly — same TDC/Polaris loaders,
same sklearn linear/MLP heads, same ``results/<encoder>_results.csv`` row layout
— so the dashboard notebook can aggregate KPGT alongside MiniMol / MolE without
changes.

Embedding extraction is delegated to ``downstream.model.kpgt.KPGTEncoder``,
which subprocesses to a sibling ``kpgt`` conda env (DGL 2.4 + descriptastorus,
incompatible with graphium's torch 2.7 + cu128 stack). Per-SMILES embeddings
get cached at ``datacache/kpgt_embeddings/{benchmark}_{task}.pt`` so subsequent
seeds don't re-embed.

Usage:
    python scripts/kpgt/kpgt_eval.py --benchmark tdc --task caco2_wang --seed 0
    python scripts/kpgt/kpgt_eval.py --benchmark polaris --task adme_fang_hclint --seed 0
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT))

from downstream.model import load_encoder  # noqa: E402

RESULTS_CSV = ROOT / "results" / "kpgt_results.csv"
EMB_CACHE_DIR = ROOT / "datacache" / "kpgt_embeddings"
POLARIS_CACHE_DIR = ROOT / "datacache" / "polaris_adme_fang"
TDC_CACHE_DIR = ROOT / "expts" / "data" / "admet"

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


def load_tdc_task(task: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from tdc.benchmark_group import admet_group
    TDC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(TDC_CACHE_DIR))
    benchmark = group.get(task)
    train, val = group.get_train_valid_split(benchmark=task, split_type="default", seed=seed)
    return train, val, benchmark["test"]


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
    valid = raw_df[target_col].notna()
    train_idx = [i for i in train_idx if valid.iloc[i]]
    test_idx = [i for i in test_idx if valid.iloc[i]]
    train_idx, val_idx = train_test_split(train_idx, test_size=val_fraction, random_state=seed, shuffle=True)

    def subset(idx: List[int]) -> pd.DataFrame:
        d = raw_df.iloc[idx][["MOL_smiles", target_col]].reset_index(drop=True)
        return d.rename(columns={"MOL_smiles": "Drug", target_col: "Y"})

    return subset(train_idx), subset(val_idx), subset(test_idx)


def fit_and_eval(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    task_type: str, head: str, seed: int,
) -> Dict[str, float]:
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    if task_type == "regression":
        model = (
            Ridge(alpha=1.0, random_state=seed) if head == "linear"
            else MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=500,
                              random_state=seed, early_stopping=True)
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
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
            "mean_target": float(np.mean(y_test)),
            "std_target": float(np.std(y_test)),
        }

    model = (
        LogisticRegression(max_iter=1000, random_state=seed) if head == "linear"
        else MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=500,
                           random_state=seed, early_stopping=True)
    )
    model.fit(x_fit, y_fit)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "auprc": float(average_precision_score(y_test, y_prob)),
        "accuracy": float((model.predict(x_test) == y_test).mean()),
        "mean_pred": float(np.mean(y_prob)),
        "mean_target": float(np.mean(y_test)),
    }


def append_results_row(row: Dict[str, object]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[dict] = []
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
    # CPU default: the kpgt env's pip ``dgl`` lacks CUDA, and torch 2.4+cu121
    # doesn't support Blackwell sm_120 either. CPU inference for LiGhT-base
    # is ~30s/1k mols; cache hides the cost on subsequent seeds.
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["tdc", "polaris"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--device", default=default_device(),
                        help="Device passed to the kpgt-env subprocess (cuda:0 or cpu).")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--ckpt-path", default=None,
                        help="Override the default downloads/kpgt/base.pth path.")
    args = parser.parse_args()

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

    print(f"[{args.benchmark}:{args.task} seed={args.seed}] train={len(train)} val={len(val)} test={len(test)}")
    cache_path = EMB_CACHE_DIR / f"{args.benchmark}_{args.task}.pt"
    smiles_col = "Drug"

    encoder_kwargs: dict = dict(
        device=args.device, batch_size=args.embed_batch_size,
    )
    if args.ckpt_path:
        encoder_kwargs["ckpt_path"] = args.ckpt_path
    encoder = load_encoder("kpgt", **encoder_kwargs)

    all_smiles = (
        pd.concat([train[smiles_col], val[smiles_col], test[smiles_col]])
        .unique().tolist()
    )
    feats_all, mask_all = encoder.extract_cached(all_smiles, cache_path)
    if not mask_all.all():
        # Track which SMILES the encoder dropped so the per-split lookups
        # below can skip them.
        bad_smiles = {s for s, ok in zip(all_smiles, mask_all) if not ok}
        print(f"  [kpgt] WARN: {len(bad_smiles)} SMILES un-embeddable; rows dropped.")
    else:
        bad_smiles = set()

    def split_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        keep = df[~df[smiles_col].isin(bad_smiles)].reset_index(drop=True)
        x_split, mask_split = encoder.extract_cached(keep[smiles_col].tolist(), cache_path)
        if not mask_split.all():
            keep = keep.iloc[mask_split].reset_index(drop=True)
        y_split = keep["Y"].to_numpy(dtype=float)
        return x_split, y_split

    x_train, y_train = split_features(train)
    x_val,   y_val   = split_features(val)
    x_test,  y_test  = split_features(test)

    metrics = fit_and_eval(x_train, y_train, x_val, y_val, x_test, y_test,
                           task_type, args.head, args.seed)
    elapsed = time.time() - t0
    print(f"[{args.benchmark}:{args.task} seed={args.seed}] task_type={task_type} "
          f"head={args.head} ({elapsed:.1f}s)")
    for key, value in metrics.items():
        print(f"    {key:12s} = {value:.4f}")

    row = {
        "benchmark": args.benchmark,
        "task": args.task,
        "seed": args.seed,
        "head": args.head,
        "model": "kpgt_base",
        "embedding_model": "kpgt_base",
        "device": args.device,
        "task_type": task_type,
        "n_train": int(len(y_train)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_test)),
        "feature_dim": int(encoder.out_dim),
        "elapsed_sec": round(elapsed, 3),
        **metrics,
    }
    append_results_row(row)
    print(f"Wrote results row to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
