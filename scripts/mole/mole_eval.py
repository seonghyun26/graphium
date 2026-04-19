#!/usr/bin/env python
"""Evaluate MolE embeddings as a feature extractor on TDC ADMET or Polaris ADME-Fang.

Usage:
    python scripts/mole/mole_eval.py --benchmark tdc --task caco2_wang --seed 0
    python scripts/mole/mole_eval.py --benchmark polaris --task adme_fang_hclint --seed 0

Writes results to results/mole_results.csv and caches per-SMILES embeddings at
datacache/mole_embeddings/{benchmark}_{task}.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import numpy as np
import pandas as pd
import torch
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
RESULTS_CSV = ROOT / "results" / "mole_results.csv"
EMB_CACHE_DIR = ROOT / "datacache" / "mole_embeddings"
POLARIS_CACHE_DIR = ROOT / "datacache" / "polaris_adme_fang"
TDC_CACHE_DIR = ROOT / "expts" / "data" / "admet"
MOLE_ASSET_DIR = ROOT / "downloads" / "mole"
UPSTREAM_PARENT = MOLE_ASSET_DIR / "upstream"
UPSTREAM_DIR = UPSTREAM_PARENT / "MolE"
UPSTREAM_REPO = "https://github.com/rolayoalarcon/MolE.git"
UPSTREAM_COMMIT = "f01b5321c41a230cf1046371ffd6f9f9631c51ea"
MOLE_MODEL_NAME = "gin_concat_R1000_E8000_lambda0.0001"

TDC_REGRESSION = {
    "caco2_wang",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "half_life_obach",
    "ld50_zhu",
    "lipophilicity_astrazeneca",
    "ppbr_az",
    "solubility_aqsoldb",
    "vdss_lombardo",
}
TDC_CLASSIFICATION = {
    "ames",
    "bbb_martins",
    "bioavailability_ma",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "dili",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
}

POLARIS_BENCHMARKS = {
    "adme_fang_hclint": ("adme-fang-hclint-reg-v1", "LOG_HLM_CLint"),
    "adme_fang_rclint": ("adme-fang-rclint-reg-v1", "LOG_RLM_CLint"),
    "adme_fang_perm": ("adme-fang-perm-reg-v1", "LOG_MDR1-MDCK_ER"),
    "adme_fang_hppb": ("adme-fang-hppb-reg-v1", "LOG_HPPB"),
    "adme_fang_rppb": ("adme-fang-rppb-reg-v1", "LOG_RPPB"),
    "adme_fang_solu": ("adme-fang-solu-reg-v1", "LOG_SOLUBILITY"),
}
POLARIS_DATASET_URL = "https://data.polarishub.io/dataset/biogen/adme-fang-v1/table.parquet"
POLARIS_BENCHMARK_API = "https://polarishub.io/api/v1/benchmark/biogen/{slug}"


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def ensure_mole_assets() -> None:
    download_script = ROOT / "scripts" / "mole" / "download_mole_baseline.sh"
    _run(["bash", str(download_script)])


def ensure_upstream_checkout() -> Path:
    UPSTREAM_PARENT.mkdir(parents=True, exist_ok=True)
    if not UPSTREAM_DIR.exists():
        _run(["git", "clone", "--depth", "1", UPSTREAM_REPO, str(UPSTREAM_DIR)])
    current_commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if current_commit != UPSTREAM_COMMIT:
        _run(["git", "-C", str(UPSTREAM_DIR), "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT])
        _run(["git", "-C", str(UPSTREAM_DIR), "checkout", UPSTREAM_COMMIT])
    return UPSTREAM_DIR


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.resolve() == src.resolve():
            return
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        dest.write_bytes(src.read_bytes())


def ensure_upstream_checkpoint_layout() -> Path:
    ensure_mole_assets()
    repo_dir = ensure_upstream_checkout()
    ckpt_dir = repo_dir / "ckpt" / MOLE_MODEL_NAME / "checkpoints"
    _link_or_copy(MOLE_ASSET_DIR / "config.yaml", ckpt_dir / "config.yaml")
    _link_or_copy(MOLE_ASSET_DIR / "model.pth", ckpt_dir / "model.pth")
    return repo_dir


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
        data = raw_df.iloc[idx][["MOL_smiles", target_col]].reset_index(drop=True)
        return data.rename(columns={"MOL_smiles": "Drug", target_col: "Y"})

    return subset(train_idx), subset(val_idx), subset(test_idx)


def embed_smiles(smiles: List[str], cache_path: Path, device: str, batch_size: int) -> np.ndarray:
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, np.ndarray] = torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    missing = [smile for smile in smiles if smile not in cache]
    if missing:
        print(f"  [mole] embedding {len(missing)} / {len(smiles)} new SMILES...", flush=True)
        repo_dir = ensure_upstream_checkpoint_layout()
        sys.path.insert(0, str(repo_dir))
        from dataset.dataset_representation import batch_representation, load_pretrained_model

        with contextlib.redirect_stdout(io.StringIO()):
            model = load_pretrained_model(
                pretrain_architecture="gin_concat",
                pretrained_model=MOLE_MODEL_NAME,
                pretrained_dir=str(repo_dir / "ckpt"),
                device=device,
            )
        unique_smiles = list(dict.fromkeys(missing))
        smile_df = pd.DataFrame({"chem_id": unique_smiles, "smiles": unique_smiles})
        embeddings = batch_representation(smile_df, model, batch_size=batch_size, device=device)
        for chem_id, row in embeddings.iterrows():
            cache[str(chem_id)] = row.to_numpy(dtype=np.float32)
        torch.save(cache, cache_path)
    return np.stack([cache[smile] for smile in smiles], axis=0)


def fit_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    head: str,
    seed: int,
) -> Dict[str, float]:
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    if task_type == "regression":
        model = Ridge(alpha=1.0, random_state=seed) if head == "linear" else MLPRegressor(
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
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
            "mean_target": float(np.mean(y_test)),
            "std_target": float(np.std(y_test)),
        }

    model = LogisticRegression(max_iter=1000, random_state=seed) if head == "linear" else MLPClassifier(
        hidden_layer_sizes=(256, 256),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
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
    parser.add_argument("--benchmark", choices=["tdc", "polaris"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--embed-batch-size", type=int, default=2048)
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
    all_smiles = pd.concat([train[smiles_col], val[smiles_col], test[smiles_col]]).unique().tolist()
    _ = embed_smiles(all_smiles, cache_path, args.device, args.embed_batch_size)

    x_train = embed_smiles(train[smiles_col].tolist(), cache_path, args.device, args.embed_batch_size)
    x_val = embed_smiles(val[smiles_col].tolist(), cache_path, args.device, args.embed_batch_size)
    x_test = embed_smiles(test[smiles_col].tolist(), cache_path, args.device, args.embed_batch_size)
    y_train = train["Y"].to_numpy(dtype=float)
    y_val = val["Y"].to_numpy(dtype=float)
    y_test = test["Y"].to_numpy(dtype=float)

    metrics = fit_and_eval(x_train, y_train, x_val, y_val, x_test, y_test, task_type, args.head, args.seed)
    elapsed = time.time() - t0
    print(f"[{args.benchmark}:{args.task} seed={args.seed}] task_type={task_type} head={args.head} ({elapsed:.1f}s)")
    for key, value in metrics.items():
        print(f"    {key:12s} = {value:.4f}")

    row = {
        "benchmark": args.benchmark,
        "task": args.task,
        "seed": args.seed,
        "head": args.head,
        "model": "mole_gin_concat",
        "embedding_model": MOLE_MODEL_NAME,
        "upstream_commit": UPSTREAM_COMMIT,
        "device": args.device,
        "task_type": task_type,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "elapsed_sec": round(elapsed, 3),
        **metrics,
    }
    append_results_row(row)
    print(f"Wrote results row to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
