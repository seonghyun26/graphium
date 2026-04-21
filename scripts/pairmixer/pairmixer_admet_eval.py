#!/usr/bin/env python
"""Evaluate PairMixer embeddings as a frozen feature extractor on TDC ADMET or Polaris ADME-Fang.

Mirrors scripts/{minimol,mole}/{minimol,mole}_eval.py — same data layout, same splits,
same metrics, same sklearn head — but extracts the molecule embedding from a graphium
PairMixer checkpoint instead of an external molecular FM. Use this for apples-to-apples
comparison vs MiniMol / MolE under one consistent eval protocol.

Pipeline (embedding half copied from scripts/pairmixer/pairmixer_dti_eval.py):
  1. Hydra-compose a tasks=dti_eval config so we get a MultitaskFromSmilesDataModule
     whose featurization matches what the PairMixer checkpoint was trained with.
  2. Load PairMixer checkpoint via PredictorModule.load_pretrained_model.
  3. Forward each unique SMILES through pre_nn -> gnn -> graph_output_nn['graph']
     to get the graph-level embedding z_mol (256-d for pairmixer_12M).
  4. Fit sklearn Ridge/LogReg or MLPRegressor/MLPClassifier on train+val combined,
     evaluate on test.
  5. Append metrics to results/pairmixer_admet_results.csv.

Usage:
    python scripts/pairmixer/pairmixer_admet_eval.py \\
        --ckpt models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/.../last.ckpt \\
        --benchmark polaris --task adme_fang_hclint --seed 0
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import joblib
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from joblib import Parallel, delayed
from omegaconf import OmegaConf
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphium.config._loader import load_accelerator, load_datamodule
from graphium.data.collate import graphium_collate_fn
from graphium.trainer.predictor import PredictorModule


ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = ROOT / "results" / "pairmixer_admet_results.csv"
EMB_CACHE_DIR = ROOT / "datacache" / "pairmixer_admet_embeddings"
POLARIS_CACHE_DIR = ROOT / "datacache" / "polaris_adme_fang"
TDC_CACHE_DIR = ROOT / "expts" / "data" / "admet"
HYDRA_CFG_DIR = ROOT / "expts" / "hydra-configs"

# ── Benchmark task metadata (lifted from mole_eval.py for shape-compat) ─────
TDC_REGRESSION = {
    "caco2_wang", "clearance_hepatocyte_az", "clearance_microsome_az",
    "half_life_obach", "ld50_zhu", "lipophilicity_astrazeneca",
    "ppbr_az", "solubility_aqsoldb", "vdss_lombardo",
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


# ── Data loading (mirrors mole_eval.py for shape-compat) ────────────────────
def load_tdc_task(task: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TDC ADMET train/val/test splits (same split as graphium's datamodule).

    NB: `pyTDC`'s admet_group chdir's on import and breaks editable graphium imports,
    so we defer this import until after all graphium modules are loaded.
    """
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


def load_polaris_task(
    task: str, seed: int, val_fraction: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slug, target_col = POLARIS_BENCHMARKS[task]
    raw_df = _load_polaris_parquet()
    meta = _load_polaris_benchmark_meta(slug)
    train_idx, test_idx = meta["split"]
    valid = raw_df[target_col].notna()
    train_idx = [i for i in train_idx if valid.iloc[i]]
    test_idx = [i for i in test_idx if valid.iloc[i]]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_fraction, random_state=seed, shuffle=True,
    )

    def subset(idx: List[int]) -> pd.DataFrame:
        data = raw_df.iloc[idx][["MOL_smiles", target_col]].reset_index(drop=True)
        return data.rename(columns={"MOL_smiles": "Drug", target_col: "Y"})

    return subset(train_idx), subset(val_idx), subset(test_idx)


# ── PairMixer backbone loading + embedding extraction ───────────────────────
# Everything below until ── Metrics ── is lifted verbatim from
# scripts/pairmixer/pairmixer_dti_eval.py so both ADMET and DTI evals share the
# exact same mol-embedding code path.
def _build_datamodule_for_featurization(model_name: str) -> Any:
    """Compose a minimal Hydra config to get the graphium featurizer.

    We only need `datamodule.smiles_transformer` (the featurizer partial) — the
    dataset itself is loaded from our benchmark-specific splits. The featurization
    config is inherited from architecture/toymix.yaml, which matches what the
    PairMixer ESMC ckpt was trained with.
    """
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CFG_DIR)):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model={model_name}",
                "accelerator=gpu",
                "tasks=dti_eval",
                "training=dti_eval",
                "++constants.dti_subset=DAVIS",
                "++constants.dti_split_method=random",
                "++constants.dti_split_seed=0",
                "++constants.seed=0",
            ],
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg, accelerator_type = load_accelerator(cfg)
    datamodule = load_datamodule(cfg, accelerator_type)
    return datamodule


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _featurize_one(smiles: str, smiles_transformer):
    try:
        g = smiles_transformer(smiles, mask_nan=0.0)
    except Exception:
        return None
    if g is None or isinstance(g, str):
        return None
    return smiles, g


def _featurize_smiles(
    smiles_list: List[str], smiles_transformer, n_jobs: int = 8,
) -> Tuple[List[Any], List[str]]:
    n_jobs = max(1, int(n_jobs))
    desc = f"featurizing SMILES (n_jobs={n_jobs})"
    if n_jobs == 1:
        results = [
            _featurize_one(s, smiles_transformer)
            for s in tqdm(smiles_list, desc=desc, unit="mol", smoothing=0.05)
        ]
    else:
        with _tqdm_joblib(tqdm(total=len(smiles_list), desc=desc, unit="mol", smoothing=0.05)):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_featurize_one)(s, smiles_transformer) for s in smiles_list
            )

    kept, graphs = [], []
    failed = 0
    for r in results:
        if r is None:
            failed += 1
            continue
        kept.append(r[0])
        graphs.append(r[1])
    if failed:
        print(f"  [pairmixer] WARN: {failed:,} SMILES failed featurization", flush=True)
    return graphs, kept


def _extract_embeddings(
    backbone, graphs: List[Any], device: torch.device, batch_size: int,
) -> torch.Tensor:
    """Forward graphs through pre_nn + gnn + graph_output_nn['graph']."""
    if "graph" not in backbone.task_heads.graph_output_nn:
        sys.exit(
            "ERROR: backbone.task_heads.graph_output_nn['graph'] missing — "
            "checkpoint has no graph-level head."
        )
    graph_output_nn = backbone.task_heads.graph_output_nn["graph"]
    target_dtype = next(backbone.parameters()).dtype
    collate_fn = partial(graphium_collate_fn, mask_nan=0)
    loader = DataLoader(
        graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    chunks: List[torch.Tensor] = []
    backbone.eval()
    n_batches = (len(graphs) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc="GNN forward", unit="batch", smoothing=0.05):
            batch = batch.to(device)
            keys = batch.keys() if callable(getattr(batch, "keys", None)) else list(batch.keys)
            for k in list(keys):
                v = batch[k]
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    batch[k] = v.to(target_dtype)
            g = backbone.encoder_manager(batch)
            if backbone.pre_nn is not None:
                g["feat"] = backbone.pre_nn.forward(g["feat"])
            if backbone.pre_nn_edges is not None:
                e = g["edge_feat"]
                if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                    e = torch.zeros(
                        list(e.shape[:-1]) + [backbone.pre_nn_edges.out_dim],
                        device=e.device, dtype=e.dtype,
                    )
                else:
                    e = backbone.pre_nn_edges.forward(e)
                g["edge_feat"] = e
            g = backbone.gnn.forward(g)
            if backbone.gnn_layer_pooling is not None:
                g["feat"] = backbone.gnn_layer_pooling(backbone.gnn._readout_cache)
            z = graph_output_nn(g)
            chunks.append(z.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def embed_smiles_cached(
    smiles: List[str], cache_path: Path, ckpt_path: str, model_name: str,
    device: str, batch_size: int, featurize_n_jobs: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed SMILES with a per-checkpoint, per-SMILES cache.

    Returns (embeddings, mask) where `mask[i]` is False for SMILES that failed
    featurization; caller is responsible for dropping corresponding label rows.
    """
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, np.ndarray] = (
        torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    )
    unique = list(dict.fromkeys(smiles))
    missing = [s for s in unique if s not in cache]

    if missing:
        print(
            f"  [pairmixer] embedding {len(missing):,} / {len(unique):,} new SMILES...",
            flush=True,
        )
        datamodule = _build_datamodule_for_featurization(model_name)
        smiles_transformer = datamodule.smiles_transformer
        torch_device = torch.device(device)
        predictor = PredictorModule.load_pretrained_model(
            name_or_path=ckpt_path, device=str(torch_device),
        )
        backbone = predictor.model
        backbone.to(torch_device)

        graphs, kept = _featurize_smiles(missing, smiles_transformer, n_jobs=featurize_n_jobs)
        if graphs:
            z = _extract_embeddings(backbone, graphs, torch_device, batch_size).numpy()
            for s, row in zip(kept, z):
                cache[s] = row.astype(np.float32)
        else:
            print("  [pairmixer] no new SMILES survived featurization in this batch.")
        for s in set(missing) - set(kept):
            cache[s] = None
        torch.save(cache, cache_path)

    mask = np.array(
        [s in cache and cache[s] is not None for s in smiles], dtype=bool,
    )
    if mask.any():
        out = np.stack(
            [cache[s] for s in smiles if s in cache and cache[s] is not None], axis=0,
        )
    else:
        out = np.zeros((0, 0), dtype=np.float32)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  [pairmixer] WARN: {dropped:,} SMILES couldn't be embedded; dropping those rows.")
    return out, mask


# ── Metrics + sklearn head (mirrors mole_eval.py so numbers are comparable) ──
def fit_and_eval(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    task_type: str, head: str, seed: int,
) -> Dict[str, float]:
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    if task_type == "regression":
        model = Ridge(alpha=1.0, random_state=seed) if head == "linear" else MLPRegressor(
            hidden_layer_sizes=(256, 256), max_iter=500, random_state=seed, early_stopping=True,
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
        hidden_layer_sizes=(256, 256), max_iter=500, random_state=seed, early_stopping=True,
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
    parser.add_argument("--ckpt", required=True, help="PairMixer checkpoint (.ckpt) path")
    parser.add_argument(
        "--model", default="pairmixer_12M",
        help="model config name (must have a matching expts/hydra-configs/model/<model>.yaml)",
    )
    parser.add_argument("--benchmark", choices=["tdc", "polaris"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument(
        "--featurize-n-jobs", type=int, default=8,
        help="Parallel workers for graphium featurization (joblib loky). 1 disables parallelism.",
    )
    parser.add_argument(
        "--ckpt-tag", default=None,
        help="Short label for this checkpoint in the results CSV (default: ckpt filename stem)",
    )
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"ERROR: checkpoint not found: {args.ckpt}")

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
        f"[pairmixer_admet:{args.benchmark}/{args.task} seed={args.seed}] "
        f"train={len(train)} val={len(val)} test={len(test)}  task_type={task_type}"
    )

    # Cache key = (benchmark, task, model, ckpt-hash). Seed-independent since
    # splits draw from the same pool of SMILES.
    ckpt_hash = hashlib.sha256(args.ckpt.encode()).hexdigest()[:12]
    cache_path = EMB_CACHE_DIR / f"{args.benchmark}_{args.task}_{args.model}_{ckpt_hash}.pt"

    all_smiles = (
        pd.concat([train[smiles_col], val[smiles_col], test[smiles_col]]).unique().tolist()
    )
    embed_smiles_cached(
        all_smiles, cache_path, args.ckpt, args.model,
        args.device, args.embed_batch_size, args.featurize_n_jobs,
    )

    def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        z_mol, mask = embed_smiles_cached(
            df[smiles_col].tolist(), cache_path, args.ckpt, args.model,
            args.device, args.embed_batch_size, args.featurize_n_jobs,
        )
        df_ok = df[mask].reset_index(drop=True)
        y = df_ok["Y"].to_numpy(dtype=float)
        return z_mol, y

    x_train, y_train = build_features(train)
    x_val, y_val = build_features(val)
    x_test, y_test = build_features(test)
    z_mol_dim = x_train.shape[1] if x_train.size else 0
    print(f"  feature shape: {x_train.shape[1]}-d  (pairmixer graph embedding)")

    metrics = fit_and_eval(
        x_train, y_train, x_val, y_val, x_test, y_test, task_type, args.head, args.seed,
    )
    elapsed = time.time() - t0
    print(
        f"[pairmixer_admet:{args.benchmark}/{args.task} seed={args.seed}] "
        f"head={args.head} task_type={task_type} ({elapsed:.1f}s)"
    )
    for key, value in metrics.items():
        print(f"    {key:18s} = {value:.4f}")

    ckpt_tag = args.ckpt_tag or Path(args.ckpt).stem
    row = {
        "benchmark": args.benchmark,
        "task": args.task,
        "seed": args.seed,
        "head": args.head,
        "model": args.model,
        "ckpt_tag": ckpt_tag,
        "ckpt_path": args.ckpt,
        "task_type": task_type,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "z_mol_dim": int(z_mol_dim),
        "elapsed_sec": round(elapsed, 1),
        **metrics,
    }
    append_results_row(row)
    print(f"  -> appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
