#!/usr/bin/env python
"""Evaluate PairMixer embeddings as a frozen feature extractor on TDC DTI subsets.

Mirrors scripts/{minimol,mole}/{minimol,mole}_dti_eval.py — same data layout,
same metrics, same sklearn head — but extracts the molecule embedding from a
graphium PairMixer checkpoint instead of an external molecular FM. Use this for
apples-to-apples comparison vs MiniMol / MolE under one consistent eval protocol.

Pipeline:
  1. Hydra-compose the dti_eval config so we get a MultitaskFromSmilesDataModule
     whose featurization matches what was used at pretrain time.
  2. Load PairMixer checkpoint via PredictorModule.load_pretrained_model.
  3. Forward each unique SMILES through pre_nn -> gnn -> graph_output_nn['graph']
     to get a graph-level embedding z_mol.
  4. Concatenate with the 1152-d ESM-C protein embedding from the prepped CSV.
  5. Fit sklearn Ridge or MLPRegressor on train+val combined, evaluate on test.
  6. Append metrics (MAE/MSE/Pearson/Spearman/CI) to results/pairmixer_dti_results.csv.

Usage:
    python scripts/pairmixer/pairmixer_dti_eval.py \\
        --ckpt models_checkpoints/small-dataset/pairmixer_12M/.../*.ckpt \\
        --subset DAVIS --method random --seed 0
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphium.config._loader import load_accelerator, load_datamodule
from graphium.data.collate import graphium_collate_fn
from graphium.trainer.predictor import PredictorModule


ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = ROOT / "results" / "pairmixer_dti_results.csv"
EMB_CACHE_DIR = ROOT / "datacache" / "pairmixer_dti_embeddings"
DTI_EVAL_DIR = ROOT / "data" / "dti-eval"
HYDRA_CFG_DIR = ROOT / "expts" / "hydra-configs"

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


# ── DTI data loading (mirrors minimol_dti_eval.py) ──────────────────────────
def load_dti_split(
    subset: str, method: str, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    parquet_path = DTI_EVAL_DIR / f"{subset}.parquet"
    splits_path = DTI_EVAL_DIR / "splits" / f"{subset}_{method}_seed{seed}.pt"
    if not parquet_path.exists():
        sys.exit(f"ERROR: {parquet_path} missing. Run scripts/data/dti_eval/01_prepare_dti_eval.py.")
    if not splits_path.exists():
        sys.exit(f"ERROR: {splits_path} missing.")

    label_col = SUBSET_LABEL_COL[subset]
    df = pd.read_parquet(parquet_path, columns=["SMILES_nometa", label_col, *PROT_EMB_COLS])
    splits = torch.load(splits_path, weights_only=False)
    return (
        df.iloc[splits["train"]].reset_index(drop=True),
        df.iloc[splits["val"]].reset_index(drop=True),
        df.iloc[splits["test"]].reset_index(drop=True),
        label_col,
    )


# ── PairMixer backbone loading + embedding extraction ────────────────────────
def _build_datamodule_for_featurization(model_name: str, subset: str) -> Any:
    """Compose the dti_eval Hydra config and instantiate the datamodule.

    We only need it for `datamodule.smiles_transformer` (the graphium featurizer
    partial) — the dataset itself is loaded from our prepped CSV. Featurization
    config is inherited from architecture/toymix.yaml so it matches what the
    PairMixer checkpoint was trained with.
    """
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CFG_DIR)):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model={model_name}",
                "accelerator=gpu",
                "tasks=dti_eval",
                "training=dti_eval",
                f"++constants.dti_subset={subset}",
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
    """Patch joblib so each completed task ticks a tqdm bar.

    joblib's BatchCompletionCallBack fires once per batch (default batch_size=auto);
    we tick by self.batch_size so the bar tracks per-task progress accurately.
    """
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
    """Worker-side featurization; returns (smiles, graph) or None on failure."""
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
    """Featurize SMILES in parallel via joblib; drop failures.

    The graphium smiles_transformer is a pickle-safe partial (the datamodule uses
    the same trick to parallelize featurization), so it ships across loky workers
    cleanly. Falls back to single-threaded if n_jobs <= 1.
    """
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
    backbone, graphs: List[Any], device: torch.device, batch_size: int
) -> torch.Tensor:
    """Forward graphs through pre_nn + gnn + graph_output_nn['graph'].

    Replicates the graph-level embedding path from
    graphium/finetuning/linear_probe.py:264-317 (the in-training linear-probe
    callback). Returns one row per input graph.
    """
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
    subset: str, device: str, batch_size: int, featurize_n_jobs: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed SMILES with a per-checkpoint, per-SMILES cache.

    Cache key: SMILES string. Cache path is suffixed by a hash of the checkpoint
    path so different checkpoints don't clobber each other.
    """
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, np.ndarray] = (
        torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    )
    unique = list(dict.fromkeys(smiles))
    missing = [s for s in unique if s not in cache]

    if missing:
        print(f"  [pairmixer] embedding {len(missing):,} / {len(unique):,} new SMILES...", flush=True)

        # Lazy datamodule + checkpoint load — only when there's actual work.
        datamodule = _build_datamodule_for_featurization(model_name, subset)
        smiles_transformer = datamodule.smiles_transformer
        torch_device = torch.device(device)
        # PredictorModule.load_pretrained_model forwards `device` to torch.load's
        # map_location, which needs a real torch device string ("cpu" / "cuda:0"),
        # not Lightning's accelerator tag "gpu".
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

        # Mark featurization failures with a None sentinel so we don't retry them
        # on every per-split call (which would re-load the checkpoint each time).
        for s in set(missing) - set(kept):
            cache[s] = None

        torch.save(cache, cache_path)

    # Return embeddings only for SMILES the cache could produce (graphium's PE
    # computation can fail on pathological molecules — kekulize failures, invalid
    # valence, etc.). Cache values of None are sentinels marking known-bad SMILES.
    # The mask lets the caller drop matching rows from labels / prot_emb / y so
    # x and y stay aligned.
    mask = np.array(
        [s in cache and cache[s] is not None for s in smiles], dtype=bool
    )
    if mask.any():
        out = np.stack(
            [cache[s] for s in smiles if s in cache and cache[s] is not None], axis=0
        )
    else:
        out = np.zeros((0, 0), dtype=np.float32)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  [pairmixer] WARN: {dropped:,} SMILES couldn't be embedded; dropping those rows.")
    return out, mask


# ── Metrics + sklearn head (mirrors minimol_dti_eval.py) ─────────────────────
def _concordance_index(y: np.ndarray, p: np.ndarray) -> float:
    """Pairwise CI — DTI-literature metric. Ties counted as 0.5."""
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


def fit_and_eval(
    x_train, y_train, x_val, y_val, x_test, y_test, head: str, seed: int,
) -> Dict[str, float]:
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    if head == "linear":
        model = Ridge(alpha=1.0, random_state=seed)
    else:
        model = MLPRegressor(
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
        "concordance_index": _concordance_index(y_test, y_pred),
        "mean_pred": float(np.mean(y_pred)),
        "std_pred": float(np.std(y_pred)),
        "mean_target": float(np.mean(y_test)),
        "std_target": float(np.std(y_test)),
    }


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


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="PairMixer checkpoint (.ckpt) path")
    parser.add_argument("--model", default="pairmixer_12M",
                        help="model config name (must have a matching expts/hydra-configs/model/<model>.yaml)")
    parser.add_argument("--subset", required=True, choices=list(SUBSET_LABEL_COL.keys()))
    parser.add_argument("--method", required=True, choices=["random", "cold_target", "temporal"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--featurize-n-jobs", type=int, default=8,
                        help="Parallel workers for graphium featurization (joblib loky backend). "
                             "Set to 1 to disable parallelism. Default: 8.")
    parser.add_argument("--ckpt-tag", default=None,
                        help="Short label for this checkpoint in the results CSV (default: filename stem)")
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"ERROR: checkpoint not found: {args.ckpt}")

    t0 = time.time()
    train, val, test, label_col = load_dti_split(args.subset, args.method, args.seed)
    print(
        f"[pairmixer_dti:{args.subset}/{args.method}/seed={args.seed}] "
        f"train={len(train):,} val={len(val):,} test={len(test):,}  label={label_col}"
    )

    # Cache embeddings per (subset, checkpoint) since seed/method only affect the split.
    ckpt_hash = hashlib.sha256(args.ckpt.encode()).hexdigest()[:12]
    cache_path = EMB_CACHE_DIR / f"{args.subset}_{args.model}_{ckpt_hash}.pt"

    all_smiles = (
        pd.concat([train["SMILES_nometa"], val["SMILES_nometa"], test["SMILES_nometa"]])
        .unique().tolist()
    )
    # Pre-warm cache; mask is recomputed per split below.
    embed_smiles_cached(
        all_smiles, cache_path, args.ckpt, args.model, args.subset, args.device, args.embed_batch_size, args.featurize_n_jobs,
    )

    def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        z_mol, mask = embed_smiles_cached(
            df["SMILES_nometa"].tolist(), cache_path, args.ckpt, args.model, args.subset,
            args.device, args.embed_batch_size, args.featurize_n_jobs,
        )
        df_ok = df[mask].reset_index(drop=True)
        z_prot = df_ok[PROT_EMB_COLS].to_numpy(dtype=np.float32)
        x = np.concatenate([z_mol, z_prot], axis=1)
        y = df_ok[label_col].to_numpy(dtype=np.float32)
        return x, y

    x_train, y_train = build_features(train)
    x_val, y_val = build_features(val)
    x_test, y_test = build_features(test)
    z_mol_dim = x_train.shape[1] - ESMC_DIM
    print(f"  feature shape: {x_train.shape[1]}-d  (pairmixer {z_mol_dim} + prot_emb {ESMC_DIM})")

    metrics = fit_and_eval(x_train, y_train, x_val, y_val, x_test, y_test, args.head, args.seed)
    elapsed = time.time() - t0
    print(
        f"[pairmixer_dti:{args.subset}/{args.method}/seed={args.seed}] "
        f"head={args.head} ({elapsed:.1f}s)"
    )
    for key, value in metrics.items():
        print(f"    {key:18s} = {value:.4f}")

    ckpt_tag = args.ckpt_tag or Path(args.ckpt).stem
    row = {
        "model": args.model,
        "ckpt_tag": ckpt_tag,
        "ckpt_path": args.ckpt,
        "subset": args.subset,
        "method": args.method,
        "seed": args.seed,
        "head": args.head,
        "label_col": label_col,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "z_mol_dim": int(z_mol_dim),
        "feature_dim": int(x_train.shape[1]),
        "elapsed_sec": round(elapsed, 1),
        **metrics,
    }
    append_results_row(row)
    print(f"  -> appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
