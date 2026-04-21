#!/usr/bin/env python
"""TDC DTI regression eval CLI.

    python -m downstream.tasks.tdc_dti_regression.eval \\
        --encoder minimol --head mlp \\
        --subsets DAVIS KIBA --methods random cold_target --seeds 0 1 2 3 4

    python -m downstream.tasks.tdc_dti_regression.eval \\
        --encoder pairmixer --ckpt ./ckpt.ckpt \\
        --subsets BindingDB_Patent_DG --methods temporal --seeds 0 1 2 3 4

Rows append to ``results/downstream/tdc_dti_regression.csv`` (one per
encoder × subset × method × seed × head).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from downstream.model import available_encoders, load_encoder
from downstream.tasks.common import (
    append_result_row,
    regression_metrics,
    train_head,
)
from downstream.tasks.common.heads import HEAD_CHOICES
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MOL_CACHE_DIR,
    DEFAULT_RESULTS_CSV,
    METHODS,
    SUBSETS,
)
from .data import load_split, load_subset


ROOT = Path(__file__).resolve().parents[3]


def _build_encoder(args) -> "downstream.model.base.MoleculeEncoder":
    if args.encoder == "pairmixer":
        if not args.ckpt:
            sys.exit("ERROR: --ckpt is required for the pairmixer encoder.")
        return load_encoder(
            "pairmixer",
            ckpt_path=args.ckpt, model_name=args.model_name,
            device=args.device, batch_size=args.embed_batch_size,
            featurize_n_jobs=args.featurize_n_jobs, ckpt_tag=args.ckpt_tag,
        )
    if args.encoder == "mole":
        return load_encoder("mole", device=args.device, batch_size=args.embed_batch_size_mole)
    if args.encoder == "minimol":
        return load_encoder("minimol", chunk_size=args.embed_batch_size_minimol)
    raise ValueError(args.encoder)


def _run_sweep(args) -> None:
    encoder = _build_encoder(args)
    results_csv = Path(args.results_csv)
    mol_cache = Path(args.mol_cache_dir) / f"{args.encoder}__{encoder.encoder_tag}.pt"
    extra_meta: dict = getattr(encoder, "metadata", {}) or {}
    extra_meta.setdefault("encoder", args.encoder)

    for subset in args.subsets:
        arrays = load_subset(args.data_dir, subset)
        mol_feats, mask = encoder.extract_cached(arrays.smiles, mol_cache)
        if encoder.out_dim <= 0:
            sys.exit(f"ERROR: encoder {args.encoder} did not set out_dim after extract_cached.")
        if not mask.any():
            print(f"  [{subset}] no SMILES embedded; skipping.")
            continue

        X_all = np.concatenate([mol_feats, arrays.prot_emb[mask]], axis=1)
        y_all = arrays.label[mask]
        row_map = np.full(len(mask), -1, dtype=np.int64)
        row_map[np.where(mask)[0]] = np.arange(int(mask.sum()))

        for method in args.methods:
            for seed in args.seeds:
                try:
                    splits = load_split(args.data_dir, subset, method, seed)
                except FileNotFoundError as exc:
                    print(f"  [{subset}/{method}/seed={seed}] {exc}; skipping.")
                    continue

                tr_idx = row_map[np.asarray(splits["train"], dtype=np.int64)]
                va_idx = row_map[np.asarray(splits["val"],   dtype=np.int64)]
                te_idx = row_map[np.asarray(splits["test"],  dtype=np.int64)]
                tr_idx = tr_idx[tr_idx >= 0]
                va_idx = va_idx[va_idx >= 0]
                te_idx = te_idx[te_idx >= 0]
                if len(tr_idx) == 0 or len(te_idx) == 0:
                    print(f"  [{subset}/{method}/seed={seed}] empty split after masking; skip")
                    continue

                t0 = time.time()
                # Fit on train+val combined (regression baselines don't need a val set
                # for hyperparam selection — matches the existing DTI regression scripts).
                X_fit = np.concatenate([X_all[tr_idx], X_all[va_idx]], axis=0)
                y_fit = np.concatenate([y_all[tr_idx], y_all[va_idx]], axis=0)
                model = train_head(
                    X_fit, y_fit,
                    task="regression", head_type=args.head,
                    random_state=seed,
                    autogluon_time_limit=args.autogluon_time_limit,
                    autogluon_preset=args.autogluon_preset,
                )
                y_pred = model.predict(X_all[te_idx])
                metrics = regression_metrics(y_all[te_idx], y_pred)

                row = {
                    "task":        "tdc_dti_regression",
                    "encoder":     args.encoder,
                    "encoder_tag": encoder.encoder_tag,
                    "head":        args.head,
                    "subset":      subset,
                    "method":      method,
                    "seed":        int(seed),
                    "label_col":   arrays.label_col,
                    "n_train":     int(len(tr_idx)),
                    "n_val":       int(len(va_idx)),
                    "n_test":      int(len(te_idx)),
                    "feature_dim":      int(X_all.shape[1]),
                    "feature_dim_mol":  int(encoder.out_dim),
                    "feature_dim_prot": int(arrays.prot_emb.shape[1]),
                    "elapsed_sec": round(time.time() - t0, 1),
                    **metrics,
                    **extra_meta,
                }
                append_result_row(results_csv, row)
                print(
                    f"  [{args.encoder}/{subset}/{method}/seed={seed}/head={args.head}] "
                    f"Pearson={metrics['pearsonr']:.4f} CI={metrics['concordance_index']:.4f} "
                    f"MAE={metrics['mae']:.4f} -> {results_csv}"
                )


def default_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--encoder", required=True, choices=available_encoders())
    p.add_argument("--head", choices=HEAD_CHOICES, default="mlp")
    p.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    p.add_argument("--ckpt", default=None)
    p.add_argument("--model-name", default="pairmixer_12M")
    p.add_argument("--ckpt-tag", default=None)
    p.add_argument("--featurize-n-jobs", type=int, default=8)

    p.add_argument("--device", default=default_device())
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--embed-batch-size-mole", type=int, default=2048)
    p.add_argument("--embed-batch-size-minimol", type=int, default=128)

    p.add_argument("--autogluon-time-limit", type=int, default=600)
    p.add_argument("--autogluon-preset", default="medium_quality")

    p.add_argument("--data-dir", default=str(ROOT / DEFAULT_DATA_DIR))
    p.add_argument("--results-csv", default=str(ROOT / DEFAULT_RESULTS_CSV))
    p.add_argument("--mol-cache-dir", default=str(ROOT / DEFAULT_MOL_CACHE_DIR))

    args = p.parse_args()
    _run_sweep(args)


if __name__ == "__main__":
    main()
