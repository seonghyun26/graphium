#!/usr/bin/env python
"""Bioactivity eval CLI — swap molecule (or cell-painting) encoder; Butina 6-fold CV.

    python -m downstream.tasks.bioactivity.eval --encoder ecfp    --cv
    python -m downstream.tasks.bioactivity.eval --encoder minimol --cv
    python -m downstream.tasks.bioactivity.eval --encoder mole    --cv
    python -m downstream.tasks.bioactivity.eval --encoder cpcnn   --cv \\
        --cpcnn-csv /path/to/jump_cpcnn_smiles_embeddings.csv
    python -m downstream.tasks.bioactivity.eval --encoder pairmixer --cv \\
        --ckpt models_checkpoints/.../*.ckpt

Rows append to ``results/downstream/bioactivity.csv`` — one row per
(encoder, fold) for CV runs, or a single row for the default single-split run
(test=fold 5, val=fold 4).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from downstream.model import available_encoders, load_encoder
from downstream.tasks.common import append_result_row
from .report import EF_TOP_FRACTIONS, assay_threshold_summary, ef_macro_column, flatten_per_assay_metrics
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MOL_CACHE_DIR,
    DEFAULT_RESULTS_CSV,
    N_FOLDS,
)
from .data import load_dataset, split_indices
from .head import train_and_eval_fold, train_and_eval_fold_ensemble


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
    if args.encoder == "kpgt":
        # KPGT runs in a sibling conda env; its DGL build has no CUDA backend,
        # so pin to CPU regardless of what --device says.
        return load_encoder("kpgt", device="cpu", batch_size=args.embed_batch_size)
    if args.encoder == "ecfp":
        return load_encoder("ecfp", n_bits=args.n_bits)
    if args.encoder == "cpcnn":
        if not args.cpcnn_csv:
            sys.exit("ERROR: --cpcnn-csv is required for the cpcnn encoder.")
        return load_encoder("cpcnn", cpcnn_csv=args.cpcnn_csv)
    raise ValueError(args.encoder)


def _resolve_head_defaults(args) -> None:
    """Pick sensible optimizer / LR per encoder when left on 'auto'.

    Matches ``dep_baseline_mlp_cell_bioactivity.py``: ECFP (sparse binary) uses
    SGD + lr=2.0 (the paper default); dense embeddings (CPCNN / MiniMol / MolE /
    PairMixer) need Adam + lr=1e-3 (SGD+2.0 explodes gradients on continuous inputs).
    """
    if args.optimizer == "auto":
        args.optimizer = "sgd" if args.encoder == "ecfp" else "adam"
    if args.lr is None:
        args.lr = 2.0 if args.optimizer == "sgd" else 1e-3


def _run_sweep(args) -> None:
    encoder = _build_encoder(args)
    arrays = load_dataset(args.data_dir, csv_name=args.csv_name)

    mol_cache_dir = Path(args.mol_cache_dir)
    mol_cache_path = mol_cache_dir / f"{args.encoder}__{encoder.encoder_tag}.pt"

    X, mask = encoder.extract_cached(arrays.smiles, mol_cache_path)
    if encoder.out_dim <= 0:
        sys.exit(f"ERROR: encoder {args.encoder} did not set out_dim after extract_cached.")
    print(f"features: {X.shape[0]:,}/{len(arrays.smiles):,} compounds × {X.shape[1]}-d")

    # Align labels/folds to the encoder-survivor subset.
    kept_idx = np.where(mask)[0]
    Y = arrays.labels[kept_idx]
    folds = arrays.folds[kept_idx]
    valid = np.ones(len(kept_idx), dtype=bool)

    test_folds = list(range(N_FOLDS)) if args.cv else [args.single_test_fold]
    standardize = args.encoder != "ecfp"   # binary fingerprints stay raw
    extra_meta: dict = getattr(encoder, "metadata", {}) or {}
    extra_meta.setdefault("encoder", args.encoder)

    results_csv = Path(args.results_csv)
    per_fold: list[dict] = []
    for k in test_folds:
        splits = split_indices(folds, valid, test_fold=k, n_folds=N_FOLDS)
        if len(splits["train"]) == 0 or len(splits["test"]) == 0:
            print(f"  [fold={k}] empty split; skip")
            continue
        tag = f"fold{k}" if args.cv else "single"
        # Reseed per-fold so fold N isn't biased by fold N-1 state.
        t0 = time.time()
        if args.ensemble_size > 1:
            metrics = train_and_eval_fold_ensemble(
                X, Y, splits, arrays.assay_cols,
                standardize=standardize,
                hidden=args.hidden_dim, num_hidden=args.num_hidden, dropout=args.dropout,
                epochs=args.epochs, batch_size=args.batch_size,
                optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
                lr_patience=args.lr_patience, min_lr=args.min_lr,
                seed=args.seed + k, device=args.device, tag=tag,
                log_every=args.log_every,
                ensemble_size=args.ensemble_size,
            )
        else:
            metrics = train_and_eval_fold(
                X, Y, splits, arrays.assay_cols,
                standardize=standardize,
                hidden=args.hidden_dim, num_hidden=args.num_hidden, dropout=args.dropout,
                epochs=args.epochs, batch_size=args.batch_size,
                optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
                lr_patience=args.lr_patience, min_lr=args.min_lr,
                seed=args.seed + k, device=args.device, tag=tag,
                log_every=args.log_every,
            )
        per_fold.append(metrics)
        per_assay_metrics = flatten_per_assay_metrics(
            per_assay_auroc=metrics["per_assay_auroc"],
            per_assay_auprc=metrics["per_assay_auprc"],
            per_assay_ef=metrics["per_assay_ef"],
        )
        threshold_metrics = assay_threshold_summary(
            metrics["per_assay_auroc"],
            denominator=len(arrays.assay_cols),
        )

        row = {
            "task":         "bioactivity",
            "encoder":      args.encoder,
            "encoder_tag":  encoder.encoder_tag,
            "head":         "focal_bce_pytorch",
            "split":        "cv6" if args.cv else "single",
            "fold":         int(k),
            "n_train":      int(len(splits["train"])),
            "n_val":        int(len(splits["val"])),
            "n_test":       int(len(splits["test"])),
            "feature_dim":  int(X.shape[1]),
            "n_assays_total":   int(len(arrays.assay_cols)),
            "test_auroc_macro": metrics["auroc_macro"],
            "test_auprc_macro": metrics["auprc_macro"],
            "n_assays_scored":  metrics["n_assays_scored"],
            **{ef_macro_column(top_fraction): metrics[ef_macro_column(top_fraction)] for top_fraction in EF_TOP_FRACTIONS},
            "best_val_auroc":   metrics.get("best_val_auroc"),
            "elapsed_sec":  round(time.time() - t0, 1),
            "optimizer":    args.optimizer,
            "lr":           args.lr,
            "ensemble_size": int(metrics.get("ensemble_size", 1)),
            "member_auroc_mean": metrics.get("member_auroc_mean"),
            "member_auroc_std":  metrics.get("member_auroc_std"),
            "member_auprc_mean": metrics.get("member_auprc_mean"),
            "member_auprc_std":  metrics.get("member_auprc_std"),
            **threshold_metrics,
            **per_assay_metrics,
            **extra_meta,
        }
        append_result_row(results_csv, row)
        print(f"  -> appended row (fold={k}) to {results_csv}")

    if args.cv and per_fold:
        aurocs = np.array([m["auroc_macro"] for m in per_fold])
        auprcs = np.array([m["auprc_macro"] for m in per_fold])
        print(
            f"\nCV SUMMARY ({len(per_fold)} folds)  "
            f"auroc={aurocs.mean():.4f} ± {aurocs.std(ddof=0):.4f}  "
            f"auprc={auprcs.mean():.4f} ± {auprcs.std(ddof=0):.4f}"
        )


def default_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--encoder", required=True, choices=available_encoders())
    p.add_argument("--cv", action="store_true",
                   help="6-fold CV rotation (one row per fold). Default: single split (test=5, val=4).")
    p.add_argument("--single-test-fold", type=int, default=5,
                   help="Test fold id for single-split runs (default: 5).")

    # encoder-specific
    p.add_argument("--ckpt", default=None, help="(pairmixer) checkpoint .ckpt path")
    p.add_argument("--model-name", default="pairmixer_12M",
                   help="(pairmixer) Hydra model config name")
    p.add_argument("--ckpt-tag", default=None,
                   help="(pairmixer) short label in results CSV; default = filename stem")
    p.add_argument("--featurize-n-jobs", type=int, default=8)
    p.add_argument("--cpcnn-csv", default=None,
                   help="(cpcnn) per-well CSV with SMILES_nometa + feature_* columns")
    p.add_argument("--n-bits", type=int, default=1024,
                   help="(ecfp) fingerprint bit count (default: 1024)")

    # device / batch
    p.add_argument("--device", default=default_device())
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--embed-batch-size-mole", type=int, default=2048)
    p.add_argument("--embed-batch-size-minimol", type=int, default=128)

    # head / training
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-hidden", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", choices=("sgd", "adam", "auto"), default="auto")
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 2.0 for SGD/ecfp; 1e-3 for Adam/dense encoders.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ensemble-size", type=int, default=1,
                   help="Inner ensemble: train N MLP heads per fold with different seeds, "
                        "average sigmoid probabilities across members, recompute metrics. "
                        "Default 1 (single fit, original behavior).")

    # paths
    p.add_argument("--data-dir", default=str(ROOT / DEFAULT_DATA_DIR))
    p.add_argument("--csv-name", default="cell_bioactivity.csv")
    p.add_argument("--results-csv", default=str(ROOT / DEFAULT_RESULTS_CSV))
    p.add_argument("--mol-cache-dir", default=str(ROOT / DEFAULT_MOL_CACHE_DIR))

    args = p.parse_args()
    _resolve_head_defaults(args)
    _run_sweep(args)


if __name__ == "__main__":
    main()
