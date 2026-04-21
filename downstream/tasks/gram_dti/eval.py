#!/usr/bin/env python
"""GRAM-DTI eval CLI — swap molecule encoder, fix ESM-2 protein features.

    python -m downstream.tasks.gram_dti.eval --encoder minimol --head mlp
    python -m downstream.tasks.gram_dti.eval --encoder mole    --head autogluon
    python -m downstream.tasks.gram_dti.eval --encoder pairmixer \\
        --ckpt models_checkpoints/.../foo.ckpt --head mlp \\
        --subsets activation --methods target_cold --folds 0  # smoke test

All encoders / heads / subsets / folds share one CSV:
``results/downstream/gram_dti.csv``. Each row = one (encoder, subset, method,
fold, head) combination. Use the dashboard notebook to aggregate.

CPU usage: sklearn MLP / LogReg and CPU-side torch default to one BLAS thread
per physical core, which on high-core hosts pushes load averages > 300 and
starves any concurrent GPU run of its dataloader cores. Cap with
``--cpu-threads N`` (or ``GRAMDTI_CPU_THREADS=N``); default is 8.
"""
from __future__ import annotations

# ---- CPU thread caps (must come before numpy / sklearn / torch import) ----
import os
import sys as _sys


def _cap_cpu_threads() -> int:
    # CLI flag wins over env. Peek sys.argv early since argparse hasn't run.
    n = int(os.environ.get("GRAMDTI_CPU_THREADS", "8"))
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
import sys
import time
from pathlib import Path

import numpy as np

# Belt-and-suspenders: some libs cache thread counts at import time so the env
# vars above don't always stick. Cap torch's internal pools too.
try:
    import torch as _torch
    _torch.set_num_threads(_CPU_THREADS)
    _torch.set_num_interop_threads(max(1, _CPU_THREADS // 2))
except ImportError:
    pass
except RuntimeError:
    # torch refuses to set interop threads twice; ignore on subsequent imports.
    pass

from downstream.model import available_encoders, load_encoder
from downstream.tasks.common import (
    append_result_row,
    classification_metrics,
    train_head,
)
from downstream.tasks.common.heads import HEAD_CHOICES, predict_proba_positive
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MOL_CACHE_DIR,
    DEFAULT_RESULTS_CSV,
    METHODS,
    SUBSETS,
    SUBSET_TO_FOLDS,
)
from .data import load_split, load_subset


ROOT = Path(__file__).resolve().parents[3]


def _build_encoder(args) -> "downstream.model.base.MoleculeEncoder":
    kwargs: dict = {}
    if args.encoder == "pairmixer":
        if not args.ckpt:
            sys.exit("ERROR: --ckpt is required for the pairmixer encoder.")
        kwargs = dict(
            ckpt_path=args.ckpt, model_name=args.model_name,
            device=args.device, batch_size=args.embed_batch_size,
            featurize_n_jobs=args.featurize_n_jobs,
            ckpt_tag=args.ckpt_tag,
        )
    elif args.encoder == "mole":
        kwargs = dict(device=args.device, batch_size=args.embed_batch_size_mole)
    elif args.encoder == "minimol":
        kwargs = dict(chunk_size=args.embed_batch_size_minimol)
    return load_encoder(args.encoder, **kwargs)


def _run_sweep(args) -> None:
    encoder = _build_encoder(args)
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    mol_cache = Path(args.mol_cache_dir) / f"{args.encoder}__{encoder.encoder_tag}.pt"

    folds_map: dict[str, list[int]] = (
        {s: list(args.folds) for s in args.subsets} if args.folds
        else {s: list(range(SUBSET_TO_FOLDS[s])) for s in args.subsets}
    )

    extra_meta: dict = getattr(encoder, "metadata", {}) or {}
    extra_meta.setdefault("encoder", args.encoder)

    for subset in args.subsets:
        arrays = load_subset(args.data_dir, subset)
        mol_feats, mask = encoder.extract_cached(arrays.smiles, mol_cache)
        # Record out_dim once (after first extract pass, PairMixer now has it).
        if encoder.out_dim <= 0:
            sys.exit(f"ERROR: encoder {args.encoder} did not set out_dim after extract_cached.")
        if not mask.any():
            print(f"  [{subset}] WARN: no SMILES embedded; skipping.")
            continue

        X_all = np.concatenate(
            [mol_feats, arrays.prot_emb[mask].astype(np.float32)], axis=1,
        )
        y_all = arrays.label[mask].astype(np.int64)
        # Row indices are wrt the original parquet; remap to the masked frame.
        row_map = np.full(len(mask), -1, dtype=np.int64)
        row_map[np.where(mask)[0]] = np.arange(int(mask.sum()))

        for method in args.methods:
            for fold in folds_map[subset]:
                splits = load_split(args.data_dir, subset, method, fold)
                tr_idx = row_map[np.asarray(splits["train"], dtype=np.int64)]
                va_idx = row_map[np.asarray(splits["val"],   dtype=np.int64)]
                te_idx = row_map[np.asarray(splits["test"],  dtype=np.int64)]
                tr_idx = tr_idx[tr_idx >= 0]
                va_idx = va_idx[va_idx >= 0]
                te_idx = te_idx[te_idx >= 0]

                if len(tr_idx) == 0 or len(te_idx) == 0:
                    print(f"  [{subset}/{method}/fold={fold}] empty split after masking; skip")
                    continue

                t0 = time.time()
                clf = train_head(
                    X_all[tr_idx], y_all[tr_idx],
                    task="classification", head_type=args.head,
                    random_state=args.random_state,
                    autogluon_time_limit=args.autogluon_time_limit,
                    autogluon_preset=args.autogluon_preset,
                )
                proba_val  = predict_proba_positive(clf, X_all[va_idx]) if len(va_idx) else np.zeros(0)
                proba_test = predict_proba_positive(clf, X_all[te_idx])

                val_m = (classification_metrics(y_all[va_idx], proba_val)
                         if len(va_idx) else {k: float("nan") for k in
                                              ("auroc", "auprc", "f1", "sensitivity", "accuracy")})
                test_m = classification_metrics(y_all[te_idx], proba_test)

                row = {
                    "task":        "gram_dti",
                    "encoder":     args.encoder,
                    "encoder_tag": encoder.encoder_tag,
                    "head":        args.head,
                    "subset":      subset,
                    "method":      method,
                    "fold":        int(fold),
                    "n_train":     int(len(tr_idx)),
                    "n_val":       int(len(va_idx)),
                    "n_test":      int(len(te_idx)),
                    "feature_dim":      int(X_all.shape[1]),
                    "feature_dim_mol":  int(encoder.out_dim),
                    "feature_dim_prot": int(arrays.prot_emb.shape[1]),
                    "elapsed_sec": round(time.time() - t0, 1),
                    **{f"val_{k}":  v for k, v in val_m.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                    **extra_meta,
                }
                append_result_row(results_csv, row)
                print(
                    f"  [{args.encoder}/{subset}/{method}/fold={fold}/head={args.head}] "
                    f"test AUPRC={test_m['auprc']:.4f} AUROC={test_m['auroc']:.4f} "
                    f"F1={test_m['f1']:.4f}  -> {results_csv}"
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
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="Subset of fold indices (default: all — 10 DTI / 5 MoA)")
    p.add_argument("--random-state", type=int, default=0)

    # pairmixer-only
    p.add_argument("--ckpt", default=None, help="(pairmixer) checkpoint .ckpt path")
    p.add_argument("--model-name", default="pairmixer_12M",
                   help="(pairmixer) Hydra model config name")
    p.add_argument("--ckpt-tag", default=None,
                   help="(pairmixer) short label in results CSV; default = filename stem")
    p.add_argument("--featurize-n-jobs", type=int, default=8,
                   help="(pairmixer) joblib workers for graphium featurization")

    # device / batch
    p.add_argument("--device", default=default_device())
    p.add_argument("--embed-batch-size", type=int, default=32,
                   help="(pairmixer) GNN forward batch size")
    p.add_argument("--embed-batch-size-mole", type=int, default=2048)
    p.add_argument("--embed-batch-size-minimol", type=int, default=128)

    # autogluon-only
    p.add_argument("--autogluon-time-limit", type=int, default=600)
    p.add_argument("--autogluon-preset", default="medium_quality")

    # cpu throttling (already applied at import time; listed here for --help)
    p.add_argument("--cpu-threads", type=int, default=_CPU_THREADS,
                   help=f"BLAS/OpenMP thread cap (already applied; default {_CPU_THREADS}, "
                        "override with --cpu-threads N or GRAMDTI_CPU_THREADS=N)")

    # paths
    p.add_argument("--data-dir", default=str(ROOT / DEFAULT_DATA_DIR))
    p.add_argument("--results-csv", default=str(ROOT / DEFAULT_RESULTS_CSV))
    p.add_argument("--mol-cache-dir", default=str(ROOT / DEFAULT_MOL_CACHE_DIR))

    args = p.parse_args()
    _run_sweep(args)


if __name__ == "__main__":
    main()
