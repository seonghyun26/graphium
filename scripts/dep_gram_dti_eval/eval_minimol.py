#!/usr/bin/env python
"""Evaluate Minimol embeddings on the GRAM-DTI binary-classification benchmark.

Mirrors scripts/minimol/minimol_dti_eval.py (regression on DAVIS/KIBA/BindingDB)
but targets the GRAM-DTI 4-dataset suite (Yamanishi 08 / Hetionet / DTIAM
Activation / DTIAM Inhibition) with binary classification heads, K-fold CV, and
3 split scenarios per dataset. See ``common.py`` and
``memory/project_gram_dti_benchmark.md`` for the protocol.

Feature vector per (drug, target):
    [ minimol_512  ||  esmc_protein_1152 ]   = 1664-d

Usage:
    python scripts/gram_dti_eval/eval_minimol.py                # full sweep, logreg head
    python scripts/gram_dti_eval/eval_minimol.py --head mlp     # MLP head
    python scripts/gram_dti_eval/eval_minimol.py \\
        --subsets activation --methods target_cold --folds 0    # smoke test

Results append to results/gram_dti_eval/minimol_results.csv (one row per fold).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from minimol import Minimol  # imported early since some pipelines chdir() at import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402  (import after sys.path mutation)
    METHODS, SUBSETS, SUBSET_TO_FOLDS, run_eval_sweep,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET_DIR = ROOT / "data" / "dti-classif-eval"
DEFAULT_RESULTS_CSV = ROOT / "results" / "gram_dti_eval" / "minimol_results.csv"
DEFAULT_MOL_CACHE_DIR = ROOT / "datacache" / "gram_dti_eval" / "minimol"


def make_extract_fn():
    """Return a SMILES → ndarray(N, 512) function backed by a single Minimol instance."""
    model = Minimol()

    def extract(smiles: list[str]) -> np.ndarray:
        embs = model(smiles)
        return np.stack([e.detach().cpu().float().numpy() for e in embs], axis=0)

    return extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Subset of fold indices to run (default: all folds — 10 for DTI, 5 for MoA).",
    )
    parser.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--mol-cache-dir", default=str(DEFAULT_MOL_CACHE_DIR))
    args = parser.parse_args()

    folds_map = (
        {s: args.folds for s in args.subsets} if args.folds is not None
        else {s: list(range(SUBSET_TO_FOLDS[s])) for s in args.subsets}
    )

    run_eval_sweep(
        encoder_tag="minimol_v1",
        extract_fn=make_extract_fn(),
        parquet_dir=args.parquet_dir,
        results_csv=args.results_csv,
        mol_cache_dir=args.mol_cache_dir,
        subsets=args.subsets,
        methods=args.methods,
        folds=folds_map,
        head_type=args.head,
        random_state=args.random_state,
        extra_row_meta={"feature_dim_mol": 512, "feature_dim_prot": 1152},
    )


if __name__ == "__main__":
    main()
