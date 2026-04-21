#!/usr/bin/env python
"""Evaluate MolE embeddings on the GRAM-DTI binary-classification benchmark.

Mirrors scripts/mole/mole_dti_eval.py (regression on DAVIS/KIBA/BindingDB) but
targets the GRAM-DTI 4-dataset suite (Yamanishi 08 / Hetionet / DTIAM
Activation / DTIAM Inhibition) with binary classification heads, K-fold CV, and
3 split scenarios per dataset. See ``common.py`` and
``memory/project_gram_dti_benchmark.md`` for the protocol.

Feature vector per (drug, target):
    [ mole_gin_concat_256 || esmc_protein_1152 ] = 1408-d

Usage:
    python scripts/gram_dti_eval/eval_mole.py                # full sweep, logreg head
    python scripts/gram_dti_eval/eval_mole.py --head mlp --device cuda:7
    python scripts/gram_dti_eval/eval_mole.py \\
        --subsets activation --methods target_cold --folds 0  # smoke test

Results append to results/gram_dti_eval/mole_results.csv.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse asset-management helpers from the existing MolE script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mole"))
from mole_eval import (  # noqa: E402
    MOLE_MODEL_NAME,
    UPSTREAM_COMMIT,
    ensure_upstream_checkpoint_layout,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    METHODS, SUBSETS, SUBSET_TO_FOLDS, run_eval_sweep,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET_DIR = ROOT / "data" / "dti-classif-eval"
DEFAULT_RESULTS_CSV = ROOT / "results" / "gram_dti_eval" / "mole_results.csv"
DEFAULT_MOL_CACHE_DIR = ROOT / "datacache" / "gram_dti_eval" / "mole"


def make_extract_fn(device: str, batch_size: int):
    """Return a SMILES → ndarray(N, 256) function backed by a single MolE instance."""
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

    def extract(smiles: list[str]) -> np.ndarray:
        smile_df = pd.DataFrame({"chem_id": smiles, "smiles": smiles})
        embeddings = batch_representation(smile_df, model, batch_size=batch_size, device=device)
        # batch_representation returns DF indexed by chem_id; preserve input order.
        return embeddings.loc[smiles].to_numpy(dtype=np.float32)

    return extract


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


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
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--embed-batch-size", type=int, default=2048)
    parser.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--mol-cache-dir", default=str(DEFAULT_MOL_CACHE_DIR))
    args = parser.parse_args()

    folds_map = (
        {s: args.folds for s in args.subsets} if args.folds is not None
        else {s: list(range(SUBSET_TO_FOLDS[s])) for s in args.subsets}
    )

    run_eval_sweep(
        encoder_tag="mole_gin_concat",
        extract_fn=make_extract_fn(args.device, args.embed_batch_size),
        parquet_dir=args.parquet_dir,
        results_csv=args.results_csv,
        mol_cache_dir=args.mol_cache_dir,
        subsets=args.subsets,
        methods=args.methods,
        folds=folds_map,
        head_type=args.head,
        random_state=args.random_state,
        extra_row_meta={
            "embedding_model": MOLE_MODEL_NAME,
            "upstream_commit": UPSTREAM_COMMIT,
            "device": args.device,
            "feature_dim_mol": 256,
            "feature_dim_prot": 1152,
        },
    )


if __name__ == "__main__":
    main()
