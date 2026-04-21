#!/usr/bin/env python
"""Evaluate PairMixer embeddings on the GRAM-DTI binary-classification benchmark.

Mirrors scripts/pairmixer/pairmixer_dti_eval.py (regression on DAVIS/KIBA/BindingDB)
but targets the GRAM-DTI 4-dataset suite (Yamanishi 08 / Hetionet / DTIAM
Activation / DTIAM Inhibition) with binary classification heads, K-fold CV, and
3 split scenarios per dataset. Heavy lifting (Hydra-composed datamodule for
featurization, encoder forward pass) is reused from the regression script —
only the head + metrics differ.

Feature vector per (drug, target):
    [ pairmixer_z_mol  ||  esmc_protein_1152 ]   = (z_mol_dim + 1152)-d

Usage:
    python scripts/gram_dti_eval/eval_pairmixer.py \\
        --ckpt models_checkpoints/.../pairmixer_12M/.../*.ckpt
    python scripts/gram_dti_eval/eval_pairmixer.py \\
        --ckpt path/to.ckpt --subsets activation --methods target_cold --folds 0  # smoke test

Results append to results/gram_dti_eval/pairmixer_results.csv (one row per fold).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse heavy lifting from the regression script (same Hydra composition,
# featurization, and encoder-forward path that it already validated).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pairmixer"))
from pairmixer_dti_eval import (  # noqa: E402
    _build_datamodule_for_featurization,
    _extract_embeddings,
    _featurize_smiles,
)
from graphium.trainer.predictor import PredictorModule  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    METHODS, SUBSETS, SUBSET_TO_FOLDS, run_eval_sweep,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET_DIR = ROOT / "data" / "dti-classif-eval"
DEFAULT_RESULTS_CSV = ROOT / "results" / "gram_dti_eval" / "pairmixer_results.csv"
DEFAULT_MOL_CACHE_DIR = ROOT / "datacache" / "gram_dti_eval" / "pairmixer"


def make_extract_fn(
    ckpt_path: str, model_name: str, device: str,
    batch_size: int, n_jobs: int,
):
    """Return a SMILES → ndarray(N, D) function that loads PairMixer once and reuses it.

    The Hydra-composed datamodule, the checkpoint, and the backbone are all
    initialized lazily on first call and cached in a closure — so a multi-subset
    sweep pays the load cost once, not per-subset.
    """
    state: dict = {}

    def _init_once() -> None:
        if state:
            return
        # Subset is only used to compose the Hydra config; the actual SMILES we
        # featurize come from the GRAM-DTI parquet, not the dti_eval CSV. Pass
        # any valid subset that exists in dti_eval's choice list.
        datamodule = _build_datamodule_for_featurization(model_name, "DAVIS")
        torch_device = torch.device(device)
        predictor = PredictorModule.load_pretrained_model(
            name_or_path=ckpt_path, device=str(torch_device),
        )
        backbone = predictor.model
        backbone.to(torch_device)
        state["smiles_transformer"] = datamodule.smiles_transformer
        state["backbone"] = backbone
        state["device"] = torch_device

    def extract(smiles: list[str]) -> np.ndarray:
        _init_once()
        graphs, kept = _featurize_smiles(smiles, state["smiles_transformer"], n_jobs=n_jobs)
        if not graphs:
            raise RuntimeError("No SMILES survived PairMixer featurization.")
        if len(kept) != len(smiles):
            failed = set(smiles) - set(kept)
            raise RuntimeError(
                f"{len(failed)} SMILES failed featurization (e.g. {sorted(failed)[:3]}). "
                "common.py's per-SMILES cache requires extract_fn to return one row per input."
            )
        z = _extract_embeddings(state["backbone"], graphs, state["device"], batch_size).numpy()
        return z

    return extract


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="PairMixer checkpoint (.ckpt) path")
    parser.add_argument(
        "--model", default="pairmixer_12M",
        help="Hydra model config name (must have a matching expts/hydra-configs/model/<model>.yaml)",
    )
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Subset of fold indices to run (default: all folds — 10 for DTI, 5 for MoA).",
    )
    parser.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--featurize-n-jobs", type=int, default=8)
    parser.add_argument(
        "--ckpt-tag", default=None,
        help="Short label for this checkpoint in the results CSV (default: filename stem).",
    )
    parser.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--mol-cache-dir", default=str(DEFAULT_MOL_CACHE_DIR))
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"ERROR: checkpoint not found: {args.ckpt}")

    # Tag the cache with the checkpoint hash so different ckpts don't share a cache file.
    ckpt_hash = hashlib.sha256(args.ckpt.encode()).hexdigest()[:12]
    encoder_tag = f"{args.model}_{ckpt_hash}"
    ckpt_tag = args.ckpt_tag or Path(args.ckpt).stem

    folds_map = (
        {s: args.folds for s in args.subsets} if args.folds is not None
        else {s: list(range(SUBSET_TO_FOLDS[s])) for s in args.subsets}
    )

    run_eval_sweep(
        encoder_tag=encoder_tag,
        extract_fn=make_extract_fn(
            args.ckpt, args.model, args.device, args.embed_batch_size, args.featurize_n_jobs,
        ),
        parquet_dir=args.parquet_dir,
        results_csv=args.results_csv,
        mol_cache_dir=args.mol_cache_dir,
        subsets=args.subsets,
        methods=args.methods,
        folds=folds_map,
        head_type=args.head,
        random_state=args.random_state,
        extra_row_meta={
            "model":     args.model,
            "ckpt_tag":  ckpt_tag,
            "ckpt_path": args.ckpt,
            "device":    args.device,
        },
    )


if __name__ == "__main__":
    main()
