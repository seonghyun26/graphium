#!/usr/bin/env python
"""Build per-subset DTI evaluation datasets from TDC.

For each TDC DTI subset, produces:
  - data/dti-eval/<subset>.csv                 : SMILES_nometa, label, prot_emb_0..1151
  - data/dti-eval/<subset>_norm_stats.pt       : per-subset z-score stats for the label
                                                  and per-feature stats for prot_emb
  - data/dti-eval/splits/<subset>_<method>_seed<k>.pt : {'train','val','test'} index lists

Splits (one .pt per (method, seed)):
  - method=random        : data.get_split(method='random',     seed=k, frac=[0.7,0.1,0.2])
  - method=cold_target   : data.get_split(method='cold_split', column_name='Target',
                                          seed=k, frac=[0.7,0.1,0.2])

Label conversion:
  - DAVIS / BindingDB_*  : pY = 9 - log10(Y_nM), rows with Y <= 0 dropped, then z-scored
  - KIBA                 : raw KIBA score (no log; different scale), then z-scored

Z-score stats are computed once globally per subset (cheap, avoids per-seed CSV blowup).
The .pt stats file lets you denormalize predictions for paper-style metric reporting.

Usage:
    python scripts/data/dti_eval/01_prepare_dti_eval.py \\
        --subsets DAVIS KIBA \\
        --methods random cold_target \\
        --seeds 0 1 2 3 4 \\
        --protein-emb graphium/data/dti/protein-esmc.parquet \\
        --output-dir data/dti-eval \\
        --tdc-cache data/tdc_cache
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SUBSETS = ["DAVIS", "KIBA", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"]
METHODS = ["random", "cold_target"]
ESMC_DIM = 1152
PROT_EMB_COLS = [f"prot_emb_{i}" for i in range(ESMC_DIM)]


def _import_tdc_dti():
    """Import TDC's DTI loader, working around gget's Python 3.10+ syntax on 3.9."""
    sys.modules.setdefault("gget", types.ModuleType("gget"))
    try:
        from tdc.multi_pred import DTI
    except ImportError:
        sys.exit("ERROR: PyTDC required. `pip install PyTDC`.")
    return DTI


def _load_protein_embeddings(path: str) -> pd.DataFrame:
    """Return a DataFrame indexed by protein_id with columns feature_0..feature_1151."""
    print(f"Loading protein embeddings from {path}")
    df = pd.read_parquet(path)
    feat_cols = [f"feature_{i}" for i in range(ESMC_DIM)]
    missing = set(feat_cols) - set(df.columns)
    if missing:
        sys.exit(f"ERROR: protein embedding file missing columns: {sorted(missing)[:5]}...")
    df = df.set_index("protein_id")[feat_cols]
    print(f"  -> {len(df):,} proteins, {len(feat_cols)}-dim embeddings")
    return df


def _build_subset_csv(
    subset: str,
    tdc_data,
    full_df: pd.DataFrame,
    protein_emb_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """Filter, label-convert, normalize, and save the per-subset CSV.

    Returns (filtered_df, label_col_name) for downstream split-index generation.
    """
    label_col = "kiba_score" if subset == "KIBA" else "pY"

    # --- Filter: known proteins; for non-KIBA also Y > 0 (log-transform precondition).
    n_before = len(full_df)
    keep = full_df["Target_ID"].isin(protein_emb_df.index)
    if subset != "KIBA":
        keep &= full_df["Y"] > 0
    full_df = full_df[keep].reset_index(drop=True)
    full_df["row_idx"] = np.arange(len(full_df), dtype=np.int64)
    print(
        f"  [{subset}] {n_before:,} -> {len(full_df):,} rows "
        f"(dropped missing proteins{' / non-positive Y' if subset != 'KIBA' else ''})"
    )

    # --- Compute scalar label.
    y = full_df["Y"].values.astype(np.float64)
    if subset == "KIBA":
        labels = y.astype(np.float32)
    else:
        labels = (9.0 - np.log10(y)).clip(0.0, 14.0).astype(np.float32)
    label_mean = float(labels.mean())
    label_std = float(max(labels.std(), 1e-8))
    label_norm = ((labels - label_mean) / label_std).astype(np.float32)
    print(
        f"  [{subset}] {label_col}: raw mean={labels.mean():.3f} std={labels.std():.3f}"
        f" min={labels.min():.3f} max={labels.max():.3f}"
    )

    # --- Build prot_emb matrix in row order, z-score per feature.
    prot_emb = protein_emb_df.loc[full_df["Target_ID"].values].values.astype(np.float32)
    prot_mean = prot_emb.mean(axis=0)
    prot_std = prot_emb.std(axis=0)
    prot_std = np.where(prot_std < 1e-8, 1.0, prot_std).astype(np.float32)
    prot_emb_norm = ((prot_emb - prot_mean) / prot_std).astype(np.float32)

    # --- Assemble output frame: SMILES_nometa, label, prot_emb_0..1151.
    out = pd.DataFrame({"SMILES_nometa": full_df["Drug"].values})
    out[label_col] = label_norm
    out[PROT_EMB_COLS] = prot_emb_norm

    csv_path = output_dir / f"{subset}.csv"
    out.to_csv(csv_path, index=False)
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  [{subset}] saved CSV: {csv_path} ({len(out):,} rows, {size_mb:.1f} MB)")

    stats_path = output_dir / f"{subset}_norm_stats.pt"
    torch.save(
        {
            "label_col": label_col,
            "label_mean": torch.tensor(label_mean),
            "label_std": torch.tensor(label_std),
            "prot_emb_mean": torch.from_numpy(prot_mean),
            "prot_emb_std": torch.from_numpy(prot_std),
        },
        stats_path,
    )
    print(f"  [{subset}] saved stats: {stats_path}")

    return full_df, label_col


def _generate_splits(
    subset: str,
    tdc_data,
    full_df: pd.DataFrame,
    methods: list[str],
    seeds: list[int],
    splits_dir: Path,
) -> None:
    """For each (method, seed), call TDC's get_split and resolve to row indices in full_df.

    Rows that didn't survive filtering (missing protein / non-positive Y) are simply
    absent from the final index lists — preserving TDC split semantics over the kept rows.
    """
    # Index full_df by (Drug, Target_ID) once for cheap lookup across all (method, seed).
    pair_to_row = pd.Series(
        full_df["row_idx"].values,
        index=pd.MultiIndex.from_arrays(
            [full_df["Drug"].values, full_df["Target_ID"].values],
            names=["Drug", "Target_ID"],
        ),
    )
    # Multiple rows may share the same (Drug, Target_ID); collapse to lists.
    pair_to_rows = pair_to_row.groupby(level=[0, 1]).apply(list)

    for method in methods:
        for seed in seeds:
            if method == "random":
                splits_raw = tdc_data.get_split(
                    method="random", seed=seed, frac=[0.7, 0.1, 0.2]
                )
            elif method == "cold_target":
                splits_raw = tdc_data.get_split(
                    method="cold_split",
                    column_name="Target",
                    seed=seed,
                    frac=[0.7, 0.1, 0.2],
                )
            else:
                raise ValueError(f"Unknown split method: {method}")

            # TDC historically returns 'valid'; graphium's MultitaskFromSmilesDataModule
            # expects 'val'. Map keys, drop anything not in {train, valid, test}.
            key_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
            split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            for raw_key, df_split in splits_raw.items():
                if raw_key not in key_map:
                    continue
                target_key = key_map[raw_key]
                lookup_idx = pd.MultiIndex.from_arrays(
                    [df_split["Drug"].values, df_split["Target_ID"].values],
                    names=["Drug", "Target_ID"],
                )
                # .reindex returns NaN for pairs that were filtered out; drop them.
                hits = pair_to_rows.reindex(lookup_idx).dropna()
                # Each entry is a list of ints (collapsed duplicates); flatten.
                rows: list[int] = [int(i) for sub in hits.values for i in sub]
                split_indices[target_key] = rows

            out_path = splits_dir / f"{subset}_{method}_seed{seed}.pt"
            torch.save(split_indices, out_path)
            n_train = len(split_indices["train"])
            n_val = len(split_indices["val"])
            n_test = len(split_indices["test"])
            print(
                f"  [{subset}/{method}/seed={seed}] train={n_train:,} val={n_val:,}"
                f" test={n_test:,}  -> {out_path.name}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build per-subset TDC DTI evaluation CSVs and split index files."
    )
    parser.add_argument("--subsets", nargs="+", default=["DAVIS", "KIBA"], choices=SUBSETS)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Seeds for split generation (one .pt file per seed)."
    )
    parser.add_argument(
        "--protein-emb",
        default="graphium/data/dti/protein-esmc.parquet",
        help="Per-protein ESM-C embedding parquet (protein_id + sequence + feature_0..1151).",
    )
    parser.add_argument(
        "--output-dir", default="data/dti-eval",
        help="Destination directory for per-subset CSV + splits/.",
    )
    parser.add_argument(
        "--tdc-cache", default="data/tdc_cache",
        help="TDC download cache directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    DTI = _import_tdc_dti()
    protein_emb_df = _load_protein_embeddings(args.protein_emb)

    for subset in args.subsets:
        print(f"\n=== {subset} ===")
        tdc_data = DTI(name=subset, path=args.tdc_cache)
        full_df = tdc_data.get_data()
        # 'Target_ID' missing in some BindingDB variants — fall back to 'Target' (the seq).
        if "Target_ID" not in full_df.columns:
            sys.exit(f"ERROR: subset {subset} has no 'Target_ID' column; got {list(full_df.columns)}")

        filtered_df, label_col = _build_subset_csv(
            subset, tdc_data, full_df, protein_emb_df, output_dir,
        )
        _generate_splits(subset, tdc_data, filtered_df, args.methods, args.seeds, splits_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
