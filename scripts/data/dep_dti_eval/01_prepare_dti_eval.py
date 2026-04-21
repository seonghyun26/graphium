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


SUBSETS = [
    "DAVIS", "KIBA", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50",
    "BindingDB_Patent_DG",   # special: uses tdc.BenchmarkGroup('dti_dg_group') with fixed temporal split
]
METHODS = ["random", "cold_target", "temporal"]
ESMC_DIM = 1152
PROT_EMB_COLS = [f"prot_emb_{i}" for i in range(ESMC_DIM)]

# Subsets that go through the DTI-DG benchmark-group path (temporal-split leaderboard).
# These do NOT use multi_pred.DTI; they use tdc.BenchmarkGroup('dti_dg_group').
DG_SUBSET_TO_BENCHMARK = {
    "BindingDB_Patent_DG": "bindingdb_patent",
}


def _import_tdc_dti():
    """Import TDC's DTI loader, working around gget's Python 3.10+ syntax on 3.9."""
    sys.modules.setdefault("gget", types.ModuleType("gget"))
    try:
        from tdc.multi_pred import DTI
    except ImportError:
        sys.exit("ERROR: PyTDC required. `pip install PyTDC`.")
    return DTI


def _import_tdc_benchmark_group():
    """Import TDC's BenchmarkGroup (used for the DTI-DG temporal-split leaderboard)."""
    sys.modules.setdefault("gget", types.ModuleType("gget"))
    try:
        from tdc import BenchmarkGroup
    except ImportError:
        sys.exit("ERROR: PyTDC required. `pip install PyTDC`.")
    return BenchmarkGroup


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

    # Parquet instead of CSV: columnar + compressed -> 5-10x smaller, 10x faster
    # to write and read at the 1M-row scale of BindingDB_Patent. Eval scripts
    # use pd.read_parquet(columns=...) for the same projection semantics.
    parquet_path = output_dir / f"{subset}.parquet"
    out.to_parquet(parquet_path, index=False)
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  [{subset}] saved parquet: {parquet_path} ({len(out):,} rows, {size_mb:.1f} MB)")

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


def _build_dtidg_subset_and_splits(
    subset: str,
    protein_emb_df: pd.DataFrame,
    seeds: list[int],
    output_dir: Path,
    splits_dir: Path,
    cache_dir: str,
) -> None:
    """Build CSV + per-seed temporal splits for the DTI-DG (BindingDB_Patent) leaderboard.

    Test split is fixed (temporal: train on early-year patents, test on later years);
    only the train/valid sub-split varies with seed. CSV is built from train_val + test
    concatenated so we can index both via row indices in one file.
    """
    BenchmarkGroup = _import_tdc_benchmark_group()
    bench_name = DG_SUBSET_TO_BENCHMARK[subset]
    print(f"  [{subset}] loading dti_dg_group::{bench_name} (cache={cache_dir})")
    group = BenchmarkGroup(name="dti_dg_group", path=cache_dir)
    benchmark = group.get(bench_name)
    train_val = benchmark["train_val"]
    test = benchmark["test"]
    full_df = pd.concat([train_val, test], ignore_index=True)
    n_train_val = len(train_val)  # boundary index between train_val and test
    print(f"  [{subset}] raw: train_val={len(train_val):,}  test={len(test):,}")

    if "Target_ID" not in full_df.columns:
        sys.exit(f"ERROR: dti_dg_group::{bench_name} has no 'Target_ID' column; got {list(full_df.columns)}")

    # Reuse the regular CSV-builder (filter by protein, pY conversion, z-score). It
    # appends a fresh row_idx column to the returned filtered_df we'll use for splits.
    # Pass tdc_data=None — _build_subset_csv only uses it for diagnostic prints.
    filtered_df, label_col = _build_subset_csv(
        subset, None, full_df, protein_emb_df, output_dir,
    )

    # Pair-to-row-index lookup, identical pattern to _generate_splits.
    pair_to_row = pd.Series(
        filtered_df["row_idx"].values,
        index=pd.MultiIndex.from_arrays(
            [filtered_df["Drug"].values, filtered_df["Target_ID"].values],
            names=["Drug", "Target_ID"],
        ),
    )
    pair_to_rows = pair_to_row.groupby(level=[0, 1]).apply(list)

    def _resolve(df_split: pd.DataFrame) -> list[int]:
        idx = pd.MultiIndex.from_arrays(
            [df_split["Drug"].values, df_split["Target_ID"].values],
            names=["Drug", "Target_ID"],
        )
        hits = pair_to_rows.reindex(idx).dropna()
        return [int(i) for sub in hits.values for i in sub]

    # The temporal test split is fixed across seeds.
    test_indices = _resolve(test)

    for seed in seeds:
        train_df, val_df = group.get_train_valid_split(
            benchmark=bench_name, split_type="default", seed=seed,
        )
        split_indices = {
            "train": _resolve(train_df),
            "val": _resolve(val_df),
            "test": test_indices,
        }
        out_path = splits_dir / f"{subset}_temporal_seed{seed}.pt"
        torch.save(split_indices, out_path)
        print(
            f"  [{subset}/temporal/seed={seed}] train={len(split_indices['train']):,}"
            f" val={len(split_indices['val']):,} test={len(split_indices['test']):,}"
            f"  -> {out_path.name}"
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

    protein_emb_df = _load_protein_embeddings(args.protein_emb)

    # Validate method/subset compatibility: temporal only with DG subsets, and DG
    # subsets only with temporal (they don't have a multi_pred.DTI loader path).
    dg_subsets = [s for s in args.subsets if s in DG_SUBSET_TO_BENCHMARK]
    regular_subsets = [s for s in args.subsets if s not in DG_SUBSET_TO_BENCHMARK]
    regular_methods = [m for m in args.methods if m != "temporal"]
    if dg_subsets and (set(args.methods) - {"temporal"}):
        sys.exit(
            f"ERROR: DG subsets {dg_subsets} only support method 'temporal'; "
            f"got {args.methods}. Run them in a separate invocation."
        )
    if "temporal" in args.methods and not dg_subsets:
        sys.exit(
            "ERROR: 'temporal' method requires a DG subset (e.g. BindingDB_Patent_DG)."
        )

    # ── Regular DTI subsets (random / cold_target) ────────────────────────────
    if regular_subsets:
        DTI = _import_tdc_dti()
        for subset in regular_subsets:
            print(f"\n=== {subset} ===")
            tdc_data = DTI(name=subset, path=args.tdc_cache)
            full_df = tdc_data.get_data()
            if "Target_ID" not in full_df.columns:
                sys.exit(f"ERROR: subset {subset} has no 'Target_ID' column; got {list(full_df.columns)}")
            filtered_df, _ = _build_subset_csv(
                subset, tdc_data, full_df, protein_emb_df, output_dir,
            )
            _generate_splits(subset, tdc_data, filtered_df, regular_methods, args.seeds, splits_dir)

    # ── DTI-DG subsets (fixed temporal split, per-seed train/val rotation) ────
    for subset in dg_subsets:
        print(f"\n=== {subset} (dti_dg_group, temporal) ===")
        _build_dtidg_subset_and_splits(
            subset, protein_emb_df, args.seeds, output_dir, splits_dir, args.tdc_cache,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
