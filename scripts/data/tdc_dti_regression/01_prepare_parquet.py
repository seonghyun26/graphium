#!/usr/bin/env python
"""Build ``data/downstream/tdc_dti_regression/<subset>.parquet`` + splits.

ESM-2-powered version of ``scripts/data/dti_eval/01_prepare_dti_eval.py``:
same regression labels and split logic, but protein features come from the
ESM-2 parquet produced by ``scripts/data/dti_esm2/`` / ``02_consolidate_esm2.py``
(dim inferred from ``feature_*`` columns — 2560 for the 3B model).

Subsets / methods:
  - DAVIS, KIBA, BindingDB_{Kd,Ki,IC50}: random / cold_target, per-seed
  - BindingDB_Patent_DG: fixed temporal split from dti_dg_group (per-seed
    train/val rotation only)
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SUBSETS = [
    "DAVIS", "KIBA",
    "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50",
    "BindingDB_Patent_DG",
]
METHODS = ["random", "cold_target", "temporal"]
DG_SUBSET_TO_BENCHMARK = {"BindingDB_Patent_DG": "bindingdb_patent"}


def _import_tdc_dti():
    sys.modules.setdefault("gget", types.ModuleType("gget"))
    try:
        from tdc.multi_pred import DTI
    except ImportError:
        sys.exit("ERROR: PyTDC required. `pip install PyTDC`.")
    return DTI


def _import_benchmark_group():
    sys.modules.setdefault("gget", types.ModuleType("gget"))
    try:
        from tdc import BenchmarkGroup
    except ImportError:
        sys.exit("ERROR: PyTDC required. `pip install PyTDC`.")
    return BenchmarkGroup


def _load_protein_embeddings(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(path)
    feat_cols = sorted(
        (c for c in df.columns if c.startswith("feature_")),
        key=lambda c: int(c.split("_")[1]),
    )
    if not feat_cols:
        sys.exit(f"ERROR: {path} has no feature_* columns.")
    prot_cols = [f"prot_emb_{i}" for i in range(len(feat_cols))]
    df = df.set_index("protein_id")[feat_cols]
    print(f"Loaded {len(df):,} proteins × {len(feat_cols)}-d from {path}")
    return df, prot_cols


def _build_subset_csv(
    subset: str, full_df: pd.DataFrame,
    prot_df: pd.DataFrame, prot_cols: list[str], output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    label_col = "kiba_score" if subset == "KIBA" else "pY"
    n_before = len(full_df)
    keep = full_df["Target_ID"].isin(prot_df.index)
    if subset != "KIBA":
        keep &= full_df["Y"] > 0
    full_df = full_df[keep].reset_index(drop=True)
    full_df["row_idx"] = np.arange(len(full_df), dtype=np.int64)
    print(f"  [{subset}] {n_before:,} -> {len(full_df):,} rows kept")

    y = full_df["Y"].values.astype(np.float64)
    labels = (y.astype(np.float32) if subset == "KIBA"
              else (9.0 - np.log10(y)).clip(0.0, 14.0).astype(np.float32))
    mu, sd = float(labels.mean()), float(max(labels.std(), 1e-8))
    labels_norm = ((labels - mu) / sd).astype(np.float32)

    prot_emb = prot_df.loc[full_df["Target_ID"].values].values.astype(np.float32)
    p_mu = prot_emb.mean(axis=0).astype(np.float32)
    p_sd = prot_emb.std(axis=0)
    p_sd = np.where(p_sd < 1e-8, 1.0, p_sd).astype(np.float32)
    prot_norm = ((prot_emb - p_mu) / p_sd).astype(np.float32)

    out = pd.DataFrame({"SMILES_nometa": full_df["Drug"].values})
    out[label_col] = labels_norm
    out[prot_cols] = prot_norm
    parquet_path = output_dir / f"{subset}.parquet"
    out.to_parquet(parquet_path, index=False)
    size = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  [{subset}] saved {parquet_path} ({size:.1f} MB)")

    torch.save(
        {"label_col": label_col,
         "label_mean": torch.tensor(mu), "label_std": torch.tensor(sd),
         "prot_emb_mean": torch.from_numpy(p_mu), "prot_emb_std": torch.from_numpy(p_sd),
         "esm2_dim": len(prot_cols)},
        output_dir / f"{subset}_norm_stats.pt",
    )
    return full_df, label_col


def _generate_splits(
    subset: str, tdc_data, full_df: pd.DataFrame,
    methods: list[str], seeds: list[int], splits_dir: Path,
) -> None:
    pair_to_row = pd.Series(
        full_df["row_idx"].values,
        index=pd.MultiIndex.from_arrays(
            [full_df["Drug"].values, full_df["Target_ID"].values],
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

    for method in methods:
        for seed in seeds:
            if method == "random":
                splits_raw = tdc_data.get_split(method="random", seed=seed, frac=[0.7, 0.1, 0.2])
            elif method == "cold_target":
                splits_raw = tdc_data.get_split(
                    method="cold_split", column_name="Target", seed=seed, frac=[0.7, 0.1, 0.2],
                )
            else:
                raise ValueError(method)
            key_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
            indices = {"train": [], "val": [], "test": []}
            for raw_key, df_split in splits_raw.items():
                if raw_key not in key_map:
                    continue
                indices[key_map[raw_key]] = _resolve(df_split)
            path = splits_dir / f"{subset}_{method}_seed{seed}.pt"
            torch.save(indices, path)
            print(f"  [{subset}/{method}/seed={seed}] "
                  f"train={len(indices['train']):,} val={len(indices['val']):,} test={len(indices['test']):,}")


def _build_dtidg(
    subset: str, prot_df: pd.DataFrame, prot_cols: list[str],
    seeds: list[int], output_dir: Path, splits_dir: Path, cache_dir: str,
) -> None:
    BenchmarkGroup = _import_benchmark_group()
    bench = DG_SUBSET_TO_BENCHMARK[subset]
    group = BenchmarkGroup(name="dti_dg_group", path=cache_dir)
    b = group.get(bench)
    full_df = pd.concat([b["train_val"], b["test"]], ignore_index=True)

    filtered, _ = _build_subset_csv(subset, full_df, prot_df, prot_cols, output_dir)
    pair_to_row = pd.Series(
        filtered["row_idx"].values,
        index=pd.MultiIndex.from_arrays(
            [filtered["Drug"].values, filtered["Target_ID"].values],
            names=["Drug", "Target_ID"],
        ),
    )
    pair_to_rows = pair_to_row.groupby(level=[0, 1]).apply(list)

    def _resolve(df: pd.DataFrame) -> list[int]:
        idx = pd.MultiIndex.from_arrays(
            [df["Drug"].values, df["Target_ID"].values], names=["Drug", "Target_ID"])
        hits = pair_to_rows.reindex(idx).dropna()
        return [int(i) for sub in hits.values for i in sub]

    test_idx = _resolve(b["test"])  # fixed across seeds
    for seed in seeds:
        tr, va = group.get_train_valid_split(benchmark=bench, split_type="default", seed=seed)
        path = splits_dir / f"{subset}_temporal_seed{seed}.pt"
        torch.save({"train": _resolve(tr), "val": _resolve(va), "test": test_idx}, path)
        print(f"  [{subset}/temporal/seed={seed}] wrote {path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--protein-emb", required=True, help="Consolidated ESM-2 parquet")
    p.add_argument("--output-dir", default="data/downstream/tdc_dti_regression")
    p.add_argument("--subsets", nargs="+", default=["DAVIS", "KIBA"], choices=SUBSETS)
    p.add_argument("--methods", nargs="+", default=["random", "cold_target"], choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--tdc-cache", default="data/tdc_cache")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    dg = [s for s in args.subsets if s in DG_SUBSET_TO_BENCHMARK]
    regular = [s for s in args.subsets if s not in DG_SUBSET_TO_BENCHMARK]
    reg_methods = [m for m in args.methods if m != "temporal"]
    if dg and (set(args.methods) - {"temporal"}):
        sys.exit("ERROR: DG subsets support only method 'temporal'; run them separately.")
    if "temporal" in args.methods and not dg:
        sys.exit("ERROR: 'temporal' requires a DG subset (BindingDB_Patent_DG).")

    prot_df, prot_cols = _load_protein_embeddings(args.protein_emb)

    if regular:
        DTI = _import_tdc_dti()
        for subset in regular:
            print(f"\n=== {subset} ===")
            tdc_data = DTI(name=subset, path=args.tdc_cache)
            full = tdc_data.get_data()
            if "Target_ID" not in full.columns:
                sys.exit(f"ERROR: {subset} missing Target_ID column")
            filtered, _ = _build_subset_csv(subset, full, prot_df, prot_cols, output_dir)
            _generate_splits(subset, tdc_data, filtered, reg_methods, args.seeds, splits_dir)

    for subset in dg:
        print(f"\n=== {subset} (dti_dg_group) ===")
        _build_dtidg(subset, prot_df, prot_cols, args.seeds, output_dir, splits_dir, args.tdc_cache)

    print("\nDone.")


if __name__ == "__main__":
    main()
