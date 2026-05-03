#!/usr/bin/env python
"""Stage 7: Add a binary pActivity column to the DTI pActivity dataset.

Threshold-binarizes the z-scored `pY` column back to its raw pIC50 scale and
adds a `pY_bin` column (1 if pIC50 >= threshold, else 0). Updates the CSV and
parquet sibling in place. Idempotent: if `pY_bin` is already present and the
threshold matches the recorded one, the script is a no-op.

Why a binary head: lets us train a CE+label-smoothing head alongside the MAE
regression head, recovering hard-label robustness while keeping the rank signal.

Usage:
    python scripts/data/dti_esmc/07_add_pactivity_binary.py \
        --csv-path data/dti-processed/dti_pactivity_esmc_100k.csv \
        --norm-stats data/dti-processed/dti_pactivity_esmc_100k_norm_stats.pt \
        --threshold 7.0
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/dti-processed/dti_pactivity_esmc_100k.csv",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="data/dti-processed/dti_pactivity_esmc_100k_norm_stats.pt",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="pIC50 cutoff for the active class. Default: 7.0 (100 nM).",
    )
    parser.add_argument(
        "--bin-col",
        type=str,
        default="pY_bin",
        help="Output column name for the binary label.",
    )
    parser.add_argument(
        "--also-parquet",
        action="store_true",
        default=True,
        help="Also rewrite the parquet sibling next to the CSV.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(args.csv_path)
    if not os.path.exists(args.norm_stats):
        raise FileNotFoundError(args.norm_stats)

    stats = torch.load(args.norm_stats, weights_only=True)
    py_mean = float(stats["pY_mean"])
    py_std = float(stats["pY_std"])
    z_threshold = (args.threshold - py_mean) / py_std
    print(
        f"pY_mean={py_mean:.4f}  pY_std={py_std:.4f}  "
        f"pIC50>={args.threshold} -> z>={z_threshold:.4f}"
    )

    print(f"Loading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)
    print(f"  shape={df.shape}")

    if args.bin_col in df.columns:
        existing = df[args.bin_col].astype(np.int8)
        recomputed = (df["pY"].values >= z_threshold).astype(np.int8)
        if np.array_equal(existing.values, recomputed):
            print(f"  '{args.bin_col}' already present and matches threshold; nothing to do.")
            return
        print(f"  '{args.bin_col}' present but stale -> overwriting with new threshold.")

    df[args.bin_col] = (df["pY"].values >= z_threshold).astype(np.int8)
    n_pos = int(df[args.bin_col].sum())
    n_total = int(len(df))
    print(f"  positives: {n_pos} / {n_total} ({100.0 * n_pos / n_total:.2f}%)")

    print(f"Writing CSV back to {args.csv_path} ...")
    df.to_csv(args.csv_path, index=False)
    print(f"  size: {os.path.getsize(args.csv_path) / (1024 ** 2):.1f} MB")

    if args.also_parquet:
        parquet_path = os.path.splitext(args.csv_path)[0] + ".parquet"
        if os.path.exists(parquet_path):
            print(f"Writing parquet back to {parquet_path} ...")
            df.to_parquet(parquet_path, index=False)
            print(f"  size: {os.path.getsize(parquet_path) / (1024 ** 2):.1f} MB")
        else:
            print(f"  (no parquet sibling at {parquet_path}; skipping)")

    print("Done.")


if __name__ == "__main__":
    main()
