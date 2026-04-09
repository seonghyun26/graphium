#!/usr/bin/env python
"""Stage 6: Prepare DTI pActivity dataset for protein-supervised pre-training.

Merges the filtered ESMC protein embeddings with raw TDC binding affinity (Y)
labels to create a training-ready CSV where:
  - Column 0: SMILES_nometa (molecule SMILES, used for graph featurization)
  - Column 1: pY (pActivity = -log10(Y_nM * 1e-9), z-score normalized)
  - Columns 2+: prot_emb_0 .. prot_emb_1151 (ESMC 600M embeddings, z-score normalized)

The model consumes this as:  MLP(GNN(molecule) || prot_emb) -> pY

pActivity conversion:
  pY = -log10(Y_nM * 1e-9) = 9 - log10(Y_nM)
  Higher pY = stronger binding.  Y=1nM -> pY=9, Y=1uM -> pY=6, Y=10uM -> pY=5.

KIBA scores are excluded (different scale, not nM).

Normalization: both pY and prot_emb are z-score normalized independently.
Stats are saved for optional denormalization.

Usage:
    python 06_prepare_dti_pactivity.py \\
        --filtered-parquet data/dti-scratch-v2/tdcdti-esmc-filtered.parquet \\
        --raw-csv data/dti-scratch-v2/protein-drug-trainonly.csv \\
        --output-dir data/dti-processed \\
        --output-prefix dti_pactivity_esmc_100k
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch


ESMC_DIM = 1152
PROT_EMB_COLS = [f"prot_emb_{i}" for i in range(ESMC_DIM)]


def load_and_merge(filtered_path: str, raw_csv_path: str) -> pd.DataFrame:
    """Load filtered ESMC data and merge with raw TDC data to recover Y labels."""
    print("Loading filtered ESMC parquet...")
    filt = pd.read_parquet(filtered_path)
    print(f"  Shape: {filt.shape}, unique proteins: {filt['protein_id'].nunique()}")

    print("Loading raw TDC CSV...")
    raw = pd.read_csv(raw_csv_path, low_memory=False)
    print(f"  Shape: {raw.shape}")

    # Exclude KIBA (different scale, not nM)
    n_before = len(raw)
    raw = raw[raw["dti_dataset"] != "kiba"].copy()
    print(f"  Excluded KIBA: {n_before} -> {len(raw)} rows")

    # Merge on (SMILES, protein_id) = (Drug, Target_ID)
    merged = filt.merge(
        raw[["Drug", "Target_ID", "Y", "dti_dataset"]],
        left_on=["SMILES", "protein_id"],
        right_on=["Drug", "Target_ID"],
        how="inner",
    )
    print(f"  Merged: {len(merged)} rows (from {len(filt)} filtered)")

    # Some (SMILES, protein) pairs appear in multiple datasets with different Y.
    # Take the median Y per (SMILES, protein_id) pair.
    n_before = len(merged)
    merged = merged.groupby(["SMILES", "protein_id"], as_index=False).agg(
        {
            "Y": "median",
            **{f"feature_{i}": "first" for i in range(ESMC_DIM)},
        }
    )
    print(f"  After median aggregation: {n_before} -> {len(merged)} unique pairs")

    return merged


def compute_pactivity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pActivity = -log10(Y_nM * 1e-9) = 9 - log10(Y_nM).

    Filters out invalid Y values (Y <= 0).
    Clips pY to [0, 14] range (Y from 100M nM to sub-femtomolar).
    """
    n_before = len(df)
    df = df[df["Y"] > 0].copy()
    print(f"  Filtered Y<=0: {n_before} -> {len(df)} rows")

    df["pY"] = 9.0 - np.log10(df["Y"].values)
    df["pY"] = df["pY"].clip(0.0, 14.0)

    print(f"  pY stats: min={df['pY'].min():.2f}, max={df['pY'].max():.2f}, "
          f"mean={df['pY'].mean():.2f}, std={df['pY'].std():.2f}")

    return df


def normalize_and_save(
    df: pd.DataFrame,
    output_dir: str,
    output_prefix: str,
    seed: int,
) -> None:
    """Z-score normalize pY and prot_emb, save CSV + stats."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    # --- Normalize pY ---
    pY = df["pY"].values.astype(np.float32)
    pY_mean, pY_std = pY.mean(), pY.std()
    pY_norm = (pY - pY_mean) / pY_std
    print(f"  pY normalization: mean={pY_mean:.4f}, std={pY_std:.4f}")

    # --- Normalize protein embeddings ---
    feature_cols = [f"feature_{i}" for i in range(ESMC_DIM)]
    prot_emb = df[feature_cols].values.astype(np.float32)
    prot_mean = prot_emb.mean(axis=0)
    prot_std = prot_emb.std(axis=0)
    prot_emb_norm = (prot_emb - prot_mean) / prot_std
    print(f"  prot_emb normalization: per-feature mean of normalized={prot_emb_norm.mean(axis=0).mean():.6e}")

    # --- Build output DataFrame ---
    out_df = pd.DataFrame({"SMILES_nometa": df["SMILES"].values})
    out_df["pY"] = pY_norm
    for i, col in enumerate(PROT_EMB_COLS):
        out_df[col] = prot_emb_norm[:, i]

    # --- Save CSV ---
    csv_path = os.path.join(output_dir, f"{output_prefix}.csv")
    out_df.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  Saved CSV: {csv_path}  ({len(out_df)} rows, {size_mb:.1f} MB)")

    # --- Save parquet ---
    parquet_path = os.path.join(output_dir, f"{output_prefix}.parquet")
    out_df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Saved parquet: {parquet_path}  ({len(out_df)} rows, {size_mb:.1f} MB)")

    # --- Save normalization stats ---
    stats_path = os.path.join(output_dir, f"{output_prefix}_norm_stats.pt")
    torch.save(
        {
            "pY_mean": torch.tensor(pY_mean),
            "pY_std": torch.tensor(pY_std),
            "prot_emb_mean": torch.FloatTensor(prot_mean),
            "prot_emb_std": torch.FloatTensor(prot_std),
        },
        stats_path,
    )
    print(f"  Saved norm stats: {stats_path}")

    # --- Verify ---
    print("\nVerification:")
    df_check = pd.read_csv(csv_path, nrows=5)
    print(f"  Columns: {df_check.columns.tolist()[:5]} ... ({len(df_check.columns)} total)")
    print(f"  Expected columns: 1 (SMILES) + 1 (pY) + {ESMC_DIM} (prot_emb) = {2 + ESMC_DIM}")
    assert len(out_df.columns) == 2 + ESMC_DIM, f"Column count mismatch: {len(out_df.columns)}"
    print(f"  Total rows: {len(out_df)}")
    print(f"  Unique SMILES: {out_df['SMILES_nometa'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Prepare DTI pActivity dataset for protein-supervised pre-training"
    )
    parser.add_argument(
        "--filtered-parquet",
        type=str,
        default="data/dti-scratch-v2/tdcdti-esmc-filtered.parquet",
        help="Filtered ESMC parquet from Stage 5 (default: data/dti-scratch-v2/tdcdti-esmc-filtered.parquet)",
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="data/dti-scratch-v2/protein-drug-trainonly.csv",
        help="Raw TDC CSV with Y labels (default: data/dti-scratch-v2/protein-drug-trainonly.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dti-processed",
        help="Output directory (default: data/dti-processed)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dti_pactivity_esmc_100k",
        help="Output filename prefix (default: dti_pactivity_esmc_100k)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("Stage 6: Prepare DTI pActivity dataset")
    print("=" * 60)

    print("\nStep 1: Load and merge data")
    df = load_and_merge(args.filtered_parquet, args.raw_csv)

    print("\nStep 2: Compute pActivity")
    df = compute_pactivity(df)

    print("\nStep 3: Normalize and save")
    normalize_and_save(df, args.output_dir, args.output_prefix, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
