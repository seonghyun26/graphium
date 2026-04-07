#!/usr/bin/env python
"""Stage 3 (LPM-24): Build the final LPM-24 dataset for Graphium training.

Takes the PubMedBERT embeddings from Stage 2 and builds the final CSV/parquet
files for use as a Graphium pre-training task.

Normalization:
    Z-score: Per-feature standardization (mean=0, std=1 per column).
             The mean/std vectors are saved separately for denormalization.

Output files (in --final-dir):
    - lpm24_pubmedbert.csv               : SMILES_nometa + 768 feature columns
    - lpm24_pubmedbert.parquet            : same, compressed
    - lpm24_pubmedbert_norm_stats.pt      : mean/std vectors for denormalization

Output schema:
    SMILES_nometa, feature_0, feature_1, ..., feature_767

Typical output: 160,560 rows x 769 columns.

Usage:
    python 03_build_lpm24_dataset.py \\
        --input data/lpm24/lpm24_embeddings.pt \\
        --final-dir data/dti-processed \\
        --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch


EMBEDDING_DIM = 768
FEATURE_COLS = [f"feature_{i}" for i in range(EMBEDDING_DIM)]


def normalize_zscore(embeddings):
    """Z-score normalize per feature (mean=0, std=1 per column).

    Returns:
        (normalized_embeddings, mean_vector, std_vector)
    """
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    # Guard against zero-std features
    std[std < 1e-8] = 1.0
    normalized = (embeddings - mean) / std
    return normalized, mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 (LPM-24): Build final dataset for Graphium"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/lpm24/lpm24_embeddings.pt",
        help="Embeddings .pt from Stage 2 (default: data/lpm24/lpm24_embeddings.pt)",
    )
    parser.add_argument(
        "--final-dir",
        type=str,
        default="data/dti-processed",
        help="Final output directory (default: data/dti-processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    os.makedirs(args.final_dir, exist_ok=True)

    print("Stage 3 (LPM-24): Building final dataset")
    print("=" * 60)

    # Load embeddings
    print(f"  Loading {args.input} ...")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"].numpy().astype(np.float32)
    smiles = data["smiles"]

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique SMILES:    {len(set(smiles)):,}")

    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"Expected dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
    )

    # Z-score normalization
    print(f"\n  Applying Z-score normalization ...")
    embeddings_norm, mean_vec, std_vec = normalize_zscore(embeddings)
    print(f"  Per-feature mean after norm: {embeddings_norm.mean(axis=0).mean():.2e}")
    print(f"  Per-feature std after norm:  {embeddings_norm.std(axis=0).mean():.4f}")

    # Build DataFrame
    df = pd.DataFrame(embeddings_norm, columns=FEATURE_COLS)
    df.insert(0, "SMILES_nometa", smiles)

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Save CSV
    csv_path = os.path.join(args.final_dir, "lpm24_pubmedbert.csv")
    df.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\n  Saved CSV:     {csv_path}  ({len(df):,} rows, {size_mb:.1f} MB)")

    # Save Parquet
    parquet_path = os.path.join(args.final_dir, "lpm24_pubmedbert.parquet")
    df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Saved Parquet: {parquet_path}  ({size_mb:.1f} MB)")

    # Save normalization stats
    stats_path = os.path.join(args.final_dir, "lpm24_pubmedbert_norm_stats.pt")
    torch.save(
        {"mean": torch.FloatTensor(mean_vec), "std": torch.FloatTensor(std_vec)},
        stats_path,
    )
    size_kb = os.path.getsize(stats_path) / 1024
    print(f"  Saved stats:   {stats_path}  ({size_kb:.1f} KB)")

    # Verification
    print(f"\n  Verification:")
    df_check = pd.read_parquet(parquet_path)
    feat_vals = df_check[FEATURE_COLS].values
    print(f"    Shape:      {df_check.shape}")
    print(f"    Mean range: [{feat_vals.mean(axis=0).min():.4f}, {feat_vals.mean(axis=0).max():.4f}]")
    print(f"    Std range:  [{feat_vals.std(axis=0).min():.4f}, {feat_vals.std(axis=0).max():.4f}]")
    print(f"    SMILES col: {df_check['SMILES_nometa'].nunique():,} unique")

    print("\nDone.")


if __name__ == "__main__":
    main()
