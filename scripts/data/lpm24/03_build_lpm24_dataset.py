#!/usr/bin/env python
"""Stage 3 (LPM-24): Build the final LPM-24 dataset for Graphium training.

Takes the text-encoder embeddings from Stage 2 and builds the final CSV/parquet
files for use as a Graphium pre-training task. The embedding dimension is
auto-detected from the input, so the same script works for PubMedBERT (768-d),
Galactica-125m (768-d), BioLinkBERT-large (1024-d), or any other Stage-2 output.

Normalization:
    Z-score: Per-feature standardization (mean=0, std=1 per column).
             The mean/std vectors are saved separately for denormalization.

Output files (in --final-dir):
    - lpm24_<model>.csv              : SMILES_nometa + D feature columns
    - lpm24_<model>.parquet          : same, compressed
    - lpm24_<model>_norm_stats.pt    : mean/std vectors for denormalization

Output schema:
    SMILES_nometa, feature_0, feature_1, ..., feature_{D-1}

Typical output: 160,560 rows x (1 + D) columns.

Usage:
    python 03_build_lpm24_dataset.py --model galactica
    python 03_build_lpm24_dataset.py --model biolinkbert
    python 03_build_lpm24_dataset.py --model pubmedbert
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch


VALID_MODELS = ("pubmedbert", "galactica", "biolinkbert")


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
        "--model",
        type=str,
        required=True,
        choices=VALID_MODELS,
        help="Which text encoder's embeddings to consume",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Embeddings .pt from Stage 2. "
            "Default: data/lpm24/lpm24_embeddings_<model>.pt"
        ),
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

    # Resolve input path.  Fall back to the legacy PubMedBERT filename
    # (lpm24_embeddings.pt) if the new per-model file doesn't exist.
    if args.input is None:
        per_model = f"data/lpm24/lpm24_embeddings_{args.model}.pt"
        legacy = "data/lpm24/lpm24_embeddings.pt"
        if os.path.exists(per_model):
            input_path = per_model
        elif args.model == "pubmedbert" and os.path.exists(legacy):
            input_path = legacy
            print(f"  Note: using legacy path {legacy} for pubmedbert.")
        else:
            raise FileNotFoundError(
                f"Could not find embeddings for '{args.model}'. "
                f"Expected: {per_model}"
            )
    else:
        input_path = args.input

    print(f"Stage 3 (LPM-24): Building final dataset ({args.model})")
    print("=" * 60)

    # Load embeddings
    print(f"  Loading {input_path} ...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"].numpy().astype(np.float32)
    smiles = data["smiles"]

    embedding_dim = embeddings.shape[1]
    feature_cols = [f"feature_{i}" for i in range(embedding_dim)]

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embedding dim:    {embedding_dim}")
    print(f"  Unique SMILES:    {len(set(smiles)):,}")
    if "model_id" in data:
        print(f"  Source model:     {data['model_id']}")

    # Z-score normalization
    print(f"\n  Applying Z-score normalization ...")
    embeddings_norm, mean_vec, std_vec = normalize_zscore(embeddings)
    print(f"  Per-feature mean after norm: {embeddings_norm.mean(axis=0).mean():.2e}")
    print(f"  Per-feature std after norm:  {embeddings_norm.std(axis=0).mean():.4f}")

    # Build DataFrame
    df = pd.DataFrame(embeddings_norm, columns=feature_cols)
    df.insert(0, "SMILES_nometa", smiles)

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Save CSV
    stem = f"lpm24_{args.model}"
    csv_path = os.path.join(args.final_dir, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\n  Saved CSV:     {csv_path}  ({len(df):,} rows, {size_mb:.1f} MB)")

    # Save Parquet
    parquet_path = os.path.join(args.final_dir, f"{stem}.parquet")
    df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Saved Parquet: {parquet_path}  ({size_mb:.1f} MB)")

    # Save normalization stats
    stats_path = os.path.join(args.final_dir, f"{stem}_norm_stats.pt")
    torch.save(
        {"mean": torch.FloatTensor(mean_vec), "std": torch.FloatTensor(std_vec)},
        stats_path,
    )
    size_kb = os.path.getsize(stats_path) / 1024
    print(f"  Saved stats:   {stats_path}  ({size_kb:.1f} KB)")

    # Verification
    print(f"\n  Verification:")
    df_check = pd.read_parquet(parquet_path)
    feat_vals = df_check[feature_cols].values
    print(f"    Shape:      {df_check.shape}")
    print(f"    Mean range: [{feat_vals.mean(axis=0).min():.4f}, {feat_vals.mean(axis=0).max():.4f}]")
    print(f"    Std range:  [{feat_vals.std(axis=0).min():.4f}, {feat_vals.std(axis=0).max():.4f}]")
    print(f"    SMILES col: {df_check['SMILES_nometa'].nunique():,} unique")

    print("\nDone.")


if __name__ == "__main__":
    main()
