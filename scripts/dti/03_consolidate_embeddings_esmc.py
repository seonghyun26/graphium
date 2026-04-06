#!/usr/bin/env python
"""Stage 3 (ESM-C): Consolidate per-protein ESM-C embeddings into single files.

Reads individual .pt files produced by Stage 2 (ESM-C) -- each containing
{"mean_embedding": tensor([1152])} -- and consolidates them into a single
CSV, Parquet, and PyTorch file.

Output schema (protein-esmc.csv / protein-esmc.parquet):
    - protein_id                  : protein identifier string
    - sequence                    : full amino acid sequence
    - feature_0 .. feature_1151  : 1152-dim ESM-C 600M mean-pooled embedding

Output .pt format:
    {
        "protein_id": list[str],          # N protein IDs
        "smiles": list[str],              # N sequences (legacy key name)
        "embedding": np.ndarray[N, 1152]  # embedding matrix
    }

Usage:
    python 03_consolidate_embeddings_esmc.py \\
        --embedding-dir data/protein-esmc/embedding \\
        --protein-csv data/dti-scratch/protein-esm2.csv \\
        --output-dir graphium/data/dti
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


EMBEDDING_DIM = 1152


def load_single_embedding(pt_path: str) -> np.ndarray:
    """Load a single .pt file and extract the mean-pooled embedding vector.

    Expected file format: {"mean_embedding": tensor([1152])}
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    embedding = data["mean_embedding"]
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    return embedding.flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 (ESM-C): Consolidate per-protein ESM-C embeddings"
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/protein-esmc/embedding",
        help="Directory containing individual .pt files (default: data/protein-esmc/embedding)",
    )
    parser.add_argument(
        "--protein-csv",
        type=str,
        default="data/dti-scratch/protein-esm2.csv",
        help="CSV with protein_id and sequence columns (default: data/dti-scratch/protein-esm2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graphium/data/dti",
        help="Output directory (default: graphium/data/dti)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Stage 3 (ESM-C): Consolidating ESM-C embeddings")
    print("=" * 60)

    # Build protein_id -> sequence mapping from existing protein CSV
    print(f"  Loading protein sequences from {args.protein_csv} ...")
    df_prot = pd.read_csv(args.protein_csv, usecols=["protein_id", "sequence"])
    seq_map = (
        df_prot[["protein_id", "sequence"]]
        .drop_duplicates(subset=["protein_id"])
        .set_index("protein_id")["sequence"]
        .to_dict()
    )
    print(f"  Known proteins from DTI data: {len(seq_map):,}")

    # Find all .pt files
    pt_files = sorted(Path(args.embedding_dir).glob("*.pt"))
    print(f"  Found {len(pt_files):,} .pt files in {args.embedding_dir}")

    if len(pt_files) == 0:
        print("  ERROR: No .pt files found. Run Stage 2 (ESM-C) first.")
        sys.exit(1)

    # Load all embeddings
    protein_ids = []
    sequences = []
    embeddings = []
    skipped = 0

    for pt_path in tqdm(pt_files, desc="  Loading embeddings"):
        protein_id = pt_path.stem
        embedding = load_single_embedding(str(pt_path))

        if embedding.shape[0] != EMBEDDING_DIM:
            print(
                f"  WARNING: {protein_id} has dim {embedding.shape[0]}, "
                f"expected {EMBEDDING_DIM}. Skipping."
            )
            skipped += 1
            continue

        sequence = seq_map.get(protein_id, "")
        if not sequence:
            print(f"  WARNING: No sequence found for {protein_id} in DTI data.")

        protein_ids.append(protein_id)
        sequences.append(sequence)
        embeddings.append(embedding)

    if skipped > 0:
        print(f"  Skipped {skipped} proteins with wrong embedding dimension.")

    embeddings_arr = np.stack(embeddings, axis=0)
    print(
        f"  Consolidated: {len(protein_ids):,} proteins, "
        f"embedding shape: {embeddings_arr.shape}"
    )

    # Build DataFrame: protein_id, sequence, feature_0 .. feature_1151
    feature_cols = [f"feature_{i}" for i in range(EMBEDDING_DIM)]
    df = pd.DataFrame(embeddings_arr, columns=feature_cols)
    df.insert(0, "protein_id", protein_ids)
    df.insert(1, "sequence", sequences)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "protein-esmc.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV:     {csv_path}  ({len(df):,} rows x {len(df.columns):,} cols)")

    # Save Parquet (compressed, faster to read)
    parquet_path = os.path.join(args.output_dir, "protein-esmc.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved Parquet: {parquet_path}")

    # Save PyTorch format (legacy key names preserved)
    pt_out = {
        "protein_id": protein_ids,
        "smiles": sequences,  # legacy key name -- actually stores sequences
        "embedding": embeddings_arr,
    }
    pt_path = os.path.join(args.output_dir, "protein-esmc.pt")
    torch.save(pt_out, pt_path)
    print(f"  Saved PyTorch: {pt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
