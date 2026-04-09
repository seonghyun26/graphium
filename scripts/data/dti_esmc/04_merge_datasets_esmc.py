#!/usr/bin/env python
"""Stage 4 (ESM-C): Merge protein ESM-C embeddings with drug SMILES from TDC DTI data.

Joins the consolidated ESM-C protein embedding table (from Stage 3) with the
SMILES pairings from the existing ESM-2 merged dataset to create a dataset
where each row is a (protein_embedding, drug_SMILES) pair.

Join logic (many-to-many):
    protein-esmc  LEFT JOIN  tdcdti-esm2
    ON (protein_id, sequence)

Output schema (tdcdti-esmc.parquet):
    - protein_id                  : protein identifier
    - sequence                    : amino acid sequence
    - feature_0 .. feature_1151  : 1152-dim ESM-C embedding
    - SMILES                      : drug SMILES string

Usage:
    python 04_merge_datasets_esmc.py \\
        --embeddings graphium/data/dti/protein-esmc.csv \\
        --dti data/dti-scratch/tdcdti-esm2.parquet \\
        --output graphium/data/dti/tdcdti-esmc.parquet
"""

import argparse
import os

import pandas as pd


EMBEDDING_DIM = 1152


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 (ESM-C): Merge protein embeddings with drug SMILES"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="graphium/data/dti/protein-esmc.csv",
        help="Consolidated protein embeddings CSV from Stage 3 (default: graphium/data/dti/protein-esmc.csv)",
    )
    parser.add_argument(
        "--dti",
        type=str,
        default="data/dti-scratch/tdcdti-esm2.parquet",
        help="Existing DTI parquet with protein_id, sequence, SMILES columns (default: data/dti-scratch/tdcdti-esm2.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="graphium/data/dti/tdcdti-esmc.parquet",
        help="Output parquet path (default: graphium/data/dti/tdcdti-esmc.parquet)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Stage 4 (ESM-C): Merging protein embeddings with drug SMILES")
    print("=" * 60)

    # Load protein embeddings
    print(f"  Loading embeddings from {args.embeddings} ...")
    df_emb = pd.read_csv(args.embeddings)
    print(f"    Shape:    {df_emb.shape}")
    print(f"    Proteins: {df_emb['protein_id'].nunique():,}")

    # Load DTI interaction data
    # Supports two formats:
    #   1. Parquet with protein_id/sequence/SMILES (from ESM-2 merged pipeline)
    #   2. CSV with Target_ID/Target/Drug (raw TDC output from stage 1)
    print(f"  Loading DTI data from {args.dti} ...")
    if args.dti.endswith(".parquet"):
        df_dti = pd.read_parquet(args.dti, columns=["protein_id", "sequence", "SMILES"])
    else:
        df_raw = pd.read_csv(args.dti)
        df_dti = df_raw.rename(columns={
            "Target_ID": "protein_id",
            "Target": "sequence",
            "Drug": "SMILES",
        })[["protein_id", "sequence", "SMILES"]]
    print(f"    Shape:        {df_dti.shape}")
    print(f"    Unique drugs: {df_dti['SMILES'].nunique():,}")

    # Prepare mapping table: unique (protein_id, sequence, SMILES) triples
    map_df = df_dti[["protein_id", "sequence", "SMILES"]].drop_duplicates()
    print(f"    Unique (protein, drug) pairs: {len(map_df):,}")

    # Merge: join on (protein_id, sequence)
    print("\n  Merging on (protein_id, sequence) ...")
    df_out = df_emb.merge(
        map_df,
        on=["protein_id", "sequence"],
        how="left",
        validate="m:m",
    )

    # Remove rows with no SMILES match
    before = len(df_out)
    df_out = df_out.dropna(subset=["SMILES"]).reset_index(drop=True)
    dropped = before - len(df_out)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with no SMILES match")

    print(f"\n  Merged result:")
    print(f"    Total rows:      {len(df_out):,}")
    print(f"    Unique proteins: {df_out['protein_id'].nunique():,}")
    print(f"    Unique SMILES:   {df_out['SMILES'].nunique():,}")

    # Enforce column order: protein_id, sequence, feature_0..feature_1151, SMILES
    feature_cols = [f"feature_{i}" for i in range(EMBEDDING_DIM)]
    expected_cols = ["protein_id", "sequence"] + feature_cols + ["SMILES"]
    df_out = df_out[expected_cols]

    # Save as Parquet
    df_out.to_parquet(args.output, index=False)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Saved to {args.output}  ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
