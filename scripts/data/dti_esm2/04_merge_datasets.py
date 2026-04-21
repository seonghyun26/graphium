#!/usr/bin/env python
"""Stage 4: Merge protein ESM2 embeddings with drug SMILES from TDC DTI data.

Joins the consolidated protein embedding table (from Stage 3) with the
protein-drug interaction table (from Stage 1) to create a dataset where
each row is a (protein_embedding, drug_SMILES) pair.

Join logic (many-to-many):
    protein-esm2  LEFT JOIN  protein-drug
    ON (protein_id = Target_ID) AND (sequence = Target)

One protein can appear hundreds or thousands of times (once per drug partner).

Output schema (tdcdti-esm2.parquet):
    - protein_id                  : protein identifier
    - sequence                    : amino acid sequence
    - feature_0 .. feature_2559  : 2560-dim ESM2 embedding
    - SMILES                      : drug SMILES string

Typical output: ~1,620,095 rows x 2,563 columns, 764,618 unique SMILES.

Usage:
    python 04_merge_datasets.py \\
        --embeddings graphium/data/dti/protein-esm2.csv \\
        --dti data/protein-drug.csv \\
        --output graphium/data/dti/tdcdti-esm2.parquet
"""

import argparse
import os
import sys

import pandas as pd


EMBEDDING_DIM = 2560


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Merge protein embeddings with drug SMILES"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="graphium/data/dti/protein-esm2.csv",
        help="Consolidated protein embeddings CSV from Stage 3 (default: graphium/data/dti/protein-esm2.csv)",
    )
    parser.add_argument(
        "--dti",
        type=str,
        default="data/protein-drug.csv",
        help="DTI data CSV from Stage 1 (default: data/protein-drug.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="graphium/data/dti/tdcdti-esm2.parquet",
        help="Output parquet path (default: graphium/data/dti/tdcdti-esm2.parquet)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Stage 4: Merging protein embeddings with drug SMILES")
    print("=" * 60)

    # Load protein embeddings (5,222 proteins x 2,562 cols)
    print(f"  Loading embeddings from {args.embeddings} ...")
    df_emb = pd.read_csv(args.embeddings)
    print(f"    Shape:    {df_emb.shape}")
    print(f"    Proteins: {df_emb['protein_id'].nunique():,}")

    # Load DTI interaction data (~1.79M rows)
    print(f"  Loading DTI data from {args.dti} ...")
    df_dti = pd.read_csv(args.dti)
    print(f"    Shape:        {df_dti.shape}")
    print(f"    Unique drugs: {df_dti['Drug'].nunique():,}")

    # Prepare mapping table: (Target_ID, Target, SMILES)
    # Rename Drug -> SMILES to match the output schema
    map_df = df_dti[["Target_ID", "Target", "Drug"]].copy()
    map_df = map_df.rename(columns={"Drug": "SMILES"})
    map_df = map_df.drop_duplicates()
    print(f"    Unique (protein, drug) pairs: {len(map_df):,}")

    # Merge: join on (protein_id = Target_ID) AND (sequence = Target)
    # This is a many-to-many join: 1 protein -> many drugs
    print("\n  Merging on (protein_id, sequence) = (Target_ID, Target) ...")
    df_out = df_emb.merge(
        map_df,
        left_on=["protein_id", "sequence"],
        right_on=["Target_ID", "Target"],
        how="left",
        validate="m:m",
    )

    # Drop redundant join-key columns from the right side
    df_out = df_out.drop(columns=["Target_ID", "Target"], errors="ignore")

    # Remove rows with no SMILES match (proteins not found in DTI data)
    before = len(df_out)
    df_out = df_out.dropna(subset=["SMILES"]).reset_index(drop=True)
    dropped = before - len(df_out)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with no SMILES match")

    print(f"\n  Merged result:")
    print(f"    Total rows:      {len(df_out):,}")
    print(f"    Unique proteins: {df_out['protein_id'].nunique():,}")
    print(f"    Unique SMILES:   {df_out['SMILES'].nunique():,}")

    # Enforce column order: protein_id, sequence, feature_0..feature_2559, SMILES
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
