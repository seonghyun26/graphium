#!/usr/bin/env python
"""Stage 1 (LPM-24): Download the L+M-24 dataset from HuggingFace.

Downloads the Language+Molecules ACL 2024 dataset, which pairs molecular
SMILES strings with natural language captions describing molecular properties.

Source: https://huggingface.co/datasets/language-plus-molecules/LPM-24_train
Paper:  L+M-24: Building a Dataset for Language+Molecules @ ACL 2024

Output schema (lpm24_raw.parquet):
    - molecule   : SMILES string
    - caption    : natural language description of molecular properties
    - properties : list of property tags the caption was generated from

Typical output: 160,560 rows, all unique SMILES.

Usage:
    python 01_download_lpm24.py --output-dir data/lpm24
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (LPM-24): Download L+M-24 dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lpm24",
        help="Output directory (default: data/lpm24)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace split to download (default: train)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Stage 1 (LPM-24): Downloading L+M-24 dataset")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        raise SystemExit(1)

    print(f"  Loading language-plus-molecules/LPM-24_train split={args.split} ...")
    ds = load_dataset("language-plus-molecules/LPM-24_train", split=args.split)

    # Convert properties list to semicolon-separated string for CSV/parquet
    df = ds.to_pandas()
    df["properties_str"] = df["properties"].apply(lambda x: "; ".join(x))

    print(f"  Total rows:      {len(df):,}")
    print(f"  Unique SMILES:   {df['molecule'].nunique():,}")
    print(f"  Columns:         {df.columns.tolist()}")

    # Caption length stats
    cap_lens = df["caption"].str.len()
    print(f"  Caption length:  min={cap_lens.min()}, median={int(cap_lens.median())}, max={cap_lens.max()}")

    # Save as parquet
    parquet_path = os.path.join(args.output_dir, "lpm24_raw.parquet")
    df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"\n  Saved: {parquet_path}  ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
