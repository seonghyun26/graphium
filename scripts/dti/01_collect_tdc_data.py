#!/usr/bin/env python
"""Stage 1: Collect raw Drug-Target Interaction (DTI) data from TDC.

Downloads multiple DTI datasets from the Therapeutics Data Commons (TDC),
concatenates them, deduplicates, and saves as a single CSV.

Output schema (protein-drug.csv):
    - Drug_ID     : compound identifier
    - Drug        : SMILES string of the drug molecule
    - Target_ID   : protein identifier (e.g. "AAK1", "P45877", "Q9UBY0")
    - Target      : full amino acid sequence of the protein
    - Y           : binding affinity value
    - dti_dataset : source dataset name (e.g. "davis", "bindingdb_patent")
    - Year        : publication year (may be NaN)

This is a many-to-many relationship: one protein binds many drugs, one drug
may bind multiple proteins. Typical output: ~1.79M rows.

Usage:
    python 01_collect_tdc_data.py --output data/protein-drug.csv
"""

import argparse
import os
import sys

import pandas as pd


# All DTI dataset names available in TDC that were used in the original pipeline
TDC_DTI_DATASETS = [
    "DAVIS",
    "KIBA",
    "BindingDB_Kd",
    "BindingDB_IC50",
    "BindingDB_Ki",
    "BindingDB_Patent",
]


def download_tdc_datasets(cache_dir: str) -> list[pd.DataFrame]:
    """Download all DTI datasets from TDC and return as list of DataFrames.

    Each DataFrame gets an additional 'dti_dataset' column recording which
    TDC dataset it came from.
    """
    try:
        from tdc.multi_pred import DTI
    except ImportError:
        print("ERROR: PyTDC is required.  Install with:  pip install PyTDC")
        sys.exit(1)

    frames = []
    for name in TDC_DTI_DATASETS:
        print(f"  Downloading TDC DTI dataset: {name} ...")
        data = DTI(name=name, path=cache_dir)
        df = data.get_data()
        df["dti_dataset"] = name.lower()
        frames.append(df)
        print(f"    -> {len(df):,} rows")
    return frames


def collect_and_merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all frames, standardise columns, deduplicate."""
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows after concat: {len(df):,}")

    # Verify expected TDC columns are present
    expected = {"Drug_ID", "Drug", "Target_ID", "Target", "Y"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in TDC data: {missing}")

    # Add Year column if not present (some TDC datasets include it)
    if "Year" not in df.columns:
        df["Year"] = float("nan")

    # Deduplicate on (Drug, Target, dti_dataset) -- keep first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=["Drug", "Target", "dti_dataset"], keep="first")
    print(f"  Rows after dedup: {len(df):,}  (removed {before - len(df):,} duplicates)")

    # Keep only canonical columns in the documented order
    columns = ["Drug_ID", "Drug", "Target_ID", "Target", "Y", "dti_dataset", "Year"]
    df = df[columns].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Collect raw DTI data from Therapeutics Data Commons"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/protein-drug.csv",
        help="Output CSV path (default: data/protein-drug.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/tdc_cache",
        help="TDC download cache directory (default: data/tdc_cache)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Stage 1: Collecting DTI data from TDC")
    print("=" * 60)

    frames = download_tdc_datasets(args.cache_dir)
    df = collect_and_merge(frames)

    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} rows to {args.output}")
    print(f"  Unique drugs (SMILES): {df['Drug'].nunique():,}")
    print(f"  Unique proteins:       {df['Target_ID'].nunique():,}")
    print(f"  Datasets:              {sorted(df['dti_dataset'].unique())}")
    print("Done.")


if __name__ == "__main__":
    main()
