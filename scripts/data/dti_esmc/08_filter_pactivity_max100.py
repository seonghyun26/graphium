#!/usr/bin/env python
"""Stage 8: Cap heavy-atom count at 100 in the DTI pActivity dataset.

Drops rows whose SMILES has more than `max_heavy` heavy atoms (default 100).
Mirrors the `_max100` filter applied to the embedding-regression CSV
(`dti_esmc_100k_v2_max100_admetfilt.csv`). PairMixer's triangle/pair op is
O(N^2) in atom count; even a single 400+-atom outlier (peptide/polymer) blows
the bf16 pair tensor past the GPU memory ceiling.

Preserves all columns including `pY_bin` (added by Stage 7) — the binary label
is computed from z-scored `pY` and is unaffected by row filtering.

Usage:
    python scripts/data/dti_esmc/08_filter_pactivity_max100.py \
        --input data/dti-processed/dti_pactivity_esmc_100k.parquet \
        --output-prefix data/dti-processed/dti_pactivity_esmc_100k_max100 \
        --max-heavy 100
"""

import argparse
import os
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


def heavy_atom_count(smiles: str) -> int:
    if not isinstance(smiles, str) or not smiles:
        return -1
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="data/dti-processed/dti_pactivity_esmc_100k.parquet",
        help="Input parquet (preferred) or CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="data/dti-processed/dti_pactivity_esmc_100k_max100",
        help="Output path prefix; .csv and .parquet are both written.",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="SMILES_nometa",
    )
    parser.add_argument(
        "--max-heavy",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    print(f"Loading {args.input} ...")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    n_total = len(df)
    print(f"  shape={df.shape}  smiles_col={args.smiles_col!r}")

    print(f"Computing heavy-atom counts with {args.n_jobs} workers ...")
    smiles = df[args.smiles_col].tolist()
    with Pool(args.n_jobs) as pool:
        counts = pool.map(heavy_atom_count, smiles, chunksize=2048)

    n_invalid = sum(1 for c in counts if c < 0)
    n_over = sum(1 for c in counts if c > args.max_heavy)
    n_keep = n_total - n_invalid - n_over
    print(
        f"  invalid SMILES: {n_invalid}  "
        f"over {args.max_heavy} atoms: {n_over}  "
        f"keeping: {n_keep} / {n_total} ({100.0 * n_keep / n_total:.2f}%)"
    )

    mask = [(0 < c <= args.max_heavy) for c in counts]
    df_filt = df.loc[mask].reset_index(drop=True)

    csv_path = args.output_prefix + ".csv"
    parquet_path = args.output_prefix + ".parquet"
    print(f"Writing parquet to {parquet_path} ...")
    df_filt.to_parquet(parquet_path, index=False)
    print(f"  size: {os.path.getsize(parquet_path) / (1024 ** 2):.1f} MB")

    print(f"Writing CSV to {csv_path} ...")
    df_filt.to_csv(csv_path, index=False)
    print(f"  size: {os.path.getsize(csv_path) / (1024 ** 2):.1f} MB")

    print(f"\nFinal shape: {df_filt.shape}")
    if "pY_bin" in df_filt.columns:
        n_pos = int(df_filt["pY_bin"].sum())
        print(f"  pY_bin positives: {n_pos} / {len(df_filt)} ({100.0 * n_pos / len(df_filt):.2f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
