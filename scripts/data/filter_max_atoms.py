"""Filter a SMILES CSV to rows whose molecule has at most N heavy atoms.

Used to prune the O(N^2) pair-tensor tail for PairMixer pretraining on
large-molecule datasets (lpm24, dti_esmc_v2, bbbc047). The filter preserves
all columns (including label and embedding columns) and only drops rows.

A sibling ``.parquet`` is always written next to the ``--output`` CSV,
since parquet loads ~30-50x faster for wide embedding files and training
configs generally prefer it for this reason. Disable with
``--no-parquet``.

Usage:
    python scripts/data/filter_max_atoms.py \
        --input  data/dti-processed/lpm24_pubmedbert.csv \
        --output data/dti-processed/lpm24_pubmedbert_max100.csv \
        --smiles-col SMILES_nometa \
        --max-atoms 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def num_heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles))
    return mol.GetNumAtoms() if mol is not None else -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-col", required=True)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument(
        "--no-parquet", action="store_true",
        help="Skip writing a sibling .parquet next to the CSV output.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} does not exist", file=sys.stderr)
        return 1

    t0 = time.time()
    df = pd.read_csv(args.input)
    n_in = len(df)
    if args.smiles_col not in df.columns:
        print(
            f"ERROR: column {args.smiles_col!r} not in {args.input.name}; "
            f"columns: {list(df.columns)[:10]}...",
            file=sys.stderr,
        )
        return 2

    atom_counts = df[args.smiles_col].map(num_heavy_atoms)
    bad_parse = (atom_counts < 0).sum()
    keep_mask = (atom_counts >= 0) & (atom_counts <= args.max_atoms)
    df_out = df.loc[keep_mask].reset_index(drop=True)
    n_out = len(df_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    parquet_path = args.output.with_suffix(".parquet")
    if not args.no_parquet:
        df_out.to_parquet(parquet_path, index=False)

    elapsed = time.time() - t0
    pct_kept = 100.0 * n_out / n_in if n_in else 0.0
    print(
        f"[filter_max_atoms] {args.input.name}: {n_in:,} -> {n_out:,} rows "
        f"({pct_kept:.2f}% kept, max_atoms={args.max_atoms}, "
        f"unparseable={bad_parse}, {elapsed:.1f}s) -> {args.output}"
        + (f" (+ {parquet_path.name})" if not args.no_parquet else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
