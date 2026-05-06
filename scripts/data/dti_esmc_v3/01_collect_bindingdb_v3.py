#!/usr/bin/env python
"""Stage 1 (v3): Build a fresh DTI training set directly from local BindingDB.

Reads BindingDB IC50, Ki, and Patent (already cached as TDC CSVs), filters,
deduplicates, samples down to 100,000 (mol, target) pairs, and writes:

  protein-esmc-v3.csv   — protein_id, sequence            (unique proteins)
  dti-bindingdb-v3.csv  — Drug_ID, Drug, Target_ID, Target, Y, dti_dataset, Year

Filters:
  - canonical SMILES via RDKit; drop invalid
  - max 100 heavy atoms per molecule
  - max 1500 amino acids per protein sequence
  - random-sample MAX_PAIRS=100_000 (mol, target) pairs (seed=42)

The BindingDB TDC tables use:
  ID1=Drug_ID,  X1=SMILES,  ID2=UniProt,  X2=sequence,  Y=affinity (-log).

Stage 2 onward (the existing v2 pipeline at scripts/data/dti_esmc/02..08)
runs unchanged on the v3 outputs.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


REPO_ROOT = Path("/home/shpark/prj-molrepr/graphium")
TABLES = [
    ("bindingdb_ic50",   REPO_ROOT / "data/tdc_cache/bindingdb_ic50.csv"),
    ("bindingdb_ki",     REPO_ROOT / "data/tdc_cache/bindingdb_ki.csv"),
    ("bindingdb_patent", REPO_ROOT / "data/tdc_cache/bindingdb_patent.csv"),
]


def _canon_and_count(smiles: str):
    """Return (canon_smiles, n_heavy_atoms) or (None, None) if invalid."""
    if not isinstance(smiles, str):
        return None, None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None
    return Chem.MolToSmiles(m, canonical=True), m.GetNumHeavyAtoms()


def _canonicalize_unique(smis, n_jobs):
    """Map raw SMILES → (canon, n_heavy) for a list of UNIQUE SMILES strings."""
    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size=512)(
        delayed(_canon_and_count)(s) for s in tqdm(smis, desc="canonicalize", unit="mol")
    )
    return {s: out[i] for i, s in enumerate(smis)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data/dti-scratch-v3"))
    p.add_argument("--max-atoms", type=int, default=100)
    p.add_argument("--max-seq-len", type=int, default=1500)
    p.add_argument("--max-pairs", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read all 3 BindingDB tables and concat.
    parts = []
    for src, path in TABLES:
        if not path.exists():
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        t0 = time.time()
        df = pd.read_csv(path, usecols=["ID1", "X1", "ID2", "X2", "Y"], low_memory=False)
        df = df.assign(dti_dataset=src)
        print(f"[{src}] {len(df):,} rows  ({time.time()-t0:.1f}s)")
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.dropna(subset=["X1", "ID2", "X2", "Y"])
    print(f"\nconcat rows: {len(raw):,}")

    # 2. Sequence length filter (drop huge proteins early).
    seq_lens = raw["X2"].str.len()
    keep_seq = seq_lens <= args.max_seq_len
    dropped_seq = (~keep_seq).sum()
    raw = raw[keep_seq].copy()
    print(f"dropped {dropped_seq:,} rows with seq>{args.max_seq_len} aa  → {len(raw):,}")

    # 3. Canonicalize unique SMILES once and compute heavy-atom counts.
    uniq_smi = raw["X1"].drop_duplicates().tolist()
    print(f"\ncanonicalizing {len(uniq_smi):,} unique SMILES on {args.n_jobs} workers ...")
    smi_map = _canonicalize_unique(uniq_smi, args.n_jobs)
    raw["canon"]   = raw["X1"].map(lambda s: smi_map[s][0] if s in smi_map else None)
    raw["n_heavy"] = raw["X1"].map(lambda s: smi_map[s][1] if s in smi_map else None)
    raw = raw.dropna(subset=["canon", "n_heavy"])
    raw["n_heavy"] = raw["n_heavy"].astype(int)
    print(f"after canonicalize: {len(raw):,}")

    # 4. Atom-count filter.
    keep_atoms = raw["n_heavy"] <= args.max_atoms
    dropped_atoms = (~keep_atoms).sum()
    raw = raw[keep_atoms].copy()
    print(f"dropped {dropped_atoms:,} rows with >{args.max_atoms} heavy atoms  → {len(raw):,}")

    # 5. Deduplicate (canon, ID2, dti_dataset) — same molecule+protein from same
    # source counted once. Pick the median Y across replicate measurements.
    raw = (
        raw.groupby(["canon", "ID2", "dti_dataset"], as_index=False, sort=False)
        .agg({"X2": "first", "ID1": "first", "Y": "median"})
    )
    print(f"after triple-dedup (Y=median): {len(raw):,}")

    # 6. Random sample MAX_PAIRS pairs.
    n_total = len(raw)
    if n_total > args.max_pairs:
        raw = raw.sample(n=args.max_pairs, random_state=args.seed).reset_index(drop=True)
        print(f"sampled {args.max_pairs:,} of {n_total:,} pairs (seed={args.seed})")
    else:
        print(f"under cap; keeping all {n_total:,} pairs")

    # 7. Write the binding-pair table (Drug_ID, Drug, Target_ID, Target, Y,
    # dti_dataset, Year) — schema mirrors scripts/data/dti_esm2/01_collect_tdc_data.py
    pairs = raw.rename(
        columns={"ID1": "Drug_ID", "canon": "Drug", "ID2": "Target_ID", "X2": "Target"}
    )[["Drug_ID", "Drug", "Target_ID", "Target", "Y", "dti_dataset"]].copy()
    pairs["Year"] = float("nan")
    pairs_path = out_dir / "dti-bindingdb-v3.csv"
    pairs.to_csv(pairs_path, index=False)
    print(f"\nwrote {pairs_path}  ({len(pairs):,} rows)")

    # 8. Write the unique-protein table for stage 02 ESM-C embedding.
    proteins = (
        pairs[["Target_ID", "Target"]]
        .drop_duplicates(subset=["Target_ID"])
        .rename(columns={"Target_ID": "protein_id", "Target": "sequence"})
        .reset_index(drop=True)
    )
    prot_path = out_dir / "protein-esmc-v3.csv"
    proteins.to_csv(prot_path, index=False)
    print(f"wrote {prot_path}  ({len(proteins):,} unique proteins)")

    print("\n=== summary ===")
    print(f"  pairs:    {len(pairs):,}")
    print(f"  unique mols:     {pairs.Drug.nunique():,}")
    print(f"  unique proteins: {len(proteins):,}")
    print(f"  per-source breakdown: {pairs.dti_dataset.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
