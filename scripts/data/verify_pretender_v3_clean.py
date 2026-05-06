#!/usr/bin/env python
"""Verify the pretender_v3 filtered parquets contain no downstream-test SMILES.

Rebuilds the union of test SMILES (ADMET TDC + ADME Fang Polaris + DTIAM 4ds
+ cell bioactivity), canonicalizes each filtered pretrain parquet, and counts
intersections. Expected: 0 leak rows everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse the loaders from the build script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_v3_4ds_dsfilt import (
    canon_list,
    load_admet_tdc,
    load_adme_fang,
    load_dtiam,
    load_bioactivity,
)

import pandas as pd

REPO = Path("/home/shpark/prj-molrepr/graphium")
TOYMIX = REPO / "data" / "graphium" / "neurips2023" / "small-dataset"

FILTERED = [
    ("dti_v3",  REPO / "data/dti-processed/protein_v3_esmc_filtered.parquet",          "SMILES_nometa"),
    ("lpm24",   REPO / "data/dti-processed/literature_v2_molformer_filtered.parquet",  "SMILES_nometa"),
    ("bbbc047", Path("/home/shpark/prj-molrepr/data/bbbc047/cell_v1_bbbc047_filtered.parquet"), "SMILES"),
    ("qm9",     TOYMIX / "qm9_filtered.parquet",                       "smiles"),
    ("tox21",   TOYMIX / "Tox21-7k-12-labels_filtered.parquet",        "smiles"),
    ("zinc",    TOYMIX / "ZINC12k_filtered.parquet",                   "smiles"),
]


def main():
    print("Rebuilding downstream test-SMILES sets ...")
    sets = {
        "admet_tdc":   load_admet_tdc(),
        "adme_fang":   load_adme_fang(),
        "dtiam":       load_dtiam(),
        "bioactivity": load_bioactivity(),
    }
    union = set().union(*sets.values())
    print(f"  union size: {len(union):,}\n")

    print(f"{'source':10s}  {'rows':>8s}  {'admet':>6s}  {'fang':>6s}  {'dtiam':>6s}  {'bio':>6s}  {'TOTAL':>6s}")
    print("-" * 64)
    all_clean = True
    for name, path, col in FILTERED:
        if not path.exists():
            print(f"{name:10s}  MISSING")
            all_clean = False
            continue
        smis = pd.read_parquet(path, columns=[col])[col].astype(str).tolist()
        cano = canon_list(smis, n_jobs=8, desc=name)
        rows_leak_admet = sum(1 for c in cano if c in sets["admet_tdc"])
        rows_leak_fang  = sum(1 for c in cano if c in sets["adme_fang"])
        rows_leak_dtiam = sum(1 for c in cano if c in sets["dtiam"])
        rows_leak_bio   = sum(1 for c in cano if c in sets["bioactivity"])
        rows_leak_union = sum(1 for c in cano if c in union)
        print(f"{name:10s}  {len(smis):>8,}  {rows_leak_admet:>6}  {rows_leak_fang:>6}  "
              f"{rows_leak_dtiam:>6}  {rows_leak_bio:>6}  {rows_leak_union:>6}")
        if rows_leak_union > 0:
            all_clean = False

    print()
    print("CLEAN ✓" if all_clean else "LEAKS DETECTED — re-run filtering.")
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
