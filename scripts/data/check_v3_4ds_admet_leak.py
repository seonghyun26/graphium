#!/usr/bin/env python
"""Check if any TDC-ADMET test SMILES leak into the v3 '4ds' pretrain pool.

4ds = ToyMix (QM9 + Tox21 + ZINC12k) + DTI ESM-C v3 (100k)
    + LPM-24 (litmolformer_v2) + BBBC047 (admetfilt).

Canonicalizes every SMILES via RDKit and intersects each source against the
union of test.csv across 22 TDC ADMET tasks. Per-source overlap counts and
example leaks are printed.
"""

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO = Path("/home/shpark/prj-molrepr/graphium")
ADMET_ROOT = REPO / "data" / "tdc" / "admet_group"

SOURCES = [
    ("dti_v3",   REPO / "data/dti-processed/dti_esmc_100k_v3.csv",                                  "SMILES_nometa"),
    ("lpm24",    REPO / "data/dti-processed/litmolformer_v2.parquet",                               "SMILES_nometa"),
    ("bbbc047",  Path("/home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings_max100_admetfilt.csv"), "SMILES"),
    ("qm9",      REPO / "data/graphium/neurips2023/small-dataset/qm9.csv",                          "smiles"),
    ("tox21",    REPO / "data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv",           "smiles"),
    ("zinc",     REPO / "data/graphium/neurips2023/small-dataset/ZINC12k.csv",                      "smiles"),
]


def canon(s, isomeric=True):
    if not isinstance(s, str):
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=isomeric) if m is not None else None


def canon_set(smis, n_jobs=8, desc="canon", isomeric=True):
    uniq = list({s for s in smis if isinstance(s, str)})
    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size=512)(
        delayed(canon)(s, isomeric) for s in tqdm(uniq, desc=desc, unit="mol")
    )
    return {c for c in out if c is not None}


def load_col(path: Path, col: str):
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=[col])[col].dropna().astype(str).tolist()
    return pd.read_csv(path, usecols=[col])[col].dropna().astype(str).tolist()


def load_admet_test():
    smis = []
    per_task_count = {}
    for d in sorted(p for p in ADMET_ROOT.iterdir() if p.is_dir()):
        f = d / "test.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        smi_col = next((c for c in df.columns if c.lower() in ("drug", "smiles", "smiles_nometa")), None)
        if smi_col is None:
            continue
        rows = df[smi_col].dropna().astype(str).tolist()
        per_task_count[d.name] = len(rows)
        smis.extend(rows)
    print("ADMET test split sizes per task:")
    for k, v in per_task_count.items():
        print(f"  {k:35s}  {v:>6d}")
    print(f"  total raw rows: {len(smis):,}")
    return canon_set(smis, desc="admet_test")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-strip-stereo", dest="strip_stereo", action="store_false",
                    help="Use stereo-preserving canonicalization only (old behaviour).")
    ap.set_defaults(strip_stereo=True)
    args = ap.parse_args()

    stereo_label = "stereo-stripped" if args.strip_stereo else "stereo-preserving"
    print("=" * 72)
    print(f"Loading ADMET TDC test SMILES  [{stereo_label} comparison]")
    print("=" * 72)
    admet = load_admet_test()
    admet_nostreo = canon_set(
        list(admet), desc="admet_test_nostreo", isomeric=False
    ) if args.strip_stereo else set()
    print(f"  unique canonical ADMET-test SMILES: {len(admet):,}")
    if args.strip_stereo:
        print(f"  unique stereo-stripped:             {len(admet_nostreo):,}\n")

    for name, path, col in SOURCES:
        if not path.exists():
            print(f"[{name}] MISSING: {path}")
            continue
        print("=" * 72)
        print(f"[{name}] {path}  (col={col})")
        print("=" * 72)
        raw = load_col(path, col)
        s = canon_set(raw, desc=name)
        leak_stereo = s & admet
        if args.strip_stereo:
            s_nostreo = canon_set(raw, desc=f"{name}_nostreo", isomeric=False)
            leak_nostreo = s_nostreo & admet_nostreo
            leak_extra = leak_nostreo - {canon(c, isomeric=False) for c in leak_stereo}
            leak_total = len(leak_stereo) + len(leak_extra)
        else:
            leak_nostreo = set()
            leak_extra = set()
            leak_total = len(leak_stereo)
        print(f"  total rows in source:         {len(raw):,}")
        print(f"  unique canonical SMILES:      {len(s):,}")
        print(f"  ADMET-test ∩ source (stereo): {len(leak_stereo):,}")
        if args.strip_stereo:
            print(f"  ADMET-test ∩ source (nostreo, new): {len(leak_extra):,}")
            print(f"  total leaks:                  {leak_total:,}")
        if leak_stereo:
            print(f"  stereo examples: {list(sorted(leak_stereo))[:3]}")
        if leak_extra:
            print(f"  nostreo examples: {list(sorted(leak_extra))[:3]}")
        print()


if __name__ == "__main__":
    main()
