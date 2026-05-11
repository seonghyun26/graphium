#!/usr/bin/env python
"""Build a 'downstream-filtered' v3 4ds pretrain pool.

Removes any pretrain SMILES that appears as a test SMILES in any of these
downstream evals:

  1. ADMET TDC (22 tasks)        — fixed test split, test.csv per task
  2. ADME Fang (Polaris, 6)      — fixed test indices in *-reg-v1.json
  3. DTIAM (4 datasets)          — CV-split, treat every SMILES as test
  4. Cell bioactivity (Fredinh)  — 6-fold CV, treat every SMILES as test

Pretrain sources filtered (the 4ds composition):
  - DTI ESM-C v3   data/dti-processed/dti_esmc_100k_v3.csv
  - LPM-24         data/dti-processed/litmolformer_v2.parquet
  - BBBC047        data/bbbc047/bbbc047_smiles_embeddings_max100_admetfilt.csv
  - QM9 / Tox21 / ZINC12k  (ToyMix)

Phase 1 (always): print leakage % per (source × downstream) and per source.
Phase 2 (--write): write filtered copies. ToyMix split .pt indices are
remapped onto the filtered row order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO       = Path("/home/shpark/prj-molrepr/graphium")
ADMET_ROOT = REPO / "data" / "tdc" / "admet_group"
POLARIS_DIR = REPO / "datacache" / "polaris_adme_fang"
POLARIS_PARQUET = POLARIS_DIR / "adme-fang-v1.parquet"
DTIAM_DIR  = REPO / "data" / "dti-classif-eval"
BIOACT_PT  = REPO / "datacache" / "minimol_embeddings" / "cell_bioactivity.pt"
TOYMIX_DIR = REPO / "data" / "graphium" / "neurips2023" / "small-dataset"
BBBC047    = Path("/home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings_max100_admetfilt.csv")
BBBC047_OUT_DIR = Path("/home/shpark/prj-molrepr/data/bbbc047")

POLARIS_SLUGS = [
    "adme-fang-hclint-reg-v1",
    "adme-fang-rclint-reg-v1",
    "adme-fang-perm-reg-v1",
    "adme-fang-hppb-reg-v1",
    "adme-fang-rppb-reg-v1",
    "adme-fang-solu-reg-v1",
]

DTIAM_PARQUETS = ["activation", "hetionet", "inhibition", "yamanishi_08"]


# Pretrain sources: (name, path, smiles_col, kind, has_split_pt)
SOURCES = [
    ("dti_v3",   REPO / "data/dti-processed/dti_esmc_100k_v3.csv",                "SMILES_nometa", "csv",     None),
    ("lpm24",    REPO / "data/dti-processed/litmolformer_v2.parquet",             "SMILES_nometa", "parquet", None),
    ("bbbc047",  BBBC047,                                                          "SMILES",        "csv",     None),
    ("qm9",      TOYMIX_DIR / "qm9.csv",                                           "smiles",        "csv",     TOYMIX_DIR / "qm9_random_splits.pt"),
    ("tox21",    TOYMIX_DIR / "Tox21-7k-12-labels.csv",                            "smiles",        "csv",     TOYMIX_DIR / "Tox21_random_splits.pt"),
    ("zinc",     TOYMIX_DIR / "ZINC12k.csv",                                       "smiles",        "csv",     TOYMIX_DIR / "ZINC12k_random_splits.pt"),
]


# ---------- canonicalization --------------------------------------------------

def canon(s, isomeric=True):
    if not isinstance(s, str):
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=isomeric) if m is not None else None


def canon_list(smis, n_jobs=8, desc="canon", isomeric=True):
    """Return list of canonical SMILES (None for invalid), preserving order."""
    return Parallel(n_jobs=n_jobs, backend="loky", batch_size=1024)(
        delayed(canon)(s, isomeric) for s in tqdm(smis, desc=desc, unit="mol")
    )


def canon_set(smis, n_jobs=8, desc="canon", isomeric=True):
    uniq = list({s for s in smis if isinstance(s, str)})
    out = canon_list(uniq, n_jobs=n_jobs, desc=desc, isomeric=isomeric)
    return {c for c in out if c is not None}


# ---------- downstream test SMILES loaders -----------------------------------

def load_admet_tdc():
    smis = []
    for d in sorted(p for p in ADMET_ROOT.iterdir() if p.is_dir()):
        f = d / "test.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        col = next((c for c in df.columns if c.lower() in ("drug", "smiles", "smiles_nometa")), None)
        if col is None:
            continue
        smis.extend(df[col].dropna().astype(str).tolist())
    return canon_set(smis, desc="admet_tdc")


def load_adme_fang():
    if not POLARIS_PARQUET.exists():
        print(f"WARN: {POLARIS_PARQUET} missing; ADME-Fang will be empty", file=sys.stderr)
        return set()
    parquet = pd.read_parquet(POLARIS_PARQUET, columns=["MOL_smiles"])["MOL_smiles"].astype(str).tolist()
    test_smis = []
    for slug in POLARIS_SLUGS:
        meta_path = POLARIS_DIR / f"{slug}.json"
        if not meta_path.exists():
            print(f"  skip {slug}: missing JSON")
            continue
        meta = json.loads(meta_path.read_text())
        train_idx, test_idx = meta["split"]
        for i in test_idx:
            test_smis.append(parquet[i])
    return canon_set(test_smis, desc="adme_fang")


def load_dtiam():
    smis = []
    for stem in DTIAM_PARQUETS:
        p = DTIAM_DIR / f"{stem}.parquet"
        if not p.exists():
            print(f"  skip {stem}: missing")
            continue
        smis.extend(pd.read_parquet(p, columns=["SMILES_nometa"])["SMILES_nometa"].dropna().astype(str).tolist())
    return canon_set(smis, desc="dtiam")


def load_bioactivity():
    if not BIOACT_PT.exists():
        print(f"WARN: {BIOACT_PT} missing; bioactivity will be empty", file=sys.stderr)
        return set()
    d = torch.load(BIOACT_PT, map_location="cpu", weights_only=False)
    return canon_set(list(d.keys()), desc="bioactivity")


# ---------- per-source canonicalization --------------------------------------

def load_source_smiles(name, path, col, kind):
    if kind == "parquet":
        return pd.read_parquet(path, columns=[col])[col].astype(str).tolist()
    return pd.read_csv(path, usecols=[col])[col].astype(str).tolist()


def load_source_full(path: Path, kind: str) -> pd.DataFrame:
    return pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)


# ---------- writers -----------------------------------------------------------

def write_parquet(df_filt: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_filt.to_parquet(out_path, index=False)


def remap_splits(splits_pt_in: Path, splits_pt_out: Path, kept_old_indices: np.ndarray):
    """Remap old row indices in a splits .pt file to the new (filtered) row order."""
    old_to_new = -np.ones(int(kept_old_indices.max()) + 1 if len(kept_old_indices) else 0, dtype=np.int64)
    for new_i, old_i in enumerate(kept_old_indices):
        old_to_new[old_i] = new_i
    raw = torch.load(splits_pt_in, map_location="cpu", weights_only=False)
    out = {}
    for split, idx_list in raw.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        in_range = idx < len(old_to_new)
        new_idx = np.where(in_range, old_to_new[np.where(in_range, idx, 0)], -1)
        kept = new_idx[new_idx >= 0]
        out[split] = kept.tolist()
    splits_pt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, splits_pt_out)
    return {k: len(v) for k, v in out.items()}


# ---------- main --------------------------------------------------------------

def fmt_pct(n, d):
    return f"{n:>6,d} / {d:>7,d}  ({100.0 * n / d:5.2f}%)" if d else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Phase 2: write filtered output files. Without this flag, only the leakage table is printed.")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--no-strip-stereo", dest="strip_stereo", action="store_false",
                    help="Use stereo-preserving canonicalization only (old behaviour).")
    ap.set_defaults(strip_stereo=True)
    args = ap.parse_args()

    # ---------------- downstream test sets ---------------------------------
    print("=" * 78)
    print("Building downstream test-SMILES sets")
    print("=" * 78)
    print("\n[1/4] ADMET TDC (22 tasks)")
    admet = load_admet_tdc()
    print(f"  unique canon: {len(admet):,}")

    print("\n[2/4] ADME Fang Polaris (6 benchmarks)")
    fang = load_adme_fang()
    print(f"  unique canon: {len(fang):,}")

    print("\n[3/4] DTIAM (4 datasets, all SMILES)")
    dtiam = load_dtiam()
    print(f"  unique canon: {len(dtiam):,}")

    print("\n[4/4] Cell bioactivity (all SMILES)")
    bioact = load_bioactivity()
    print(f"  unique canon: {len(bioact):,}")

    downstream = {
        "admet_tdc":  admet,
        "adme_fang":  fang,
        "dtiam":      dtiam,
        "bioactivity": bioact,
    }
    union = set().union(*downstream.values())
    print(f"\nUNION of all downstream test SMILES: {len(union):,}")
    if args.strip_stereo:
        union_nostreo = canon_set(list(union), n_jobs=args.n_jobs,
                                  desc="union_nostreo", isomeric=False)
        print(f"UNION stereo-stripped (nostreo):     {len(union_nostreo):,}")
    else:
        union_nostreo = set()

    # ---------------- per-source leakage report ----------------------------
    print("\n" + "=" * 78)
    print("Leakage report (per pretrain source x per downstream)")
    print("=" * 78)

    cached_canon = {}         # name -> list of canonical SMILES (stereo), None for invalid
    cached_canon_nostreo = {} # name -> list of canonical SMILES (nostreo), None for invalid
    cached_dfs = {}           # name -> DataFrame of source

    for name, path, col, kind, _splits in SOURCES:
        if not path.exists():
            print(f"\n[{name}] MISSING: {path}")
            continue
        print(f"\n[{name}]  {path}")
        df = load_source_full(path, kind)
        smis = df[col].astype(str).tolist()
        cano = canon_list(smis, n_jobs=args.n_jobs, desc=name)
        cached_canon[name] = cano
        cached_dfs[name] = df

        n_total = len(smis)
        n_invalid = sum(1 for c in cano if c is None)
        unique = {c for c in cano if c is not None}
        print(f"  rows={n_total:,}  unique_canon={len(unique):,}  invalid={n_invalid:,}")

        leak_stereo = unique & union
        for ds_name, ds in downstream.items():
            inter = unique & ds
            print(f"    vs {ds_name:11s}  {fmt_pct(len(inter), len(unique))}")
        n_leak_rows = sum(1 for c in cano if c is not None and c in union)
        print(f"    UNION leak rows (stereo): {fmt_pct(n_leak_rows, n_total)}  "
              f"(unique: {len(leak_stereo):,})")

        if args.strip_stereo:
            cano_ns = canon_list(smis, n_jobs=args.n_jobs, desc=f"{name}_nostreo", isomeric=False)
            cached_canon_nostreo[name] = cano_ns
            unique_ns = {c for c in cano_ns if c is not None}
            leak_ns = unique_ns & union_nostreo
            n_leak_ns_rows = sum(1 for c in cano_ns if c is not None and c in union_nostreo)
            extra = len(leak_ns) - len({canon(c, isomeric=False) for c in leak_stereo if c is not None})
            print(f"    UNION leak rows (nostreo): {fmt_pct(n_leak_ns_rows, n_total)}  "
                  f"(unique: {len(leak_ns):,}, extra vs stereo: {extra:+,})")

    if not args.write:
        print("\n" + "=" * 78)
        print("Phase 1 only. Re-run with --write to emit filtered files.")
        print("=" * 78)
        return

    # ---------------- phase 2: write filtered copies -----------------------
    print("\n" + "=" * 78)
    print("Phase 2: writing filtered parquet files")
    print("=" * 78)

    # Output filename for each source. Pattern:
    #   {modality}_{ver}_{model}_filtered.parquet  for protein/cell/literature
    #   {original_stem}_filtered.parquet           for ToyMix
    output_paths = {
        "dti_v3":  REPO / "data/dti-processed/protein_v3_esmc_filtered.parquet",
        "lpm24":   REPO / "data/dti-processed/literature_v2_molformer_filtered.parquet",
        "bbbc047": BBBC047_OUT_DIR / "cell_v1_bbbc047_filtered.parquet",
        "qm9":     TOYMIX_DIR / "qm9_filtered.parquet",
        "tox21":   TOYMIX_DIR / "Tox21-7k-12-labels_filtered.parquet",
        "zinc":    TOYMIX_DIR / "ZINC12k_filtered.parquet",
    }

    for name, path, col, kind, splits_pt in SOURCES:
        if name not in cached_canon:
            continue
        df = cached_dfs[name]
        cano = cached_canon[name]
        cano_ns = cached_canon_nostreo.get(name) if args.strip_stereo else None
        # keep rows whose canon is valid AND not in any downstream test set (stereo or nostreo)
        if cano_ns is not None:
            keep_mask = np.array(
                [(c is not None) and (c not in union) and (cn not in union_nostreo)
                 for c, cn in zip(cano, cano_ns)],
                dtype=bool,
            )
        else:
            keep_mask = np.array([(c is not None) and (c not in union) for c in cano], dtype=bool)
        df_filt = df.loc[keep_mask].reset_index(drop=True)
        kept_old = np.where(keep_mask)[0]

        out_path = output_paths[name]
        write_parquet(df_filt, out_path)

        # DTI v3: also recompute and write per-feature norm stats
        if name == "dti_v3":
            feat_cols = [c for c in df_filt.columns if c.startswith("feature_")]
            feats = df_filt[feat_cols].to_numpy(dtype=np.float32)
            mean = feats.mean(axis=0)
            std = feats.std(axis=0).clip(min=1e-6)
            n_targets = int(df_filt["Target_ID"].nunique()) if "Target_ID" in df_filt.columns else -1
            norm_path = out_path.with_name(out_path.stem + "_norm_stats.pt")
            torch.save(
                {
                    "mean": torch.from_numpy(mean.astype(np.float32)),
                    "std":  torch.from_numpy(std.astype(np.float32)),
                    "n_pairs": len(df_filt),
                    "n_targets": n_targets,
                    "model": "esmc_600m",
                },
                norm_path,
            )

        # ToyMix: remap and write a fresh splits .pt aligned to the filtered row order
        if splits_pt is not None and splits_pt.exists():
            out_splits = out_path.with_name(out_path.stem + "_random_splits.pt")
            sizes = remap_splits(splits_pt, out_splits, kept_old)
            print(f"  [{name}] split sizes after remap: {sizes}  -> {out_splits.name}")

        kept = int(keep_mask.sum())
        dropped = int((~keep_mask).sum())
        print(f"  [{name}] kept={kept:,}  dropped={dropped:,}  "
              f"({100.0 * dropped / max(len(df), 1):5.2f}%)  -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
