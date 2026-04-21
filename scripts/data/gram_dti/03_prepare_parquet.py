#!/usr/bin/env python
"""Build ``data/downstream/gram_dti/<subset>.parquet`` + per-fold split ``.pt`` files.

Consumes:
  - Clone of DTIAM (https://github.com/CSUBioGroup/DTIAM) — positive pairs +
    drug SMILES + target sequences per subset.
  - Consolidated ESM-2 embedding parquet produced by
    ``scripts/data/gram_dti/02_consolidate_esm2.py`` (protein_id + feature_0..).

Produces, for each of ``yamanishi_08 / hetionet / activation / inhibition``:
  - ``<subset>.parquet``: SMILES_nometa, drug_id, target_id, label,
    prot_emb_0..{D-1} where D = ESM-2 dim (2560 for ESM2-3B).
  - ``splits/<subset>_<method>_fold<k>.pt``: {train, val, test} row-index dict.
    3 methods × n_folds files per subset (n_folds = 10 for DTI, 5 for MoA).

1:10 negative sampling, KFold(shuffle=True, random_state=0) — matches DTIAM.
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold


METHODS = ("warm", "drug_cold", "target_cold")

# DTIAM draws the drug / target "universe" for negative sampling differently
# across subsets. For DTI (Yamanishi, Hetionet) the universes are the unique
# IDs *in the interaction file*; for MoA (Activation, Inhibition) they are
# every row in the drug/target CSVs. Preserving this so our negative pool
# matches the paper's.
SUBSET_CONFIG = {
    "yamanishi_08": {
        "root": "data/dti/yamanishi_08",
        "drug_file": "drug_smiles.csv", "drug_id": "drug_id", "drug_smiles": "smiles",
        "target_file": "protein_seq.csv", "target_id": "pro_id",
        "dti_file": "dti.csv", "dti_header": False, "dti_drug": 0, "dti_target": 2, "dti_sep": "\t",
        "drug_source": "dti", "target_source": "dti", "n_folds": 10,
    },
    "hetionet": {
        "root": "data/dti/hetionet",
        "drug_file": "drug_smiles.csv", "drug_id": "drug_id", "drug_smiles": "smiles",
        "target_file": "protein_seq.csv", "target_id": "pro_id",
        "dti_file": "dti.csv", "dti_header": False, "dti_drug": 0, "dti_target": 2, "dti_sep": "\t",
        "drug_source": "dti", "target_source": "dti", "n_folds": 10,
    },
    "activation": {
        "root": "data/moa/activation",
        "drug_file": "drug_smi.csv", "drug_id": "DrugID", "drug_smiles": "smi",
        "target_file": "tar_seq.csv", "target_id": "TargetID",
        "dti_file": "dti.csv", "dti_header": True, "dti_drug": "DrugID", "dti_target": "TargetID", "dti_sep": "\t",
        "drug_source": "file", "target_source": "file", "n_folds": 5,
    },
    "inhibition": {
        "root": "data/moa/inhibition",
        "drug_file": "drug_smi.csv", "drug_id": "DrugID", "drug_smiles": "smi",
        "target_file": "tar_seq.csv", "target_id": "TargetID",
        "dti_file": "dti.csv", "dti_header": True, "dti_drug": "DrugID", "dti_target": "TargetID", "dti_sep": "\t",
        "drug_source": "file", "target_source": "file", "n_folds": 5,
    },
}


def _load_protein_embeddings(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read the consolidated ESM-2 parquet; return (indexed_df, prot_emb_cols)."""
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feat_cols:
        sys.exit(f"ERROR: {path} has no ``feature_*`` columns.")
    feat_cols.sort(key=lambda c: int(c.split("_")[1]))
    dim = len(feat_cols)
    prot_emb_cols = [f"prot_emb_{i}" for i in range(dim)]
    df = df.set_index("protein_id")[feat_cols]
    print(f"Loaded {len(df):,} proteins × {dim}-d from {path}")
    return df, prot_emb_cols


def _load_tables(dtiam_root: Path, cfg: dict) -> dict:
    root = dtiam_root / cfg["root"]
    drugs = pd.read_csv(root / cfg["drug_file"], sep="\t").rename(
        columns={cfg["drug_id"]: "drug_id", cfg["drug_smiles"]: "smiles"}
    )
    drug_to_smiles = dict(zip(drugs["drug_id"].astype(str), drugs["smiles"]))
    targets = pd.read_csv(root / cfg["target_file"], sep="\t").rename(
        columns={cfg["target_id"]: "target_id"}
    )
    target_ids_file = targets["target_id"].astype(str).drop_duplicates().tolist()

    if cfg["dti_header"]:
        dti_df = pd.read_csv(root / cfg["dti_file"], sep=cfg["dti_sep"])
    else:
        dti_df = pd.read_csv(root / cfg["dti_file"], sep=cfg["dti_sep"], header=None)
    pos_pairs = [
        (str(d), str(t))
        for d, t in zip(dti_df[cfg["dti_drug"]].tolist(), dti_df[cfg["dti_target"]].tolist())
    ]

    drug_ids = (sorted({p[0] for p in pos_pairs}) if cfg["drug_source"] == "dti"
                else drugs["drug_id"].astype(str).drop_duplicates().tolist())
    target_ids = (sorted({p[1] for p in pos_pairs}) if cfg["target_source"] == "dti"
                  else target_ids_file)
    return {"drug_ids": drug_ids, "target_ids": target_ids,
            "drug_to_smiles": drug_to_smiles, "pos_pairs": pos_pairs}


def _sample_pairs(tables: dict, neg_ratio: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pos_set = set(tables["pos_pairs"])
    all_pairs = list(itertools.product(tables["drug_ids"], tables["target_ids"]))
    neg_pool = [p for p in all_pairs if p not in pos_set]
    n_neg = len(tables["pos_pairs"]) * neg_ratio
    if n_neg > len(neg_pool):
        sys.exit(f"ERROR: requested {n_neg:,} negatives, only {len(neg_pool):,} available.")
    neg = rng.sample(neg_pool, n_neg)
    rows = [(d, t, 1) for (d, t) in tables["pos_pairs"]] + [(d, t, 0) for (d, t) in neg]
    return pd.DataFrame(rows, columns=["drug_id", "target_id", "label"])


def _build_parquet(
    subset: str, pair_df: pd.DataFrame, tables: dict,
    prot_df: pd.DataFrame, prot_emb_cols: list[str], output_dir: Path,
) -> pd.DataFrame:
    drug_to_smiles = tables["drug_to_smiles"]
    n_before = len(pair_df)
    pair_df = pair_df[
        pair_df["drug_id"].isin(drug_to_smiles) & pair_df["target_id"].isin(prot_df.index)
    ].reset_index(drop=True)
    if n_before != len(pair_df):
        print(f"  [{subset}] dropped {n_before - len(pair_df):,} rows with missing SMILES/embedding")

    pair_df["SMILES_nometa"] = pair_df["drug_id"].map(drug_to_smiles)
    prot_emb = prot_df.loc[pair_df["target_id"].values].values.astype(np.float32)
    # Per-feature z-score (labels are binary so no label normalization).
    mu = prot_emb.mean(axis=0).astype(np.float32)
    sd = prot_emb.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    prot_norm = ((prot_emb - mu) / sd).astype(np.float32)

    out = pd.DataFrame({
        "SMILES_nometa": pair_df["SMILES_nometa"].values,
        "drug_id":       pair_df["drug_id"].values,
        "target_id":     pair_df["target_id"].values,
        "label":         pair_df["label"].astype(np.float32).values,
    })
    out[prot_emb_cols] = prot_norm
    out["row_idx"] = np.arange(len(out), dtype=np.int64)  # retained only for split generation

    parquet_path = output_dir / f"{subset}.parquet"
    out.drop(columns=["row_idx"]).to_parquet(parquet_path, index=False)
    size = parquet_path.stat().st_size / (1024 * 1024)
    pos, neg = int((out["label"] == 1).sum()), int((out["label"] == 0).sum())
    print(f"  [{subset}] wrote {parquet_path}  ({len(out):,} rows pos={pos:,} neg={neg:,}, {size:.1f} MB)")

    torch.save(
        {"prot_emb_mean": torch.from_numpy(mu), "prot_emb_std": torch.from_numpy(sd),
         "n_positive": pos, "n_negative": neg, "esm2_dim": len(prot_emb_cols)},
        output_dir / f"{subset}_norm_stats.pt",
    )
    return out


def _generate_splits(
    subset: str, cfg: dict, pair_df: pd.DataFrame,
    val_fraction: float, val_seed: int, splits_dir: Path,
) -> None:
    n_folds = cfg["n_folds"]
    n = len(pair_df)
    drug = pair_df["drug_id"].values
    target = pair_df["target_id"].values
    unique_drugs = np.array(sorted(pd.unique(drug)))
    unique_targets = np.array(sorted(pd.unique(target)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    val_rng = random.Random(val_seed)

    def _carve_val(train_idx: np.ndarray) -> tuple[list[int], list[int]]:
        idx = train_idx.tolist()
        val_rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        return idx[n_val:], idx[:n_val]

    all_rows = np.arange(n)
    for method in METHODS:
        domain = (all_rows if method == "warm"
                  else unique_drugs if method == "drug_cold" else unique_targets)
        for fold, (train_dom, test_dom) in enumerate(kf.split(domain)):
            if method == "warm":
                train_rows, test_rows = all_rows[train_dom], all_rows[test_dom]
            elif method == "drug_cold":
                test_set = set(unique_drugs[test_dom].tolist())
                mask = np.isin(drug, list(test_set))
                train_rows, test_rows = all_rows[~mask], all_rows[mask]
            else:
                test_set = set(unique_targets[test_dom].tolist())
                mask = np.isin(target, list(test_set))
                train_rows, test_rows = all_rows[~mask], all_rows[mask]

            train_final, val_final = _carve_val(train_rows)
            out = {"train": train_final, "val": val_final, "test": test_rows.tolist()}
            path = splits_dir / f"{subset}_{method}_fold{fold}.pt"
            torch.save(out, path)
            print(f"  [{subset}/{method}/fold={fold}] "
                  f"train={len(out['train']):,} val={len(out['val']):,} test={len(out['test']):,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dtiam-root", required=True, help="Clone of CSUBioGroup/DTIAM")
    p.add_argument("--protein-emb", required=True,
                   help="Consolidated ESM-2 parquet (protein_id + feature_0..).")
    p.add_argument("--output-dir", default="data/downstream/gram_dti")
    p.add_argument("--subsets", nargs="+", default=list(SUBSET_CONFIG),
                   choices=list(SUBSET_CONFIG))
    p.add_argument("--neg-ratio", type=int, default=10)
    p.add_argument("--neg-seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=42)
    args = p.parse_args()

    dtiam_root = Path(args.dtiam_root)
    if not dtiam_root.is_dir():
        sys.exit(f"ERROR: --dtiam-root {dtiam_root} not found.")
    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    prot_df, prot_emb_cols = _load_protein_embeddings(args.protein_emb)

    for subset in args.subsets:
        cfg = SUBSET_CONFIG[subset]
        print(f"\n=== {subset} ===")
        tables = _load_tables(dtiam_root, cfg)
        print(f"  drugs={len(tables['drug_ids']):,} targets={len(tables['target_ids']):,} "
              f"positives={len(tables['pos_pairs']):,}")
        pair_df = _sample_pairs(tables, args.neg_ratio, args.neg_seed)
        pair_df = _build_parquet(subset, pair_df, tables, prot_df, prot_emb_cols, output_dir)
        _generate_splits(subset, cfg, pair_df, args.val_fraction, args.val_seed, splits_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
