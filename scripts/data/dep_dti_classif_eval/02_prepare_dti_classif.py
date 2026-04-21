#!/usr/bin/env python
"""Stage 2: Build per-subset parquet files and K-fold split indices for the
GRAM-DTI (ICLR 2026) benchmark over the 4 DTIAM datasets.

Subsets, cross-validation and splits (frozen to paper-parity; see
``memory/project_gram_dti_benchmark.md`` for the rationale):

    Yamanishi 08   DTI  10-fold CV   {warm, drug_cold, target_cold}
    Hetionet       DTI  10-fold CV   {warm, drug_cold, target_cold}
    Activation     MoA  5-fold  CV   {warm, drug_cold, target_cold}
    Inhibition     MoA  5-fold  CV   {warm, drug_cold, target_cold}

For each subset, this script:

  1. Loads the positive (drug, target) pairs and the drug/target candidate sets
     from the DTIAM repo (the drug/target source differs between DTI and MoA,
     matching DTIAM's original code at ``code/data_process/data_split_{dti,moa}.py``).
  2. Samples ``neg_ratio`` * N_positive negatives uniformly from the complement
     (reseeded deterministically; DTIAM's original code is unseeded).
  3. Joins drug ID → SMILES and target ID → ESM-C embedding, writes a flat
     parquet with columns {``SMILES_nometa``, ``label``, ``prot_emb_0..1151``}.
  4. Generates 3 × K fold index files ({warm, drug_cold, target_cold} × K), each
     a dict {train, val, test} of int row-indices into the parquet. Val is a
     seeded random 10% slice of train (DTIAM's code emits only train/test; we
     need val for early stopping / monitoring).

Outputs (under ``--output-dir``, default ``data/dti-classif-eval/``):

    <subset>.parquet                         # SMILES_nometa, label, prot_emb_0..1151
    splits/<subset>_<method>_fold<k>.pt      # {'train','val','test'} lists of int
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


ESMC_DIM = 1152
PROT_EMB_COLS = [f"prot_emb_{i}" for i in range(ESMC_DIM)]
METHODS = ("warm", "drug_cold", "target_cold")

# DTIAM packs DTI (Yamanishi, Hetionet) and MoA (Activation, Inhibition) with
# different file / column / header conventions. Normalize them here.
#
# ``drug_source`` / ``target_source``: DTIAM's original code draws the drug and
# target universes differently — for DTI the universes are the unique IDs *in
# the interaction file*, while for MoA they are *every row* in the drug/target
# CSVs (so Inhibition counts 14,049 drugs, not just the ~5k with known
# interactions). We preserve that asymmetry so our negative pool matches theirs.
SUBSET_CONFIG = {
    "yamanishi_08": {
        "root":            "data/dti/yamanishi_08",
        "drug_file":       "drug_smiles.csv",  "drug_id_col": "drug_id",  "drug_smiles_col": "smiles",
        "target_file":     "protein_seq.csv",  "target_id_col": "pro_id",
        "dti_file":        "dti.csv",
        "dti_has_header":  False, "dti_drug_col": 0, "dti_target_col": 2, "dti_sep": "\t",
        "drug_source":     "dti",    "target_source": "dti",
        "n_folds":         10,
    },
    "hetionet": {
        "root":            "data/dti/hetionet",
        "drug_file":       "drug_smiles.csv",  "drug_id_col": "drug_id",  "drug_smiles_col": "smiles",
        "target_file":     "protein_seq.csv",  "target_id_col": "pro_id",
        "dti_file":        "dti.csv",
        "dti_has_header":  False, "dti_drug_col": 0, "dti_target_col": 2, "dti_sep": "\t",
        "drug_source":     "dti",    "target_source": "dti",
        "n_folds":         10,
    },
    "activation": {
        "root":            "data/moa/activation",
        "drug_file":       "drug_smi.csv",     "drug_id_col": "DrugID",   "drug_smiles_col": "smi",
        "target_file":     "tar_seq.csv",      "target_id_col": "TargetID",
        "dti_file":        "dti.csv",
        "dti_has_header":  True,  "dti_drug_col": "DrugID", "dti_target_col": "TargetID", "dti_sep": "\t",
        "drug_source":     "file",   "target_source": "file",
        "n_folds":         5,
    },
    "inhibition": {
        "root":            "data/moa/inhibition",
        "drug_file":       "drug_smi.csv",     "drug_id_col": "DrugID",   "drug_smiles_col": "smi",
        "target_file":     "tar_seq.csv",      "target_id_col": "TargetID",
        "dti_file":        "dti.csv",
        "dti_has_header":  True,  "dti_drug_col": "DrugID", "dti_target_col": "TargetID", "dti_sep": "\t",
        "drug_source":     "file",   "target_source": "file",
        "n_folds":         5,
    },
}


def _load_protein_embeddings(path: str) -> pd.DataFrame:
    """Return a DataFrame indexed by protein_id with columns feature_0..feature_1151."""
    print(f"Loading protein embeddings from {path}")
    df = pd.read_parquet(path)
    feat_cols = [f"feature_{i}" for i in range(ESMC_DIM)]
    missing = set(feat_cols) - set(df.columns)
    if missing:
        sys.exit(f"ERROR: protein embedding file missing columns: {sorted(missing)[:5]}...")
    df = df.set_index("protein_id")[feat_cols]
    print(f"  -> {len(df):,} proteins, {ESMC_DIM}-dim embeddings")
    return df


def _load_subset_tables(subset: str, cfg: dict, dtiam_root: Path) -> tuple[dict, dict, list]:
    """Return (drug_to_smiles, target_id_list, positive_pairs).

    ``drug_to_smiles`` is keyed by drug_id (every drug in the candidate set);
    ``target_id_list`` is the *ordered* target universe (list, not set, so the
    later KFold split on targets is deterministic).
    ``positive_pairs`` is a list of (drug_id, target_id).
    """
    root = dtiam_root / cfg["root"]
    # Drug CSV: id + SMILES.
    drugs = pd.read_csv(root / cfg["drug_file"], sep="\t")
    drugs = drugs.rename(columns={cfg["drug_id_col"]: "drug_id", cfg["drug_smiles_col"]: "smiles"})
    drug_to_smiles = dict(zip(drugs["drug_id"].astype(str), drugs["smiles"]))

    # Target CSV (sequence column not needed — embeddings come from the parquet).
    targets = pd.read_csv(root / cfg["target_file"], sep="\t")
    targets = targets.rename(columns={cfg["target_id_col"]: "target_id"})
    target_ids_file = targets["target_id"].astype(str).drop_duplicates().tolist()

    # DTI file (positives).
    if cfg["dti_has_header"]:
        dti_df = pd.read_csv(root / cfg["dti_file"], sep=cfg["dti_sep"])
        drug_col = cfg["dti_drug_col"]
        target_col = cfg["dti_target_col"]
    else:
        dti_df = pd.read_csv(root / cfg["dti_file"], sep=cfg["dti_sep"], header=None)
        drug_col = cfg["dti_drug_col"]
        target_col = cfg["dti_target_col"]
    pos_pairs = [
        (str(d), str(t))
        for d, t in zip(dti_df[drug_col].tolist(), dti_df[target_col].tolist())
    ]

    # Drug / target universe — DTIAM's source-of-truth convention.
    if cfg["drug_source"] == "dti":
        drug_ids = sorted({p[0] for p in pos_pairs})
    else:
        drug_ids = drugs["drug_id"].astype(str).drop_duplicates().tolist()
    if cfg["target_source"] == "dti":
        target_ids = sorted({p[1] for p in pos_pairs})
    else:
        target_ids = target_ids_file

    return {"drug_ids": drug_ids, "target_ids": target_ids,
            "drug_to_smiles": drug_to_smiles, "pos_pairs": pos_pairs}


def _sample_pairs(tables: dict, neg_ratio: int, neg_seed: int) -> pd.DataFrame:
    """Build the full (positive + sampled-negative) pair table for one subset.

    Follows DTIAM's algorithm (``code/data_process/data_split_{dti,moa}.py``):
    enumerate all (drug, target), subtract positives, uniform-sample ``neg_ratio
    * N_pos`` negatives. DTIAM leaves RNG unseeded; we seed it so reruns are
    byte-identical (paper parity on *distribution* only, not on individual
    pairs — see project memory).
    """
    drug_ids = tables["drug_ids"]
    target_ids = tables["target_ids"]
    pos_pairs = tables["pos_pairs"]

    rng = random.Random(neg_seed)  # isolated RNG — does not mutate global random state
    pos_set = set(pos_pairs)

    n_all = len(drug_ids) * len(target_ids)
    n_pos = len(pos_pairs)
    n_neg_target = n_pos * neg_ratio
    n_neg_available = n_all - n_pos
    if n_neg_target > n_neg_available:
        sys.exit(f"ERROR: requested {n_neg_target:,} negatives but only {n_neg_available:,} available.")

    # For modest sizes (≤ ~1M pairs) materialize the complement directly — this
    # matches DTIAM's code line-for-line. For Hetionet the complement is ~8M
    # tuples (~800 MB of Python objects) which is still tractable on a workstation.
    all_pairs = list(itertools.product(drug_ids, target_ids))
    neg_all = [p for p in all_pairs if p not in pos_set]
    assert len(neg_all) == n_neg_available
    neg_sampled = rng.sample(neg_all, n_neg_target)

    rows: list[tuple[str, str, int]] = []
    rows.extend((d, t, 1) for (d, t) in pos_pairs)
    rows.extend((d, t, 0) for (d, t) in neg_sampled)
    df = pd.DataFrame(rows, columns=["drug_id", "target_id", "label"])
    return df


def _build_parquet(
    subset: str,
    cfg: dict,
    pair_df: pd.DataFrame,
    tables: dict,
    protein_emb_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Join SMILES + protein embedding onto the pair table and write parquet.

    Rows whose drug has no SMILES or whose target has no ESM-C embedding are
    dropped (with a loud count); the split indices emitted later refer to this
    filtered frame's row order, so splits stay consistent with the parquet.
    """
    drug_to_smiles = tables["drug_to_smiles"]
    n_before = len(pair_df)
    known_drugs = set(drug_to_smiles.keys())
    known_targets = set(protein_emb_df.index)

    pair_df = pair_df[
        pair_df["drug_id"].isin(known_drugs) & pair_df["target_id"].isin(known_targets)
    ].reset_index(drop=True)
    n_after = len(pair_df)
    dropped_drug = sum(~pair_df["drug_id"].isin(known_drugs))  # always 0 after filter; report on original
    if n_before != n_after:
        # Report what was lost, by cause.
        orig_has_drug = pd.Series([d in known_drugs for d in pair_df["drug_id"]])  # placeholder
        print(f"  [{subset}] dropped {n_before - n_after:,} rows with missing SMILES or protein embedding")

    # Smiles column.
    pair_df["SMILES_nometa"] = pair_df["drug_id"].map(drug_to_smiles)

    # Protein embedding matrix in row order.
    prot_emb = protein_emb_df.loc[pair_df["target_id"].values].values.astype(np.float32)

    # Per-feature z-score. Unlike the existing DTI regression eval where labels
    # are scalars, here the label is a binary 0/1 so we only normalize the aux
    # features. Stats are computed once globally per subset.
    prot_mean = prot_emb.mean(axis=0).astype(np.float32)
    prot_std = prot_emb.std(axis=0)
    prot_std = np.where(prot_std < 1e-8, 1.0, prot_std).astype(np.float32)
    prot_emb_norm = ((prot_emb - prot_mean) / prot_std).astype(np.float32)

    out = pd.DataFrame({
        "SMILES_nometa": pair_df["SMILES_nometa"].values,
        "drug_id":       pair_df["drug_id"].values,
        "target_id":     pair_df["target_id"].values,
        "label":         pair_df["label"].astype(np.float32).values,
    })
    out[PROT_EMB_COLS] = prot_emb_norm

    parquet_path = output_dir / f"{subset}.parquet"
    out.to_parquet(parquet_path, index=False)
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    pos = int((out["label"] == 1).sum())
    neg = int((out["label"] == 0).sum())
    print(f"  [{subset}] wrote {parquet_path} ({len(out):,} rows, pos={pos:,} neg={neg:,}, {size_mb:.1f} MB)")

    # Stats file — mirrors the regression eval's norm_stats.pt shape (labels are
    # already 0/1 here; we only record protein-embedding stats).
    stats_path = output_dir / f"{subset}_norm_stats.pt"
    torch.save(
        {
            "prot_emb_mean": torch.from_numpy(prot_mean),
            "prot_emb_std":  torch.from_numpy(prot_std),
            "n_positive":    pos,
            "n_negative":    neg,
        },
        stats_path,
    )
    return out


def _generate_splits(
    subset: str,
    cfg: dict,
    pair_df: pd.DataFrame,
    val_fraction: float,
    val_seed: int,
    splits_dir: Path,
) -> None:
    """Emit 3 × n_folds split files. K-fold matches DTIAM's KFold(shuffle=True, random_state=0)."""
    n_folds = cfg["n_folds"]
    n_rows = len(pair_df)
    drug_arr = pair_df["drug_id"].values
    target_arr = pair_df["target_id"].values

    unique_drugs = np.array(sorted(pd.unique(drug_arr)))
    unique_targets = np.array(sorted(pd.unique(target_arr)))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    val_rng = random.Random(val_seed)

    def _carve_val(train_idx: np.ndarray) -> tuple[list[int], list[int]]:
        """Deterministically carve ``val_fraction`` of ``train_idx`` as val."""
        idx = train_idx.tolist()
        val_rng.shuffle(idx)  # uses val_rng's state — per-fold advancement is OK
        n_val = max(1, int(round(len(idx) * val_fraction)))
        return idx[n_val:], idx[:n_val]

    all_rows = np.arange(n_rows)

    for method in METHODS:
        # KFold yields (train_in_split_domain, test_in_split_domain). For warm
        # the domain is rows; for drug_cold it's the drug universe; for
        # target_cold it's the target universe.
        if method == "warm":
            split_iter = kf.split(all_rows)
        elif method == "drug_cold":
            split_iter = kf.split(unique_drugs)
        elif method == "target_cold":
            split_iter = kf.split(unique_targets)
        else:
            raise ValueError(method)

        for fold_idx, (train_dom, test_dom) in enumerate(split_iter):
            if method == "warm":
                train_rows = all_rows[train_dom]
                test_rows = all_rows[test_dom]
            elif method == "drug_cold":
                test_drugs = set(unique_drugs[test_dom].tolist())
                mask_test = np.isin(drug_arr, list(test_drugs))
                test_rows = all_rows[mask_test]
                train_rows = all_rows[~mask_test]
            else:  # target_cold
                test_targets = set(unique_targets[test_dom].tolist())
                mask_test = np.isin(target_arr, list(test_targets))
                test_rows = all_rows[mask_test]
                train_rows = all_rows[~mask_test]

            train_final, val_final = _carve_val(train_rows)
            split_indices = {
                "train": train_final,
                "val":   val_final,
                "test":  test_rows.tolist(),
            }

            out_path = splits_dir / f"{subset}_{method}_fold{fold_idx}.pt"
            torch.save(split_indices, out_path)
            print(
                f"  [{subset}/{method}/fold={fold_idx}] "
                f"train={len(split_indices['train']):,} "
                f"val={len(split_indices['val']):,} "
                f"test={len(split_indices['test']):,}  -> {out_path.name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtiam-root", required=True, help="Clone of CSUBioGroup/DTIAM.")
    parser.add_argument(
        "--protein-emb", required=True,
        help="Path to consolidated ESM-C embedding parquet (protein_id + feature_0..1151).",
    )
    parser.add_argument(
        "--output-dir", default="data/dti-classif-eval",
        help="Destination for per-subset parquet + splits/.",
    )
    parser.add_argument(
        "--subsets", nargs="+", default=list(SUBSET_CONFIG.keys()),
        choices=list(SUBSET_CONFIG.keys()),
    )
    parser.add_argument("--neg-ratio", type=int, default=10)
    parser.add_argument("--neg-seed", type=int, default=0,
                        help="Seed for negative sampling. DTIAM's code is unseeded; we seed for reproducibility.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-seed", type=int, default=42)
    args = parser.parse_args()

    dtiam_root = Path(args.dtiam_root)
    if not dtiam_root.is_dir():
        sys.exit(f"ERROR: --dtiam-root {dtiam_root} not found.")

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    protein_emb_df = _load_protein_embeddings(args.protein_emb)

    for subset in args.subsets:
        cfg = SUBSET_CONFIG[subset]
        print(f"\n=== {subset} ===")
        tables = _load_subset_tables(subset, cfg, dtiam_root)
        print(
            f"  drugs={len(tables['drug_ids']):,}  targets={len(tables['target_ids']):,}  "
            f"positives={len(tables['pos_pairs']):,}"
        )
        pair_df = _sample_pairs(tables, args.neg_ratio, args.neg_seed)
        pair_df = _build_parquet(subset, cfg, pair_df, tables, protein_emb_df, output_dir)
        _generate_splits(subset, cfg, pair_df, args.val_fraction, args.val_seed, splits_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
