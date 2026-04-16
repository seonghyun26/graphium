"""
Prepare the 29-assay cell_bioactivity dataset (Fredinh et al., Nat. Commun. 2024).

This is a faithful port of the upstream data-prep pipeline from
github.com/cfredinh/bioactive (data_prep/prep_activity_data.py +
data_prep/data_splitting.py), adapted to graphium's CSV conventions:

- Output `cell_bioactivity.csv` has columns [smiles, assay_<id>, ...] with
  0/1/NaN labels (upstream uses -1/1/0 where 0 means unknown; we remap).
- Output `cell_bioactivity_split.csv` has [train, val, test] columns of
  positional row indices, padded with NaN (graphium splits format).

Inputs required
---------------
- ChEMBL 33 SQLite DB (`chembl_33.db`) — download from
  https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ (~20 GB). Pass
  its path via --chembl-db.
- JUMP-CP metadata (well.csv.gz, compound.csv.gz) under --jump-metadata-dir.
  Defaults to /home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata which
  already has both files.

Pipeline
--------
1. Join ChEMBL `compound_structures` with JUMP `compound.csv.gz` on InChIKey
   to get the overlapping compound set (`molregno`).
2. Pull `activities` for those molregnos; keep
   standard_type == "Potency" and activity_comment in {Active, active,
   Not Active, inactive}. Binarize to +1/-1.
3. Keep assays with >100 records. Pivot to (molregno x assay_id) with
   aggfunc=median. NaN-fill to 0 (0 = "unknown"). Keep assays with >50
   positives and >50 negatives overall.
4. Restrict to JUMP `source_11` compounds and re-apply the >50/>50 filter
   on that subset (mirrors the paper).
5. Butina-cluster on ECFP4 (r=2, 1024 bits, cutoff=0.7); distribute clusters
   greedily into 6 folds, assign train/val/test = {0,1,2,3}/{4}/{5}.
6. Write graphium-format CSV + splits CSV.

Usage
-----
    python scripts/prep_cell_bioactivity.py \\
        --chembl-db /path/to/chembl_33.db \\
        [--jump-metadata-dir /home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata] \\
        [--out-dir /home/shpark/prj-molrepr/datacache/cell_bioactivity] \\
        [--seed 0]

Notes
-----
The 29 assay_ids listed in expts/hydra-configs/tasks/loss_metrics_datamodule/
cell_bioactivity.yaml were produced by the upstream repo from ChEMBL 33. If
this script's filter output disagrees with that list (e.g. a newer ChEMBL
release dropped records), the CSV is still written for whatever passed the
filter and a warning is logged — update the yaml `label_cols` to match.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

# Assay IDs hard-coded in the graphium yaml. We write the CSV's assay_* columns
# in this order; any missing assays become all-NaN columns (with a warning).
EXPECTED_ASSAY_IDS: List[int] = [
    688128, 688238, 688360, 688546, 688549, 688612, 688687, 688812, 688816,
    736947, 737187, 737287, 737344, 752347, 752407, 752434, 752493, 752563,
    752590, 752594, 845045, 845102, 845164, 845169, 845173, 845177, 845196,
    954338, 1495346,
]

ACTIVE_COMMENTS = {"Active", "active"}
INACTIVE_COMMENTS = {"Not Active", "inactive"}


def load_chembl_tables(chembl_db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (compound_structures, activities) from the ChEMBL SQLite DB.

    We only pull rows we'll actually use: compound_structures needs molregno +
    standard_inchi_key + canonical_smiles; activities needs molregno, assay_id,
    standard_type, activity_comment.
    """
    logger.info(f"opening ChEMBL DB at {chembl_db}")
    con = sqlite3.connect(str(chembl_db))
    try:
        compounds = pd.read_sql_query(
            "SELECT molregno, standard_inchi_key, canonical_smiles "
            "FROM compound_structures",
            con,
        )
        activities = pd.read_sql_query(
            "SELECT molregno, assay_id, standard_type, activity_comment "
            "FROM activities "
            "WHERE standard_type = 'Potency' "
            "AND activity_comment IN ('Active', 'active', 'Not Active', 'inactive')",
            con,
        )
    finally:
        con.close()
    logger.info(f"loaded {len(compounds):,} compound_structures, "
                f"{len(activities):,} filtered activities")
    return compounds, activities


def binarize_activities(activities: pd.DataFrame) -> pd.DataFrame:
    """Map activity_comment -> {+1, -1}. Returns columns [molregno, assay_id, y]."""
    y = np.zeros(len(activities), dtype=np.int8)
    y[activities.activity_comment.isin(ACTIVE_COMMENTS)] = 1
    y[activities.activity_comment.isin(INACTIVE_COMMENTS)] = -1
    out = activities[["molregno", "assay_id"]].copy()
    out["y"] = y
    # Paper drops assays with < 100 records before pivoting.
    counts = out.assay_id.value_counts()
    keep = counts[counts > 100].index
    out = out[out.assay_id.isin(keep)]
    logger.info(f"after >100-record filter: {out.assay_id.nunique():,} assays")
    return out


def build_label_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to molregno x assay_id with aggfunc=median (mirrors upstream).

    Upstream code then `.fillna(0)`'s this so 0 encodes "unknown"; we keep the
    same encoding here and defer NaN-conversion to write_csv.
    """
    mat = long_df.pivot_table(
        values="y", index="molregno", columns="assay_id", aggfunc=np.median
    ).fillna(0)
    return mat


def prune_assays_by_balance(
    label_matrix: pd.DataFrame, min_pos: int = 50, min_neg: int = 50
) -> pd.DataFrame:
    """Drop assays lacking > min_pos positives OR > min_neg negatives."""
    pos_ok = (label_matrix == 1.0).sum() > min_pos
    neg_ok = (label_matrix == -1.0).sum() > min_neg
    keep = label_matrix.columns[pos_ok & neg_ok]
    logger.info(f"balance filter ({min_pos}+/{min_neg}-): kept {len(keep)}/"
                f"{label_matrix.shape[1]} assays")
    return label_matrix[keep]


def load_jump_source11_inchikeys(jump_metadata_dir: Path) -> set[str]:
    """Return the set of InChIKeys of compounds measured in JUMP source_11."""
    compound = pd.read_csv(jump_metadata_dir / "compound.csv.gz")
    well = pd.read_csv(jump_metadata_dir / "well.csv.gz")

    src11_jcp = well.loc[well.Metadata_Source == "source_11", "Metadata_JCP2022"].unique()
    src11_inchi = compound.loc[
        compound.Metadata_JCP2022.isin(src11_jcp), "Metadata_InChIKey"
    ].unique()
    logger.info(f"JUMP source_11: {len(src11_jcp):,} wells -> "
                f"{len(src11_inchi):,} unique InChIKeys")
    return set(src11_inchi)


def ecfp4_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


def butina_clusters(fps: list, cutoff: float = 0.7) -> list[list[int]]:
    """Upstream `ClusterFps`: all-pairs Tanimoto, cluster via Butina."""
    nfps = len(fps)
    dists: list[float] = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in sims)
    return [list(c) for c in Butina.ClusterData(dists, nfps, cutoff, isDistData=True)]


def assign_folds(clusters: list[list[int]], n_rows: int, n_folds: int = 6,
                 seed: int = 0) -> np.ndarray:
    """Shuffle clusters and greedily fill n_folds so each fold gets ~n_rows/n_folds
    compounds. Mirrors `split_data_cross_validation_structure_clustering`.
    """
    rng = random.Random(seed)
    clusters = clusters[:]
    rng.shuffle(clusters)

    target = n_rows // n_folds
    folds = np.full(n_rows, -1, dtype=np.int8)
    ci = 0
    for f in range(n_folds):
        filled = 0
        while filled <= target and ci < len(clusters):
            for idx in clusters[ci]:
                folds[idx] = f
                filled += 1
            ci += 1
    # Any leftover clusters (can happen when target*n_folds < n_rows) land in last fold.
    while ci < len(clusters):
        for idx in clusters[ci]:
            folds[idx] = n_folds - 1
        ci += 1
    return folds


def write_graphium_csv(
    smiles: pd.Series, label_matrix: pd.DataFrame, folds: np.ndarray, out_csv: Path
) -> pd.DataFrame:
    """Encode label_matrix (-1/0/+1) as graphium-friendly (0/NaN/1) CSV with a
    leading `smiles` column and a trailing `fold` column (Butina fold id 0..5 per
    row). Writes assay columns in EXPECTED_ASSAY_IDS order; missing assays become
    all-NaN.

    The `fold` column exists so downstream code can do 6-fold CV without parsing
    the pre-packed splits CSV (which lumps folds 0-3 into a single `train` list).
    """
    # -1 inactive -> 0, +1 active -> 1, 0 unknown -> NaN.
    out = label_matrix.astype(float).replace({-1.0: 0.0, 0.0: np.nan, 1.0: 1.0})

    missing = [a for a in EXPECTED_ASSAY_IDS if a not in out.columns]
    if missing:
        logger.warning(
            f"{len(missing)} expected assays absent after filtering "
            f"(will be written as all-NaN): {missing}"
        )
        for a in missing:
            out[a] = np.nan

    out = out[EXPECTED_ASSAY_IDS]
    out.columns = [f"assay_{a}" for a in EXPECTED_ASSAY_IDS]
    out.insert(0, "smiles", smiles.values)
    out["fold"] = folds

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info(f"wrote {out_csv}  ({len(out):,} rows x {out.shape[1]} cols)")
    return out


def write_splits_csv(folds: np.ndarray, out_split: Path) -> None:
    train = np.where(np.isin(folds, [0, 1, 2, 3]))[0].tolist()
    val = np.where(folds == 4)[0].tolist()
    test = np.where(folds == 5)[0].tolist()
    width = max(len(train), len(val), len(test))

    def pad(xs):
        return xs + [np.nan] * (width - len(xs))

    pd.DataFrame({"train": pad(train), "val": pad(val), "test": pad(test)}).to_csv(
        out_split, index=False
    )
    logger.info(f"wrote {out_split}  (train={len(train)} val={len(val)} test={len(test)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chembl-db", type=Path, required=True,
                   help="Path to chembl_33.db (SQLite).")
    p.add_argument("--jump-metadata-dir", type=Path,
                   default=Path("/home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/home/shpark/prj-molrepr/datacache/cell_bioactivity"))
    p.add_argument("--seed", type=int, default=0, help="Butina cluster shuffle seed.")
    args = p.parse_args()

    # 1. ChEMBL <-> JUMP overlap on InChIKey.
    compounds, activities = load_chembl_tables(args.chembl_db)
    jump_compound = pd.read_csv(args.jump_metadata_dir / "compound.csv.gz")
    jump_inchi = set(jump_compound.Metadata_InChIKey.unique())
    overlap = compounds[compounds.standard_inchi_key.isin(jump_inchi)]
    logger.info(f"ChEMBL <-> JUMP overlap: {len(overlap):,} compounds")
    activities = activities[activities.molregno.isin(overlap.molregno)]

    # 2-3. Binarize, pivot, balance-filter (overall).
    long_df = binarize_activities(activities)
    label_matrix = build_label_matrix(long_df)
    label_matrix = prune_assays_by_balance(label_matrix, min_pos=50, min_neg=50)

    # 4. Restrict to JUMP source_11 and rebalance.
    src11_inchi = load_jump_source11_inchikeys(args.jump_metadata_dir)
    src11_molregno = overlap.loc[
        overlap.standard_inchi_key.isin(src11_inchi), "molregno"
    ].values
    mat = label_matrix.loc[label_matrix.index.intersection(src11_molregno)]
    mat = prune_assays_by_balance(mat, min_pos=50, min_neg=50)
    logger.info(f"source_11 subset: {len(mat):,} compounds x {mat.shape[1]} assays")

    # 5. Compute SMILES via ChEMBL's canonical_smiles (preferred over InChI roundtrip).
    smiles_lookup = compounds.set_index("molregno").canonical_smiles
    smiles = mat.index.to_series().map(smiles_lookup)
    # Drop rows without canonical SMILES or with unparseable SMILES.
    fps = smiles.apply(ecfp4_fingerprint)
    valid = fps.notna()
    dropped = (~valid).sum()
    if dropped:
        logger.warning(f"{dropped} compounds dropped (missing/unparseable SMILES)")
        mat = mat.loc[valid.values]
        smiles = smiles.loc[valid.values]
        fps = fps.loc[valid.values]

    # 6. Butina fold assignment.
    logger.info(f"Butina-clustering {len(fps):,} compounds (ECFP4, cutoff=0.7)")
    clusters = butina_clusters(list(fps), cutoff=0.7)
    logger.info(f"got {len(clusters):,} clusters")
    folds = assign_folds(clusters, n_rows=len(fps), n_folds=6, seed=args.seed)

    # 7. Write graphium-format outputs.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_graphium_csv(smiles, mat, folds, args.out_dir / "cell_bioactivity.csv")
    write_splits_csv(folds, args.out_dir / "cell_bioactivity_split.csv")
    logger.info("done")


if __name__ == "__main__":
    main()
