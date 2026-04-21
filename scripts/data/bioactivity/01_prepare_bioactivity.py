#!/usr/bin/env python
"""Prepare the 29-assay cell_bioactivity dataset (Fredinh et al. 2024).

Faithful port of ``github.com/cfredinh/bioactive`` data-prep
(``data_prep/prep_activity_data.py`` + ``data_prep/data_splitting.py``),
writing to the new downstream layout (``data/downstream/bioactivity/``).

Pipeline
--------
1. Join ChEMBL ``compound_structures`` with JUMP ``compound.csv.gz`` on
   InChIKey to get the overlapping compound set.
2. Pull ``activities`` where ``standard_type == Potency`` and
   ``activity_comment in {Active, active, Not Active, inactive}``. Binarize
   to +1 / -1.
3. Keep assays with >100 activity records. Pivot to (molregno x assay_id) with
   aggfunc=median. NaN-fill to 0 ("unknown"). Keep assays with >50 pos AND
   >50 neg overall.
4. Restrict to JUMP ``source_11`` compounds; re-apply the >50/>50 filter on
   that subset (mirrors the paper).
5. Butina-cluster on ECFP4 (r=2, 1024 bits, cutoff=0.7); greedily distribute
   clusters into 6 folds.
6. Write ``cell_bioactivity.csv`` (``smiles`` + ``assay_<id>`` + ``fold``) and
   ``cell_bioactivity_split.csv`` (train/val/test row indices, for legacy
   consumers that don't use the per-row ``fold`` column).

Prerequisites
-------------
- ChEMBL 33 SQLite DB (``chembl_33.db``, ~20 GB). Download from
  https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
- JUMP-CP metadata (``well.csv.gz``, ``compound.csv.gz``).

Usage
-----
    python scripts/data/bioactivity/01_prepare_bioactivity.py \\
        --chembl-db /path/to/chembl_33.db \\
        [--jump-metadata-dir /home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata] \\
        [--out-dir data/downstream/bioactivity] \\
        [--seed 0]
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


EXPECTED_ASSAY_IDS: List[int] = [
    688128, 688238, 688360, 688546, 688549, 688612, 688687, 688812, 688816,
    736947, 737187, 737287, 737344, 752347, 752407, 752434, 752493, 752563,
    752590, 752594, 845045, 845102, 845164, 845169, 845173, 845177, 845196,
    954338, 1495346,
]

ACTIVE_COMMENTS = {"Active", "active"}
INACTIVE_COMMENTS = {"Not Active", "inactive"}


def _load_chembl(chembl_db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    logger.info(f"ChEMBL: {len(compounds):,} compounds, {len(activities):,} filtered activities")
    return compounds, activities


def _binarize(activities: pd.DataFrame) -> pd.DataFrame:
    y = np.zeros(len(activities), dtype=np.int8)
    y[activities.activity_comment.isin(ACTIVE_COMMENTS)] = 1
    y[activities.activity_comment.isin(INACTIVE_COMMENTS)] = -1
    out = activities[["molregno", "assay_id"]].copy()
    out["y"] = y
    counts = out.assay_id.value_counts()
    keep = counts[counts > 100].index
    out = out[out.assay_id.isin(keep)]
    logger.info(f">100-record filter: {out.assay_id.nunique():,} assays")
    return out


def _label_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(
        values="y", index="molregno", columns="assay_id", aggfunc=np.median,
    ).fillna(0)


def _balance_filter(mat: pd.DataFrame, min_pos: int = 50, min_neg: int = 50) -> pd.DataFrame:
    pos = (mat == 1.0).sum() > min_pos
    neg = (mat == -1.0).sum() > min_neg
    keep = mat.columns[pos & neg]
    logger.info(f"balance filter ({min_pos}+/{min_neg}-): kept {len(keep)}/{mat.shape[1]} assays")
    return mat[keep]


def _source11_inchikeys(jump_metadata_dir: Path) -> set[str]:
    compound = pd.read_csv(jump_metadata_dir / "compound.csv.gz")
    well = pd.read_csv(jump_metadata_dir / "well.csv.gz")
    src11_jcp = well.loc[well.Metadata_Source == "source_11", "Metadata_JCP2022"].unique()
    src11_inchi = compound.loc[
        compound.Metadata_JCP2022.isin(src11_jcp), "Metadata_InChIKey"
    ].unique()
    logger.info(f"JUMP source_11: {len(src11_jcp):,} wells -> {len(src11_inchi):,} InChIKeys")
    return set(src11_inchi)


def _ecfp4(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


def _butina_clusters(fps: list, cutoff: float = 0.7) -> list[list[int]]:
    nfps = len(fps)
    dists: list[float] = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in sims)
    return [list(c) for c in Butina.ClusterData(dists, nfps, cutoff, isDistData=True)]


def _assign_folds(clusters: list[list[int]], n_rows: int, n_folds: int = 6, seed: int = 0) -> np.ndarray:
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
    while ci < len(clusters):
        for idx in clusters[ci]:
            folds[idx] = n_folds - 1
        ci += 1
    return folds


def _write_csv(smiles: pd.Series, label_matrix: pd.DataFrame, folds: np.ndarray, out_csv: Path) -> None:
    out = label_matrix.astype(float).replace({-1.0: 0.0, 0.0: np.nan, 1.0: 1.0})
    missing = [a for a in EXPECTED_ASSAY_IDS if a not in out.columns]
    if missing:
        logger.warning(f"{len(missing)} expected assays absent; writing as all-NaN")
        for a in missing:
            out[a] = np.nan
    out = out[EXPECTED_ASSAY_IDS]
    out.columns = [f"assay_{a}" for a in EXPECTED_ASSAY_IDS]
    out.insert(0, "smiles", smiles.values)
    out["fold"] = folds
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info(f"wrote {out_csv}  ({len(out):,} rows x {out.shape[1]} cols)")


def _write_splits(folds: np.ndarray, out_split: Path) -> None:
    train = np.where(np.isin(folds, [0, 1, 2, 3]))[0].tolist()
    val = np.where(folds == 4)[0].tolist()
    test = np.where(folds == 5)[0].tolist()
    width = max(len(train), len(val), len(test))
    pad = lambda xs: xs + [np.nan] * (width - len(xs))
    pd.DataFrame({"train": pad(train), "val": pad(val), "test": pad(test)}).to_csv(out_split, index=False)
    logger.info(f"wrote {out_split}  (train={len(train)} val={len(val)} test={len(test)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chembl-db", type=Path, required=True,
                   help="Path to chembl_33.db (SQLite).")
    p.add_argument("--jump-metadata-dir", type=Path,
                   default=Path("/home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata"))
    p.add_argument("--out-dir", type=Path, default=Path("data/downstream/bioactivity"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    compounds, activities = _load_chembl(args.chembl_db)
    jump_compound = pd.read_csv(args.jump_metadata_dir / "compound.csv.gz")
    overlap = compounds[compounds.standard_inchi_key.isin(jump_compound.Metadata_InChIKey)]
    logger.info(f"ChEMBL <-> JUMP overlap: {len(overlap):,} compounds")
    activities = activities[activities.molregno.isin(overlap.molregno)]

    long_df = _binarize(activities)
    mat = _balance_filter(_label_matrix(long_df))

    src11 = _source11_inchikeys(args.jump_metadata_dir)
    src11_molregno = overlap.loc[overlap.standard_inchi_key.isin(src11), "molregno"].values
    mat = mat.loc[mat.index.intersection(src11_molregno)]
    mat = _balance_filter(mat)
    logger.info(f"source_11 subset: {len(mat):,} compounds × {mat.shape[1]} assays")

    smiles_lookup = compounds.set_index("molregno").canonical_smiles
    smiles = mat.index.to_series().map(smiles_lookup)
    fps = smiles.apply(_ecfp4)
    valid = fps.notna()
    if (~valid).any():
        logger.warning(f"{int((~valid).sum())} compounds dropped (bad SMILES)")
        mat = mat.loc[valid.values]
        smiles = smiles.loc[valid.values]
        fps = fps.loc[valid.values]

    logger.info(f"Butina-clustering {len(fps):,} compounds (ECFP4, cutoff=0.7)")
    clusters = _butina_clusters(list(fps), cutoff=0.7)
    folds = _assign_folds(clusters, n_rows=len(fps), n_folds=6, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(smiles, mat, folds, args.out_dir / "cell_bioactivity.csv")
    _write_splits(folds, args.out_dir / "cell_bioactivity_split.csv")
    logger.info("done")


if __name__ == "__main__":
    main()
