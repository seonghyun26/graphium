#!/usr/bin/env python
"""Build downstream task SQLite database from raw source files.

Tables:
  tdc_admet        22 TDC ADMET tasks (scaffold split, train_val / test)
  adme_fang        6 Polaris ADME-Fang tasks (per-task scaffold split by seed)
  gram_dti         4 DTIAM datasets × 3 split types × 5 folds (binary DTI)
  cell_bioactivity 30 cell bioactivity assays (Fredinh et al. 2024)

Usage (from graphium/):
  python data/build_downstream_db.py
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPHIUM    = Path(__file__).parent.parent
DATA        = GRAPHIUM / "data"
DATACACHE   = Path("/home/shpark/prj-molrepr/datacache")
DOWNSTREAM  = Path("/home/shpark/prj-molrepr/data/downstream")
DB          = DATA / "db" / "downstream.db"

# ---------------------------------------------------------------------------
# SMILES canonicalization helpers
# ---------------------------------------------------------------------------

def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol) if mol is not None else smi


def canon_series(s: pd.Series) -> pd.Series:
    """Vectorised canonicalize: de-dup, map, fill in one pass."""
    unique = s.dropna().unique()
    m = {x: _canon(x) for x in unique}
    return s.map(m)


def n_atoms_series(s: pd.Series) -> pd.Series:
    """Heavy-atom count per SMILES; de-duped for speed."""
    unique = s.dropna().unique()
    m = {}
    for smi in unique:
        mol = Chem.MolFromSmiles(smi)
        m[smi] = mol.GetNumAtoms() if mol is not None else None
    return s.map(m)


# ---------------------------------------------------------------------------
# Split-file loader  (Polaris / cell-bioactivity style)
# Each column contains row-indices into the source DataFrame, NaN-padded.
# ---------------------------------------------------------------------------

def load_split_df(path: Path) -> dict[str, np.ndarray]:
    """Return {split_name: int-array of row indices}."""
    df = pd.read_csv(path)
    result = {}
    for col in ["train", "val", "test"]:
        if col in df.columns:
            result[col] = df[col].dropna().astype(int).values
    return result


def make_split_col(n: int, idx_map: dict[str, np.ndarray]) -> np.ndarray:
    arr = np.full(n, None, dtype=object)
    for name, idxs in idx_map.items():
        arr[idxs] = name
    return arr


# ---------------------------------------------------------------------------
# Table 1 – TDC ADMET
# ---------------------------------------------------------------------------

_TDC_TYPE = {
    **{t: "cls" for t in [
        "ames", "bbb_martins", "bioavailability_ma",
        "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels", "cyp2d6_veith",
        "cyp3a4_substrate_carbonmangels", "cyp3a4_veith",
        "dili", "herg", "hia_hou", "pgp_broccatelli",
    ]},
    **{t: "reg" for t in [
        "caco2_wang", "clearance_hepatocyte_az", "clearance_microsome_az",
        "half_life_obach", "ld50_zhu", "lipophilicity_astrazeneca",
        "ppbr_az", "solubility_aqsoldb", "vdss_lombardo",
    ]},
}


def build_tdc_admet(conn: sqlite3.Connection) -> None:
    tdc_root = DATA / "tdc" / "admet_group"
    rows = []
    for task in sorted(os.listdir(tdc_root)):
        if task.startswith("."):
            continue
        td = tdc_root / task
        for fname, split in [("train_val.csv", "train_val"), ("test.csv", "test")]:
            df = pd.read_csv(td / fname)
            df["smiles_canon"] = canon_series(df["Drug"])
            for _, r in df.iterrows():
                rows.append((
                    r["smiles_canon"],
                    str(r["Drug_ID"]),
                    task,
                    float(r["Y"]),
                    _TDC_TYPE.get(task, "unknown"),
                    split,
                ))

    out = pd.DataFrame(rows, columns=["smiles", "drug_id", "task", "value", "task_type", "split"])
    out["n_atoms"] = n_atoms_series(out["smiles"])
    out.to_sql("tdc_admet", conn, if_exists="replace", index=False)
    print(f"  tdc_admet:        {len(out):>8,} rows | {out['task'].nunique()} tasks")


# ---------------------------------------------------------------------------
# Table 2 – ADME-Fang (Polaris biogen/adme-fang-v1)
# One row per SMILES; each task gets a value column + split_seed0/seed42 column.
# ---------------------------------------------------------------------------

_FANG_TASKS = {
    "hclint": "LOG_HLM_CLint",
    "rclint": "LOG_RLM_CLint",
    "perm":   "LOG_MDR1-MDCK_ER",
    "hppb":   "LOG_HPPB",
    "rppb":   "LOG_RPPB",
    "solu":   "LOG_SOLUBILITY",
}


def build_adme_fang(conn: sqlite3.Connection) -> None:
    root = DATACACHE / "polaris_adme_fang"
    df = pd.read_parquet(root / "adme-fang-v1.parquet")
    df.index = range(len(df))

    out = pd.DataFrame({
        "smiles":    canon_series(df["SMILES"]),
        "unique_id": df["UNIQUE_ID"],
    })
    for short, col in _FANG_TASKS.items():
        out[short] = df[col].values

    n_split_cols = 0
    for task in _FANG_TASKS:
        for seed in [0, 42]:
            col = f"{task}_split_seed{seed}"
            fname = root / f"adme-fang-{task}-reg-v1_seed{seed}_split.csv"
            if fname.exists():
                out[col] = make_split_col(len(df), load_split_df(fname))
                n_split_cols += 1
            else:
                out[col] = None

    out["n_atoms"] = n_atoms_series(out["smiles"])
    out.to_sql("adme_fang", conn, if_exists="replace", index=False)
    print(f"  adme_fang:        {len(out):>8,} rows | 6 tasks | {n_split_cols} split cols")


# ---------------------------------------------------------------------------
# Table 3 – GRAM-DTI (dti-classif-eval)
# Exploded: one row per (smiles, target_id, dataset, split_type, fold).
# ---------------------------------------------------------------------------

_DTI_DATASETS    = ["activation", "inhibition", "hetionet", "yamanishi_08"]
_DTI_SPLIT_TYPES = ["warm", "drug_cold", "target_cold"]
_DTI_N_FOLDS     = 5


def build_gram_dti(conn: sqlite3.Connection) -> None:
    dti_root   = DATA / "dti-classif-eval"
    total_rows = 0
    first      = True

    for dataset in _DTI_DATASETS:
        df = pd.read_parquet(
            dti_root / f"{dataset}.parquet",
            columns=["SMILES_nometa", "drug_id", "target_id", "label"],
        )
        df["smiles"]  = canon_series(df["SMILES_nometa"])
        df["label"]   = df["label"].astype(int)
        df["dataset"] = dataset
        core = df[["smiles", "drug_id", "target_id", "dataset", "label"]].reset_index(drop=True)
        core["n_atoms"] = n_atoms_series(core["smiles"])
        n = len(core)

        dataset_rows = 0
        for split_type in _DTI_SPLIT_TYPES:
            for fold in range(_DTI_N_FOLDS):
                sf = dti_root / "splits" / f"{dataset}_{split_type}_fold{fold}.pt"
                if not sf.exists():
                    continue

                split_dict = torch.load(sf)
                split_arr  = np.full(n, None, dtype=object)
                for sname, idxs in split_dict.items():
                    split_arr[np.asarray(idxs)] = sname

                chunk = core.assign(split_type=split_type, fold=fold, split=split_arr)
                chunk.to_sql(
                    "gram_dti", conn,
                    if_exists="replace" if first else "append",
                    index=False, chunksize=100_000,
                )
                first = False
                dataset_rows += len(chunk)

        print(f"    {dataset:<20}: {dataset_rows:>9,} rows")
        total_rows += dataset_rows

    print(f"  gram_dti total:   {total_rows:>8,} rows")


# ---------------------------------------------------------------------------
# Table 4 – Cell bioactivity (Fredinh et al. 2024)
# ---------------------------------------------------------------------------

def build_cell_bioactivity(conn: sqlite3.Connection) -> None:
    bio_root = DOWNSTREAM / "bioactivity"
    df       = pd.read_csv(bio_root / "cell_bioactivity.csv")
    df["smiles"] = canon_series(df["smiles"])

    idx_map      = load_split_df(bio_root / "cell_bioactivity_split.csv")
    df["split"]  = make_split_col(len(df), idx_map)

    if "fold" in df.columns:
        df = df.rename(columns={"fold": "cv_fold"})

    df["n_atoms"] = n_atoms_series(df["smiles"])
    assay_cols = [c for c in df.columns if c.startswith("assay_")]
    df.to_sql("cell_bioactivity", conn, if_exists="replace", index=False)
    print(f"  cell_bioactivity: {len(df):>8,} rows | {len(assay_cols)} assays")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()

    conn = sqlite3.connect(DB)

    print("Building tdc_admet ...")
    build_tdc_admet(conn)

    print("Building adme_fang ...")
    build_adme_fang(conn)

    print("Building gram_dti ...")
    build_gram_dti(conn)

    print("Building cell_bioactivity ...")
    build_cell_bioactivity(conn)

    print("Creating indices ...")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_tdc_smiles  ON tdc_admet(smiles);
        CREATE INDEX IF NOT EXISTS idx_tdc_task    ON tdc_admet(task, split);
        CREATE INDEX IF NOT EXISTS idx_fang_smiles ON adme_fang(smiles);
        CREATE INDEX IF NOT EXISTS idx_dti_smiles  ON gram_dti(smiles);
        CREATE INDEX IF NOT EXISTS idx_dti_lookup  ON gram_dti(dataset, split_type, fold, split);
        CREATE INDEX IF NOT EXISTS idx_bio_smiles  ON cell_bioactivity(smiles);
    """)

    conn.commit()

    # Summary
    print("\n── Table summary ──────────────────────────────────")
    for table in ["tdc_admet", "adme_fang", "gram_dti", "cell_bioactivity"]:
        (cnt,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table:<20}: {cnt:>10,} rows")

    conn.close()
    sz = DB.stat().st_size
    print(f"\nDatabase → {DB}  ({sz / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
