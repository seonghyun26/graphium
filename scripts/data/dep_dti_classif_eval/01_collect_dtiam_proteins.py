#!/usr/bin/env python
"""Stage 1: Collect unique (protein_id, sequence) rows across the 4 DTIAM datasets,
restricted to proteins that will actually be referenced by the GRAM-DTI eval.

Produces a single CSV suitable as input to the existing ESM-C pipeline at
``scripts/data/dti_esmc/02_extract_embeddings_esmc.py``. The downstream GRAM-DTI
benchmark (Yamanishi 08, Hetionet, Activation, Inhibition) uses three different
protein-ID conventions (``hsa:NNNN`` for Yamanishi, ``Gene::NNNNN`` for Hetionet,
``TNNNNN`` for MoA), which don't overlap. We keep the DTIAM IDs as-is so the
later prep stage can index by them directly.

Used-protein sets — matches ``02_prepare_dti_classif.py``'s target universe:

    Yamanishi 08 / Hetionet (DTI) : proteins that appear in dti.csv (positives)
    Activation / Inhibition (MoA) : all proteins in tar_seq.csv (candidate set
                                    used for negative sampling)

This matters for Hetionet: its protein_seq.csv is a ~3x superset (~19k rows) of
the 5,763 target proteins the benchmark actually uses, and embedding the extras
would triple the ESM-C wall-clock for nothing.

Usage:
    python scripts/data/dti_classif_eval/01_collect_dtiam_proteins.py \\
        --dtiam-root /tmp/DTIAM \\
        --output data/dti-classif-eval/dtiam-proteins.csv

Then (activate the esmc env):
    conda activate esmc
    python scripts/data/dti_esmc/02_extract_embeddings_esmc.py \\
        --input  data/dti-classif-eval/dtiam-proteins.csv \\
        --output-dir data/dti-classif-eval/protein-esmc \\
        --gpus 0,1
    python scripts/data/dti_esmc/03_consolidate_embeddings_esmc.py \\
        --embedding-dir data/dti-classif-eval/protein-esmc/embedding \\
        --protein-csv  data/dti-classif-eval/dtiam-proteins.csv \\
        --output-dir   data/dti-classif-eval
    # -> data/dti-classif-eval/protein-esmc.parquet (feeds stage 2 of data prep)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Per-dataset: (seq_file, seq_id_col, seq_col, usage_file, usage_id_col, usage_has_header, usage_sep)
# ``usage_file`` is the file we read to determine the *used* protein set. For
# DTI, that's dti.csv column 2 (headerless, tab-sep). For MoA, it's tar_seq.csv
# itself — every row is a candidate, so the filter is a no-op.
SUBSETS = {
    "yamanishi_08": {
        "seq_file":  "data/dti/yamanishi_08/protein_seq.csv",
        "seq_id_col": "pro_id", "seq_col": "seq",
        "usage_file": "data/dti/yamanishi_08/dti.csv",
        "usage_id_col": 2, "usage_has_header": False, "usage_sep": "\t",
    },
    "hetionet": {
        "seq_file":  "data/dti/hetionet/protein_seq.csv",
        "seq_id_col": "pro_id", "seq_col": "seq",
        "usage_file": "data/dti/hetionet/dti.csv",
        "usage_id_col": 2, "usage_has_header": False, "usage_sep": "\t",
    },
    "activation": {
        "seq_file":  "data/moa/activation/tar_seq.csv",
        "seq_id_col": "TargetID", "seq_col": "seq",
        "usage_file": "data/moa/activation/tar_seq.csv",  # same file; every row is a candidate
        "usage_id_col": "TargetID", "usage_has_header": True, "usage_sep": "\t",
    },
    "inhibition": {
        "seq_file":  "data/moa/inhibition/tar_seq.csv",
        "seq_id_col": "TargetID", "seq_col": "seq",
        "usage_file": "data/moa/inhibition/tar_seq.csv",
        "usage_id_col": "TargetID", "usage_has_header": True, "usage_sep": "\t",
    },
}


def _read_used_ids(root: Path, cfg: dict) -> set[str]:
    """Return the set of protein IDs actually referenced by the benchmark."""
    path = root / cfg["usage_file"]
    if cfg["usage_has_header"]:
        df = pd.read_csv(path, sep=cfg["usage_sep"])
        col = cfg["usage_id_col"]
    else:
        df = pd.read_csv(path, sep=cfg["usage_sep"], header=None)
        col = cfg["usage_id_col"]
    return set(df[col].astype(str).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dtiam-root", required=True,
        help="Clone of https://github.com/CSUBioGroup/DTIAM (expects data/dti/* and data/moa/* inside).",
    )
    parser.add_argument(
        "--output", default="data/dti-classif-eval/dtiam-proteins.csv",
        help="Destination CSV. Columns: protein_id, sequence.",
    )
    args = parser.parse_args()

    dtiam_root = Path(args.dtiam_root)
    if not dtiam_root.is_dir():
        sys.exit(f"ERROR: --dtiam-root {dtiam_root} not found.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for subset, cfg in SUBSETS.items():
        seq_path = dtiam_root / cfg["seq_file"]
        if not seq_path.is_file():
            sys.exit(f"ERROR: {seq_path} missing.")
        seq_df = pd.read_csv(seq_path, sep="\t").rename(
            columns={cfg["seq_id_col"]: "protein_id", cfg["seq_col"]: "sequence"}
        )[["protein_id", "sequence"]]
        seq_df["protein_id"] = seq_df["protein_id"].astype(str)

        used_ids = _read_used_ids(dtiam_root, cfg)
        n_before = len(seq_df)
        seq_df = seq_df[seq_df["protein_id"].isin(used_ids)].copy()
        n_after = len(seq_df)
        seq_df["source"] = subset

        missing = used_ids - set(seq_df["protein_id"])
        if missing:
            print(f"  WARN [{subset}] {len(missing)} used protein_id(s) have no sequence in {cfg['seq_file']}; first few: {sorted(missing)[:3]}")
        print(f"  [{subset}] {n_before:,} seqs in file -> {n_after:,} used (of {len(used_ids):,} referenced)")
        frames.append(seq_df)

    all_df = pd.concat(frames, ignore_index=True)

    # Hard-fail on protein_id → multiple sequences: ESM-C keys by protein_id
    # and a collision would silently corrupt embeddings. DTIAM IDs don't overlap
    # across subsets, but guard anyway.
    conflicts = all_df.groupby("protein_id")["sequence"].nunique().loc[lambda s: s > 1]
    if len(conflicts):
        print(f"ERROR: {len(conflicts)} protein_id(s) map to multiple sequences:", file=sys.stderr)
        for pid in conflicts.index[:5]:
            rows = all_df[all_df["protein_id"] == pid]
            print(f"  {pid}: {sorted(rows['source'].unique())}", file=sys.stderr)
        sys.exit(1)

    out_df = (
        all_df.drop_duplicates(subset="protein_id", keep="first")[["protein_id", "sequence"]]
        .sort_values("protein_id")
        .reset_index(drop=True)
    )
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(out_df):,} unique proteins -> {out_path}")
    print(f"  by source: " + ", ".join(f"{s}={int((all_df['source'] == s).sum()):,}" for s in SUBSETS))
    print("Next step: feed this CSV into scripts/data/dti_esmc/02_extract_embeddings_esmc.py (esmc env).")


if __name__ == "__main__":
    main()
