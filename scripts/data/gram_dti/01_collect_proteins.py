#!/usr/bin/env python
"""Collect (protein_id, sequence) pairs across the 4 DTIAM datasets used by the
GRAM-DTI benchmark (Yamanishi 08 / Hetionet / Activation / Inhibition).

Restrict to proteins actually referenced by the benchmark — Hetionet's
``protein_seq.csv`` is a ~3x superset of the 5,763 actually-used proteins, and
embedding the rest would triple ESM-2 wall-clock for no gain.

Emits a CSV with columns ``Target_ID`` and ``Target`` (sequence), matching the
schema consumed by ``scripts/data/dti_esm2/02_extract_embeddings.py`` so the
existing ESM-2 extraction pipeline runs unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Per-dataset file locations + column names.  ``usage_file`` gives the set of
# proteins we actually embed (benchmark coverage); for DTI that's the dti.csv
# target column, for MoA that's tar_seq.csv itself (every row is a candidate).
SUBSETS = {
    "yamanishi_08": {
        "seq_file":   "data/dti/yamanishi_08/protein_seq.csv",
        "seq_id":     "pro_id",    "seq_col":   "seq",
        "usage_file": "data/dti/yamanishi_08/dti.csv",
        "usage_id":   2, "usage_header": False, "usage_sep": "\t",
    },
    "hetionet": {
        "seq_file":   "data/dti/hetionet/protein_seq.csv",
        "seq_id":     "pro_id",    "seq_col":   "seq",
        "usage_file": "data/dti/hetionet/dti.csv",
        "usage_id":   2, "usage_header": False, "usage_sep": "\t",
    },
    "activation": {
        "seq_file":   "data/moa/activation/tar_seq.csv",
        "seq_id":     "TargetID",  "seq_col":   "seq",
        "usage_file": "data/moa/activation/tar_seq.csv",
        "usage_id":   "TargetID", "usage_header": True, "usage_sep": "\t",
    },
    "inhibition": {
        "seq_file":   "data/moa/inhibition/tar_seq.csv",
        "seq_id":     "TargetID",  "seq_col":   "seq",
        "usage_file": "data/moa/inhibition/tar_seq.csv",
        "usage_id":   "TargetID", "usage_header": True, "usage_sep": "\t",
    },
}


def _used_ids(root: Path, cfg: dict) -> set[str]:
    path = root / cfg["usage_file"]
    if cfg["usage_header"]:
        df = pd.read_csv(path, sep=cfg["usage_sep"])
    else:
        df = pd.read_csv(path, sep=cfg["usage_sep"], header=None)
    return set(df[cfg["usage_id"]].astype(str).tolist())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dtiam-root", required=True,
                   help="Clone of https://github.com/CSUBioGroup/DTIAM")
    p.add_argument("--output", default="data/downstream/gram_dti/dtiam-proteins.csv",
                   help="Destination CSV (columns: Target_ID, Target)")
    args = p.parse_args()

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
            columns={cfg["seq_id"]: "Target_ID", cfg["seq_col"]: "Target"}
        )[["Target_ID", "Target"]]
        seq_df["Target_ID"] = seq_df["Target_ID"].astype(str)

        used = _used_ids(dtiam_root, cfg)
        n_before = len(seq_df)
        seq_df = seq_df[seq_df["Target_ID"].isin(used)].copy()
        n_after = len(seq_df)
        missing = used - set(seq_df["Target_ID"])
        if missing:
            print(f"  WARN [{subset}] {len(missing)} used id(s) have no sequence")
        print(f"  [{subset}] {n_before:,} seqs -> {n_after:,} used (of {len(used):,} referenced)")
        seq_df["source"] = subset
        frames.append(seq_df)

    all_df = pd.concat(frames, ignore_index=True)

    # Guard against protein_id → multiple sequences. DTIAM IDs don't overlap
    # across subsets in practice but silent corruption here would be bad.
    conflicts = all_df.groupby("Target_ID")["Target"].nunique().loc[lambda s: s > 1]
    if len(conflicts):
        sys.exit(f"ERROR: {len(conflicts)} Target_ID(s) map to multiple sequences")

    out_df = (
        all_df.drop_duplicates(subset="Target_ID", keep="first")[["Target_ID", "Target"]]
        .sort_values("Target_ID")
        .reset_index(drop=True)
    )
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df):,} unique proteins -> {out_path}")
    print(
        "\nNext: pipe into the existing ESM-2 extractor\n"
        f"  python scripts/data/dti_esm2/02_extract_embeddings.py \\\n"
        f"      --input {out_path} \\\n"
        f"      --output-dir data/downstream/gram_dti/protein-esm2 \\\n"
        f"      --gpus 0,1"
    )


if __name__ == "__main__":
    main()
