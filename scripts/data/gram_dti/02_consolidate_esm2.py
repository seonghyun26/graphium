#!/usr/bin/env python
"""Consolidate per-protein ESM-2 ``.pt`` files into a single parquet.

``scripts/data/dti_esm2/02_extract_embeddings.py`` writes one ``<pid>.pt`` per
protein under ``<output_dir>/embedding/``. This stage reads those files, stacks
the mean representations, and writes a columnar parquet keyed by ``protein_id``
with columns ``feature_0..feature_{ESM2_DIM-1}``.

Matches the schema the downstream prep stage expects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Default matches the GRAM-DTI paper (ESM-2 t33_650M_UR50D → layer 33 → 1280-d).
# If only one layer is present in the ``mean_representations`` dict the script
# uses it unconditionally, so this default rarely matters.
DEFAULT_REPR_LAYER = 33


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--embedding-dir", required=True,
                   help="Directory containing <protein_id>.pt files")
    p.add_argument("--protein-csv", required=True,
                   help="CSV listing proteins to include (columns: Target_ID, Target). "
                        "IDs not present in embedding-dir are skipped with a warning.")
    p.add_argument("--output", default="data/downstream/protein-esm2.parquet",
                   help="Destination parquet (protein_id + feature_0..feature_{D-1})")
    p.add_argument("--repr-layer", type=int, default=DEFAULT_REPR_LAYER,
                   help="ESM-2 layer to read. Overridden silently if the .pt file "
                        "contains exactly one layer.")
    args = p.parse_args()

    embedding_dir = Path(args.embedding_dir)
    if not embedding_dir.is_dir():
        sys.exit(f"ERROR: {embedding_dir} not found.")

    proteins = pd.read_csv(args.protein_csv)
    if "Target_ID" not in proteins.columns:
        sys.exit(f"ERROR: {args.protein_csv} missing Target_ID column")
    protein_ids = proteins["Target_ID"].astype(str).drop_duplicates().tolist()

    rows: list[dict] = []
    dim: int | None = None
    repr_layer: int | None = None
    missing = 0
    for pid in tqdm(protein_ids, desc="consolidating", unit="prot"):
        pt = embedding_dir / f"{pid}.pt"
        if not pt.exists():
            missing += 1
            continue
        data = torch.load(pt, map_location="cpu", weights_only=False)
        mean_reprs = data["mean_representations"]
        if repr_layer is None:
            if len(mean_reprs) == 1:
                repr_layer = next(iter(mean_reprs))
            else:
                repr_layer = args.repr_layer
            if repr_layer not in mean_reprs:
                sys.exit(
                    f"ERROR: layer {repr_layer} not in {pt} (available: {sorted(mean_reprs)}). "
                    "Pass --repr-layer to pick one."
                )
        mean_repr = mean_reprs[repr_layer]
        vec = mean_repr.detach().float().numpy().astype(np.float32)
        if dim is None:
            dim = int(vec.shape[0])
        elif vec.shape[0] != dim:
            sys.exit(
                f"ERROR: {pid} has dim {vec.shape[0]} but first file was {dim}. "
                "Mixed ESM-2 variants in embedding_dir."
            )
        rows.append({"protein_id": pid, **{f"feature_{i}": vec[i] for i in range(dim)}})

    if missing:
        print(f"  WARN: {missing:,} protein(s) in {args.protein_csv} have no embedding .pt")
    if not rows:
        sys.exit("ERROR: no embeddings loaded.")

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {len(out):,} proteins × {dim}-d -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
