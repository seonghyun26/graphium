#!/usr/bin/env python
"""Stage 2 (v3): Assemble the final v3 DTI dataset that graphium consumes.

Joins:
  - 100k (drug, target, Y) rows from data/dti-scratch-v3/dti-bindingdb-v3.csv
  - 1152-d ESM-C embedding per target from data/protein-esmc-v3/embedding/*.pt

Produces:
  data/dti-processed/dti_esmc_100k_v3.csv
  data/dti-processed/dti_esmc_100k_v3.parquet
  data/dti-processed/dti_esmc_100k_v3_norm_stats.pt   (mean/std per feature)

Schema (mirrors dti_esmc_100k_v2.csv consumed by tasks/dti_esmc_v2.yaml):
  SMILES_nometa, feature_0, feature_1, ..., feature_1151
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


REPO = Path("/home/shpark/prj-molrepr/graphium")
PAIRS_CSV = REPO / "data" / "dti-scratch-v3" / "dti-bindingdb-v3.csv"
EMB_DIR = REPO / "data" / "protein-esmc-v3" / "embedding"
OUT_DIR = REPO / "data" / "dti-processed"
STEM = "dti_esmc_100k_v3"
EMBED_DIM = 1152


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs-csv", default=str(PAIRS_CSV))
    ap.add_argument("--embedding-dir", default=str(EMB_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--stem", default=STEM)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== loading pairs CSV ===")
    pairs = pd.read_csv(
        args.pairs_csv,
        usecols=["Drug", "Target_ID", "Y"],
    )
    print(f"  pairs: {len(pairs):,}")
    print(f"  unique drugs: {pairs.Drug.nunique():,}")
    print(f"  unique targets: {pairs.Target_ID.nunique():,}")

    targets = pairs["Target_ID"].drop_duplicates().tolist()

    print("\n=== loading per-target ESM-C embeddings ===")
    emb_dir = Path(args.embedding_dir)
    emb_table = np.empty((len(targets), EMBED_DIM), dtype=np.float32)
    missing = []
    for i, tid in enumerate(tqdm(targets, desc="proteins", unit="prot")):
        pt = emb_dir / f"{tid}.pt"
        if not pt.exists():
            missing.append(tid)
            emb_table[i] = 0.0
            continue
        d = torch.load(pt, map_location="cpu", weights_only=False)
        emb_table[i] = d["mean_embedding"].numpy()
    if missing:
        print(f"WARN: {len(missing)} targets missing .pt embeddings", file=sys.stderr)

    target_idx = {tid: i for i, tid in enumerate(targets)}

    print("\n=== expanding to per-pair rows ===")
    t0 = time.time()
    pair_idx = pairs["Target_ID"].map(target_idx).to_numpy()
    feats = emb_table[pair_idx]
    print(f"  feature matrix shape: {feats.shape}  ({time.time() - t0:.1f}s)")

    print("\n=== writing outputs ===")
    feat_cols = [f"feature_{i}" for i in range(EMBED_DIM)]
    df = pd.DataFrame({"SMILES_nometa": pairs["Drug"].to_numpy()})
    df = pd.concat([df, pd.DataFrame(feats, columns=feat_cols)], axis=1)

    csv_path = out_dir / f"{args.stem}.csv"
    pq_path = out_dir / f"{args.stem}.parquet"
    norm_path = out_dir / f"{args.stem}_norm_stats.pt"

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0).clip(min=1e-6)
    torch.save(
        {
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std":  torch.from_numpy(std.astype(np.float32)),
            "n_pairs": len(df),
            "n_targets": len(targets),
            "model": "esmc_600m",
        },
        norm_path,
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {pq_path}")
    print(f"  wrote {norm_path}")
    print(f"\nfinal shape: {df.shape}")


if __name__ == "__main__":
    main()
