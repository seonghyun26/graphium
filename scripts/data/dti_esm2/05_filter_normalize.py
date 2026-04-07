#!/usr/bin/env python
"""Stage 5: Filter and normalize the merged DTI-ESM2 dataset.

Applies protein-stratified sampling and SMILES diversity filtering to reduce
the ~1.62M-row dataset to exactly 63,405 rows (matching the rxrx3 microscopy
dataset size). Then applies L2 and Z-score normalization.

Filtering strategy (seed=42, target=63,405):
    Phase 1: Sample 1 random row per protein (5,222 rows).
             Guarantees every protein is represented at least once.
    Phase 2: From the remaining ~1.61M rows, select 58,183 rows whose SMILES
             are NOT already in Phase 1, deduplicated by SMILES.

Result: 63,405 rows, 5,222 proteins (all represented), ~62,585 unique SMILES.

Normalization:
    L2:      Each embedding vector divided by its L2 norm -> unit vectors.
    Z-score: Per-feature standardization -> mean=0, std=1 per feature.
             The mean/std vectors are saved separately for denormalization.

Output files:
    - tdcdti-esm2-filtered.parquet          : filtered (raw embeddings, no norm)
    - tdcdti-esm2-filtered-l2.parquet       : L2-normalized embeddings
    - tdcdti-esm2-filtered-l2.pt            : same, PyTorch dict format
    - tdcdti-esm2-filtered-zscore.parquet   : Z-score normalized embeddings
    - tdcdti-esm2-filtered-zscore.pt        : same, PyTorch dict format
    - tdcdti-esm2-filtered-zscore-stats.pt  : mean/std vectors for denorm

Output .parquet schema:
    protein_id, SMILES, feature_0 .. feature_2559
    (sequence column is dropped to save space)

Output .pt format:
    {"embeddings": FloatTensor[63405, 2560],
     "smiles": list[str],
     "protein_id": list[str]}

Usage:
    python 05_filter_normalize.py \\
        --input graphium/data/dti/tdcdti-esm2.parquet \\
        --output-dir graphium/data/dti \\
        --seed 42 --target-size 63405
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


EMBEDDING_DIM = 2560
FEATURE_COLS = [f"feature_{i}" for i in range(EMBEDDING_DIM)]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_dataset(
    df: pd.DataFrame, target_size: int, seed: int
) -> pd.DataFrame:
    """Apply protein-stratified + SMILES diversity filtering.

    Phase 1: 1 random row per protein (guarantees all proteins present).
    Phase 2: Fill remaining slots with novel-SMILES rows, dedup by SMILES.

    Args:
        df: Full merged dataset with columns protein_id, SMILES, feature_*.
        target_size: Desired number of output rows (63,405).
        seed: Random seed for reproducibility.

    Returns:
        Filtered DataFrame with exactly target_size rows (or fewer if
        insufficient novel SMILES are available).
    """
    rng = np.random.RandomState(seed)

    n_proteins = df["protein_id"].nunique()
    phase2_target = target_size - n_proteins
    print(f"  Target size:         {target_size:,}")
    print(f"  Unique proteins:     {n_proteins:,}")
    print(f"  Phase 1 (stratified): {n_proteins:,} rows (1 per protein)")
    print(f"  Phase 2 (novel SMILES): {phase2_target:,} rows needed")

    # -- Phase 1: sample 1 random row per protein --
    phase1 = df.groupby("protein_id", group_keys=False).apply(
        lambda g: g.sample(n=1, random_state=rng)
    )
    phase1_smiles = set(phase1["SMILES"].values)
    print(
        f"  Phase 1 done: {len(phase1):,} rows, "
        f"{len(phase1_smiles):,} unique SMILES"
    )

    # -- Phase 2: from remaining rows, get novel SMILES --
    remaining = df.drop(phase1.index)
    novel_mask = ~remaining["SMILES"].isin(phase1_smiles)
    novel_rows = remaining[novel_mask].copy()
    print(f"  Remaining rows with novel SMILES: {len(novel_rows):,}")

    # Deduplicate by SMILES (keep first occurrence)
    novel_dedup = novel_rows.drop_duplicates(subset=["SMILES"], keep="first")
    print(f"  After SMILES dedup: {len(novel_dedup):,}")

    if len(novel_dedup) < phase2_target:
        print(
            f"  WARNING: Only {len(novel_dedup):,} novel-SMILES rows available, "
            f"need {phase2_target:,}. Using all available."
        )
        phase2 = novel_dedup
    else:
        phase2 = novel_dedup.sample(n=phase2_target, random_state=rng)

    # Combine Phase 1 + Phase 2 and shuffle deterministically
    filtered = pd.concat([phase1, phase2], ignore_index=True)
    filtered = filtered.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    print(f"\n  Filtered dataset:")
    print(f"    Total rows:      {len(filtered):,}")
    print(f"    Unique proteins: {filtered['protein_id'].nunique():,}")
    print(f"    Unique SMILES:   {filtered['SMILES'].nunique():,}")

    return filtered


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length (norm=1)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def normalize_zscore(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize per feature (mean=0, std=1 per column).

    Returns:
        (normalized_embeddings, mean_vector, std_vector)
    """
    mean = embeddings.mean(axis=0)  # shape [2560]
    std = embeddings.std(axis=0)    # shape [2560]
    normalized = (embeddings - mean) / std
    return normalized, mean, std


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, embeddings: np.ndarray, path: str):
    """Save as parquet: protein_id, SMILES, feature_0..feature_2559."""
    out_df = pd.DataFrame(embeddings, columns=FEATURE_COLS)
    out_df.insert(0, "protein_id", df["protein_id"].values)
    out_df.insert(1, "SMILES", df["SMILES"].values)
    out_df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved parquet: {path}  ({len(out_df):,} rows, {size_mb:.1f} MB)")


def save_pt(
    smiles: list[str],
    protein_ids: list[str],
    embeddings: np.ndarray,
    path: str,
):
    """Save as PyTorch dict: embeddings, smiles, protein_id."""
    data = {
        "embeddings": torch.FloatTensor(embeddings),
        "smiles": smiles,
        "protein_id": protein_ids,
    }
    torch.save(data, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(
        f"  Saved pt:      {path}  "
        f"(embeddings shape: {data['embeddings'].shape}, {size_mb:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Filter and normalize the DTI-ESM2 dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="graphium/data/dti/tdcdti-esm2.parquet",
        help="Input parquet from Stage 4 (default: graphium/data/dti/tdcdti-esm2.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graphium/data/dti",
        help="Output directory (default: graphium/data/dti)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=63405,
        help="Target number of rows in filtered dataset (default: 63405)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Stage 5: Filter + Normalize DTI-ESM2 dataset")
    print("=" * 60)

    # ---- Step 1: Load data ----
    print(f"\nStep 1: Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  Shape:           {df.shape}")
    print(f"  Columns (first): {df.columns[:5].tolist()}")
    print(f"  Columns (last):  {df.columns[-3:].tolist()}")
    print(f"  Unique proteins: {df['protein_id'].nunique():,}")
    print(f"  Unique SMILES:   {df['SMILES'].nunique():,}")

    # ---- Step 2: Filter ----
    print(f"\nStep 2: Filtering to {args.target_size:,} rows ...")
    filtered = filter_dataset(df, args.target_size, args.seed)

    # Save unmodified filtered dataset (raw embeddings, no normalization)
    filtered_path = os.path.join(args.output_dir, "tdcdti-esm2-filtered.parquet")
    raw_embeddings = filtered[FEATURE_COLS].values.astype(np.float32)
    save_parquet(filtered, raw_embeddings, filtered_path)

    smiles_list = filtered["SMILES"].tolist()
    protein_id_list = filtered["protein_id"].tolist()

    # ---- Step 3: L2 normalization ----
    print(f"\nStep 3: L2 normalization ...")
    embeddings_l2 = normalize_l2(raw_embeddings)
    norms_check = np.linalg.norm(embeddings_l2, axis=1)
    print(
        f"  L2 norms after normalization: "
        f"mean={norms_check.mean():.6f}, std={norms_check.std():.2e}"
    )

    l2_parquet = os.path.join(args.output_dir, "tdcdti-esm2-filtered-l2.parquet")
    l2_pt = os.path.join(args.output_dir, "tdcdti-esm2-filtered-l2.pt")
    save_parquet(filtered, embeddings_l2, l2_parquet)
    save_pt(smiles_list, protein_id_list, embeddings_l2, l2_pt)

    # ---- Step 4: Z-score normalization ----
    print(f"\nStep 4: Z-score normalization ...")
    embeddings_zscore, mean_vec, std_vec = normalize_zscore(raw_embeddings)
    print(
        f"  Per-feature mean of normalized: "
        f"{embeddings_zscore.mean(axis=0).mean():.6e}"
    )
    print(
        f"  Per-feature std of normalized:  "
        f"{embeddings_zscore.std(axis=0).mean():.6f}"
    )

    zscore_parquet = os.path.join(
        args.output_dir, "tdcdti-esm2-filtered-zscore.parquet"
    )
    zscore_pt = os.path.join(args.output_dir, "tdcdti-esm2-filtered-zscore.pt")
    save_parquet(filtered, embeddings_zscore, zscore_parquet)
    save_pt(smiles_list, protein_id_list, embeddings_zscore, zscore_pt)

    # Save mean/std vectors for denormalization at inference time
    stats_pt = os.path.join(
        args.output_dir, "tdcdti-esm2-filtered-zscore-stats.pt"
    )
    torch.save(
        {
            "mean": torch.FloatTensor(mean_vec),
            "std": torch.FloatTensor(std_vec),
        },
        stats_pt,
    )
    size_kb = os.path.getsize(stats_pt) / 1024
    print(f"  Saved zscore stats: {stats_pt}  ({size_kb:.1f} KB)")

    # ---- Step 5: Verify ----
    print(f"\nStep 5: Verification ...")
    df_l2 = pd.read_parquet(l2_parquet)
    df_zs = pd.read_parquet(zscore_parquet)
    pt_l2 = torch.load(l2_pt, map_location="cpu", weights_only=False)
    pt_zs = torch.load(zscore_pt, map_location="cpu", weights_only=False)
    stats = torch.load(stats_pt, map_location="cpu", weights_only=False)

    checks_passed = 0
    total_checks = 0

    def check(condition, msg):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
        else:
            print(f"  FAIL: {msg}")

    check(
        len(df_l2) == args.target_size,
        f"L2 parquet rows: {len(df_l2)} != {args.target_size}",
    )
    check(
        len(df_zs) == args.target_size,
        f"Zscore parquet rows: {len(df_zs)} != {args.target_size}",
    )
    check(
        pt_l2["embeddings"].shape == (args.target_size, EMBEDDING_DIM),
        f"L2 pt shape: {pt_l2['embeddings'].shape}",
    )
    check(
        pt_zs["embeddings"].shape == (args.target_size, EMBEDDING_DIM),
        f"Zscore pt shape: {pt_zs['embeddings'].shape}",
    )
    check(
        len(pt_l2["smiles"]) == args.target_size,
        f"L2 pt smiles count: {len(pt_l2['smiles'])}",
    )
    check(
        len(pt_zs["smiles"]) == args.target_size,
        f"Zscore pt smiles count: {len(pt_zs['smiles'])}",
    )
    check(
        len(pt_l2["protein_id"]) == args.target_size,
        f"L2 pt protein_id count: {len(pt_l2['protein_id'])}",
    )
    check(
        stats["mean"].shape == (EMBEDDING_DIM,),
        f"Stats mean shape: {stats['mean'].shape}",
    )
    check(
        stats["std"].shape == (EMBEDDING_DIM,),
        f"Stats std shape: {stats['std'].shape}",
    )

    # Verify L2 norms are all ~1.0
    l2_norms = np.linalg.norm(df_l2[FEATURE_COLS].values, axis=1)
    check(
        np.allclose(l2_norms, 1.0, atol=1e-5),
        f"L2 norms not close to 1.0 (max deviation: {np.max(np.abs(l2_norms - 1.0)):.2e})",
    )

    print(f"  Checks passed: {checks_passed}/{total_checks}")
    if checks_passed == total_checks:
        print("  All verifications passed.")
    else:
        print(f"  WARNING: {total_checks - checks_passed} check(s) failed!")

    # ---- Summary ----
    print("\nDone. Output files:")
    all_outputs = [
        filtered_path,
        l2_parquet,
        l2_pt,
        zscore_parquet,
        zscore_pt,
        stats_pt,
    ]
    for f in all_outputs:
        size = os.path.getsize(f)
        if size > 1024 * 1024:
            print(f"  {f}  ({size / (1024 * 1024):.1f} MB)")
        else:
            print(f"  {f}  ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
