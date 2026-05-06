#!/usr/bin/env python
"""Build `litmolformer_v2`: MolFormer-XL embeddings over the union of the
other pretraining sources, prioritizing maximum non-overlapping coverage.

Embeds canonical SMILES drawn from the union of:
  * ToyMix      (QM9 ∪ Tox21 ∪ ZINC12k)
  * DTI ESM-C v3
  * BBBC047

Selection policy:
  1. Canonicalize via RDKit; drop invalid
  2. Drop SMILES with >100 heavy atoms
  3. Optionally drop SMILES in any TDC-ADMET test split
  4. Prioritize source-unique SMILES first (belonging to exactly one source),
     then 2-way overlaps, then 3-way overlaps
  5. Within each overlap tier, round-robin across the source-combos to spread
     coverage instead of letting one dominant source fill the entire budget
  6. Keep at most 100,000 molecules

Output:
    data/dti-processed/litmolformer_v2.csv
    data/dti-processed/litmolformer_v2.parquet
    data/dti-processed/litmolformer_v2_norm_stats.pt
schema: SMILES_nometa, feature_0..feature_767

Usage:
    python scripts/data/literature/02_build_litmolformer.py --gpu 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


REPO = Path("/home/shpark/prj-molrepr/graphium")
SMALL = REPO / "data" / "graphium" / "neurips2023" / "small-dataset"
DTI_V3_CSV = REPO / "data" / "dti-processed" / "dti_esmc_100k_v3.csv"
BBBC047_CSV = Path(
    "/home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings_max100_admetfilt.csv"
)

OUT_DIR = REPO / "data" / "dti-processed"
STEM = "litmolformer_v2"

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
EMBED_DIM = 768


# ----- canonicalization & filtering ---------------------------------------------

def _canon_atoms(smi: str):
    """Return (canon_smiles, n_heavy) or (None, None)."""
    if not isinstance(smi, str):
        return None, None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    return Chem.MolToSmiles(m, canonical=True), m.GetNumHeavyAtoms()


def canon_unique(smis, n_jobs=8):
    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1024)(
        delayed(_canon_atoms)(s) for s in tqdm(smis, desc="canonicalize", unit="mol")
    )
    return {s: out[i] for i, s in enumerate(smis)}


# ----- ADMET test exclusion (reuse logic from 01_build_litopenai.py) ------------

def load_admet_test_canon():
    """Return canonical-SMILES set across all TDC-ADMET task test splits."""
    root = REPO / "data" / "tdc" / "admet_group"
    if not root.exists():
        print(f"WARN: ADMET test root not found: {root}", file=sys.stderr)
        return set()
    test_smis = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_csv = task_dir / "test.csv"
        if not test_csv.exists():
            continue
        df = pd.read_csv(test_csv)
        smi_col = next(
            (c for c in df.columns if c.lower() in ("drug", "smiles", "smiles_nometa")),
            None,
        )
        if smi_col is None:
            continue
        test_smis.extend(df[smi_col].dropna().tolist())
    print(f"  ADMET-test raw SMILES gathered: {len(test_smis):,}")
    cmap = canon_unique(list(set(test_smis)), n_jobs=8)
    return {v[0] for v in cmap.values() if v[0] is not None}


def load_canonical_set(path: Path, col: str, *, n_jobs: int, name: str) -> set[str]:
    """Canonicalize a dataset column into a unique SMILES set."""
    if path.suffix == ".parquet":
        vals = pd.read_parquet(path, columns=[col])[col].dropna().astype(str).tolist()
    else:
        vals = pd.read_csv(path, usecols=[col])[col].dropna().astype(str).tolist()
    print(f"  {name:<10} raw rows: {len(vals):,}")
    cmap = canon_unique(list(dict.fromkeys(vals)), n_jobs=n_jobs)
    out = {v[0] for v in cmap.values() if v[0] is not None and v[1] is not None and v[1] <= 100}
    print(f"  {name:<10} canonical <=100 atoms: {len(out):,}")
    return out


def load_union_source_sets(n_jobs: int) -> dict[str, set[str]]:
    """Load canonical <=100-heavy-atom sets for ToyMix, DTI_v3, and BBBC047."""
    qm9 = load_canonical_set(SMALL / "qm9.csv", "smiles", n_jobs=n_jobs, name="QM9")
    tox21 = load_canonical_set(
        SMALL / "Tox21-7k-12-labels.csv", "smiles", n_jobs=n_jobs, name="Tox21"
    )
    zinc = load_canonical_set(SMALL / "ZINC12k.csv", "smiles", n_jobs=n_jobs, name="ZINC12k")
    toy = qm9 | tox21 | zinc
    print(f"  {'ToyMix':<10} canonical <=100 atoms: {len(toy):,}")
    dti = load_canonical_set(DTI_V3_CSV, "SMILES_nometa", n_jobs=n_jobs, name="DTI_v3")
    bbbc = load_canonical_set(BBBC047_CSV, "SMILES", n_jobs=n_jobs, name="BBBC047")
    return {"ToyMix": toy, "DTI_v3": dti, "BBBC047": bbbc}


def prioritize_union_smiles(source_sets: dict[str, set[str]], max_size: int) -> tuple[list[str], dict[tuple[str, ...], int]]:
    """Select up to `max_size` SMILES, preferring lower-overlap source regions."""
    membership: dict[str, tuple[str, ...]] = {}
    for source, smis in source_sets.items():
        for smi in smis:
            membership.setdefault(smi, []).append(source)
    regions: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for smi, sources in membership.items():
        regions[tuple(sorted(sources))].append(smi)
    for key in regions:
        regions[key].sort()

    selected: list[str] = []
    region_counts: dict[tuple[str, ...], int] = {}
    for overlap_order in sorted({len(k) for k in regions}):
        active = {k: list(v) for k, v in regions.items() if len(k) == overlap_order}
        while active and len(selected) < max_size:
            exhausted = []
            for region in sorted(active):
                if not active[region]:
                    exhausted.append(region)
                    continue
                selected.append(active[region].pop(0))
                region_counts[region] = region_counts.get(region, 0) + 1
                if len(selected) >= max_size:
                    break
            for region in exhausted:
                active.pop(region, None)
            active = {k: v for k, v in active.items() if v}
    return selected, region_counts


# ----- MolFormer embedding ------------------------------------------------------

def load_molformer(device):
    from transformers import AutoTokenizer, AutoModel

    print(f"  loading {MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, deterministic_eval=True
    ).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def embed_batch(tok, model, smis, device):
    """Mean-pool MolFormer hidden states (excl. padding) → (B, 768)."""
    enc = tok(smis, padding=True, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model(**enc)
    h = out.last_hidden_state                                   # (B, L, 768)
    mask = enc.attention_mask.unsqueeze(-1).to(h.dtype)         # (B, L, 1)
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.float().cpu().numpy()


# ----- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-atoms", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--max-size", type=int, default=100_000)
    ap.add_argument("--keep-admet-test", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build the union pool from the three source modalities.
    print("=== Stage 1: load source unions (ToyMix / DTI_v3 / BBBC047) ===")
    source_sets = load_union_source_sets(n_jobs=args.n_jobs)
    union_all = set().union(*source_sets.values())
    print(f"  UNION canonical <=100 atoms: {len(union_all):,}")

    # 2. Optional ADMET-test exclusion.
    pool = set(union_all)
    if not args.keep_admet_test:
        print("\n=== Stage 2: subtract ADMET-test SMILES ===")
        admet_test = load_admet_test_canon()
        print(f"  ADMET test canonical SMILES: {len(admet_test):,}")
        leaked = pool & admet_test
        pool = pool - admet_test
        print(f"  ADMET-test leak removed: {len(leaked):,}")
        source_sets = {k: (v & pool) for k, v in source_sets.items()}
        print(f"  UNION after ADMET exclusion: {len(set().union(*source_sets.values())):,}")
    else:
        print("\n=== Stage 2: keeping ADMET-test SMILES (per --keep-admet-test) ===")

    # 3. Prioritize unique-source SMILES first, then overlap regions.
    print("\n=== Stage 3: prioritize maximal non-overlap coverage ===")
    canon_list, region_counts = prioritize_union_smiles(source_sets, max_size=args.max_size)
    n = len(canon_list)
    print(f"  selected molecules: {n:,} / max_size={args.max_size:,}")
    for region, count in sorted(region_counts.items(), key=lambda kv: (len(kv[0]), kv[0])):
        print(f"    {' + '.join(region):<28} -> {count:>7,}")

    # 5. Embed via MolFormer-XL on the requested GPU.
    print(f"\n=== Stage 4: MolFormer-XL embedding on cuda:{args.gpu} ===")
    device = torch.device(f"cuda:{args.gpu}")
    tok, model = load_molformer(device)

    feats = np.empty((n, EMBED_DIM), dtype=np.float32)

    bs = args.batch_size
    t0 = time.time()
    for i in tqdm(range(0, n, bs), desc="embed", unit="batch"):
        chunk = canon_list[i:i + bs]
        feats[i:i + len(chunk)] = embed_batch(tok, model, chunk, device)
    dt = time.time() - t0
    print(f"  embedded {n:,} mols in {dt:.1f}s  ({n / dt:.1f} mol/s)")

    # 5. Save: csv + parquet + norm_stats.pt.
    print("\n=== Stage 5: write outputs ===")
    cols = ["SMILES_nometa"] + [f"feature_{i}" for i in range(EMBED_DIM)]
    df = pd.DataFrame({"SMILES_nometa": canon_list})
    df = pd.concat([df, pd.DataFrame(feats, columns=cols[1:])], axis=1)

    csv_path = OUT_DIR / f"{STEM}.csv"
    pq_path = OUT_DIR / f"{STEM}.parquet"
    norm_path = OUT_DIR / f"{STEM}_norm_stats.pt"

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0).clip(min=1e-6)
    torch.save(
        {
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std":  torch.from_numpy(std.astype(np.float32)),
            "n_mols": n,
            "model": MODEL_NAME,
            "selection_max_size": args.max_size,
            "region_counts": {" + ".join(k): v for k, v in region_counts.items()},
        },
        norm_path,
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {pq_path}")
    print(f"  wrote {norm_path}")
    print(f"  feats shape: {feats.shape}  dtype: {feats.dtype}")


if __name__ == "__main__":
    main()
