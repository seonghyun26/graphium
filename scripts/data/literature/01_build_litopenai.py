#!/usr/bin/env python
"""Build `litopenai`: OpenAI text-embedding-3-small embeddings over the
protein+cell pre-training SMILES pool, with downstream-test SMILES filtered
out.

Source pool (union of RDKit-canonical SMILES):
  - data/dti-processed/dti_esmc_100k_v2.csv   [SMILES_nometa]   (protein side)
  - /home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings.csv
                                              [SMILES]          (cell side)

Downstream-test SMILES to exclude (union, canonical form):
  - ADMET 22 (TDC):        data/tdc/admet_group/<task>/test.csv
  - Polaris adme-fang (6): datacache/polaris_adme_fang/adme-fang-v1.parquet
                           indexed by adme-fang-*_seed0_split.csv `test` col
  - TDC DTI eval:          data/dti-eval/<subset>.csv indexed by
                           data/dti-eval/splits/*.pt `test`
  - Cell bioactivity:      datacache/cell_bioactivity/cell_bioactivity.csv
                           indexed by cell_bioactivity_split.csv `test`

Embedding: text-embedding-3-small (1536-d). Resumable — per-SMILES embeddings
are checkpointed after each flush, so re-running skips already embedded ones.

Output (drop-in parallel to lpm24_<model>.*):
  - data/dti-processed/litopenai_small.csv
  - data/dti-processed/litopenai_small.parquet
  - data/dti-processed/litopenai_small_norm_stats.pt

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/data/literature/01_build_litopenai.py
    python scripts/data/literature/01_build_litopenai.py --dry-run
    python scripts/data/literature/01_build_litopenai.py --sources protein
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up OPENAI_API_KEY from ./.env (searches upward)
except ImportError:
    pass

RDLogger.DisableLog("rdApp.*")


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_DATA = Path("data")  # under graphium/
EXTERN_BBBC047 = Path("/home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings.csv")
EXTERN_POLARIS = Path("/home/shpark/prj-molrepr/datacache/polaris_adme_fang")
EXTERN_CELLBIO = Path("/home/shpark/prj-molrepr/datacache/cell_bioactivity")

DTI_V2_CSV = REPO_DATA / "dti-processed" / "dti_esmc_100k_v2.csv"
TDC_ADMET_ROOT = REPO_DATA / "tdc" / "admet_group"
DTI_EVAL_ROOT = REPO_DATA / "dti-eval"

DEFAULT_OUT_DIR = REPO_DATA / "dti-processed"
DEFAULT_STEM = "litopenai_small"

DTI_EVAL_SUBSETS = ["DAVIS", "KIBA", "BindingDB_Patent_DG"]


# ----------------------------------------------------------------------------
# Canonicalization
# ----------------------------------------------------------------------------
def canonicalize(smiles: str) -> str | None:
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def canon_set(values) -> set[str]:
    out: set[str] = set()
    for s in values:
        c = canonicalize(s)
        if c is not None:
            out.add(c)
    return out


# ----------------------------------------------------------------------------
# Source pool
# ----------------------------------------------------------------------------
def load_source_pool(sources: list[str]) -> set[str]:
    pool: set[str] = set()
    if "protein" in sources:
        if not DTI_V2_CSV.exists():
            sys.exit(f"ERROR: missing {DTI_V2_CSV}")
        df = pd.read_csv(DTI_V2_CSV, usecols=["SMILES_nometa"])
        s = canon_set(df["SMILES_nometa"])
        print(f"  protein (DTI v2):       {len(df):>7,} rows -> {len(s):>7,} canonical")
        pool |= s
    if "cell" in sources:
        if not EXTERN_BBBC047.exists():
            sys.exit(f"ERROR: missing {EXTERN_BBBC047}")
        df = pd.read_csv(EXTERN_BBBC047, usecols=["SMILES"])
        s = canon_set(df["SMILES"])
        print(f"  cell (BBBC047):         {len(df):>7,} rows -> {len(s):>7,} canonical")
        pool |= s
    print(f"  UNION pool:             {len(pool):>7,} unique canonical SMILES")
    return pool


# ----------------------------------------------------------------------------
# Test-SMILES exclusion set
# ----------------------------------------------------------------------------
def load_admet_test() -> set[str]:
    out: set[str] = set()
    if not TDC_ADMET_ROOT.exists():
        print("  WARN: ADMET cache missing — skipping")
        return out
    subtasks = sorted(p for p in TDC_ADMET_ROOT.iterdir() if p.is_dir())
    for sub in subtasks:
        fp = sub / "test.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        col = "Drug" if "Drug" in df.columns else df.columns[1]
        out |= canon_set(df[col])
    print(f"  ADMET ({len(subtasks)}):             -> {len(out):>7,} canonical")
    return out


def load_polaris_test() -> set[str]:
    parquet = EXTERN_POLARIS / "adme-fang-v1.parquet"
    if not parquet.exists():
        print("  WARN: Polaris parquet missing — skipping")
        return set()
    df = pd.read_parquet(parquet)
    smi_col = "MOL_smiles" if "MOL_smiles" in df.columns else "SMILES"
    idx: set[int] = set()
    splits = sorted(EXTERN_POLARIS.glob("adme-fang-*_seed0_split.csv"))
    for sf in splits:
        s = pd.read_csv(sf)
        if "test" in s.columns:
            idx.update(s["test"].dropna().astype(int).tolist())
    idx = [i for i in sorted(idx) if 0 <= i < len(df)]
    out = canon_set(df.loc[idx, smi_col])
    print(f"  Polaris adme-fang ({len(splits)}):  -> {len(out):>7,} canonical")
    return out


def _dti_subset_from_split_name(name: str) -> str | None:
    for sub in sorted(DTI_EVAL_SUBSETS, key=len, reverse=True):
        if name.startswith(sub + "_"):
            return sub
    return None


def load_dti_eval_test() -> set[str]:
    out: set[str] = set()
    split_dir = DTI_EVAL_ROOT / "splits"
    if not split_dir.exists():
        print("  WARN: DTI-eval splits missing — skipping")
        return out
    subset_cache: dict[str, pd.Series] = {}
    count = 0
    for pt in sorted(split_dir.glob("*.pt")):
        subset = _dti_subset_from_split_name(pt.stem)
        if subset is None:
            continue
        csv = DTI_EVAL_ROOT / f"{subset}.csv"
        if not csv.exists():
            continue
        if subset not in subset_cache:
            subset_cache[subset] = pd.read_csv(csv, usecols=["SMILES_nometa"])["SMILES_nometa"]
        smiles = subset_cache[subset]
        d = torch.load(pt, map_location="cpu", weights_only=False)
        if not isinstance(d, dict) or "test" not in d:
            continue
        test_idx = [int(i) for i in d["test"] if 0 <= int(i) < len(smiles)]
        out |= canon_set(smiles.iloc[test_idx])
        count += 1
    print(f"  DTI eval ({count} splits):     -> {len(out):>7,} canonical")
    return out


def load_cell_bioactivity_test() -> set[str]:
    csv = EXTERN_CELLBIO / "cell_bioactivity.csv"
    split = EXTERN_CELLBIO / "cell_bioactivity_split.csv"
    if not (csv.exists() and split.exists()):
        print("  WARN: cell_bioactivity files missing — skipping")
        return set()
    smiles = pd.read_csv(csv, usecols=["smiles"])["smiles"]
    s = pd.read_csv(split)
    if "test" not in s.columns:
        print("  WARN: cell_bioactivity_split lacks `test` col — skipping")
        return set()
    test_idx = [int(i) for i in s["test"].dropna().astype(int).tolist() if 0 <= int(i) < len(smiles)]
    out = canon_set(smiles.iloc[test_idx])
    print(f"  Cell bioactivity:       -> {len(out):>7,} canonical")
    return out


def load_test_exclusion() -> set[str]:
    return (
        load_admet_test()
        | load_polaris_test()
        | load_dti_eval_test()
        | load_cell_bioactivity_test()
    )


# ----------------------------------------------------------------------------
# OpenAI embedding with resumable checkpoint
# ----------------------------------------------------------------------------
def _embed_batch_with_retry(client, model: str, inputs: list[str], max_attempts: int = 6):
    for attempt in range(max_attempts):
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            # Preserve order via index
            arr = np.zeros((len(inputs), len(resp.data[0].embedding)), dtype=np.float32)
            for item in resp.data:
                arr[item.index] = item.embedding
            return arr
        except Exception as e:  # noqa: BLE001
            if attempt == max_attempts - 1:
                raise
            sleep_s = min(60.0, 2.0**attempt + random.uniform(0.0, 1.0))
            print(f"    [retry {attempt+1}/{max_attempts}] {type(e).__name__}: {e} — sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)


def embed_smiles(
    to_embed: list[str],
    checkpoint: Path,
    model: str,
    batch_size: int,
    concurrency: int,
    flush_every: int,
) -> dict[str, np.ndarray]:
    from openai import OpenAI  # local import so --dry-run works without the lib

    # Load existing checkpoint (maps canonical SMILES -> np.ndarray)
    cache: dict[str, np.ndarray] = {}
    if checkpoint.exists():
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cache = {k: v for k, v in data.items()}
        print(f"  Loaded checkpoint: {len(cache):,} embeddings from {checkpoint}")

    pending = [s for s in to_embed if s not in cache]
    print(f"  Pending after cache: {len(pending):,} / {len(to_embed):,}")
    if not pending:
        return {s: cache[s] for s in to_embed}

    client = OpenAI()
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"  Dispatching {len(batches)} batches (size={batch_size}) @ concurrency={concurrency}")

    t0 = time.time()
    done_since_flush = 0
    done_total = 0

    def work(batch: list[str]):
        return batch, _embed_batch_with_retry(client, model, batch)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, b) for b in batches]
        for i, fut in enumerate(as_completed(futures), 1):
            batch, arr = fut.result()
            for s, v in zip(batch, arr):
                cache[s] = v
            done_total += len(batch)
            done_since_flush += len(batch)
            if done_since_flush >= flush_every or i == len(batches):
                torch.save(cache, checkpoint)
                done_since_flush = 0
            if i == 1 or i % 10 == 0 or i == len(batches):
                dt = time.time() - t0
                rate = done_total / max(dt, 1e-9)
                eta = (len(pending) - done_total) / max(rate, 1e-9)
                print(
                    f"    [{i:>4}/{len(batches)}] embedded {done_total:,}/{len(pending):,} "
                    f"({rate:.1f}/s, ETA {eta/60:.1f}min)"
                )

    return {s: cache[s] for s in to_embed}


# ----------------------------------------------------------------------------
# Finalize: z-score + write parquet/csv/norm_stats
# ----------------------------------------------------------------------------
def write_final(
    ordered_smiles: list[str],
    embeddings: dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
    seed: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.stack([embeddings[s] for s in ordered_smiles]).astype(np.float32)
    dim = arr.shape[1]

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    arr_norm = (arr - mean) / std

    feat_cols = [f"feature_{i}" for i in range(dim)]
    df = pd.DataFrame(arr_norm, columns=feat_cols)
    df.insert(0, "SMILES_nometa", ordered_smiles)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    csv_path = out_dir / f"{stem}.csv"
    parquet_path = out_dir / f"{stem}.parquet"
    stats_path = out_dir / f"{stem}_norm_stats.pt"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    torch.save(
        {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
        stats_path,
    )

    print(f"\n  Saved CSV:     {csv_path}  ({len(df):,} rows, {csv_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Saved Parquet: {parquet_path}  ({parquet_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Saved stats:   {stats_path}  (dim={dim})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="protein,cell", help="comma-sep subset of {protein,cell}")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stem", default=DEFAULT_STEM, help="output filename stem")
    p.add_argument("--batch-size", type=int, default=1024, help="SMILES per request (OpenAI hard cap 2048)")
    p.add_argument("--concurrency", type=int, default=8, help="parallel in-flight requests")
    p.add_argument("--flush-every", type=int, default=8192, help="checkpoint every N embeddings")
    p.add_argument("--checkpoint", type=Path, default=None, help="defaults to <output-dir>/<stem>_ckpt.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="report pool/excluded counts and exit")
    p.add_argument("--api-key", default=None, help="overrides $OPENAI_API_KEY")
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in {"protein", "cell"}]
    if bad:
        sys.exit(f"ERROR: unknown source(s): {bad}")

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    print("== LOAD SOURCE POOL ==")
    pool = load_source_pool(sources)
    print("\n== LOAD TEST EXCLUSION ==")
    test_set = load_test_exclusion()

    keep_set = pool - test_set
    excluded = len(pool) - len(keep_set)
    print(
        f"\n== FILTER ==\n"
        f"  pool:     {len(pool):,}\n"
        f"  test:     {len(test_set):,}\n"
        f"  excluded: {excluded:,} ({100*excluded/max(len(pool),1):.1f}%)\n"
        f"  to embed: {len(keep_set):,}"
    )

    ordered = sorted(keep_set)  # deterministic order

    if args.dry_run:
        print("\n[dry-run] skipping embedding + write. Done.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set (pass --api-key or export)")

    checkpoint = args.checkpoint or (args.output_dir / f"{args.stem}_ckpt.pt")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n== EMBED ==")
    embeddings = embed_smiles(
        to_embed=ordered,
        checkpoint=checkpoint,
        model=args.model,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        flush_every=args.flush_every,
    )

    print("\n== FINALIZE ==")
    write_final(
        ordered_smiles=ordered,
        embeddings=embeddings,
        out_dir=args.output_dir,
        stem=args.stem,
        seed=args.seed,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
