#!/usr/bin/env python
"""Build `litopenai`: OpenAI text-embedding-3-small embeddings over a
molecule pool, with downstream-test SMILES filtered out.

The **OpenAI input** depends on the selected source:
  - protein / cell sources embed the canonical SMILES STRING itself.
  - lpm24 source embeds the L+M-24 CAPTION (natural-language description)
    for each molecule; embeddings are still keyed by canonical SMILES so the
    downstream consumer (``SMILES_nometa`` + ``feature_*`` columns) stays
    identical to the SMILES-embedding path.

Source pool (union of RDKit-canonical SMILES):
  - protein (``--sources protein``): data/dti-processed/dti_esmc_100k_v2.csv
                                     [SMILES_nometa]
  - cell    (``--sources cell``):    /home/shpark/prj-molrepr/data/bbbc047/
                                     bbbc047_smiles_embeddings.csv [SMILES]
  - lpm24   (``--sources lpm24``):   data/lpm24/lpm24_raw.parquet
                                     [molecule, caption]   -- embed the caption

Downstream-test SMILES to exclude (union, canonical form):
  - ADMET 22 (TDC):        data/tdc/admet_group/<task>/test.csv
  - Polaris adme-fang (6): datacache/polaris_adme_fang/adme-fang-v1.parquet
                           indexed by adme-fang-*_seed0_split.csv `test` col
  - TDC DTI eval:          data/dti-eval/<subset>.csv indexed by
                           data/dti-eval/splits/*.pt `test`
  - Cell bioactivity:      datacache/cell_bioactivity/cell_bioactivity.csv
                           indexed by cell_bioactivity_split.csv `test`

Embedding: text-embedding-3-small (1536-d) by default. Resumable — per-
canonical-SMILES embeddings are checkpointed after each flush.

Output (drop-in parallel to lpm24_<model>.*):
  - data/dti-processed/<stem>.csv
  - data/dti-processed/<stem>.parquet
  - data/dti-processed/<stem>_norm_stats.pt

Usage:
    export OPENAI_API_KEY=sk-...  # (or put it in graphium/.env)
    # smiles-string embedding (original behavior):
    python scripts/data/literature/01_build_litopenai.py
    # caption embedding (L+M-24):
    python scripts/data/literature/01_build_litopenai.py \\
        --sources lpm24 --stem litopenai_lpm24cap_small
    # single-example smoke test:
    python scripts/data/literature/01_build_litopenai.py \\
        --sources lpm24 --limit 1 --stem _test_litopenai_lpm24cap
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
LPM24_RAW = REPO_DATA / "lpm24" / "lpm24_raw.parquet"
TDC_ADMET_ROOT = REPO_DATA / "tdc" / "admet_group"
DTI_EVAL_ROOT = REPO_DATA / "dti-eval"

DEFAULT_OUT_DIR = REPO_DATA / "dti-processed"
DEFAULT_STEM = "litopenai_small"

DTI_EVAL_SUBSETS = ["DAVIS", "KIBA", "BindingDB_Patent_DG"]
KNOWN_SOURCES = {"protein", "cell", "lpm24"}


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
def load_source_pool(sources: list[str]) -> tuple[set[str], dict[str, str]]:
    """Build the canonical-SMILES pool; when ``lpm24`` is included also
    return a ``canonical_smiles -> caption`` map for caption-mode embedding.
    """
    pool: set[str] = set()
    captions: dict[str, str] = {}
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
    if "lpm24" in sources:
        if not LPM24_RAW.exists():
            sys.exit(f"ERROR: missing {LPM24_RAW}")
        df = pd.read_parquet(LPM24_RAW, columns=["molecule", "caption"])
        added = 0
        skipped_no_caption = 0
        for smi, cap in zip(df["molecule"].tolist(), df["caption"].tolist()):
            k = canonicalize(smi)
            if k is None:
                continue
            if not isinstance(cap, str) or not cap.strip():
                skipped_no_caption += 1
                continue
            if k not in captions:
                captions[k] = cap
                added += 1
        pool |= set(captions.keys())
        print(
            f"  lpm24 (captions):       {len(df):>7,} rows -> {added:>7,} canonical w/ captions"
            + (f" (skipped {skipped_no_caption} w/o caption)" if skipped_no_caption else "")
        )
    print(f"  UNION pool:             {len(pool):>7,} unique canonical SMILES")
    return pool, captions


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
import re as _re
import threading as _threading


class TokenBucket:
    """Thread-safe rolling-window token bucket.

    Enforces a ``capacity`` ceiling over a ``window`` (default 60s). Threads
    call :meth:`reserve(n)` before dispatching; the call blocks until ``n``
    tokens can be "spent" without exceeding the ceiling. Spends older than
    ``window`` are evicted lazily. This prevents 429 TPM errors instead of
    recovering from them.
    """

    def __init__(self, capacity: int, window: float = 60.0):
        self.capacity = capacity
        self.window = window
        self._spends: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = _threading.Lock()
        self._cond = _threading.Condition(self._lock)

    def _current_used(self, now: float) -> int:
        cutoff = now - self.window
        while self._spends and self._spends[0][0] < cutoff:
            self._spends.pop(0)
        return sum(n for _, n in self._spends)

    def reserve(self, n: int) -> None:
        if n >= self.capacity:
            # Request alone exceeds budget; impossible to satisfy — wait out
            # one full window then let it through.
            time.sleep(self.window)
            n = self.capacity
        with self._cond:
            while True:
                now = time.time()
                used = self._current_used(now)
                if used + n <= self.capacity:
                    self._spends.append((now, n))
                    return
                # Time until the oldest spend ages out enough to fit this request.
                deficit = (used + n) - self.capacity
                wait = 0.0
                running = 0
                for ts, tok in self._spends:
                    running += tok
                    if running >= deficit:
                        wait = (ts + self.window) - now
                        break
                self._cond.wait(timeout=max(0.05, wait))


_RETRY_AFTER_RE = _re.compile(r"try again in ([0-9.]+)s", _re.IGNORECASE)


def _parse_retry_after(err: Exception) -> float | None:
    """Extract the server's suggested retry delay from a 429 error message."""
    msg = str(err)
    m = _RETRY_AFTER_RE.search(msg)
    return float(m.group(1)) if m else None


def _embed_batch_with_retry(
    client,
    model: str,
    inputs: list[str],
    bucket: TokenBucket | None,
    est_tokens: int,
    max_attempts: int = 6,
):
    if bucket is not None:
        bucket.reserve(est_tokens)
    for attempt in range(max_attempts):
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            arr = np.zeros((len(inputs), len(resp.data[0].embedding)), dtype=np.float32)
            for item in resp.data:
                arr[item.index] = item.embedding
            return arr
        except Exception as e:  # noqa: BLE001
            if attempt == max_attempts - 1:
                raise
            hint = _parse_retry_after(e)
            if hint is not None:
                sleep_s = hint + random.uniform(0.2, 0.8)
            else:
                sleep_s = min(30.0, 2.0**attempt + random.uniform(0.0, 1.0))
            print(
                f"    [retry {attempt+1}/{max_attempts}] {type(e).__name__}: "
                f"sleep {sleep_s:.1f}s (hint={hint})"
            )
            time.sleep(sleep_s)
            # Re-reserve after waking — the bucket view may have drifted.
            if bucket is not None:
                bucket.reserve(est_tokens)


def _estimate_tokens(texts: list[str]) -> int:
    """Rough conservative upper-bound token count for a batch.

    ``text-embedding-3-small`` uses cl100k_base-ish tokenization; ~4 chars/tok
    on average, but SMILES/captions can be denser. Use chars/3 + small batch
    overhead to keep the bucket safely below server-side accounting.
    """
    return sum(len(t) for t in texts) // 3 + 4 * len(texts)


def embed_smiles(
    to_embed: list[str],
    checkpoint: Path,
    model: str,
    batch_size: int,
    concurrency: int,
    flush_every: int,
    tpm_limit: int,
    inputs_by_key: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Embed each key in ``to_embed`` with the OpenAI embeddings API.

    The cache + returned dict are keyed by the *key* (canonical SMILES),
    regardless of what was sent to the API. When ``inputs_by_key`` is
    provided, its values (e.g. L+M-24 captions) are what get embedded; the
    SMILES key is only used for cache addressing. When it's ``None``, the
    key is passed directly as the API input (original SMILES-string path).
    """
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
    # Leave ~10% headroom vs. the raw TPM cap to cover token-estimation drift.
    bucket = TokenBucket(capacity=int(tpm_limit * 0.9), window=60.0)
    print(
        f"  Dispatching {len(batches)} batches (size={batch_size}) @ concurrency={concurrency}"
        f" — TPM budget={tpm_limit:,} (bucket cap={bucket.capacity:,})"
    )

    t0 = time.time()
    done_since_flush = 0
    done_total = 0

    def work(batch: list[str]):
        inputs = [inputs_by_key[k] for k in batch] if inputs_by_key is not None else batch
        return batch, _embed_batch_with_retry(
            client, model, inputs, bucket=bucket, est_tokens=_estimate_tokens(inputs)
        )

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
    p.add_argument("--sources", default="protein,cell",
                   help=f"comma-sep subset of {sorted(KNOWN_SOURCES)}")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stem", default=DEFAULT_STEM, help="output filename stem")
    p.add_argument("--batch-size", type=int, default=1024, help="inputs per request (OpenAI hard cap 2048)")
    p.add_argument("--concurrency", type=int, default=8, help="parallel in-flight requests")
    p.add_argument(
        "--tpm-limit",
        type=int,
        default=1_000_000,
        help=(
            "Tokens-per-minute cap used for client-side pacing (shared bucket). "
            "Default matches OpenAI tier-1 text-embedding-3-small (1M TPM). "
            "Bucket uses 90%% of this for headroom."
        ),
    )
    p.add_argument("--flush-every", type=int, default=8192, help="checkpoint every N embeddings")
    p.add_argument("--checkpoint", type=Path, default=None, help="defaults to <output-dir>/<stem>_ckpt.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="report pool/excluded counts and exit")
    p.add_argument("--limit", type=int, default=None,
                   help="Only embed the first N SMILES after filtering — useful for smoke tests.")
    p.add_argument("--api-key", default=None, help="overrides $OPENAI_API_KEY")
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in KNOWN_SOURCES]
    if bad:
        sys.exit(f"ERROR: unknown source(s): {bad} (known: {sorted(KNOWN_SOURCES)})")

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    print("== LOAD SOURCE POOL ==")
    pool, captions = load_source_pool(sources)
    embedding_mode = "caption" if captions else "smiles-string"
    print(f"  Embedding input mode:   {embedding_mode}")
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
    if args.limit is not None:
        ordered = ordered[: args.limit]
        print(f"  --limit={args.limit}: truncated to {len(ordered):,} key(s)")

    # When a caption source is selected, build the per-key input text dict.
    inputs_by_key = {k: captions[k] for k in ordered} if captions else None

    if args.dry_run:
        if inputs_by_key:
            sample_key = ordered[0] if ordered else ""
            print(f"\n[dry-run] example caption for first key ({sample_key[:60]}...):")
            print(f"          {inputs_by_key.get(sample_key, '')[:400]}")
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
        tpm_limit=args.tpm_limit,
        inputs_by_key=inputs_by_key,
    )
    if embeddings:
        first_v = next(iter(embeddings.values()))
        print(f"  Embedding dim (verified): {first_v.shape[0]}")

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
