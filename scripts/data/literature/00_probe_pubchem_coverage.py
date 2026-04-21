#!/usr/bin/env python
"""Probe PubChem coverage on a random sample of the filtered SMILES pool.

For each sampled SMILES:
  1. Compute InChIKey (RDKit).
  2. Look up PubChem CID by InChIKey.
  3. If CID exists, fetch the brief description.

Reports:
  - InChIKey-derivable rate (sanity check)
  - CID-match rate
  - Description-available rate
  - Mean description length

Throttled at 5 req/s (PubChem PUG-REST limit).

Usage:
    python scripts/data/literature/00_probe_pubchem_coverage.py
    python scripts/data/literature/00_probe_pubchem_coverage.py --sample 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO_DATA = Path("data")
EXTERN_BBBC047 = Path("/home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings.csv")
DTI_V2_CSV = REPO_DATA / "dti-processed" / "dti_esmc_100k_v2.csv"
EXTERN_POLARIS = Path("/home/shpark/prj-molrepr/datacache/polaris_adme_fang")
EXTERN_CELLBIO = Path("/home/shpark/prj-molrepr/datacache/cell_bioactivity")
TDC_ADMET_ROOT = REPO_DATA / "tdc" / "admet_group"
DTI_EVAL_ROOT = REPO_DATA / "dti-eval"
DTI_EVAL_SUBSETS = ["DAVIS", "KIBA", "BindingDB_Patent_DG"]

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RATE_PER_SEC = 5.0
USER_AGENT = "graphium-litprobe/0.1 (mailto:none)"


def canon(s):
    if not isinstance(s, str):
        return None
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except Exception:
        return None


def canon_set(values):
    out = set()
    for s in values:
        c = canon(s)
        if c:
            out.add(c)
    return out


def load_pool():
    dti = pd.read_csv(DTI_V2_CSV, usecols=["SMILES_nometa"])
    bbbc = pd.read_csv(EXTERN_BBBC047, usecols=["SMILES"])
    return canon_set(dti["SMILES_nometa"]) | canon_set(bbbc["SMILES"])


def load_test_exclusion():
    out = set()
    # ADMET
    if TDC_ADMET_ROOT.exists():
        for sub in TDC_ADMET_ROOT.iterdir():
            fp = sub / "test.csv"
            if fp.exists():
                df = pd.read_csv(fp)
                col = "Drug" if "Drug" in df.columns else df.columns[1]
                out |= canon_set(df[col])
    # Polaris
    parquet = EXTERN_POLARIS / "adme-fang-v1.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        smi_col = "MOL_smiles" if "MOL_smiles" in df.columns else "SMILES"
        idx = set()
        for sf in EXTERN_POLARIS.glob("adme-fang-*_seed0_split.csv"):
            s = pd.read_csv(sf)
            if "test" in s.columns:
                idx.update(s["test"].dropna().astype(int).tolist())
        idx = [i for i in sorted(idx) if 0 <= i < len(df)]
        out |= canon_set(df.loc[idx, smi_col])
    # DTI eval
    sd = DTI_EVAL_ROOT / "splits"
    if sd.exists():
        cache = {}
        for pt in sd.glob("*.pt"):
            for sub in sorted(DTI_EVAL_SUBSETS, key=len, reverse=True):
                if pt.stem.startswith(sub + "_"):
                    csv = DTI_EVAL_ROOT / f"{sub}.csv"
                    if not csv.exists():
                        break
                    if sub not in cache:
                        cache[sub] = pd.read_csv(csv, usecols=["SMILES_nometa"])["SMILES_nometa"]
                    d = torch.load(pt, map_location="cpu", weights_only=False)
                    if isinstance(d, dict) and "test" in d:
                        idx = [int(i) for i in d["test"] if 0 <= int(i) < len(cache[sub])]
                        out |= canon_set(cache[sub].iloc[idx])
                    break
    # Cell bioactivity
    cb = EXTERN_CELLBIO / "cell_bioactivity.csv"
    cs = EXTERN_CELLBIO / "cell_bioactivity_split.csv"
    if cb.exists() and cs.exists():
        smiles = pd.read_csv(cb, usecols=["smiles"])["smiles"]
        s = pd.read_csv(cs)
        if "test" in s.columns:
            idx = [int(i) for i in s["test"].dropna().astype(int).tolist() if 0 <= int(i) < len(smiles)]
            out |= canon_set(smiles.iloc[idx])
    return out


def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def smiles_to_inchikey(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(m) if m else None
    except Exception:
        return None


def inchikey_to_cid(key):
    url = f"{PUBCHEM_BASE}/compound/inchikey/{urllib.parse.quote(key)}/cids/JSON"
    try:
        body = http_get(url)
        data = json.loads(body)
        cids = data.get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def cid_to_description(cid):
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/description/JSON"
    try:
        body = http_get(url)
        data = json.loads(body)
        info = data.get("InformationList", {}).get("Information", [])
        for entry in info:
            if "Description" in entry:
                return entry["Description"]
        return None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/dti-processed/_pubchem_probe.json"))
    args = p.parse_args()

    print(f"Loading pool + test exclusion ...")
    pool = load_pool()
    test = load_test_exclusion()
    keep = sorted(pool - test)
    print(f"  pool={len(pool):,}  test={len(test):,}  keep={len(keep):,}")

    rng = random.Random(args.seed)
    sample = rng.sample(keep, min(args.sample, len(keep)))
    print(f"  Sampling {len(sample)} SMILES (seed={args.seed})")

    interval = 1.0 / RATE_PER_SEC
    t_next = time.time()
    results = []
    n_inchikey = n_cid = n_desc = 0
    desc_lens = []

    print("\nProbing PubChem (2 calls per SMILES, throttled to 5 req/s):")
    print("-" * 70)
    for i, smi in enumerate(sample, 1):
        key = smiles_to_inchikey(smi)
        if key is None:
            results.append({"smiles": smi, "inchikey": None, "cid": None, "desc": None})
            continue
        n_inchikey += 1

        # Throttle
        now = time.time()
        if now < t_next:
            time.sleep(t_next - now)
        t_next = time.time() + interval

        try:
            cid = inchikey_to_cid(key)
        except Exception as e:
            print(f"  [{i}] CID fetch error: {e}")
            cid = None
        if cid is None:
            results.append({"smiles": smi, "inchikey": key, "cid": None, "desc": None})
            continue
        n_cid += 1

        # Throttle for description call
        now = time.time()
        if now < t_next:
            time.sleep(t_next - now)
        t_next = time.time() + interval

        try:
            desc = cid_to_description(cid)
        except Exception as e:
            print(f"  [{i}] desc fetch error: {e}")
            desc = None
        if desc:
            n_desc += 1
            desc_lens.append(len(desc))
        results.append({"smiles": smi, "inchikey": key, "cid": cid, "desc": desc})

        if i % 50 == 0 or i == len(sample):
            print(
                f"  [{i}/{len(sample)}] inchikey={n_inchikey}  cid={n_cid}  "
                f"desc={n_desc} ({100*n_desc/i:.1f}% of sample)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"sample_size": len(sample), "results": results}, f)

    n = len(sample)
    print("\n" + "=" * 70)
    print("PUBCHEM COVERAGE PROBE")
    print("=" * 70)
    print(f"  sample size:        {n}")
    print(f"  InChIKey computed:  {n_inchikey:>5} ({100*n_inchikey/n:.1f}%)")
    print(f"  CID matched:        {n_cid:>5} ({100*n_cid/n:.1f}%)")
    print(f"  Description found:  {n_desc:>5} ({100*n_desc/n:.1f}%)")
    if desc_lens:
        print(f"  Desc length: mean={sum(desc_lens)/len(desc_lens):.0f} chars  "
              f"min={min(desc_lens)}  max={max(desc_lens)}")
    print(f"\n  Extrapolated to full pool ({len(keep):,} SMILES):")
    print(f"    Expect ~{int(len(keep) * n_desc / n):,} with descriptions "
          f"({100*n_desc/n:.1f}%)")
    print(f"\n  Raw probe results saved to: {args.out}")


if __name__ == "__main__":
    main()
