#!/usr/bin/env python3
"""
ToyMix SMILES overlap check against PubChem, ChEMBL, and DrugBank.

Outputs:
  notebooks/data/toymix_db_overlap.csv   per-molecule flags + IDs
  stdout                                  summary statistics

Checkpoints are saved to /tmp/toymix_overlap_ckp/ so partial runs survive.
Run from graphium/ directory.
"""

import pandas as pd
import pickle, time, sys
from io import StringIO
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from chembl_webresource_client.new_client import new_client

# ── Config ─────────────────────────────────────────────────────────────────
BATCH_PUB    = 800    # InChIKeys per PubChem compound request
BATCH_PUBAID = 100    # CIDs per PubChem AID request (heavy payload)
BATCH_DB     = 200    # CIDs per DrugBank xref request
BATCH_CHEMBL = 200    # InChIKeys per ChEMBL molecule request

SLEEP_PUB    = 0.22   # ~4.5 req/s (PubChem limit: 5/s)
SLEEP_CHEMBL = 1.2    # ChEMBL is slower

BASE = Path("data/graphium/neurips2023/small-dataset")
CKP  = Path("/tmp/toymix_overlap_ckp")
OUT  = Path("notebooks/data")
CKP.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def checkpoint_load(path):
    return pickle.load(open(path, "rb")) if path.exists() else None

def checkpoint_save(path, obj):
    pickle.dump(obj, open(path, "wb"))


# ── 1. Load SMILES ──────────────────────────────────────────────────────────
print("Loading ToyMix datasets...")
qm9  = pd.read_parquet(BASE / "qm9_filtered.parquet")[["smiles"]].assign(source="qm9")
tox  = pd.read_parquet(BASE / "Tox21-7k-12-labels_filtered.parquet")[["smiles"]].assign(source="tox21")
zinc = pd.read_parquet(BASE / "ZINC12k_filtered.parquet")[["smiles"]].assign(source="zinc12k")

# Track which sources each SMILES appears in (minimal overlap expected)
src_map = {}
for row in pd.concat([qm9, tox, zinc]).itertuples():
    src_map.setdefault(row.smiles, set()).add(row.source)

df = pd.concat([qm9, tox, zinc]).drop_duplicates("smiles").reset_index(drop=True)
df["sources"] = df["smiles"].map(lambda s: ",".join(sorted(src_map[s])))
print(f"  QM9: {len(qm9):,}  Tox21: {len(tox):,}  ZINC12k: {len(zinc):,}  Union: {len(df):,}")


# ── 2. InChIKey generation ──────────────────────────────────────────────────
ck_ik = CKP / "inchikeys.pkl"
inchikeys = checkpoint_load(ck_ik) or {}

if len(inchikeys) < len(df):
    print(f"Generating InChIKeys ({len(df) - len(inchikeys):,} remaining)...")
    for i, smi in enumerate(df["smiles"]):
        if smi in inchikeys:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                inchi = MolToInchi(mol)
                if inchi:
                    inchikeys[smi] = InchiToInchiKey(inchi)
        except Exception:
            pass
        if (i + 1) % 25000 == 0:
            print(f"  {i+1:,}/{len(df):,}")
            checkpoint_save(ck_ik, inchikeys)
    checkpoint_save(ck_ik, inchikeys)

df["inchikey"] = df["smiles"].map(inchikeys)
df_v = df.dropna(subset=["inchikey"]).copy()
print(f"Valid InChIKeys: {len(df_v):,} / {len(df):,}")


# ── 3. PubChem CID lookup (InChIKey → CID) ─────────────────────────────────
# POST /pug/compound/inchikey/property/InChIKey/CSV
# Returns CSV: CID, InChIKey  (only for found molecules)
ck_pub = CKP / "pubchem_cid.pkl"
pub_map = checkpoint_load(ck_pub) or {}  # inchikey -> cid

todo_keys = [k for k in df_v["inchikey"].tolist() if k not in pub_map]
if todo_keys:
    batches = [todo_keys[i:i+BATCH_PUB] for i in range(0, len(todo_keys), BATCH_PUB)]
    print(f"\nPubChem CID lookup: {len(batches)} batches × {BATCH_PUB} ...")
    for bi, batch in enumerate(batches):
        try:
            resp = requests.post(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/property/InChIKey/CSV",
                data={"inchikey": ",".join(batch)},
                timeout=60,
            )
            if resp.status_code == 200:
                tbl = pd.read_csv(StringIO(resp.text))
                # columns: CID, InChIKey
                for _, row in tbl.iterrows():
                    pub_map[row["InChIKey"]] = int(row["CID"])
            elif resp.status_code == 404:
                pass  # none in batch found
            else:
                print(f"  batch {bi}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  batch {bi} error: {e}")
        time.sleep(SLEEP_PUB)
        if (bi + 1) % 50 == 0:
            print(f"  {bi+1}/{len(batches)} done, found so far: {len(pub_map):,}")
            checkpoint_save(ck_pub, pub_map)
    checkpoint_save(ck_pub, pub_map)

df_v["pubchem_cid"] = df_v["inchikey"].map(pub_map)
df_v["in_pubchem"] = df_v["pubchem_cid"].notna()
print(f"In PubChem: {df_v['in_pubchem'].sum():,} / {len(df_v):,} ({df_v['in_pubchem'].mean()*100:.1f}%)")


# ── 4. PubChem BioAssay presence (CID → AIDs) ──────────────────────────────
# GET /pug/compound/cid/{cids}/aids/JSON  →  InformationList[{CID, AID:[...]}]
ck_pub_bio = CKP / "pubchem_bio.pkl"
pub_bio = checkpoint_load(ck_pub_bio) or {}  # cid(int) -> bool

cids_for_bio = df_v.loc[df_v["in_pubchem"], "pubchem_cid"].dropna().astype(int).tolist()
todo_cids_bio = [c for c in cids_for_bio if c not in pub_bio]
if todo_cids_bio:
    batches = [todo_cids_bio[i:i+BATCH_PUBAID] for i in range(0, len(todo_cids_bio), BATCH_PUBAID)]
    print(f"\nPubChem BioAssay check: {len(batches)} batches × {BATCH_PUBAID} ...")
    for bi, batch in enumerate(batches):
        try:
            cid_str = ",".join(map(str, batch))
            resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/aids/JSON",
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                active = {}
                for info in data.get("InformationList", {}).get("Information", []):
                    cid = info.get("CID")
                    aids = info.get("AID", [])
                    active[cid] = len(aids) > 0
                for c in batch:
                    pub_bio[c] = active.get(c, False)
            elif resp.status_code == 404:
                for c in batch:
                    pub_bio[c] = False
            else:
                print(f"  batch {bi}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  batch {bi} error: {e}")
        time.sleep(SLEEP_PUB)
        if (bi + 1) % 100 == 0:
            print(f"  {bi+1}/{len(batches)}, with_bioassay: {sum(pub_bio.values()):,}")
            checkpoint_save(ck_pub_bio, pub_bio)
    checkpoint_save(ck_pub_bio, pub_bio)

df_v["pubchem_has_bioassay"] = df_v["pubchem_cid"].map(
    lambda x: pub_bio.get(int(x), False) if pd.notna(x) else False
)
print(f"PubChem + BioAssay: {df_v['pubchem_has_bioassay'].sum():,}")


# ── 5. PubChem description text ─────────────────────────────────────────────
# GET /pug/compound/cid/{cids}/description/JSON  (batched, 100 CIDs per request)
# Concatenates all Description fields across sources into one string per molecule.
# These texts can later be embedded with e.g. OpenAI text-embedding-3-small
# (fixed 1536-dim output regardless of input text length; ~$0.28 for 46k molecules).
BATCH_DESC = 100
ck_desc = CKP / "pubchem_desc.pkl"
desc_map = checkpoint_load(ck_desc) or {}  # cid(int) -> str (concatenated descriptions)

todo_desc = [c for c in cids_for_bio if c not in desc_map]
if todo_desc:
    batches = [todo_desc[i:i+BATCH_DESC] for i in range(0, len(todo_desc), BATCH_DESC)]
    print(f"\nPubChem descriptions: {len(batches)} batches × {BATCH_DESC} ...")
    for bi, batch in enumerate(batches):
        try:
            cid_str = ",".join(map(str, batch))
            resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/description/JSON",
                timeout=60,
            )
            if resp.status_code == 200:
                by_cid = {}
                for info in resp.json().get("InformationList", {}).get("Information", []):
                    cid = info.get("CID")
                    if cid is None:
                        continue
                    if "Description" in info:
                        by_cid.setdefault(cid, []).append(info["Description"])
                    elif "Title" in info:
                        by_cid.setdefault(cid, [])  # ensure entry exists
                for c in batch:
                    parts = by_cid.get(c, [])
                    desc_map[c] = " ".join(parts) if parts else None
            elif resp.status_code == 404:
                for c in batch:
                    desc_map[c] = None
            else:
                print(f"  batch {bi}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  batch {bi} error: {e}")
        time.sleep(SLEEP_PUB)
        if (bi + 1) % 50 == 0:
            n_with = sum(1 for v in desc_map.values() if v)
            print(f"  {bi+1}/{len(batches)}, with description: {n_with:,}")
            checkpoint_save(ck_desc, desc_map)
    checkpoint_save(ck_desc, desc_map)

df_v["pubchem_description"] = df_v["pubchem_cid"].map(
    lambda x: desc_map.get(int(x)) if pd.notna(x) else None
)
df_v["has_pubchem_desc"] = df_v["pubchem_description"].notna()
n_desc = df_v["has_pubchem_desc"].sum()
print(f"PubChem descriptions: {n_desc:,} / {df_v['in_pubchem'].sum():,} PubChem-found molecules")


# ── 6. DrugBank via PubChem RegistryID xrefs ───────────────────────────────
# GET /pug/compound/cid/{cids}/xrefs/RegistryID/JSON
# DrugBank IDs match pattern DB\d{5}
ck_db = CKP / "drugbank.pkl"
db_map = checkpoint_load(ck_db) or {}  # cid(int) -> drugbank_id or None

cids_for_db = cids_for_bio  # same set: only PubChem-found CIDs
todo_cids_db = [c for c in cids_for_db if c not in db_map]
if todo_cids_db:
    batches = [todo_cids_db[i:i+BATCH_DB] for i in range(0, len(todo_cids_db), BATCH_DB)]
    print(f"\nDrugBank xref check: {len(batches)} batches × {BATCH_DB} ...")
    for bi, batch in enumerate(batches):
        try:
            cid_str = ",".join(map(str, batch))
            resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/xrefs/RegistryID/JSON",
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                for info in data.get("InformationList", {}).get("Information", []):
                    cid = info.get("CID")
                    reg_ids = info.get("RegistryID", [])
                    import re as _re
                    db_ids = [r for r in reg_ids if isinstance(r, str) and _re.fullmatch(r'DB\d{5}', r)]
                    db_map[cid] = db_ids[0] if db_ids else None
            elif resp.status_code == 404:
                for c in batch:
                    db_map[c] = None
            else:
                print(f"  batch {bi}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  batch {bi} error: {e}")
        time.sleep(SLEEP_PUB)
        if (bi + 1) % 50 == 0:
            n_found = sum(1 for v in db_map.values() if v is not None)
            print(f"  {bi+1}/{len(batches)}, in DrugBank: {n_found:,}")
            checkpoint_save(ck_db, db_map)
    checkpoint_save(ck_db, db_map)

df_v["drugbank_id"] = df_v["pubchem_cid"].map(
    lambda x: db_map.get(int(x)) if pd.notna(x) else None
)
df_v["in_drugbank"] = df_v["drugbank_id"].notna()
print(f"In DrugBank: {df_v['in_drugbank'].sum():,} / {len(df_v):,} ({df_v['in_drugbank'].mean()*100:.1f}%)")


# ── 7. ChEMBL compound lookup ───────────────────────────────────────────────
ck_chembl = CKP / "chembl_mol.pkl"
chembl_map = checkpoint_load(ck_chembl) or {}  # inchikey -> chembl_id

todo_chembl = [k for k in df_v["inchikey"].tolist() if k not in chembl_map]
if todo_chembl:
    molecule = new_client.molecule
    batches = [todo_chembl[i:i+BATCH_CHEMBL] for i in range(0, len(todo_chembl), BATCH_CHEMBL)]
    print(f"\nChEMBL compound lookup: {len(batches)} batches × {BATCH_CHEMBL} ...")
    for bi, batch in enumerate(batches):
        try:
            results = molecule.filter(
                molecule_structures__standard_inchi_key__in=batch
            ).only(["molecule_chembl_id", "molecule_structures"])
            found_keys = set()
            for r in results:
                structs = r.get("molecule_structures") or {}
                ik = structs.get("standard_inchi_key")
                if ik:
                    chembl_map[ik] = r["molecule_chembl_id"]
                    found_keys.add(ik)
            # Mark not-found keys so we don't re-query them
            for k in batch:
                if k not in chembl_map:
                    chembl_map[k] = None
        except Exception as e:
            print(f"  batch {bi} error: {e}")
        time.sleep(SLEEP_CHEMBL)
        if (bi + 1) % 20 == 0:
            n_found = sum(1 for v in chembl_map.values() if v is not None)
            print(f"  {bi+1}/{len(batches)}, found: {n_found:,}")
            checkpoint_save(ck_chembl, chembl_map)
    checkpoint_save(ck_chembl, chembl_map)

df_v["chembl_id"] = df_v["inchikey"].map(lambda k: chembl_map.get(k))
df_v["in_chembl"] = df_v["chembl_id"].notna()
print(f"In ChEMBL: {df_v['in_chembl'].sum():,} / {len(df_v):,} ({df_v['in_chembl'].mean()*100:.1f}%)")


# ── 8. ChEMBL bioassay activity ─────────────────────────────────────────────
# Uses direct REST with limit=1 per molecule to check existence only (no pagination).
# Concurrent requests via ThreadPoolExecutor for speed (~5-10 min for 6k molecules).
ck_chembl_act = CKP / "chembl_act.pkl"
chembl_act = checkpoint_load(ck_chembl_act) or {}  # chembl_id -> bool

chembl_ids = df_v.loc[df_v["in_chembl"], "chembl_id"].dropna().tolist()
todo_act = [c for c in chembl_ids if c not in chembl_act]

if todo_act:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    CHEMBL_ACT_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={}&limit=1"
    ACT_WORKERS    = 5
    SLEEP_ACT      = 0.05

    def check_chembl_activity(cid):
        try:
            resp = requests.get(CHEMBL_ACT_URL.format(cid), timeout=15)
            if resp.status_code == 200:
                return cid, resp.json()["page_meta"]["total_count"] > 0
            return cid, False
        except Exception:
            return cid, False

    print(f"\nChEMBL activity check: {len(todo_act):,} molecules, {ACT_WORKERS} workers ...")
    done = 0
    with ThreadPoolExecutor(max_workers=ACT_WORKERS) as ex:
        futures = {ex.submit(check_chembl_activity, cid): cid for cid in todo_act}
        for fut in as_completed(futures):
            cid, has_act = fut.result()
            chembl_act[cid] = has_act
            done += 1
            time.sleep(SLEEP_ACT)
            if done % 500 == 0:
                print(f"  {done:,}/{len(todo_act):,}, with_activity: {sum(chembl_act.values()):,}")
                checkpoint_save(ck_chembl_act, chembl_act)
    checkpoint_save(ck_chembl_act, chembl_act)

df_v["chembl_has_activity"] = df_v["chembl_id"].map(
    lambda x: chembl_act.get(x, False) if pd.notna(x) else False
)
print(f"ChEMBL + Activity: {df_v['chembl_has_activity'].sum():,}")


# ── 9. SureChEMBL via UniChem REST (src_id=15) ─────────────────────────────
# Only query PubChem-found molecules (others are unlikely to be in SureChEMBL)
# Uses ThreadPoolExecutor for concurrent requests since no batch API exists.
ck_schembl = CKP / "surechembl.pkl"
schembl_map = checkpoint_load(ck_schembl) or {}  # inchikey -> surechembl_id or None

UNICHEM_URL  = "https://www.ebi.ac.uk/unichem/rest/inchikey/{}"
SCHEMBL_SRC  = "15"
WORKERS      = 4
SLEEP_UC     = 0.05   # per-worker sleep; 4 workers * 20 req/s = 5 req/s total

pub_inchikeys = df_v.loc[df_v["in_pubchem"], "inchikey"].tolist()
todo_uc = [k for k in pub_inchikeys if k not in schembl_map]

if todo_uc:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def query_unichem(ik):
        try:
            resp = requests.get(UNICHEM_URL.format(ik), timeout=15)
            if resp.status_code == 200:
                hits = [x["src_compound_id"] for x in resp.json() if x.get("src_id") == SCHEMBL_SRC]
                return ik, hits[0] if hits else None
            return ik, None
        except Exception:
            return ik, None

    print(f"\nSureChEMBL (UniChem): {len(todo_uc):,} keys, {WORKERS} workers ...")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(query_unichem, k): k for k in todo_uc}
        for fut in as_completed(futures):
            ik, sid = fut.result()
            schembl_map[ik] = sid
            done += 1
            time.sleep(SLEEP_UC)
            if done % 2000 == 0:
                n_found = sum(1 for v in schembl_map.values() if v is not None)
                print(f"  {done:,}/{len(todo_uc):,}, in SureChEMBL: {n_found:,}")
                checkpoint_save(ck_schembl, schembl_map)
    checkpoint_save(ck_schembl, schembl_map)

df_v["surechembl_id"]  = df_v["inchikey"].map(lambda k: schembl_map.get(k))
df_v["in_surechembl"]  = df_v["surechembl_id"].notna()
print(f"In SureChEMBL: {df_v['in_surechembl'].sum():,} / {len(df_v):,} ({df_v['in_surechembl'].mean()*100:.1f}%)")


# ── 10. Save and summarize ──────────────────────────────────────────────────
df_v.to_csv(OUT / "toymix_db_overlap.csv", index=False)
print(f"\nSaved: {OUT / 'toymix_db_overlap.csv'}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
cols = ["in_pubchem", "pubchem_has_bioassay", "in_drugbank", "in_chembl", "chembl_has_activity", "in_surechembl"]
labels = ["In PubChem", "PubChem + BioAssay", "In DrugBank", "In ChEMBL", "ChEMBL + Activity", "In SureChEMBL"]

for src_filter, label in [("qm9", "QM9"), ("tox21", "Tox21"), ("zinc12k", "ZINC12k"), (None, "ALL")]:
    if src_filter:
        sub = df_v[df_v["sources"].str.contains(src_filter)]
    else:
        sub = df_v
    n = len(sub)
    print(f"\n{label} (n={n:,})")
    for col, lbl in zip(cols, labels):
        cnt = sub[col].sum()
        pct = cnt / n * 100
        print(f"  {lbl:<25} {cnt:6,}  ({pct:.1f}%)")

print("\nDone.")
