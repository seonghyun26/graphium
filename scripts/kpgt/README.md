# KPGT baseline

KPGT (Li et al., Nat. Commun. 2023 — "A knowledge-guided pre-training framework
for improving molecular representation learning") evaluated as a downstream
embedding extractor, alongside MiniMol and MolE.

## Why a separate conda env

KPGT pins DGL 0.7.2 + torch 1.10. The graphium env runs torch 2.7 + cu128
(Blackwell sm_120), so dropping DGL into it bricks unrelated code via a
`graphbolt` ABI mismatch. Embedding extraction therefore lives in a sibling
`kpgt` conda env (torch 2.4 + DGL 2.4 + descriptastorus, **CPU inference**)
and the graphium-side encoder shells out via subprocess.

CPU inference is the deliberate fallback: pip `dgl` ships without the CUDA
backend, and torch 2.4+cu121 doesn't support sm_120 Blackwell anyway. LiGhT-base
runs at ~77 mol/s on CPU; embeddings are cached per-SMILES so subsequent seeds
hit the cache.

## One-time setup

```bash
# 1. Create the env (skip if /home/shpark/miniforge3/envs/kpgt already exists)
mamba create -n kpgt -c conda-forge -y python=3.10 numpy=1.26 scipy=1.13 pandas tqdm 'rdkit>=2024.03' pip
mamba run -n kpgt pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
mamba run -n kpgt pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
mamba run -n kpgt pip install descriptastorus dgllife 'torchdata<0.10'

# 2. Clone upstream KPGT at the pinned commit
mkdir -p downloads/kpgt/upstream
git clone https://github.com/lihan97/KPGT.git downloads/kpgt/upstream/KPGT
( cd downloads/kpgt/upstream/KPGT && git checkout 47dc1646c70b2138a157de481d24a1ac35d174cd )

# 3. Download base.pth (~447 MB) from figshare
#    The share is JS-rendered behind AWS WAF — grab manually:
#       https://figshare.com/s/d488f30c23946cf6898f
#    Save the file as downloads/kpgt/base.pth
```

## Running

ADMET TDC + Polaris ADME-Fang (mirrors `run_mole_admet.sh`):

```bash
bash scripts/kpgt/run_kpgt_admet.sh 4              # GPU id only matters for env
SEEDS="0 1 2" HEAD=mlp bash scripts/kpgt/run_kpgt_admet.sh 4
```

The other downstream task drivers are unchanged — KPGT is a registered encoder:

```bash
python -m downstream.tasks.bioactivity.eval --encoder kpgt --cv
python -m downstream.tasks.gram_dti.eval --encoder kpgt --head mlp
python -m downstream.tasks.tdc_dti_regression.eval --encoder kpgt --head mlp
```

## Outputs

| Path                                           | Rows                                          |
|------------------------------------------------|-----------------------------------------------|
| `results/kpgt_results.csv`                     | one row per (benchmark, task, seed) — ADMET   |
| `results/downstream/bioactivity.csv`           | one row per (encoder, fold)                   |
| `results/downstream/gram_dti.csv`              | one row per (encoder, subset, method, fold)   |
| `results/downstream/tdc_dti_regression.csv`    | one row per (encoder, subset, method, seed)   |
| `datacache/kpgt_embeddings/<benchmark>_<task>.pt` | per-SMILES `.pt` cache (2304-d float32)    |
| `datacache/downstream/<task>/kpgt__kpgt_base.pt` | shared cache for the bioactivity/dti drivers |

## Provenance

Embedding pipeline mirrors KPGT's `preprocess_downstream_dataset.py` byte-for-byte:
- `smiles_to_graph_tune(max_length=5, n_virtual_nodes=2)` — DGL triplet line graph
- `Chem.RDKFingerprint(minPath=1, maxPath=7, fpSize=512)` — 512-bit FP
- `descriptastorus.descriptors.rdNormalizedDescriptors.RDKit2DNormalized` — 200-d MD vector
  (descriptastorus replaces KPGT's vendored copy, which references `scipy.stats.gilbrat` →
  renamed to `gibrat` in scipy 1.11+)
- `LiGhTPredictor.generate_fps` returns `[fp_vn, md_vn, mean_readout]` → 3 × 768 = **2304-d**

Pinned KPGT upstream commit: `47dc1646c70b2138a157de481d24a1ac35d174cd` (2024-09-17).
