# Pre-training Dataset Pipelines

All dataset download and preparation scripts for Graphium pre-training.

## Datasets

| Dataset | Source | Embedding Model | Params | Dims | Samples | Command |
|---|---|---|---|---|---|---|
| ToyMix | Zenodo 10797794 | N/A (raw features) | - | varies | ~150K | `bash scripts/data/download_graphium_dataset.sh` |
| LargeMix | Zenodo 10797794 | N/A (raw features) | - | varies | ~5.5M | `bash scripts/data/download_graphium_dataset.sh` |
| DTI (ESM2) | TDC + ESM2 | ESM2 t36 3B | 3B | 2560 | 100K | `bash scripts/data/dti_esm2/run_pipeline.sh 0` |
| DTI (ESM-C) | TDC + ESM-C | ESM-C 600M | 600M | 1152 | 100K | `bash scripts/data/dti_esmc/run_pipeline.sh 0` |
| LPM-24 | HuggingFace L+M-24 | PubMedBERT | 110M | 768 | 160K | `bash scripts/data/lpm24/run_pipeline.sh 0` |
| RxRx3 | Recursion rxrx.ai | OpenPhenom-S/16 | 22M | 384 | 17K | `bash scripts/data/rxrx3/download.sh` |
| BBBC047 | JUMP Cell Painting | CPCNN (EfficientNet-B0) | 5.3M | 672 | 109K | `bash scripts/data/bbbc047/download.sh` |

## Quick Start

Download everything:
```bash
bash scripts/data/download_all.sh
```

Or run individual pipelines:
```bash
# Molecular graph datasets (ToyMix + LargeMix)
bash scripts/data/download_graphium_dataset.sh

# DTI with ESM-2 protein embeddings (requires GPU, ~4h)
bash scripts/data/dti_esm2/run_pipeline.sh 0,1

# DTI with ESM-C protein embeddings (requires GPU + esmc conda env)
bash scripts/data/dti_esmc/run_pipeline.sh 0,1,2,3

# LPM-24 language embeddings (requires GPU, ~2h)
bash scripts/data/lpm24/run_pipeline.sh 0

# Pre-computed cell morphology embeddings
bash scripts/data/rxrx3/download.sh
bash scripts/data/bbbc047/download.sh
```

## Output Locations

| Dataset | Output Path |
|---|---|
| ToyMix | `data/graphium/neurips2023/small-dataset/` |
| LargeMix | `data/graphium/neurips2023/large-dataset/` |
| DTI (ESM2) | `data/dti-processed/dti_esm2_100k.csv` |
| DTI (ESM-C) | `data/dti-processed/dti_esmc_100k.csv` |
| LPM-24 | `data/dti-processed/lpm24_pubmedbert.csv` |
| RxRx3 | `data/rxrx3/rxrx3_smiles_embeddings.csv` |
| BBBC047 | `../../data/bbbc047/bbbc047_smiles_embeddings.csv` |

## Training

After downloading, use in training with Hydra task configs:
```bash
graphium-train tasks=toymix_dti model=gpspp_800M ...
```

See `expts/hydra-configs/tasks/` for all available task combinations.
