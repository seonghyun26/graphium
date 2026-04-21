# Literature Embeddings

## Source

`data/literature/embeddings.pt` contains 160,560 PubMedBERT (768-dim) embeddings.

These are the **same 160,560 molecules** as the LPM-24 dataset. The embeddings were pre-extracted from molecular property captions using PubMedBERT (~110M params).

## Relationship to LPM-24

The LPM-24 pipeline (`scripts/data/lpm24/`) reproduces these embeddings from scratch and additionally supports two alternative text encoders:

1. Downloads L+M-24 from HuggingFace (160,560 SMILES + captions)
2. Extracts embeddings from captions with the chosen text encoder
3. Z-score normalizes and saves as CSV

`data/literature/embeddings.pt` is the raw (unnormalized) output of step 2 for **PubMedBERT**, stored as a list of 160,560 tensors of shape `(1, 768)`.

## Supported encoders

All fit on a single GPU and output ≤1024-dim embeddings.

| Key | Model | Params | Dim | Domain |
|---|---|---|---|---|
| `pubmedbert`  | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | ~110M | 768 | Biomedical (PubMed abstracts + PMC full-text) |
| `galactica`   | `facebook/galactica-125m`                                       | ~125M | 768 | Scientific LLM: 48M papers + SMILES + IUPAC (chemistry-aware) |
| `biolinkbert` | `michiyasunaga/BioLinkBERT-large`                               | ~340M | 1024 | PubMed + citation graph (captures relational knowledge) |

Larger Galactica variants (1.3B → 2048-d, 6.7B → 4096-d, 30B+) exceed the 1024-d cap and are not included. `BioLinkBERT-base` (768-d, 110M) is also usable if lower VRAM is needed — add it to `MODEL_REGISTRY` in `02_extract_embeddings.py`.

## Reproducing

```bash
cd graphium/

# PubMedBERT (default, reproduces the legacy embeddings)
bash scripts/data/lpm24/run_lpm24_pipeline.sh --model pubmedbert --gpu 1

# Galactica (chemistry-aware scientific LM)
bash scripts/data/lpm24/run_lpm24_pipeline.sh --model galactica --gpu 1

# BioLinkBERT-large (citation-graph-aware biomedical encoder)
bash scripts/data/lpm24/run_lpm24_pipeline.sh --model biolinkbert --gpu 1
```

Outputs land in `data/dti-processed/lpm24_<model>.{csv,parquet}` and the normalization stats in `lpm24_<model>_norm_stats.pt`. Point a training task config's `df_path` at the CSV corresponding to the encoder you want to use.
