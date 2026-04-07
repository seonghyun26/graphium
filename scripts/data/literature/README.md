# Literature Embeddings

## Source

`data/literature/embeddings.pt` contains 160,560 PubMedBERT (768-dim) embeddings.

These are the **same 160,560 molecules** as the LPM-24 dataset. The embeddings were pre-extracted from molecular property captions using PubMedBERT (~110M params).

## Relationship to LPM-24

The LPM-24 pipeline (`scripts/data/lpm24/`) reproduces these embeddings from scratch:
1. Downloads L+M-24 from HuggingFace (160,560 SMILES + captions)
2. Extracts PubMedBERT embeddings from captions
3. Z-score normalizes and saves as CSV

`data/literature/embeddings.pt` is the raw (unnormalized) output of step 2, stored as a list of 160,560 tensors of shape `(1, 768)`.

## Model

- **PubMedBERT** (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- ~110M parameters
- 768-dim output embeddings
- Pre-trained on PubMed abstracts + PMC full-text articles
