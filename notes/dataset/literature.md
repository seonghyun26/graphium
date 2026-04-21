# Dataset — Literature (LPM24 / PubMedBERT)

Auxiliary pretraining dataset for learning **SMILES → literature-text
embedding** regression. The model is taught to map each compound to the
contextual text representation of its most-cited biomedical sentence.

## Source

- Compound-literature pairs curated in 2024 ("LPM24" = Literature-Paired
  Molecules 2024) by cross-referencing SMILES with PubMed abstracts / figure
  captions that mention the compound (or a standardized synonym).
- Text encoder: **PubMedBERT** (Gu et al. 2021), 768-dim `[CLS]` pooled
  representation over the matched sentence.
- Only compounds with at least one high-confidence literature match are
  retained.

## Statistics

| Property                     | Value      |
|------------------------------|-----------:|
| Unique compounds (SMILES)    |   160,560  |
| Text-embedding dimension     |        768 |
| Label columns                |        768 (`feature_0…feature_767`) |

File: `graphium/data/dti-processed/lpm24_pubmedbert.csv`.

## Task formulation

- **Input**: one SMILES per row (column `SMILES_nometa`, metadata stripped).
- **Output**: a 768-dim real-valued vector — the PubMedBERT `[CLS]` embedding
  of the associated literature sentence.
- **Loss**: MAE over all 768 output dimensions.
- **Task level**: graph (single vector per molecule).

The task teaches the molecule encoder a representation correlated with **how
the compound is described and contextualized in the biomedical literature**
— a signal distinct from quantum chemistry, cellular phenotype, or bioassay
activity.

## Splits

Random 80 / 10 / 10 split at `seed=0`, built inside the datamodule
(`split_val = 0.1, split_test = 0.1`).

## Label preprocessing

Labels are **pre-normalized** (z-score) during the pipeline build, **before**
being written to CSV. The datamodule's `label_normalization` block is
disabled (commented out in `tasks/lpm24.yaml`) to avoid double normalization.

## Metrics

- MAE (loss-aligned)
- Pearson r (per feature, averaged)

## How it's used in pretraining

Added as an extra `lpm24` task head on top of an existing pretraining mix:

- `toymix + lpm24` → `toymix_lpm24` (hydra config `tasks=toymix_lpm24`).
- Standalone `lpm24` is available for ablations.

Combined with other auxiliary signals (ESM-C protein embeddings, BBBC047
morphology), this contributes a **text-grounded** view of each compound to
the shared backbone.

## Reference

- PubMedBERT: Gu et al. 2021, _Domain-Specific Language Model Pretraining for
  Biomedical Natural Language Processing._ ACM Trans. Comput. Healthcare.
