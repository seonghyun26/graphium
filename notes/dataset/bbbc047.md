# Dataset — BBBC047 (Cell Painting)

Auxiliary pretraining dataset for learning **SMILES → cell-morphology
embedding** regression. The model is taught to map each compound to its
Cell Painting morphological profile.

## Source

- Compound library: **BBBC047** from the JUMP Cell Painting consortium
  (`cpg0016`, Cell Painting Gallery on AWS).
- Imaging protocol: Cell Painting assay on U2OS cells, six fluorescent
  channels. Images processed through the JUMP consortium standard pipeline.
- Encoder: **CPCNN** (Cell-Painting CNN) based on EfficientNet-B0, pretrained
  on the JUMP-CP collection, as published in Moshkov et al. 2024.

## Statistics

| Property                              | Value      |
|---------------------------------------|-----------:|
| Unique compounds (SMILES)             |   109,645  |
| Morphological embedding dimension     |        672 |
| Label columns                         |        672 (`feature_0…feature_671`) |

File: `/home/shpark/prj-molrepr/data/bbbc047/bbbc047_smiles_embeddings.csv`.

## Task formulation

- **Input**: one SMILES per row.
- **Output**: a 672-dim real-valued vector — the CPCNN morphological
  embedding of U2OS cells exposed to that compound.
- **Loss**: MAE over all 672 output dimensions.
- **Task level**: graph (single vector per molecule).

Morphological embeddings encode the compound's phenotypic signature:
cytoskeletal organization, organelle integrity, nuclear texture, etc. Training
a GNN to predict this vector teaches the molecule encoder a representation
correlated with **cellular phenotype**, complementary to both quantum-chemistry
and assay-activity signals.

## Splits

Random 80 / 10 / 10 split at `seed=0`, built inside the datamodule
(`split_val = 0.1, split_test = 0.1`). No scaffold split — BBBC047 is
exploratory, so random suffices.

## Label preprocessing

Per-feature **z-score normalization** applied inside the datamodule
(`normalize_val_test: True`, `method: normal`).

## Metrics

- MAE (loss-aligned)
- Pearson r (per feature, averaged)

## How it's used in pretraining

Added as an extra `bbbc047` task head on top of existing pretraining mixes:

- `toymix + bbbc047` → `toymix_bbbc047` (hydra config
  `tasks=toymix_bbbc047`).
- `bbbc047` alone can also be run standalone for ablations.

The pretrained backbone can then be used as the starting point for downstream
ADMET / DTI / cell-bioactivity finetuning.

## Reference

- Moshkov et al. 2024, _Learning representations for image-based profiling of
  perturbations._ Nature Communications 15, 1594.
- JUMP Cell Painting consortium dataset catalog: Chandrasekaran et al. 2023,
  _JUMP Cell Painting dataset: morphological impact of 136,000 chemical and
  genetic perturbations._ bioRxiv.
