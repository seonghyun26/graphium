# Dataset — ESM-C v2 (DTI)

Auxiliary pretraining dataset for learning **SMILES → protein-embedding**
regression. The model is taught to map each ligand in a drug–target pair to
the ESM-C protein embedding of its binding target, via MAE regression.

## Source

- Drug–target pairs taken from TDC's DTI benchmarks, concatenated and
  deduplicated. 100,000 pairs (subsampled from the full DTI pool to match
  toymix-scale compute budgets).
- Protein encoder: **ESM-C** (2024 release from EvolutionaryScale / Meta FAIR).
  Sequences run through ESM-C and mean-pooled to a 1,152-dim embedding per
  target.

"v2" indicates the second version of this dataset, built with ESM-C (vs the
v1 which used ESM-2). ESM-C gives a smaller, newer embedding that transfers
better in our finetune evaluations.

## Statistics

| Property                     | Value      |
|------------------------------|-----------:|
| Drug–target pairs            |   100,000  |
| Unique drugs (SMILES)        | see `dti_esmc_100k_v2.csv` |
| Protein embedding dimension  |      1,152 |
| Label columns                |      1,152 (`feature_0…feature_1151`) |

File: `graphium/data/dti-processed/dti_esmc_100k_v2.csv`.

## Task formulation

- **Input**: one SMILES per row (column `SMILES_nometa`, metadata stripped
  during prep).
- **Output**: a 1,152-dim real-valued vector — the mean-pooled ESM-C embedding
  of the paired target protein.
- **Loss**: MAE over all 1,152 output dimensions.
- **Task level**: graph (single vector per molecule).

The model effectively learns a regression surface
`f: SMILES → ℝ^{1152}` that matches the protein-embedding space each SMILES
interacts with. Because a single drug often binds multiple targets, the
labels are noisy — this is by design: the loss encourages the molecule
encoder to learn a representation **predictive of protein-interaction
propensity**.

## Splits

Random 80 / 10 / 10 split at `seed=0`, built inside the datamodule
(`split_val = 0.1, split_test = 0.1`). No pre-materialized split file — the
seed guarantees reproducibility across runs.

## Label preprocessing

Per-feature **z-score normalization** applied inside the datamodule
(`normalize_val_test: True`, `method: normal`) so all 1,152 dimensions are on
a common scale before MAE loss.

## Metrics

- MAE (loss-aligned)
- Pearson r (per feature, averaged)

## How it's used in pretraining

The `dti` task is added on top of an existing pretraining mix (`toymix`,
`largemix`, etc.) as an **extra task head** sharing the backbone:

- `toymix + dti_esmc_v2` → `toymix_dti_esmc_v2` multi-task pretrain
  (graphium hydra config `tasks=toymix_dti_esmc_v2`).
- `largemix + dti_esmc_v2` → `largemix_dti_esmc_v2` (similar pattern).

The shared GNN backbone learns ToyMix/LargeMix chemistry signals jointly with
the DTI protein-embedding signal. Downstream DTI evaluation (see
`notes/experiment/downstream - TDC DTI.md`) reuses the trained backbone plus
its `dti_pactivity` head as the rebuild source.

## Reference

- ESM-C: Hayes, Rao et al. 2024, _Simulating 500 million years of evolution
  with a language model._ bioRxiv / EvolutionaryScale.
- TDC DTI sources: DAVIS, KIBA, BindingDB (see `downstream - TDC DTI.md` for
  per-subset citations).
