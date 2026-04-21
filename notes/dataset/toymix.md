# Dataset — ToyMix

Graphium's small-dataset multi-task pretraining mix. Three public molecular
property datasets combined into a shared multi-task training objective. Used
as the fast-iteration / small-compute pretraining target (a few hours on a
single GPU).

Source: Valence Discovery, NeurIPS 2023 Datasets track. Hosted at
`https://storage.valencelabs.com/graphium/datasets/neurips_2023/Small-dataset/`.

## Tasks

| Task    | Molecules | Labels / task              | Task type                         | Label(s) |
|---------|----------:|---------------------------|-----------------------------------|---|
| `qm9`   |   133,885 | 19 per molecule            | multi-output regression (graph)   | quantum-chemistry properties (DFT): A, B, C (rotational), dipole μ, polarizability α, HOMO, LUMO, gap, ⟨r²⟩, ZPVE, U₀, U₂₉₈, H₂₉₈, G₂₉₈, C_v, plus 5 atomization-energy variants |
| `tox21` |     7,831 | 12 per molecule (sparse)   | multi-output binary classification | 7 nuclear-receptor and 5 stress-response assays (NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-γ, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53) |
| `zinc`  |    12,000 | 3 per molecule             | multi-output regression (graph)   | synthetic accessibility (SA), constrained logP, heuristic score |

**Total molecules: ~153 k.** The three datasets are trained jointly with shared
GNN layers and task-specific heads. Missing Tox21 labels are NaN and masked out
of the loss.

## Splits

Fixed random 80 / 10 / 10 split provided alongside each dataset as a
`_random_splits.pt` file:

| Task    | Train    | Val    | Test   |
|---------|---------:|-------:|-------:|
| `qm9`   |  107,109 | 13,388 | 13,388 |
| `tox21` |    6,265 |    783 |    783 |
| `zinc`  |    9,600 |  1,200 |  1,200 |

Splits are molecule-level (not scaffold); the same split file is reused across
every ToyMix run, so seeds only affect training stochasticity.

## Label preprocessing

- QM9 and ZINC regression labels are **z-score normalized** inside the
  datamodule (`method: normal`, applied to val/test too so all metrics live on
  the same scale).
- Tox21 labels are binary (0/1) with NaN for not-measured; normalization is a
  no-op.

## Loss & metrics

- **QM9 / ZINC** (regression): MAE (loss), plus Pearson r and R² for
  monitoring.
- **Tox21** (classification): BCE-with-logits, plus AUROC, AUPRC, F1@0.5,
  precision@0.5.

## Reference

- Beaini et al. 2024, _Towards Foundational Models for Molecular Learning on
  Large-Scale Multi-Task Datasets._ ICLR.
- QM9: Ramakrishnan et al. 2014. Tox21: NIH Tox21 Challenge 2014. ZINC: Irwin
  & Shoichet 2005 / ZINC-250k sampled to ZINC-12k.
