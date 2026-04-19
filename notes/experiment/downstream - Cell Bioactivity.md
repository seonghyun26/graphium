# Downstream — Cell Bioactivity (29-assay ChEMBL)

Multi-task binary classification across 29 ChEMBL bioactivity assays from
**Fredinh et al., _Nat. Commun._ 2024**. A molecule's prediction is a 29-way
vector of assay activity probabilities; the ground truth is sparse (each
molecule labeled for only a subset of assays).

## Dataset statistics

- **Molecules**: 13,210 unique SMILES.
- **Assays**: 29 binary ChEMBL assays (all from the paper's JUMP-CP linked
  subset).
- **Total molecule–assay cells**: 383,090 (13,210 × 29).
- **Observed labels**: 25,107 (**6.6 %** density — highly sparse).
- **Class balance (over observed labels)**: 37 % positive / 63 % negative.
- **Split**: pre-defined train / val / test partition shipped with the dataset
  (scaffold-based, not regenerated per run).

## Evaluation protocol

- **Unit of evaluation**: **one multi-output model** per seed — a single run
  trains the whole 29-dim head jointly. Unlike TDC ADMET, there is no per-task
  loop.
- **Missing-label handling**: labels outside a molecule's measured assays are
  NaN. The loss masks them; metrics only aggregate over observed cells.
- **Per-assay aggregation**: all metrics use `mean-per-label` aggregation —
  each assay contributes equally regardless of its label count, so
  well-measured assays don't dominate the score.
- **Headline number (paper)**: **mean per-assay AUROC on test** (mirroring
  Fredinh et al.).

## Prediction head

A single 29-output MLP head appended to the pretrained molecule encoder:

```
MLP( z_mol )  →  29-dim logit vector
  hidden_dims = 256, depth = 2, ReLU, dropout = 0.5, LayerNorm
  last_activation = none   # BCE-with-logits expects raw logits
```

## Training protocol

Backbone frozen (linear-probe-like) by default, with an optional unfreeze at
epoch 25.

| Setting                   | Value                          |
|---------------------------|--------------------------------|
| Optimizer                 | Adam                           |
| Learning rate             | 1e-4                           |
| Scheduler                 | CosineAnnealingLR              |
| T_max / η_min             | 50 / 1e-6                      |
| Max epochs                | 50                             |
| Loss                      | BCE-with-logits, NaN-masked    |
| NaN mask                  | `ignore-mean-per-label`        |

## Logged metrics

All metrics are averaged per-assay (mean over 29 assays) with NaN labels
dropped from the aggregation:

- **AUROC** (primary, progress-bar),
- AUPRC,
- accuracy (threshold = 0.5).

## Reference

- Fredinh et al., 2024. _Large-scale bioactivity prediction with cell-painting
  assay embeddings._ _Nature Communications_.
