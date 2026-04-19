# Downstream — TDC ADMET

22 single-task ADMET benchmarks from Therapeutics Data Commons
([admet_group](https://tdcommons.ai/benchmark/admet_group/overview/)). Each
benchmark is a property-prediction problem on small molecules — either
regression (MAE / Spearman ρ) or binary classification (AUROC / AUPRC).

## Benchmarks and statistics

All counts are TDC's canonical scaffold splits (`train_val` / `test`). Each
benchmark is evaluated independently as its own run.

| Benchmark                          | Type           | Train   | Test   | Total   | Primary |
|------------------------------------|----------------|--------:|-------:|--------:|---------|
| **Absorption**                     |                |         |        |         |         |
| `caco2_wang`                       | regression     |     728 |    182 |     910 | MAE |
| `hia_hou`                          | classification |     461 |    117 |     578 | AUROC |
| `pgp_broccatelli`                  | classification |     973 |    245 |   1,218 | AUROC |
| `bioavailability_ma`               | classification |     512 |    128 |     640 | AUROC |
| `lipophilicity_astrazeneca`        | regression     |   3,360 |    840 |   4,200 | MAE |
| `solubility_aqsoldb`               | regression     |   7,985 |  1,997 |   9,982 | MAE |
| **Distribution**                   |                |         |        |         |         |
| `bbb_martins`                      | classification |   1,624 |    406 |   2,030 | AUROC |
| `ppbr_az`                          | regression     |   2,231 |    559 |   2,790 | MAE |
| `vdss_lombardo`                    | regression     |     904 |    226 |   1,130 | Spearman ρ |
| **Metabolism**                     |                |         |        |         |         |
| `cyp2d6_veith`                     | classification |  10,504 |  2,626 |  13,130 | AUPRC |
| `cyp3a4_veith`                     | classification |   9,861 |  2,467 |  12,328 | AUPRC |
| `cyp2c9_veith`                     | classification |   9,673 |  2,419 |  12,092 | AUPRC |
| `cyp2d6_substrate_carbonmangels`   | classification |     532 |    135 |     667 | AUPRC |
| `cyp3a4_substrate_carbonmangels`   | classification |     535 |    135 |     670 | AUPRC |
| `cyp2c9_substrate_carbonmangels`   | classification |     534 |    135 |     669 | AUPRC |
| **Excretion**                      |                |         |        |         |         |
| `half_life_obach`                  | regression     |     532 |    135 |     667 | Spearman ρ |
| `clearance_hepatocyte_az`          | regression     |     970 |    243 |   1,213 | Spearman ρ |
| `clearance_microsome_az`           | regression     |     881 |    221 |   1,102 | Spearman ρ |
| **Toxicity**                       |                |         |        |         |         |
| `ld50_zhu`                         | regression     |   5,907 |  1,478 |   7,385 | MAE |
| `herg`                             | classification |     523 |    132 |     655 | AUROC |
| `ames`                             | classification |   5,821 |  1,457 |   7,278 | AUROC |
| `dili`                             | classification |     379 |     96 |     475 | AUROC |

Totals: **22 benchmarks, 81,419 molecules** (14,299 test). 13 classification,
9 regression. Primary-metric choices mirror the TDC leaderboard.

## Evaluation protocol

- **Split**: canonical TDC `train_val` / `test` per benchmark. The `train_val`
  partition is further split at a fixed seed into train / val.
- **Unit of evaluation**: one fitted model per (benchmark, seed). Results are
  reported as the mean over ≥ 3 seeds per benchmark.
- **Headline number (paper)**: the primary metric in the table above
  (mirroring TDC's public leaderboard).

## Prediction head

A fresh per-benchmark MLP is appended to the frozen pretrained backbone
embedding `z_mol ∈ ℝ^256`:

```
MLP( z_mol )  →  scalar logit
  hidden_dims = 256, depth = 4, ReLU, last layer is readout
```

For classification, logits are passed through sigmoid at metric time; for
regression, the scalar is the prediction directly.

## Training protocol

Two regimes reported side-by-side:

| Setting                | Scratch (no pretrain) | Finetune (frozen backbone) |
|------------------------|-----------------------|----------------------------|
| Optimizer              | Adam                  | Adam                       |
| Learning rate          | 4e-5                  | 1e-4                       |
| Scheduler              | WarmUpLinearLR        | CosineAnnealingLR          |
| Warmup / η_min         | 10 epochs             | 1e-6                       |
| Max epochs             | 100                   | 100                        |
| Backbone               | trainable (full model)| **frozen by default**      |
| Loss (regression)      | MAE                   | MAE                        |
| Loss (classification)  | BCE-with-logits       | BCE-with-logits            |

Unfreezing schedules can be enabled per-run; the default reported numbers are
with the backbone frozen throughout.

## Logged metrics (per benchmark)

- Regression: **MAE**, Spearman ρ, Pearson r, R².
- Classification: **AUROC**, AUPRC, accuracy, MCC (threshold = 0.5).

The bold metric in each row, plus the table's primary column, is what we report
in the paper.

## Reference

- Huang et al. 2021, _Therapeutics Data Commons: Machine Learning Datasets and
  Tasks for Drug Discovery and Development._ NeurIPS Datasets & Benchmarks.
