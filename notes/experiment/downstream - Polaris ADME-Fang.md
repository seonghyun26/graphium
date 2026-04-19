# Downstream — Polaris ADME-Fang

Six single-task regression benchmarks from Polaris'
[`biogen/adme-fang-v1`](https://polarishub.io/benchmarks/biogen) suite —
small-molecule ADME properties published by Fang et al. (Biogen). Each
endpoint is evaluated independently.

## Endpoints

| Benchmark           | Polaris slug              | Target                                      | Raw scale               | Normalized |
|---------------------|---------------------------|---------------------------------------------|-------------------------|:----------:|
| `adme_fang_hclint`  | `adme-fang-hclint-reg-v1` | log human liver microsomal clearance        | log10(µL/min/mg)        | ✅ |
| `adme_fang_rclint`  | `adme-fang-rclint-reg-v1` | log rat liver microsomal clearance          | log10(µL/min/mg)        | ✅ |
| `adme_fang_perm`    | `adme-fang-perm-reg-v1`   | log MDR1-MDCK efflux ratio                  | log10(ER)               |    |
| `adme_fang_hppb`    | `adme-fang-hppb-reg-v1`   | log human plasma protein binding            | log10(% unbound)        |    |
| `adme_fang_rppb`    | `adme-fang-rppb-reg-v1`   | log rat plasma protein binding              | log10(% unbound)        |    |
| `adme_fang_solu`    | `adme-fang-SOLU-reg-v1`   | log aqueous solubility                      | log10(µg/mL)            | ✅ |

`hclint`, `rclint`, and `solu` labels are z-score normalized at prep time
(their native scales differ by orders of magnitude from the other three).
Predictions can be denormalized from cached per-endpoint stats at reporting
time.

Exact per-endpoint train/test counts are defined by the frozen Polaris split
files (retrieve with `polaris login && polaris.load_benchmark(...)`); the
split is identical across runs and seeds.

## Evaluation protocol

- **Split**: canonical Polaris train / test split per endpoint (fixed, shipped
  with the benchmark). The train partition is further split at a fixed seed
  into train / val for early stopping.
- **Unit of evaluation**: one fitted model per (endpoint, seed). Results are
  reported as the mean over ≥ 3 seeds per endpoint.
- **Headline number (paper)**: **Pearson r on test** — Polaris' primary
  benchmark metric.

## Prediction head

A fresh per-endpoint MLP appended to the pretrained molecule encoder:

```
MLP( z_mol )  →  scalar regression target
  hidden_dims = 64, depth = 2, ReLU, dropout = 0.5, LayerNorm
```

Backbone is **frozen by default** (linear-probe-like). Optional unfreeze
schedules are available but not used for the headline numbers.

## Training protocol

Two regimes side-by-side:

| Setting           | Scratch              | Finetune (frozen backbone) |
|-------------------|----------------------|----------------------------|
| Optimizer         | Adam                 | Adam                       |
| Learning rate     | 4e-5                 | 1e-4                       |
| Scheduler         | WarmUpLinearLR       | CosineAnnealingLR          |
| Warmup / η_min    | 10 epochs            | 1e-6                       |
| Max epochs        | 100                  | 100                        |
| Loss              | MAE                  | MAE                        |

## Logged metrics (per endpoint)

- **Pearson r** (primary, matches Polaris),
- Spearman ρ,
- R²,
- MAE,
- MSE.

The `explained_var` metric in Polaris' definition is omitted (no direct
graphium equivalent); all other Polaris metrics map one-to-one.

## Reference

- Fang et al., _Prospective Validation of Machine Learning Algorithms for
  Absorption, Distribution, Metabolism, and Excretion Prediction._
  J. Chem. Inf. Model. 2023.
- [Polaris Benchmark Hub — biogen/adme-fang-v1](https://polarishub.io/benchmarks/biogen).
