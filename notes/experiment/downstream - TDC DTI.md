# Downstream — TDC DTI

Drug–Target Interaction affinity regression on five canonical Therapeutics Data
Commons (TDC) subsets. Each (molecule, protein) pair is mapped to a scalar
binding affinity.

## Subsets and statistics

Pair counts from TDC's raw tables. "Unique drugs/targets" counts distinct
entities before any filtering.

| Subset            | Pairs    | Unique drugs | Unique targets | Label (raw)            |
|-------------------|---------:|-------------:|---------------:|------------------------|
| `DAVIS`           |   25,772 |           68 |            379 | Kd (nM)                |
| `KIBA`            |  117,657 |        2,068 |            229 | KIBA score (unitless)  |
| `BindingDB_Kd`    |   52,274 |       10,661 |          1,090 | Kd (nM)                |
| `BindingDB_Ki`    |  374,820 |      174,547 |          2,420 | Ki (nM)                |
| `BindingDB_IC50`  |  990,630 |      548,633 |          3,975 | IC50 (nM)              |

## Label definition

- **DAVIS / BindingDB_\***: `pY = 9 − log10(Y_nM)`. Rows with `Y ≤ 0` are
  dropped before the log transform. Labels are then per-subset **z-score
  normalized** at prep time; the normalization stats are saved alongside the
  data so test predictions can be denormalized for reporting.
- **KIBA**: raw KIBA score (not log-transformed — different scale), then
  z-score normalized.

## Splits

Two split regimes per subset, five random seeds each (0 – 4). Ratios are
`train / val / test = 0.7 / 0.1 / 0.2`.

- **random** — i.i.d. split over (drug, target) pairs.
- **cold_target** — target-disjoint split (proteins seen at train time are
  absent from val/test). Stresses generalization to unseen targets.

**Total per model**: 5 subsets × 2 regimes × 5 seeds = **50 runs**; results
are reported as seed-averaged metrics (mean ± std over 5 seeds) per
(subset, regime).

## Protein features

Precomputed **ESM-C** embeddings, 1,152-dimensional, one vector per target.
Embeddings are frozen — no protein encoder is trained.

## Prediction head

Linear-probe-style: the pretrained molecule encoder is **frozen** (backbone
never updated) and a fresh MLP is fitted on the concatenation
`[ z_mol (256-d) ‖ prot_emb (1,152-d) ]`:

```
MLP( z_mol ‖ prot_emb )  →  scalar affinity prediction
  hidden_dims = 256, depth = 2, ReLU
```

## Training protocol

| Setting                   | Value                              |
|---------------------------|------------------------------------|
| Optimizer                 | Adam                               |
| Learning rate             | 1e-4                               |
| Scheduler                 | CosineAnnealingLR                  |
| T_max / η_min             | 100 / 1e-6                         |
| Max epochs                | 100                                |
| Validation cadence        | every epoch                        |
| Loss                      | MAE (on z-scored labels)           |
| Backbone                  | frozen                             |

## Logged metrics

Per run: **MAE**, MSE, **Pearson r**, **Spearman ρ**.

- **Progress-bar metric**: MAE.
- **Headline metric (paper)**: **Pearson r on the test split** — matches the
  TDC DTI-DG leaderboard convention. MAE / Spearman ρ / MSE are reported
  alongside.

## Reference

- Huang et al. 2021, _Therapeutics Data Commons._ NeurIPS Datasets &
  Benchmarks.
- Davis et al. 2011 (DAVIS); Tang et al. 2014 (KIBA); Liu et al. 2007
  (BindingDB).
