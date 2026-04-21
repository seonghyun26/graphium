# Dataset — LargeMix

Graphium's large-scale multi-task pretraining mix. Combines four datasets
covering orthogonal molecular signals — two transcriptomic (LINCS L1000),
one bioassay (PCBA), one quantum-chemistry (PCQM4M). Used as the large-scale
pretraining target (multi-day training on multi-GPU).

Source: Valence Discovery, NeurIPS 2023 Datasets track. Hosted at
`https://storage.valencelabs.com/graphium/datasets/neurips_2023/Large-dataset/`.

## Tasks

| Task            | Molecules   | Labels / task | Task type                                   | Label semantics |
|-----------------|------------:|--------------:|---------------------------------------------|---|
| `l1000_vcap`    |      15,220 |         978   | per-gene 3-way classification (graph)       | LINCS L1000 VCAP cell-line transcriptional response, discretized into {down, neutral, up} |
| `l1000_mcf7`    |      11,622 |         978   | per-gene 3-way classification (graph)       | LINCS L1000 MCF7 cell-line transcriptional response, same 3-class encoding |
| `pcba_1328`     |   1,563,664 |       1,328   | multi-output binary classification (graph)  | 1,328 PubChem BioAssay activity labels per molecule (sparse, NaN-masked) |
| `pcqm4m_g25`    |   3,810,323 |          25   | multi-output regression (graph)             | 25 graph-level DFT-derived descriptors (extended PCQM4M targets) |
| `pcqm4m_n4`     |   3,810,323 |           4 per node | multi-output regression (node)      | 4 node-level quantum descriptors — same parquet as `pcqm4m_g25` |

**Total: ~5.4 M molecule rows**, though PCQM4M dominates by 2+ orders of
magnitude over the other three. `pcqm4m_g25` and `pcqm4m_n4` share the same
3.81 M molecules at different task levels (graph vs node).

## Splits

Fixed random splits provided alongside each file:

| Task          | Train       | Val     | Test    | Test_seen  |
|---------------|------------:|--------:|--------:|-----------:|
| `l1000_vcap`  |      13,316 |     578 |     578 |        748 |
| `l1000_mcf7`  |       9,713 |     422 |     422 |      1,065 |
| `pcba_1328`   |   1,386,580 |  60,286 |  60,286 |     56,512 |
| `pcqm4m`      |   3,512,805 | 148,759 | 148,759 |          — |

The L1000 and PCBA splits additionally ship a `test_seen` partition — held-out
molecules that share scaffolds with training (used for in-distribution vs
generalization comparisons).

## Label preprocessing

- **L1000** (vcap / mcf7): per-gene response thresholded at log-fold-change
  |z| ≥ 2 into {-1, 0, +1}, encoded as 3-class.
- **PCBA_1328**: binary {0, 1}, NaN for unmeasured assays.
- **PCQM4M (both levels)**: z-score normalized at datamodule load
  (val/test included).

## Loss & metrics

- **L1000**: hybrid cross-entropy (3-way classification), `alpha = 0.5`; AUROC
  and average precision per gene.
- **PCBA_1328**: BCE-with-logits, NaN-masked, mean-per-assay AUROC / AUPRC.
- **PCQM4M_g25 / n4**: MAE, Pearson r, R².

## Epoch sampling

Each task YAML sets `epoch_sampling_fraction: 1.0` — every row is seen each
epoch. The relative weighting across tasks comes from their row counts (PCQM4M
dominates) and loss scale, not from resampling.

## Reference

- Beaini et al. 2024, _Towards Foundational Models for Molecular Learning on
  Large-Scale Multi-Task Datasets._ ICLR.
- L1000: Subramanian et al. 2017 (LINCS). PCBA: Wang et al. 2012 (PubChem
  BioAssay). PCQM4M: OGB-LSC (Hu et al. 2021).
