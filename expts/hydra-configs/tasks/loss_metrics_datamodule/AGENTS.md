<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/tasks/loss_metrics_datamodule

## Purpose
Per-task-set loss functions, metrics, and datamodule args. Each file plugs `predictor.loss_fun`, `metrics.<task>`, and `datamodule.args.task_specific_args` into the Hydra composition. Loaded transitively by `../<task_set>.yaml`.

## Key Files

| File | Description |
|------|-------------|
| `toymix.yaml` | Losses/metrics/args for QM9 / Tox21 / ZINC12k |
| `largemix.yaml` | Losses/metrics/args for L1000 / PCBA_1328 / PCQM4M_* |
| `pcqm4m.yaml`, `pcqm4m_g25.yaml`, `pcqm4m_n4.yaml` | PCQM4M single-task |
| `pcba_1328.yaml` | PCBA-1328 |
| `l1000_mcf7.yaml`, `l1000_vcap.yaml` | L1000 single-cell-line |
| `admet.yaml` | ADMET-22 (per-task loss + metrics) |
| `debug.yaml` | Tiny debug set |

## For AI Agents

### Working In This Directory
- **Stem must match** `../<name>.yaml` and `../task_heads/<name>.yaml`.
- Task names here must be identical to the head task names in `../task_heads/<name>.yaml` — mismatches fail at loader time.
- `datamodule.args.task_specific_args.<task>.sample_size` accepts an int (row count) or float (fraction) for subsampling.
- Loss / metric names are resolved via `graphium/utils/spaces.py::LOSS_DICT` and `METRIC_DICT` — new entries must be registered there first.
- NaN-masking is automatic in the loss layer — do not attempt to filter NaNs in the datamodule.

## Dependencies

### Internal
- `graphium/trainer/losses.py`, `graphium/trainer/metrics.py`
- `graphium/data/datamodule.py::MultitaskFromSmilesDataModule`
- `graphium/utils/spaces.py` — loss/metric registry

<!-- MANUAL: -->
