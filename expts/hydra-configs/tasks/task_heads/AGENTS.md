<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/tasks/task_heads

## Purpose
Per-task-set head network definitions. Each file declares `architecture.task_heads.<task>` — the MLP shape, output dimension, pooling mode, and dropout for each task in the set. Loaded transitively by `../<task_set>.yaml`.

## Key Files

| File | Description |
|------|-------------|
| `toymix.yaml` | Heads for QM9 / Tox21 / ZINC12k |
| `largemix.yaml` | Heads for L1000 / PCBA_1328 / PCQM4M_G25 / PCQM4M_N4 |
| `pcqm4m.yaml`, `pcqm4m_g25.yaml`, `pcqm4m_n4.yaml` | PCQM4M single-task heads |
| `pcba_1328.yaml` | PCBA-1328 binary-classification heads |
| `l1000_mcf7.yaml`, `l1000_vcap.yaml` | L1000 per-cell-line heads |
| `admet.yaml` | ADMET-22 classification / regression heads |
| `debug.yaml` | Tiny debug head set |

## For AI Agents

### Working In This Directory
- **Stem must match** the corresponding `../<name>.yaml` and `../loss_metrics_datamodule/<name>.yaml`.
- Head `hidden_dims` and `out_dim` should stay stable across pooling-mode changes — only touch `level_in_dim` wiring (project memory).
- Head task-level (`node`, `edge`, `graph`, `nodepair`) must match the featurization in the architecture YAML and the labels in the datamodule config.

## Dependencies

### Internal
- `graphium/nn/architectures/global_architectures.py::TaskHeads`
- `../loss_metrics_datamodule/<name>.yaml` — must declare losses and metrics for the same task names

<!-- MANUAL: -->
