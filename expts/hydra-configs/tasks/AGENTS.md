<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# expts/hydra-configs/tasks

## Purpose
Task-set definitions. Each `tasks/<name>.yaml` declares which tasks to train on, and loads matching `tasks/task_heads/<name>.yaml` (task-head network shapes) and `tasks/loss_metrics_datamodule/<name>.yaml` (loss functions, metrics, datamodule args). Tasks are decoupled from architecture so the same backbone can run on different task sets.

## Key Files (top-level task configs)

| File | Description |
|------|-------------|
| `toymix.yaml` | ToyMix benchmark — QM9 + Tox21 + ZINC12k |
| `largemix.yaml` | LargeMix — L1000 (VCAP + MCF7), PCBA_1328, PCQM4M_G25/N4 |
| `pcqm4m.yaml`, `pcqm4m_g25.yaml`, `pcqm4m_n4.yaml` | PCQM4M single-task variants |
| `pcba_1328.yaml` | PCBA-1328 only |
| `l1000_mcf7.yaml`, `l1000_vcap.yaml` | L1000 single-cell-line variants |
| `admet.yaml` | ADMET-22 benchmark (used with `+finetuning=admet_*`) |
| `bbbc047.yaml` | BBBC047 image-based cell morphology task |
| `rxrx3.yaml`, `rxrx3_dti.yaml` | RxRx3 cell morphology tasks (384-dim embeddings), optionally combined with DTI |
| `dti.yaml`, `dti_filtered.yaml`, `dti_10k.yaml`, `dti_10k_filtered.yaml` | Drug-target interaction task sets |
| `debug.yaml` | Tiny debug task set used by `00_debug.sh` |
| `toymix_*.yaml` | Combined ToyMix + RxRx3 / BBBC047 / DTI multi-modal variants |
| `largemix_*.yaml` | Combined LargeMix + RxRx3 / DTI multi-modal variants |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `task_heads/` | Per-task-set `architecture.task_heads.<task>` definitions (output MLP shapes) |
| `loss_metrics_datamodule/` | Per-task-set loss functions, metrics, and datamodule args |

## For AI Agents

### Working In This Directory
- **Naming convention is load-bearing**: `tasks/foo.yaml`, `tasks/task_heads/foo.yaml`, and `tasks/loss_metrics_datamodule/foo.yaml` must share the stem `foo`, and the top-level `tasks/foo.yaml` loads the other two via `defaults:`.
- When adding a new task set:
  1. Create `<name>.yaml` with the dataset declaration.
  2. Create `task_heads/<name>.yaml` with the per-task head MLP shapes.
  3. Create `loss_metrics_datamodule/<name>.yaml` with the loss/metric/datamodule args.
  4. Usually also create a matching `training/<name>.yaml` and `training/accelerator/<name>_<device>.yaml`.
- **Per-task `sample_size`** (int row count or float fraction) can be set under `datamodule.args.task_specific_args.<task>.sample_size`.
- The Hydra override is `tasks=<stem>`.

### Testing Requirements
- `python ../../debug_yaml.py tasks=<stem> model=gcn training=<stem> architecture=<arch>` must resolve.
- `tests/test_loaders.py`, `tests/test_multitask_datamodule.py` exercise the loader + datamodule path.

### Common Patterns
- Multi-modal task sets (e.g., `toymix_rxrx3_dti.yaml`) combine 2–3 dataset families via nested datamodule args; follow the existing `toymix_rxrx3.yaml` as a template.
- ADMET tasks are downloaded on demand via `pytdc`; the task YAML only lists task names and head shapes.

## Dependencies

### Internal
- `graphium/data/datamodule.py::MultitaskFromSmilesDataModule` — consumes `datamodule.args`
- `graphium/trainer/losses.py`, `graphium/trainer/metrics.py` — consume `loss_metrics_datamodule/`
- `graphium/nn/architectures/global_architectures.py::TaskHeads` — consumes `task_heads/`

<!-- MANUAL: -->
