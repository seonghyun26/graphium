<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/training

## Purpose
Training-loop configuration: optimizer, scheduler, trainer flags (max epochs, early stopping, gradient clipping, checkpointing). Each top-level YAML targets one dataset / task family. The `training/accelerator/` and `training/model/` subgroups layer on device-specific and (dataset, model)-specific specializations **last** in the composition chain, so they can override anything earlier.

## Key Files (dataset-level)

| File | Description |
|------|-------------|
| `toymix.yaml` | ToyMix base training config |
| `largemix.yaml` | LargeMix base training config |
| `pcqm4m.yaml` | PCQM4M single-task training |
| `rxrx3.yaml`, `rxrx3_dti.yaml` | RxRx3 cell morphology training |
| `dti.yaml`, `dti_filtered.yaml`, `dti_10k.yaml`, `dti_10k_filtered.yaml` | DTI training configs |
| `largemix_dti.yaml`, `largemix_dti_filtered.yaml`, `largemix_rxrx3_dti.yaml` | Combined multi-modal training |
| `bbbc047.yaml`, `toymix_bbbc047*.yaml`, `toymix_rxrx3*.yaml`, `toymix_dti*.yaml` | Combined modality training configs |
| `toymix_dti_cosine.yaml`, `toymix_dti_cosine_restarts.yaml`, `toymix_dti_constant.yaml` | Schedule-variant sweeps for ToyMix+DTI |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `accelerator/` | Device-specific trainer flags (precision, workers, batch size) per dataset (see `accelerator/AGENTS.md`) |
| `model/` | `(dataset, model)`-specific overrides — the **last** group in composition, highest-priority (see `model/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Top-level files here set the dataset-family defaults. The `training/accelerator/<dataset>_<device>.yaml` specialization then adjusts precision and workers, and `training/model/<dataset>_<model>.yaml` sets per-combination run name, checkpoint dir, and usually batch size.
- **Composition order**: these files are loaded after `tasks/` but before `model/`, `training/accelerator/`, and `training/model/`. Anything you set here will be overridden by those later groups if they touch the same key — do not be surprised when a scheduler you wrote gets replaced.
- When adding a schedule variant (as with `toymix_dti_cosine*.yaml`), mirror the naming in `training/accelerator/` and `training/model/` so the triple resolves cleanly.
- The Hydra override is `training=<stem>`.

### Testing Requirements
- `python ../../debug_yaml.py tasks=<tasks> training=<training> model=<model> architecture=<arch>` should print the resolved config.
- `bash ../../../scripts/00_debug.sh <model> <dataset> 0` runs a short smoke training.

## Dependencies

### Internal
- `graphium/config/_loader.py::load_predictor`, `load_trainer`
- `graphium/utils/spaces.py::SCHEDULER_DICT` — scheduler registry

<!-- MANUAL: -->
