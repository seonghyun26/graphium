<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# expts/hydra-configs

## Purpose
Root of the Hydra config tree used by `graphium-train`. Each subdirectory is a Hydra config group, and `main.yaml` composes them in a specific order so later groups can override earlier ones.

## Key Files

| File | Description |
|------|-------------|
| `main.yaml` | **Root config** — declares the `defaults` list that composes every other group |
| `README.md` | Upstream guide to composing configs (kept as user-facing documentation) |
| `__init__.py` | Package marker so `debug_yaml.py` can resolve relative imports |

## Subdirectories (Hydra groups)

| Directory | Purpose |
|-----------|---------|
| `accelerator/` | Device-level configs — CPU, GPU, IPU (see `accelerator/AGENTS.md`) |
| `architecture/` | Architecture definitions (encoders, GNN skeleton, featurization) — `toymix`, `largemix`, `pcqm4m` |
| `tasks/` | Task sets (datasets + task heads + loss/metric/datamodule) — see `tasks/AGENTS.md` |
| `training/` | Training loop config (optimizer, scheduler, trainer flags) + `training/model/` and `training/accelerator/` specializations (see `training/AGENTS.md`) |
| `model/` | Model configs — GCN, GIN, GINE, MPNN, Gated-GCN, GPS++, Pairformer, PairMixer (see `model/AGENTS.md`) |
| `finetuning/` | Finetuning overrides layered on top of a pretraining config (see `finetuning/AGENTS.md`) |
| `experiment/` | Experiment specializations (rarely used; one example: `toymix_mpnn.yaml`) |
| `hparam_search/` | Hydra multirun / sweep configs (currently only `optuna.yaml`) |

## For AI Agents

### Working In This Directory
- **Composition order** (later overrides earlier):
  ```
  accelerator → architecture → tasks → training → model → training/accelerator → training/model
  ```
  Understand this before editing any group. For example, a `batch_size` set in `training/largemix.yaml` can be overridden by `training/accelerator/largemix_gpu.yaml` which can in turn be overridden by `training/model/largemix_gpspp_800M.yaml`.
- **Every config file starts with `# @package _global_`** so its keys are merged at the top level rather than nested under the group name.
- **`constants.norm`** is referenced by `architecture/largemix.yaml` via `${constants.norm}` but not defined in base configs — the model config (e.g. `model/gpspp_800M.yaml`) or a CLI override must set it.
- **GPS++ dimension gotcha**: changing a GPS++ hidden dim requires touching `pre_nn.out_dim`, `pre_nn.hidden_dims`, `gnn.in_dim/out_dim/hidden_dims`, AND `gnn.layer_kwargs.mpnn_kwargs.in_dim/out_dim`. Audit the whole chain.
- **Prefer YAML edits over CLI `++` overrides in scripts** (project memory). Keep `scripts/*.sh` thin.
- **Head hidden_dims/out_dim**: do not change when switching pooling modes; only touch `level_in_dim` wiring (project memory).

### Testing Requirements
- Use `python ../debug_yaml.py <overrides>` to dry-run composition.
- `bash ../../scripts/00_debug.sh <model> <dataset> 0` runs a 3-epoch smoke training against real data.

### Common Patterns
- Groups pair up: for each `tasks/<name>.yaml` there are sibling files under `tasks/task_heads/<name>.yaml` and `tasks/loss_metrics_datamodule/<name>.yaml` with the same stem.
- `training/model/<dataset>_<model>.yaml` is the most-specific specialization — it overrides earlier groups with per-combination settings (batch size, LR, checkpoint directory, run name).
- New model YAMLs typically inherit from a base model YAML via Hydra `defaults: [<base>, _self_]`.

## Dependencies

### Internal
- Consumed by `../../graphium/cli/train_finetune_test.py::cli` via `@hydra.main`.
- `graphium/utils/spaces.py` maps string layer/encoder/metric names from this tree to Python classes.

### External
- `hydra-core`, `omegaconf`

<!-- MANUAL: -->
