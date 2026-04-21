<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# expts

## Purpose
Experiment artefacts: Hydra config tree (the primary way training runs are configured), a handful of one-off experiment scripts, and archived NeurIPS 2023 configs.

## Key Files

| File | Description |
|------|-------------|
| `main_run_multitask.py` | Legacy entry point for multitask runs (superseded by `graphium-train`) |
| `main_run_get_fingerprints.py` | Extract pre-trained fingerprints for downstream sklearn models |
| `main_run_predict.py` | Inference-only entry point |
| `main_run_test.py` | Test-only entry point |
| `run_validation_test.py` | Validation sweep helper |
| `dataset_benchmark.py` | Featurization / datamodule micro-benchmark |
| `debug_yaml.py` | Prints the resolved Hydra config for a given override set |
| `__init__.py` | Package init |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `hydra-configs/` | **Main** Hydra config tree consumed by `graphium-train` (see `hydra-configs/AGENTS.md`) |
| `neurips2023_configs/` | Archived NeurIPS 2023 non-Hydra YAML configs (legacy, preserved for reproducibility) |
| `configs/` | Miscellaneous non-Hydra YAML fragments |
| `data/` | Small experiment-specific data files |

## For AI Agents

### Working In This Directory
- **`hydra-configs/main.yaml` is the root config** resolved by `graphium-train`. All new configurations should plug into that tree via the existing groups (`accelerator`, `architecture`, `tasks`, `training`, `model`, `finetuning`, `experiment`, `hparam_search`).
- The legacy `main_run_*.py` scripts are **not** on the `graphium-train` code path. Avoid adding new features there; prefer Typer subcommands under `graphium/cli/`.
- `debug_yaml.py` is the fastest way to confirm a config override composes the way you expect.
- Files in `neurips2023_configs/` are frozen — only touch them to fix documentation bugs.

### Testing Requirements
- Config composition is indirectly exercised by `tests/test_loaders.py` and `tests/test_training.py`.
- Manual check: `python expts/debug_yaml.py model=gcn tasks=toymix training=toymix` should print the resolved config without errors.

## Dependencies

### Internal
- `graphium/cli/train_finetune_test.py` — the `@hydra.main` entry point that reads this tree
- `graphium/config/_loader.py` — the loader that consumes the resolved dict

### External
- `hydra-core`, `omegaconf`

<!-- MANUAL: -->
