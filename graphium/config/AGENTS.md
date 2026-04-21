<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/config

## Purpose
Runtime config loading: reads a Hydra/OmegaConf config tree, converts it, and instantiates accelerator, datamodule, architecture, metrics, predictor, and trainer from it. Also hosts a handful of legacy dummy YAML files used by unit tests.

## Key Files

| File | Description |
|------|-------------|
| `_load.py` | Top-level `load_*` functions used by the CLI orchestrator (yaml → dict → objects) |
| `_loader.py` | `load_accelerator`, `load_datamodule`, `load_architecture`, `load_predictor`, `load_trainer`, `load_metrics` — bridge the config to runtime objects |
| `config_convert.py` | Type coercion helpers (string → class via `spaces.py`, nested dict handling) |
| `zinc_default_multitask_pyg.yaml` | Legacy default for ZINC multitask (test fixture) |
| `fake_multilevel_multitask_pyg.yaml` | Fake fixture for multitask datamodule tests |
| `fake_and_missing_multilevel_multitask_pyg.yaml` | Fixture covering missing-level combinations |
| `dummy_finetuning_from_gnn.yaml` | Fixture for the `test_finetuning.py` suite |
| `dummy_finetuning_from_task_head.yaml` | Fixture for the `test_finetuning.py` suite |
| `__init__.py` | Re-exports loader helpers |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- `_loader.load_architecture` resolves class names (e.g., `FullGraphMultiTaskNetwork`) via `graphium/utils/spaces.py` — any new architecture must be registered there.
- `load_predictor` attaches metrics and loss functions from the config. Per-task metric configs live under `metrics.<task>`.
- Dummy YAML files in this directory are consumed by tests; do **not** edit them for production use.
- When adding a new top-level config section (sibling of `architecture`, `tasks`, `training`), also add a `load_*` function and wire it from the CLI orchestrator.

### Testing Requirements
- `tests/test_loaders.py` — loader sanity
- `tests/test_finetuning.py` — exercises the dummy finetuning YAMLs
- `tests/test_training.py` — end-to-end with a fixture config

### Common Patterns
- All loaders accept a plain Python dict (OmegaConf already resolved) to stay friendly to unit tests that skip Hydra.
- Config-to-object coercion goes through `spaces.py` registry lookups — avoid `eval` or `importlib` hacks.

## Dependencies

### Internal
- `graphium/utils/spaces.py` — class registry
- `graphium/nn/architectures/global_architectures.py` — target classes
- `graphium/trainer/predictor.py`, `graphium/trainer/metrics.py`, `graphium/trainer/losses.py`

### External
- `hydra-core`, `omegaconf`, `pytorch_lightning`

<!-- MANUAL: -->
