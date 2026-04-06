<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/cli

## Purpose
User-facing command-line entry points. `graphium-train` (installed via `pyproject.toml`) resolves to `main.py::main_cli`, which in turn dispatches to the training/finetuning/testing orchestration and a few auxiliary data subcommands.

## Key Files

| File | Description |
|------|-------------|
| `main.py` | Top-level Typer app; registers `train`, `data`, `finetune`, and `fingerprints` subcommands |
| `__main__.py` | Enables `python -m graphium.cli` |
| `train_finetune_test.py` | Hydra entry `cli()` → `run_training_finetuning_testing()` — loads accelerator, datamodule, architecture, predictor, runs `trainer.fit` and `trainer.test`, appends results to `results/experiment_results.csv` |
| `finetune_utils.py` | Helpers used by the finetuning CLI path (checkpoint resolution, task-head overrides) |
| `fingerprints.py` | `graphium-train fingerprints ...` — extract pre-trained embeddings into a flat file |
| `data.py` | `graphium-train data ...` — featurization/caching helpers |
| `parameters.py` | Shared CLI option/argument types |
| `__init__.py` | Package init |

## For AI Agents

### Working In This Directory
- `train_finetune_test.py::run_training_finetuning_testing` is the canonical orchestration function — read it end-to-end before changing training flow.
- The `cli()` function is decorated with `@hydra.main(config_path="../../expts/hydra-configs", config_name="main")`. Changing the config path requires updating scripts under `scripts/`.
- When adding a new subcommand, register it on the top-level Typer app in `main.py` and keep the CLI surface documented under `docs/cli/`.

### Testing Requirements
- End-to-end smoke via `bash scripts/00_debug.sh gpspp toymix 0` (3 epochs, 10 batches). This is the fastest integration path.
- Unit tests: `tests/test_loaders.py`, `tests/test_training.py`, `tests/test_finetuning.py`.

### Common Patterns
- Accelerator selection lives in `graphium/config/_loader.py::load_accelerator`, not here — keep this file focused on CLI wiring.
- Results CSV appends happen at the very end of `run_training_finetuning_testing` — if you add new metrics, thread them through `trainer.test` return value, not by writing the CSV directly from another place.

## Dependencies

### Internal
- `graphium/config/` — config loading
- `graphium/trainer/predictor.py` — the Lightning module built per run
- `graphium/data/datamodule.py` — data loading
- `graphium/finetuning/` — finetuning callback and head

### External
- `typer`, `hydra-core`, `omegaconf`, `pytorch_lightning`

<!-- MANUAL: -->
