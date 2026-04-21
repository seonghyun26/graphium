<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium (package)

## Purpose
The main Python package. Implements the full training stack: SMILES featurization, multi-task data module, GNN architectures (GCN/MPNN/GPS++/Pairformer/PairMixer), positional/structural encoders, Lightning predictor, Hydra config loaders, fine-tuning callbacks, and the `graphium-train` CLI.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package entry; re-exports submodules |
| `_version.py` | Version constant |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cli/` | `graphium-train` Typer CLI and subcommands (see `cli/AGENTS.md`) |
| `config/` | YAML config loaders and conversion helpers (see `config/AGENTS.md`) |
| `data/` | Datamodule, dataset, collate, sampler, normalization (see `data/AGENTS.md`) |
| `features/` | SMILES → PyG graph featurization and positional encodings (see `features/AGENTS.md`) |
| `finetuning/` | Finetuning architecture, Lightning callback, fingerprinting (see `finetuning/AGENTS.md`) |
| `hyper_param_search/` | Hyperparameter sweep result helpers (see `hyper_param_search/AGENTS.md`) |
| `ipu/` | Graphcore IPU dataloader, losses, metrics, wrappers (see `ipu/AGENTS.md`) |
| `nn/` | Model architectures, encoders, PyG layers, residual connections (see `nn/AGENTS.md`) |
| `trainer/` | Lightning `PredictorModule`, metrics, loss, summaries (see `trainer/AGENTS.md`) |
| `utils/` | General utilities: fs, hashing, mup, safe_run, spaces, tensor (see `utils/AGENTS.md`) |
| `expts/` | Experiment registration stub used by Hydra |

## For AI Agents

### Working In This Directory
- The entry point for a training run is `cli/train_finetune_test.py::cli` → `run_training_finetuning_testing()`, which orchestrates accelerator, datamodule, architecture, predictor, trainer.fit, trainer.test, results CSV append.
- Model stack is: `pre_nn (MLP) → pe_encoders → gnn (FeedForwardGraph) → graph_output_nn → task_heads`.
- For finetuning, the network is wrapped as `FullGraphFinetuningNetwork(PretrainedModel, FinetuningHead)` and layer freezing is controlled by the `GraphFinetuning` Lightning callback.
- Only touch `level_in_dim` wiring when switching pooling modes — keep head `hidden_dims`/`out_dim` unchanged (project memory).

### Testing Requirements
- Any change here should be covered by `../tests/test_*.py` equivalents (e.g., a new encoder → `tests/test_positional_encoders.py`).
- `pytest -m 'not ipu'` from the project root must pass before merging.

### Common Patterns
- Module config classes (Python dataclasses / OmegaConf dicts) are passed top-down from the Hydra config — modules should not reach into the global config.
- All registrable classes (layer types, encoders, losses, metrics, schedulers) must be added to the corresponding dictionary in `utils/spaces.py`.

## Dependencies

### External
- `torch`, `torch_geometric`, `pytorch_lightning`
- `hydra-core`, `omegaconf`
- `datamol`, `rdkit`, `fsspec`
- `wandb`, `mup`, `loguru`

<!-- MANUAL: -->
