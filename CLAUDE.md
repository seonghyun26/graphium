# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fork of [Graphium](https://github.com/datamol-io/graphium) extended with multi-modal pre-training (molecular properties + RxRx3 cell morphology) and ablation studies on pre-training dataset type and size. Trains GNN models (GCN, MPNN, GPS++, Pairformer) on molecular graph data, then fine-tunes on 22 ADMET benchmark tasks.

## Environment and setup

```bash
mamba activate graphium                # conda env at ~/.conda/envs/graphium
pip install --no-deps -e .             # editable install (already done)
```

The Python executable is at `/home/shpark/.conda/envs/graphium/bin/python`. Use this when running Python scripts directly (the base env does not have torch).

## Common commands

```bash
# Training
graphium-train model=gpspp_800M accelerator=gpu tasks=largemix training=largemix architecture=largemix

# Run experiment scripts (from graphium/ directory)
bash scripts/00_pretrain.sh gpspp largemix 0        # pretrain
bash scripts/00_finetune_admet.sh gpspp ./ckpt 0    # finetune (frozen backbone by default)
bash scripts/00_scratch_admet.sh gpspp 0             # no-pretrain baseline
bash scripts/00_debug.sh gpspp toymix 0              # quick sanity check (3 epochs)

# Tests
pytest -m 'not ipu'                    # all tests except IPU
pytest tests/test_architectures.py     # single test file
pytest tests/test_architectures.py::test_name -x  # single test, stop on failure

# Formatting
black .                                # format (line-length=110, py310/311)
black --check .                        # check only
```

## Architecture

### Execution flow

`graphium-train` → Hydra loads config → `cli()` in `graphium/cli/train_finetune_test.py` → `run_training_finetuning_testing()` which orchestrates:

1. `load_accelerator()` → device selection
2. `load_datamodule()` → SMILES-to-graph featurization and batching
3. `load_architecture()` → model instantiation
4. `load_predictor()` → wraps model in Lightning module
5. `trainer.fit()` → training loop (with optional `GraphFinetuning` callback)
6. `trainer.test()` → evaluation
7. Results saved to `outputs/<date>/<time>/test_results.yaml` + `results/experiment_results.csv`

### Hydra config composition

Config at `expts/hydra-configs/main.yaml` composes from these defaults groups (later overrides earlier):

```
accelerator → architecture → tasks → training → model → training/accelerator → training/model
```

Example: `model=gpspp_800M training=largemix tasks=largemix` loads:
- `model/gpspp_800M.yaml` (inherits `gpspp`, sets DIM=1504, depth=12)
- `training/model/largemix_gpspp_800M.yaml` (data paths, checkpoint dir, batch_size=192)
- `training/accelerator/largemix_gpu.yaml` (precision, workers)

The `training/model` config loads **last** and can override anything from earlier configs (including batch_size from `training/accelerator`).

### Model stack

`FullGraphMultiTaskNetwork` in `graphium/nn/architectures/global_architectures.py`:

```
pre_nn (MLP)           → embed raw atom features to hidden dim
pe_encoders            → Laplacian eigenvec/val + random walk positional encodings
gnn (FeedForwardGraph) → GCN/MPNN/GPS++/Pairformer layers with residual connections
graph_output_nn        → pooling (sum/mean) + MLP per task level (node/graph)
task_heads             → per-task output MLPs
```

For finetuning, `FullGraphFinetuningNetwork` wraps a `PretrainedModel` + optional `FinetuningHead`. The `GraphFinetuning` Lightning callback controls layer freezing/unfreezing.

### Data pipeline

`MultitaskFromSmilesDataModule` (`graphium/data/datamodule.py`) handles:
- SMILES → PyG graph conversion via `graphium/features/featurizer.py`
- Caching featurized graphs to disk (`datacache/` directories)
- Multi-task batching with `graphium_collate_fn`
- Per-task `sample_size` parameter for dataset subsampling (int or float fraction)

### Results system

Every `graphium-train` run appends to `results/experiment_results.csv` with metadata (model, task, seed, finetuning config) + all test metrics. The notebook `notebooks/07_results_dashboard.ipynb` reads this CSV for visualization.

## Key conventions

- **Finetuning is frozen by default**: `UNFREEZE_DEPTH=0`, `EPOCH_UNFREEZE_ALL=none`. Only the task head trains. Override explicitly to unfreeze backbone layers.
- **Scripts source `common.sh`**: All `scripts/0*.sh` files source `scripts/common.sh` for shared config (W&B settings, ADMET task list, dimension helpers, `RESULTS_DIR`).
- **Config dimension overrides for GPS++**: Must set `pre_nn.out_dim`, `pre_nn.hidden_dims`, `gnn.in_dim/out_dim/hidden_dims`, AND `gnn.layer_kwargs.mpnn_kwargs.in_dim/out_dim` — these are not auto-linked.
- **`constants.norm`**: Referenced by `architecture/largemix.yaml` via `${constants.norm}` but not defined in base configs. Must be set by the model config (e.g., `gpspp_800M.yaml` sets `layer_norm`) or via CLI.
- **Data paths**: ToyMix task YAML uses `${constants.data_dir}` (must be set correctly). LargeMix task YAML hardcodes paths under `./data/graphium/neurips2023/large-dataset/`.

## Datasets

| Config key | Path | Notes |
|---|---|---|
| `tasks=toymix` | `./data/graphium/neurips2023/small-dataset/` | QM9, Tox21, ZINC12k |
| `tasks=largemix` | `./data/graphium/neurips2023/large-dataset/` | L1000, PCBA, PCQM4M |
| `tasks=rxrx3` | `./data/rxrx3/rxrx3_smiles_embeddings.csv` | 384-dim cell morphology |
| `tasks=largemix_rxrx3` | Both above combined | Multi-modal |
| `tasks=admet` | Auto-downloaded via TDC | 22 ADMET benchmark tasks |

## GPUs

8x NVIDIA RTX PRO 6000 Blackwell Max-Q (96GB each). Use `CUDA_VISIBLE_DEVICES=N` to select.
