<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# tests

## Purpose
Pytest suite covering featurization, datamodules, layers, architectures, encoders, losses, metrics, the predictor, the Lightning training loop, and the optional IPU code paths.

## Key Files

| File | Description |
|------|-------------|
| `conftest.py` | Shared pytest fixtures (fake batches, dummy configs, tmp cache dirs) |
| `dummy-pretrained-model.ckpt` | Small checkpoint used by `test_finetuning.py` |
| `fake_and_missing_multilevel_data.parquet` | Multi-level fixture with missing labels |
| `converted_fake_multilevel_data.parquet` | Reference output for `sdf2csv` / data conversion tests |
| `config_test_ipu_dataloader.yaml` | IPU dataloader test config |
| `config_test_ipu_dataloader_multitask.yaml` | IPU multitask dataloader test config |
| `test_architectures.py` | `FullGraphNetwork` / `FullGraphMultiTaskNetwork` forward + loading |
| `test_mtl_architecture.py` | End-to-end multi-task wiring |
| `test_pairformer.py` | Pairformer layer correctness + shape checks |
| `test_attention.py` | Attention primitives used by Pairformer/GPS++ |
| `test_base_layers.py` | `FCLayer`, `MLP`, normalization |
| `test_residual_connections.py` | Residual variants |
| `test_ensemble_layers.py` | Ensemble MLP |
| `test_pyg_layers.py` | GCN/GIN/MPNN+/PNA/GatedGCN/GPS++ smoke tests |
| `test_positional_encoders.py` | Encoder network outputs |
| `test_positional_encodings.py` | Raw PE computation correctness |
| `test_pe_spectral.py`, `test_pe_rw.py`, `test_pe_nodepair.py` | Individual PE suites |
| `test_pos_transfer_funcs.py` | node ↔ edge ↔ nodepair PE transfer |
| `test_featurizer.py` | SMILES → graph conversion |
| `test_datamodule.py`, `test_multitask_datamodule.py`, `test_dataset.py`, `test_collate.py` | Data pipeline |
| `test_data_utils.py` | Cache path / hashing / split helpers |
| `test_loaders.py` | `graphium/config/_loader.py` end-to-end |
| `test_training.py` | Short end-to-end trainer.fit |
| `test_finetuning.py` | Frozen and unfrozen finetuning paths |
| `test_predictor.py` | `PredictorModule` wiring |
| `test_losses.py`, `test_metrics.py` | NaN masking, multi-task aggregation |
| `test_mup.py`, `test_packing.py`, `test_utils.py` | Misc utilities |
| `test_ipu_*.py` | IPU-only (marked `@pytest.mark.ipu`, skipped without Graphcore SDK) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `data/` | Small fixture molecules/CSVs used across tests |

## For AI Agents

### Working In This Directory
- **Default test run**: `pytest -m 'not ipu'` from the `graphium/` project root. IPU tests require `poptorch` and are skipped otherwise.
- New modules in `graphium/` must ship with a matching test file here. Mirror the naming convention (`graphium/foo/bar.py` → `tests/test_bar.py`).
- Fixtures live in `conftest.py` — prefer extending it over duplicating fixture code across tests.
- `dummy-pretrained-model.ckpt` is a compact checkpoint used for finetuning tests. If you regenerate it, keep it small (<1 MB) so the repo stays clean.

### Testing Requirements
- Tests must pass without network access except for explicitly marked slow tests (ADMET downloads).
- When adding IPU tests, mark them with `@pytest.mark.ipu`.
- Single-file runs: `pytest tests/test_<name>.py -x`. Single-test: `pytest tests/test_<name>.py::<test_fn> -x`.

### Common Patterns
- Tests that need a full config use the dummy YAMLs under `graphium/config/` (e.g., `fake_multilevel_multitask_pyg.yaml`).
- Smoke tests should use a small-batch, few-epoch config; there is no dedicated GPU test marker — tests must run on CPU by default.

## Dependencies

### Internal
- All of `graphium/` — this is the primary exerciser of the package.
- `graphium/config/*.yaml` — dummy configs consumed as fixtures.

### External
- `pytest`, `torch`, `torch_geometric`, `pytorch_lightning`, `datamol`, `rdkit`

<!-- MANUAL: -->
