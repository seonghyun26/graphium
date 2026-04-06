<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/data

## Purpose
Dataset classes, Lightning data module, collate function, normalization, and sampling logic for multi-task molecular graphs. Converts SMILES lists into PyG `Batch` objects with positional encodings, caches them to disk, and yields train/val/test dataloaders.

## Key Files

| File | Description |
|------|-------------|
| `datamodule.py` | `MultitaskFromSmilesDataModule` — the main Lightning `LightningDataModule` used by `graphium-train` |
| `dataset.py` | `MultitaskDataset` and `SingleTaskDataset` — torch Dataset implementations over cached PyG graphs |
| `collate.py` | `graphium_collate_fn` — PyG batching that handles multi-level labels (node/edge/graph/nodepair) and missing tasks |
| `sampler.py` | Per-task samplers for balanced multi-task batches and dataset subsampling |
| `normalization.py` | Label normalization (z-score, min-max) and inverse transforms applied during evaluation |
| `smiles_transform.py` | RDKit canonicalization and augmentation helpers |
| `multilevel_utils.py` | Utilities for handling node / edge / graph / nodepair label levels in one dataset |
| `utils.py` | Cache path helpers, hash computation, split loading |
| `sdf2csv.py` | CLI helper to convert SDF files into the expected CSV format |
| `__init__.py` | Re-exports dataset + datamodule classes |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `L1000/` | L1000 cell-profile dataset loaders / preprocessing |
| `QM9/` | QM9 dataset loader utilities |
| `micro_ZINC/` | Small ZINC subset loaders used by tests |
| `multitask/` | Legacy multitask CSV combiners |
| `make_data_splits/` | Scripts that produce train/val/test splits for new datasets |
| `single_atom_dataset/` | Tiny single-atom dataset used for debugging |

## For AI Agents

### Working In This Directory
- **Featurization cache**: graphs are cached to `../../datacache/` under a directory named by a hash of featurization kwargs. Bump `utils.py::featurization_hash` only when the featurization semantics change — otherwise existing caches will be invalidated unnecessarily.
- **Per-task `sample_size`**: `MultitaskFromSmilesDataModule` supports int (row count) or float (fraction) subsampling per task via the `task_specific_args` YAML section.
- **Multi-level labels**: a single task can have node/edge/graph/nodepair labels simultaneously. The collate function and normalization functions must agree on level keys.
- New datasets should extend `MultitaskFromSmilesDataModule` or write a sibling datamodule — do not fork `datamodule.py` wholesale.

### Testing Requirements
- `tests/test_datamodule.py`, `tests/test_multitask_datamodule.py`, `tests/test_dataset.py`, `tests/test_collate.py`, `tests/test_data_utils.py`.
- ADMET paths are tested via `tests/test_loaders.py`.

### Common Patterns
- Task YAML configs (`expts/hydra-configs/tasks/`) declare `task_specific_args` consumed here.
- Label tensors carry NaN for missing targets; loss/metric code in `graphium/trainer/` is responsible for masking.

## Dependencies

### Internal
- `graphium/features/featurizer.py` — SMILES → PyG graph
- `graphium/features/positional_encoding.py` — raw PE vectors attached to each graph

### External
- `torch`, `torch_geometric`, `pytorch_lightning`
- `datamol`, `rdkit`, `fsspec`, `loguru`, `pytdc` (for ADMET downloads)

<!-- MANUAL: -->
