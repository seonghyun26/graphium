<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# expts/hydra-configs/architecture

## Purpose
Dataset-scoped architecture skeletons. Each YAML defines the `pre_nn`, `pe_encoders`, `gnn`, `graph_output_nn` and `task_heads` **structural** shape for a dataset family, independent of the concrete layer type (GCN, GIN, GPS++, …). The layer type is set later by the `model/` group.

## Key Files

| File | Description |
|------|-------------|
| `toymix.yaml` | Architecture for the ToyMix benchmark (QM9 + Tox21 + ZINC12k) |
| `largemix.yaml` | Architecture for LargeMix (L1000, PCBA, PCQM4M). References `${constants.norm}` which must be set by the model config |
| `pcqm4m.yaml` | Architecture specialized for PCQM4M-only training |

## For AI Agents

### Working In This Directory
- These files declare placeholders (e.g., `layer_type: ${model.layer_type}`) and depth/width shapes that are filled in by the `model/` config.
- **`largemix.yaml` references `${constants.norm}`** — any model paired with it must set `constants.norm` (e.g., `gpspp_800M.yaml` sets `layer_norm`). Forgetting this leads to interpolation errors.
- Featurization settings (datamodule args under `datamodule.featurization`) live here since they must match the architecture's expected inputs. Do not duplicate them into `model/` configs.
- When adding a new dataset family, create `<name>.yaml` here together with a matching `tasks/<name>.yaml`, `tasks/task_heads/<name>.yaml`, and `tasks/loss_metrics_datamodule/<name>.yaml`.

### Testing Requirements
- `python ../../debug_yaml.py architecture=<name> model=<model> tasks=<tasks> training=<training>` should print a fully-resolved config.
- `tests/test_loaders.py::test_load_architecture` covers the loader path.

## Dependencies

### Internal
- `graphium/nn/architectures/global_architectures.py` — target class
- `graphium/features/featurizer.py` — featurization consumed via `datamodule.featurization`
- Composes with `../model/` for concrete layer selection

<!-- MANUAL: -->
