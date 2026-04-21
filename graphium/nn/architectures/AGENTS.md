<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/nn/architectures

## Purpose
Top-level model assemblies that stitch together MLPs, positional encoders, GNN layer stacks, pooling, and task heads. The main class used in training is `FullGraphMultiTaskNetwork`.

## Key Files

| File | Description |
|------|-------------|
| `global_architectures.py` | `FullGraphNetwork`, `FullGraphMultiTaskNetwork`, `FullGraphFinetuningNetwork`, `FullGraphSiameseNetwork`, `TaskHeads` — end-to-end forward paths |
| `encoder_manager.py` | `EncoderManager` — instantiates positional encoders from config and pools their outputs at the correct graph level (node/edge/graph) |
| `pyg_architectures.py` | `FeedForwardPyg` — stacks PyG layers with residual connections and is used inside the global architectures as the `gnn` block |
| `__init__.py` | Re-exports architecture classes |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- `FullGraphMultiTaskNetwork.forward` order: `pre_nn` → positional encoders (via `EncoderManager`) → `gnn` (`FeedForwardPyg`) → `graph_output_nn` (pool + post-MLP per task level) → `task_heads`.
- `FullGraphFinetuningNetwork` wraps a `PretrainedModel` plus an optional `FinetuningHead`; freezing/unfreezing is controlled externally by the `GraphFinetuning` Lightning callback in `graphium/finetuning/`.
- **GPS++ dimension gotcha**: changing hidden dim requires overriding `pre_nn.out_dim`, `pre_nn.hidden_dims`, `gnn.in_dim/out_dim/hidden_dims`, AND `gnn.layer_kwargs.mpnn_kwargs.in_dim/out_dim`. These are not auto-linked — audit any config touching GPS++ dims.
- Keep head `hidden_dims`/`out_dim` unchanged when switching pooling modes; only adjust `level_in_dim` wiring (project memory).

### Testing Requirements
- `tests/test_architectures.py`, `tests/test_mtl_architecture.py`, `tests/test_pairformer.py`.
- When adding a new architecture type, register it under `model_type` in config and add a smoke test mirroring `test_mtl_architecture.py`.

### Common Patterns
- Architectures take an `EncoderManager` instance and expose its output keys back to the caller for tracking.
- Task heads are instantiated from `TaskHeads` using the per-task dict in the Hydra `architecture.task_heads` section.

## Dependencies

### Internal
- `../base_layers.py`, `../base_graph_layer.py`, `../encoders/`, `../pyg_layers/`, `../residual_connections.py`
- `graphium/features/positional_encoding.py` — raw positional features consumed by `EncoderManager`
- `graphium/trainer/predictor.py` — the Lightning wrapper around these networks

### External
- `torch`, `torch_geometric`

<!-- MANUAL: -->
