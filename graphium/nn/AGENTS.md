<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/nn

## Purpose
Neural network building blocks: base layer classes, residual connection variants, ensemble layers, and utility functions. Deeper model components (full architectures, positional encoders, PyG layers) live in subdirectories.

## Key Files

| File | Description |
|------|-------------|
| `base_graph_layer.py` | `BaseGraphStructure` and `BaseGraphModule` — inherited by all GNN layers |
| `base_layers.py` | Building blocks: `FCLayer`, `MLP`, normalization wrappers, activation selection |
| `residual_connections.py` | `ResidualConnectionBase` and variants (simple, concat, dense, weighted, random) |
| `ensemble_layers.py` | Ensemble-MLP utilities used by mup/hparam sweeps |
| `utils.py` | Mixed-precision helpers and small nn helpers |
| `__init__.py` | Re-exports layer classes |
| `README.md` | Upstream folder guide (kept for reference) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `architectures/` | Full graph networks (`FullGraphNetwork`, `FullGraphMultiTaskNetwork`) and PyG feed-forward stacks (see `architectures/AGENTS.md`) |
| `encoders/` | Positional/structural encoder networks — LapPE, RWSE, Gaussian kernel, MLP, SignNet, Bessel (see `encoders/AGENTS.md`) |
| `pyg_layers/` | PyG-based GNN layer implementations: GCN, GIN(E), MPNN+, PNA, Gated-GCN, GPS++, Pairformer, PairMixer, DimeNet, pooling (see `pyg_layers/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- New GNN layers should inherit from `BaseGraphModule` in `base_graph_layer.py` and follow the `PNAMessagePassingPyg` example.
- `FCLayer` in `base_layers.py` is the canonical fully-connected block (dropout + norm + activation); reuse it instead of `nn.Linear` in new heads to stay consistent with the rest of the stack.
- If you add a new residual mode, register it in `utils/spaces.py::RESIDUAL_DICT`.

### Testing Requirements
- `tests/test_base_layers.py`, `tests/test_residual_connections.py`, `tests/test_ensemble_layers.py`, `tests/test_pyg_layers.py`, `tests/test_architectures.py`.

### Common Patterns
- Most modules accept a common kwargs bundle: `in_dim`, `out_dim`, `activation`, `dropout`, `normalization`, `residual_type`. Preserve this shape for drop-in swaps from YAML.
- Mixed precision is handled via helpers in `utils.py` — do not call `.half()` or `torch.autocast` directly.

## Dependencies

### Internal
- `graphium/utils/spaces.py` — registry of layer/encoder/residual classes
- `graphium/utils/mup.py` — mup scaling used by architecture depth/width sweeps

### External
- `torch`, `torch_geometric`

<!-- MANUAL: -->
