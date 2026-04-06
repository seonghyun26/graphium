<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/nn/pyg_layers

## Purpose
PyTorch Geometric implementations of individual GNN and graph-transformer layers. Each file defines one layer class that can be selected in YAML via `architecture.gnn.layer_type: "pyg:<name>"`.

## Key Files

| File | Description |
|------|-------------|
| `gcn_pyg.py` | `GCNConvPyg` — vanilla GCN layer |
| `gin_pyg.py` | `GINConvPyg`, `GINEConvPyg` — GIN and GINE layers |
| `pna_pyg.py` | `PNAMessagePassingPyg` — PNA layer; reference implementation for new layers |
| `gated_gcn_pyg.py` | `GatedGCNPyg` — gated GCN variant |
| `mpnn_pyg.py` | `MPNNPlusPyg` — MPNN+ layer with edge updates |
| `gps_pyg.py` | `GPSLayerPyg` — GraphGPS hybrid (MPNN + attention) used by the `gpspp_*` models |
| `pairformer_pyg.py` | Pairformer-style node+pair attention layer (AlphaFold2-inspired) |
| `pairmixer_pyg.py` | PairMixer layer — pair-update variant currently under active iteration |
| `dimenet_pyg.py` | `DimeNetPyg` — directional 3D message passing |
| `pooling_pyg.py` | Graph pooling strategies: sum, mean, max, virtual-node |
| `utils.py` | Shared helpers (edge aggregation, scatter wrappers) |
| `__init__.py` | Re-exports layer classes |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- **Use `pna_pyg.py` as the reference** for any new GNN layer: it shows the constructor signature (`in_dim`, `out_dim`, `activation`, `dropout`, `normalization`, aggregators, scalers, …), the PyG `MessagePassing` inheritance, and how to plug into `BaseGraphModule`.
- **Pairformer/PairMixer** maintain both node and pair (`edge-like`) representations; changes must keep both tensors consistent with the pre/post shapes expected by `FeedForwardPyg`.
- Register new layers in `graphium/utils/spaces.py::PYG_LAYERS_DICT` with the exact `"pyg:<name>"` key used in YAML configs.
- When adding a layer with a new kwargs namespace (e.g., `layer_kwargs.my_kwargs`), document the required keys in the model YAML and wire them via `FeedForwardPyg`.

### Testing Requirements
- `tests/test_pyg_layers.py` covers most layers.
- Pairformer has dedicated tests in `tests/test_pairformer.py` and `tests/test_attention.py`.
- New layers must have at least one forward-pass smoke test with a small batch.

### Common Patterns
- Pooling layers operate on the `graph_output_nn` stage, not inside the GNN loop.
- Edge features are carried as `edge_attr` and updated in-place where the layer supports it (MPNN+, GPS++, Pairformer).
- Virtual-node pooling interacts with `pre_nn.out_dim` and `gnn.in_dim` — keep dimensions aligned.

## Dependencies

### Internal
- `../base_graph_layer.py`, `../base_layers.py`
- `../architectures/pyg_architectures.py::FeedForwardPyg` — the consumer that stacks these layers

### External
- `torch`, `torch_geometric`, `torch_scatter`, `torch_sparse`

<!-- MANUAL: -->
