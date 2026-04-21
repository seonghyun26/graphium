<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# expts/hydra-configs/model

## Purpose
Concrete model definitions: layer type, depth, hidden dims, normalization, and any layer-specific kwargs. A model YAML plugs into an `architecture/` skeleton and is further specialized by `training/model/<dataset>_<model>.yaml`.

## Key Files

| File | Description |
|------|-------------|
| `gcn.yaml` | Vanilla GCN — `layer_type: "pyg:gcn"` |
| `gin.yaml` | GIN — `layer_type: "pyg:gin"` |
| `gine.yaml` | GINE with edge features |
| `gated_gcn.yaml` | Gated GCN |
| `mpnn.yaml` | MPNN+ |
| `gpspp.yaml` | GraphGPS++ base config (128 hidden) |
| `gpspp_768.yaml` | GPS++ with 768 hidden dim |
| `gpspp_800M.yaml` | GPS++ scaled to ~800M params (DIM=1504, depth=12). Sets `constants.norm: layer_norm` |
| `gpspp_800M_moe.yaml` | GPS++ 800M with mixture-of-experts FFN |
| `gpspp_deep.yaml` | Deeper GPS++ variant |
| `pairformer.yaml` | Pairformer base |
| `pairformer_17M.yaml` / `pairformer_52M.yaml` / `pairformer_boltz.yaml` | Pairformer scale variants |
| `pairmixer_10M.yaml` / `pairmixer_boltz.yaml` | PairMixer scale variants — currently under active iteration |
| Deleted — model was < 10M params

## For AI Agents

### Working In This Directory
- **GPS++ dimension gotcha**: changing a GPS++ hidden dim requires matching overrides across `pre_nn.out_dim`, `pre_nn.hidden_dims`, `gnn.in_dim/out_dim/hidden_dims`, AND `gnn.layer_kwargs.mpnn_kwargs.in_dim/out_dim`. Audit the whole chain — these are not auto-linked.
- **`constants.norm`** must be set here for any model paired with `architecture/largemix.yaml` (which references `${constants.norm}`). `gpspp_800M.yaml` is the reference for how to do this.
- **When switching pooling modes** (e.g., adding `pairmixer_10M.yaml`), keep head `hidden_dims` and `out_dim` unchanged — only touch the `level_in_dim` wiring (project memory).
- New model files should inherit from an existing base via `defaults: [<base>, _self_]` rather than copy-pasting the full config.
- The Hydra override is `model=<stem>` (e.g., `model=gpspp_800M`).

### Testing Requirements
- `bash ../../../scripts/00_debug.sh <model> toymix 0` for a 3-epoch smoke run.
- `python ../../debug_yaml.py model=<stem> architecture=toymix tasks=toymix training=toymix` to verify composition.

### Common Patterns
- Each model YAML sets `architecture.gnn.layer_type` (mandatory) and optionally `architecture.gnn.layer_kwargs.<type>_kwargs`.
- Scale variants follow a naming convention: `<family>_<size>.yaml` where size is `small`, `medium`, `large`, `boltz`, `800M`, etc.

## Dependencies

### Internal
- `graphium/nn/pyg_layers/` — the layer classes registered in `graphium/utils/spaces.py::PYG_LAYERS_DICT`
- Pairs with `../training/model/<dataset>_<model>.yaml` specializations

<!-- MANUAL: -->
