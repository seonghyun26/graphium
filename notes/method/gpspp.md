# Method — GPS++

GPS++ is the repo's GraphGPS-style hybrid architecture. It combines a local
message-passing block with a global attention block inside each GNN layer, then
applies a node MLP update.

## Core idea

GPS++ keeps a **single/node representation** backbone and mixes:

- local graph structure through an MPNN branch,
- global context through full self-attention,
- a feed-forward update after combining both.

In this repo it is the main graph-transformer-style baseline without a dense
pair backbone.

## Architecture diagram

```text
node features s0
    │
    ▼
repeated GPS++ layer
┌──────────────────────────────────────────────┐
│ local branch:    MPNN / message passing      │
│ global branch:   full self-attention         │
│ merge:           local + global              │
│ update:          FFN / MLP                   │
│ output:          projected node state s(l+1) │
└──────────────────────────────────────────────┘
    │
    ▼
final node state sL
    │
graph pooling
    │
task head
```

## Representation

- **Single / node representation**: `s`
- No persistent dense pair representation like PairMixer or PairMixer++

## Per-layer computation

Each GPS++ layer follows this pattern:

1. Apply a local **MPNN block** to node features
2. Apply a global **self-attention block**
3. Sum the local and attention outputs
4. Apply an **MLP / FFN update**
5. Project to the layer output dimension

The local and global paths are both residual-style updates on the node track.

## Important parameters

Common GPS++ parameters in this repo:

- `mpnn_type` — local message-passing layer type
- `mpnn_kwargs` — parameters for the local MPNN branch
- `attn_type` — attention type for the global branch
- `attn_kwargs` — attention parameters such as head count
- `node_residual` — residual behavior for the node path
- `hidden_dim_scaling` — expansion factor for the FFN
- `droppath_rate_attn` — stochastic depth rate on the attention path
- `droppath_rate_ffn` — stochastic depth rate on the FFN path
- `precision` — attention / mixed-precision behavior
- `biased_attention_key` — optional positional or pairwise bias source
- `force_consistent_in_dim` — keeps attention and MPNN dimensions aligned
- `moe_num_experts`, `moe_top_k`, `moe_aux_loss_coeff` — optional MoE FFN

## Positional / structural information

GPS++ can use external positional encodings through the encoder stack before the
GNN layers. The architecture is often paired with:

- Laplacian positional encodings
- random-walk positional encodings
- optional attention bias features

## Readout

GPS++ uses standard graph readout from the node backbone:

- graph pooling over node features
- task heads applied after pooled graph features

Unlike PairMixer-family models, it does not depend on direct pair pooling.

## Strengths

- Strong general-purpose graph transformer baseline
- Combines local chemistry and global context well
- Flexible because the local branch and attention branch are configurable

## Limitations

- No persistent pair representation
- Can be more expensive than pure MPNN models because of full attention
- Requires careful dimension wiring when hidden sizes change

## Main repo references

- Layer: `graphium/nn/pyg_layers/gps_pyg.py`
- Example config: `expts/hydra-configs/model/gpspp.yaml`
