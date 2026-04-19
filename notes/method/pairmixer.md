# Method — PairMixer

Lightweight pair-track architecture adapted from the PairMixer family. In this
repo it is implemented as a dense **pair representation** backbone with almost
no single-track computation inside the GNN stack.

## Core idea

PairMixer keeps a pair state `z` for all node pairs and updates it with
triangle-multiplication blocks. The node/single state `s` is largely treated as
an initializer and readout support, not as a fully updated backbone.

This makes the model:

- more relational than a standard message-passing GNN,
- simpler than Pairformer,
- cheaper than architectures that use full pair attention.

## Architecture diagram

```text
node features s0
    │
    ├─ pair initialization ───────────────┐
    │                                     ▼
    │                               pair state z0
    │                                     │
    │                    ┌────────────────┴───────────────┐
    │                    │ repeated PairMixer layer       │
    │                    │  • edge injection (optional)   │
    │                    │  • triangle mult (outgoing)    │
    │                    │  • triangle mult (incoming)    │
    │                    │  • pair transition MLP         │
    │                    └────────────────┬───────────────┘
    │                                     ▼
    └──────────────────────────────► final pair state zL
                                          │
                               pair pooling / pair stats
                                          │
                                 optional node pooling
                                          │
                                      task head
```

## Representations

- **Single / node representation**: `s`
- **Pair representation**: `z`

The most important method parameter is the **pair representation dimension**
(`pair_dim`), since it controls the width of the dense pair track.

## Per-layer computation

Each PairMixer layer follows this pattern:

1. Reuse the current pair representation `z`
2. Optionally add projected edge features to the pair track
3. Apply **triangle multiplication (outgoing)**
4. Apply **triangle multiplication (incoming)**
5. Apply a **pair transition MLP**
6. Re-mask padding pairs

The single track is not meaningfully updated inside the backbone. This is the
main distinction from Pairformer and from PairMixer++.

## Pair initialization

The first layer creates `z_0` using a pair initializer. In this repo the pair
initializer can combine:

- outer-product-based single-to-pair initialization,
- graph-structural priors,
- optional edge-feature priors.

The relevant parameters include:

- `pair_init`
- `opm_hidden`

## Important parameters

Common PairMixer parameters in this repo:

- `pair_dim` — width of the pair representation
- `pair_dropout` — dropout on the pair track
- `hidden_dim_scaling` — expansion factor in the pair transition MLP
- `opm_hidden` — hidden size used for pair initialization
- `edge_injection` — whether edge features are projected into the pair track at each layer
- `use_checkpoint` — activation checkpointing for memory savings
- `compile_mode` — optional `torch.compile` path
- `tri_mul_mode` — backend for triangle multiplication
- `force_float32_einsums` — stability option for triangle ops
- `parallel_pair_ops` — whether pair ops read the same `z` in parallel residual form
- `moe_num_experts`, `moe_top_k`, `moe_aux_loss_coeff` — optional MoE pair transition

## Readout

PairMixer commonly uses **direct pair readout**:

- graph-level pooling from `pair_feat`
- optional concatenation with pooled node features

Important readout parameters:

- `pair_pool` — pair pooling mode
- `node_pooling` — optional node pooling alongside pair pooling

## Strengths

- Strong relational inductive bias
- Efficient compared with heavier pair-attention models
- Scales well when pair width is the main capacity lever

## Limitations

- The pair track can become weak if initialization is poor
- The single/node track is not refreshed inside the backbone
- Performance depends strongly on pair initialization and pair readout quality

## Main repo references

- Layer: `graphium/nn/pyg_layers/pairmixer_pyg.py`
- Pair init: `graphium/nn/pyg_layers/pair_init.py`
- Pair readout: `graphium/nn/architectures/global_architectures.py`
- Example config: `expts/hydra-configs/model/pairmixer_12M.yaml`
