# Method — PairMixer++

PairMixer++ is a lightweight extension of PairMixer added in this repo. It
keeps the dense pair backbone, but refreshes the single/node representation at
every layer and injects that updated single state back into the pair track.

## Core idea

Standard PairMixer updates only the pair state `z`. PairMixer++ adds a cheap
single-track update so the pair state does not drift too far from the evolving
node state.

The design goal is:

- keep PairMixer's dense pair reasoning,
- avoid heavy Pairformer-style attention,
- add only a small coupling cost.

## Architecture diagram

```text
node features s0 ────────────────┐
                                 │
                                 ▼
                        repeated PairMixer++ layer
                        ┌─────────────────────────────────────┐
                        │ single track:                      │
                        │   GINE → residual/norm → s(l+1)    │
                        │                                     │
                        │ pair track:                        │
                        │   reuse/init z(l)                  │
                        │   + edge injection (optional)      │
                        │   + low-rank single→pair update    │
                        │   + triangle mult (outgoing)       │
                        │   + triangle mult (incoming)       │
                        │   + pair transition MLP            │
                        └─────────────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
             final single state sL       final pair state zL
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

The key method parameters are:

- **pair representation dimension** (`pair_dim`)
- **single-to-pair coupling rank** (`single_to_pair_rank`)

## Per-layer computation

Each PairMixer++ layer follows this pattern:

1. Update the node/single state with **one GINE block**
2. Convert the updated node state to dense form
3. Reuse or initialize the pair state `z`
4. Optionally inject projected edge features into `z`
5. Apply a **lightweight single→pair coupling**
6. Apply **triangle multiplication (outgoing)**
7. Apply **triangle multiplication (incoming)**
8. Apply a **pair transition MLP**
9. Re-mask padding pairs

## Lightweight single→pair coupling

The coupling path is intentionally simple:

- project the updated single state into a low-rank left factor,
- project it again into a low-rank right factor,
- build a broadcasted pair interaction,
- project that interaction into the pair dimension,
- gate the update before adding it to `z`.

This adds fresh single-track information to the pair backbone without adding
pair-biased attention or triangle attention.

## Important parameters

Common PairMixer++ parameters in this repo:

- `pair_dim` — width of the pair representation
- `single_to_pair_rank` — low-rank width for single→pair coupling
- `pair_dropout` — dropout on the pair track
- `hidden_dim_scaling` — expansion factor in the pair transition MLP
- `opm_hidden` — hidden size used for pair initialization
- `edge_injection` — whether edge features are injected into the pair track each layer
- `pair_init` — pair initialization configuration
- `use_checkpoint` — activation checkpointing
- `compile_mode` — optional compiled pair-track execution
- `tri_mul_mode` — triangle multiplication backend
- `force_float32_einsums` — stability option for pair ops
- `parallel_pair_ops` — parallel residual form for triangle updates
- `moe_num_experts`, `moe_top_k`, `moe_aux_loss_coeff` — optional MoE pair transition

## Readout

Like PairMixer, PairMixer++ is intended to use pair-aware readout:

- direct graph pooling from `pair_feat`
- optional concatenation with pooled node features

Important readout parameters:

- `pair_pool`
- `node_pooling`

## Why it exists

PairMixer++ targets the main weakness of PairMixer:

- PairMixer has a strong pair backbone, but its single track is mostly static
- PairMixer++ keeps a cheap node update at every layer
- The pair track therefore sees a refreshed single state every layer

## Relation to other methods

- **vs PairMixer**: adds single updates and single→pair coupling
- **vs Pairformer**: much lighter, no pair-biased single attention, no triangle attention
- **vs GPS++**: keeps a dense pair track instead of only a node backbone

## Main repo references

- Layer: `graphium/nn/pyg_layers/pairmixerpp_pyg.py`
- Pair readout: `graphium/nn/architectures/global_architectures.py`
- Example config: `expts/hydra-configs/model/pairmixerpp_12M.yaml`
