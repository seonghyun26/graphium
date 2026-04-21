<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/nn/encoders

## Purpose
Positional and structural encoder networks that map raw features (Laplacian eigenvectors/values, random walks, distances, etc.) into the hidden embedding space consumed by downstream GNN layers. Instantiated by `EncoderManager` in `../architectures/encoder_manager.py`.

## Key Files

| File | Description |
|------|-------------|
| `base_encoder.py` | `BaseEncoder` — abstract base; all encoders inherit from this |
| `mlp_encoder.py` | `MLPEncoder` — simple MLP wrapper for arbitrary PE vectors |
| `laplace_pos_encoder.py` | `LapPENodeEncoder` — Laplacian eigenvector + eigenvalue encoder |
| `signnet_pos_encoder.py` | SignNet-based sign-invariant Laplacian encoder |
| `gaussian_kernel_pos_encoder.py` | `GaussianKernelPosEncoder` — Gaussian basis over distances (used by Graphormer-3D-style layers) |
| `bessel_pos_encoder.py` | Bessel-basis radial distance encoder |
| `__init__.py` | Re-exports encoder classes |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- Inherit from `BaseEncoder` and implement `forward(batch)` returning a dict keyed by the PE name declared in the Hydra `pe_encoders` block.
- Register new encoders in `graphium/utils/spaces.py::PE_ENCODERS_DICT` so they can be selected via YAML.
- Raw PE values (what the encoder consumes) are computed in `graphium/features/positional_encoding.py` and friends (`commute.py`, `rw.py`, `spectral.py`, `electrostatic.py`, `graphormer.py`).
- Encoders may target multiple graph levels (node/edge/graph/nodepair); return the right level key so `EncoderManager` can pool it correctly.

### Testing Requirements
- `tests/test_positional_encoders.py`, `tests/test_pe_nodepair.py`, `tests/test_pe_rw.py`, `tests/test_pe_spectral.py`.

### Common Patterns
- Sign ambiguity of Laplacian eigenvectors is handled via augmentation in `LapPENodeEncoder` or explicit equivariance in SignNet — do not strip this.
- Encoder output dimension must match the downstream `pe_encoders.out_dim` config; it is usually configured once and reused by every PE.

## Dependencies

### Internal
- `graphium/features/positional_encoding.py` — input side
- `../architectures/encoder_manager.py` — consumer

### External
- `torch`, `torch_geometric`

<!-- MANUAL: -->
