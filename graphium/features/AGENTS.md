<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/features

## Purpose
SMILES-to-graph featurization. Produces the node, edge, and graph-level features and the raw positional/structural encoding vectors that the downstream encoder networks (`graphium/nn/encoders/`) will embed.

## Key Files

| File | Description |
|------|-------------|
| `featurizer.py` | Main SMILES → PyG `Data` conversion (atoms, bonds, 3D coords, positional encodings) |
| `positional_encoding.py` | `graph_positional_encoder` — dispatches to individual PE computers below |
| `spectral.py` | Laplacian eigenvalue/eigenvector computation |
| `rw.py` | Random walk structural encoding |
| `commute.py` | Commute-time / effective resistance features |
| `electrostatic.py` | Electrostatic distance-based features |
| `graphormer.py` | Graphormer-style shortest-path / centrality encoding |
| `properties.py` | Whole-molecule property computation (mass, logP, etc.) |
| `nmp.py` | `numeric_property_string` helpers — guards against non-float CSV entries |
| `transfer_pos_level.py` | Utilities to transfer PE vectors between node/edge/nodepair levels |
| `periodic_table.csv` | Element property lookup table used by the atom featurizer |
| `test_new_pes.ipynb` | Ad-hoc notebook for prototyping new PEs (not a test) |
| `__init__.py` | Re-exports featurizer entry points |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- `featurizer.mol_to_graph_signature` determines the cache key — changing its output invalidates `../../datacache/` entries.
- New PE types require: (1) a computation here, (2) registration in `positional_encoding.py::graph_positional_encoder`, (3) a matching encoder in `graphium/nn/encoders/`, (4) an entry in `graphium/utils/spaces.py::PE_ENCODERS_DICT`.
- Keep featurization **pure and deterministic** — any randomness must come from a seed passed via kwargs, otherwise cached graphs become non-reproducible.
- `periodic_table.csv` is the ground truth for atom-level features; if you add new element properties, update this file and the column reader in `featurizer.py`.

### Testing Requirements
- `tests/test_featurizer.py`, `tests/test_pe_spectral.py`, `tests/test_pe_rw.py`, `tests/test_pe_nodepair.py`, `tests/test_pos_transfer_funcs.py`, `tests/test_positional_encodings.py`.

### Common Patterns
- Each PE function returns a dict keyed by graph level (`"node"`, `"edge"`, `"graph"`, `"nodepair"`) → tensor.
- SMILES parsing goes through `datamol`/`rdkit` — do not bring in additional chemistry libraries.

## Dependencies

### Internal
- Consumed by `graphium/data/datamodule.py` during cache build.

### External
- `rdkit`, `datamol`, `numpy`, `scipy` (spectral), `networkx` (commute/graphormer)

<!-- MANUAL: -->
