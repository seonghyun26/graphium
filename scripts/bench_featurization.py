"""Benchmark dense vs sparse eigendecomposition on a subset of LargeMix SMILES."""

import time
import numpy as np
import pandas as pd
from graphium.features.featurizer import mol_to_pyggraph
from graphium.features.spectral import _get_positional_eigvecs, _get_positional_eigvecs_dense
from scipy.sparse import csr_matrix
from graphium.features.spectral import normalize_matrix

# ── Config ──────────────────────────────────────────────────────────────────
NUM_SAMPLES = 2000
NUM_POS = 8

# Featurization kwargs matching largemix architecture config
FEATURIZATION_KWARGS = dict(
    atom_property_list_onehot=["atomic-number", "group", "period", "total-valence"],
    atom_property_list_float=["degree", "formal-charge", "radical-electron", "aromatic", "in-ring"],
    edge_property_list=["bond-type-onehot", "stereo", "in-ring"],
    add_self_loop=False,
    explicit_H=False,
    use_bonds_weights=False,
    pos_encoding_as_features={
        "pos_types": {
            "lap_eigvec": {
                "pos_level": "node",
                "pos_type": "laplacian_eigvec",
                "num_pos": NUM_POS,
                "normalization": "none",
                "disconnected_comp": True,
            },
            "lap_eigval": {
                "pos_level": "node",
                "pos_type": "laplacian_eigval",
                "num_pos": NUM_POS,
                "normalization": "none",
                "disconnected_comp": True,
            },
            "rw_pos": {
                "pos_level": "node",
                "pos_type": "rw_return_probs",
                "ksteps": 16,
            },
        }
    },
)

# ── Load SMILES ─────────────────────────────────────────────────────────────
print(f"Loading SMILES from PCQM4M dataset...")
df = pd.read_parquet(
    "data/graphium/neurips2023/large-dataset/PCQM4M_G25_N4.parquet",
    columns=["ordered_smiles"],
)
smiles_list = df["ordered_smiles"].dropna().tolist()[:NUM_SAMPLES]
print(f"Loaded {len(smiles_list)} SMILES")

# ── Benchmark: full featurization with dense (default) ──────────────────────
print(f"\n=== Dense eig (original) on {NUM_SAMPLES} molecules ===")
dense_kwargs = FEATURIZATION_KWARGS.copy()
t0 = time.perf_counter()
dense_results = []
dense_failures = 0
for smi in smiles_list:
    try:
        g = mol_to_pyggraph(smi, **dense_kwargs)
        dense_results.append(g)
    except Exception:
        dense_failures += 1
t_dense = time.perf_counter() - t0
print(f"Time: {t_dense:.2f}s ({t_dense/NUM_SAMPLES*1000:.1f} ms/mol), failures: {dense_failures}")

# ── Benchmark: full featurization with sparse eigsh ─────────────────────────
print(f"\n=== Sparse eigsh on {NUM_SAMPLES} molecules ===")
sparse_kwargs = FEATURIZATION_KWARGS.copy()
# Add eig_method to both lap_eigvec and lap_eigval
sparse_pe = {
    "pos_types": {
        "lap_eigvec": {
            **FEATURIZATION_KWARGS["pos_encoding_as_features"]["pos_types"]["lap_eigvec"],
            "eig_method": "sparse",
        },
        "lap_eigval": {
            **FEATURIZATION_KWARGS["pos_encoding_as_features"]["pos_types"]["lap_eigval"],
            "eig_method": "sparse",
        },
        "rw_pos": FEATURIZATION_KWARGS["pos_encoding_as_features"]["pos_types"]["rw_pos"],
    }
}
sparse_kwargs["pos_encoding_as_features"] = sparse_pe

t0 = time.perf_counter()
sparse_results = []
sparse_failures = 0
for smi in smiles_list:
    try:
        g = mol_to_pyggraph(smi, **sparse_kwargs)
        sparse_results.append(g)
    except Exception:
        sparse_failures += 1
t_sparse = time.perf_counter() - t0
print(f"Time: {t_sparse:.2f}s ({t_sparse/NUM_SAMPLES*1000:.1f} ms/mol), failures: {sparse_failures}")

# ── Compare results ─────────────────────────────────────────────────────────
print(f"\n=== Comparison ===")
print(f"Dense:  {t_dense:.2f}s")
print(f"Sparse: {t_sparse:.2f}s")
print(f"Speedup: {t_dense/t_sparse:.2f}x")

# Verify numerical similarity
n_check = min(100, len(dense_results), len(sparse_results))
eigvec_diffs = []
eigval_diffs = []
for i in range(n_check):
    d = dense_results[i]
    s = sparse_results[i]
    # Eigenvectors can differ by sign, compare absolute values
    d_eigvec = np.abs(d["laplacian_eigvec"].numpy())
    s_eigvec = np.abs(s["laplacian_eigvec"].numpy())
    eigvec_diffs.append(np.mean(np.abs(d_eigvec - s_eigvec)))

    d_eigval = d["laplacian_eigval"].numpy()
    s_eigval = s["laplacian_eigval"].numpy()
    eigval_diffs.append(np.mean(np.abs(d_eigval - s_eigval)))

print(f"\nNumerical check (first {n_check} molecules):")
print(f"  Eigenvector mean abs diff: {np.mean(eigvec_diffs):.6f}")
print(f"  Eigenvalue mean abs diff:  {np.mean(eigval_diffs):.6f}")
