"""GRAM-DTI benchmark constants (frozen to paper parity).

Do not change without re-reading ``memory/project_gram_dti_benchmark.md`` —
the split counts, method set, and negative ratio are all load-bearing.
"""
from __future__ import annotations

SUBSETS: tuple[str, ...] = ("yamanishi_08", "hetionet", "activation", "inhibition")
METHODS: tuple[str, ...] = ("warm", "drug_cold", "target_cold")

SUBSET_TO_FOLDS: dict[str, int] = {
    "yamanishi_08": 10,
    "hetionet":     10,
    "activation":   5,
    "inhibition":   5,
}

# Protein encoder: ESM-2 ``esm2_t33_650M_UR50D`` mean-pooled over layer 33,
# yielding a 1280-d embedding. Matches the GRAM-DTI paper (arXiv:2509.21971)
# exactly. Extraction runs through ``scripts/data/dti_esm2/02_extract_embeddings.py``
# with ``--esm-model esm2_t33_650M_UR50D --repr-layer 33``.
ESM2_DIM: int = 1280
PROT_EMB_COLS: list[str] = [f"prot_emb_{i}" for i in range(ESM2_DIM)]

# Default on-disk locations (relative to repo root).
DEFAULT_DATA_DIR = "../data/downstream/gram_dti"
DEFAULT_RESULTS_CSV = "results/downstream/gram_dti.csv"
DEFAULT_MOL_CACHE_DIR = "datacache/downstream/gram_dti"
