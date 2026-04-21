"""TDC DTI regression constants."""
from __future__ import annotations

SUBSETS: tuple[str, ...] = (
    "DAVIS", "KIBA",
    "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50",
    "BindingDB_Patent_DG",
)
METHODS: tuple[str, ...] = ("random", "cold_target", "temporal")

# Per-subset label column — most are pY (9 - log10(Y)); KIBA is its own scale.
# ``BindingDB_Patent_DG`` uses pY via the ``dti_dg_group`` leaderboard loader.
SUBSET_LABEL_COL: dict[str, str] = {
    "DAVIS":               "pY",
    "KIBA":                "kiba_score",
    "BindingDB_Kd":        "pY",
    "BindingDB_Ki":        "pY",
    "BindingDB_IC50":      "pY",
    "BindingDB_Patent_DG": "pY",
}

# Protein encoder: ESM-2 ``esm2_t33_650M_UR50D`` mean-pooled over layer 33 (1280-d).
# Same configuration as the GRAM-DTI task for consistency across downstream evals.
ESM2_DIM: int = 1280
PROT_EMB_COLS: list[str] = [f"prot_emb_{i}" for i in range(ESM2_DIM)]

# Default on-disk locations (relative to repo root).
DEFAULT_DATA_DIR = "data/downstream/tdc_dti_regression"
DEFAULT_RESULTS_CSV = "results/downstream/tdc_dti_regression.csv"
DEFAULT_MOL_CACHE_DIR = "datacache/downstream/tdc_dti_regression"
