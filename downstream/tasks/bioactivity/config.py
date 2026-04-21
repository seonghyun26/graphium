"""Bioactivity benchmark constants — frozen to match the Fredinh et al. 2024 paper."""
from __future__ import annotations

# 29 ChEMBL assay IDs retained after the paper's balance filter (>50 pos / >50 neg)
# on the ChEMBL-33 ∩ JUMP-CP ``source_11`` overlap. See
# ``scripts/data/bioactivity/01_prepare_bioactivity.py`` for the filter pipeline.
EXPECTED_ASSAY_IDS: list[int] = [
    688128, 688238, 688360, 688546, 688549, 688612, 688687, 688812, 688816,
    736947, 737187, 737287, 737344, 752347, 752407, 752434, 752493, 752563,
    752590, 752594, 845045, 845102, 845164, 845169, 845173, 845177, 845196,
    954338, 1495346,
]
ASSAY_COLS: list[str] = [f"assay_{a}" for a in EXPECTED_ASSAY_IDS]
N_ASSAYS: int = len(EXPECTED_ASSAY_IDS)

# 6-fold Butina CV with rotation: test={k}, val={(k+1)%6}, train=rest.
# Matches the upstream ``cfredinh/bioactive`` splits.
N_FOLDS: int = 6

# Default on-disk locations (relative to repo root).
DEFAULT_DATA_DIR = "../data/downstream/bioactivity"
DEFAULT_RESULTS_CSV = "results/downstream/bioactivity.csv"
DEFAULT_MOL_CACHE_DIR = "datacache/downstream/bioactivity"
