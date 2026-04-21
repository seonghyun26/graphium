"""TDC DTI regression benchmark (DAVIS / KIBA / BindingDB_*).

Same encoder / ESM-2 / head toolkit as ``gram_dti``, but with regression
labels (pY for BindingDB & DAVIS, KIBA score for KIBA) and DAVIS/KIBA's
``random`` + ``cold_target`` splits plus BindingDB_Patent_DG's fixed
``temporal`` split from the TDC leaderboard.
"""
from .config import METHODS, SUBSETS, SUBSET_LABEL_COL, ESM2_DIM, PROT_EMB_COLS

__all__ = ["METHODS", "SUBSETS", "SUBSET_LABEL_COL", "ESM2_DIM", "PROT_EMB_COLS"]
