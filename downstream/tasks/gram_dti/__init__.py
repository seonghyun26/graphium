"""GRAM-DTI binary-classification benchmark (arXiv:2509.21971, ICLR 2026).

Four DTIAM datasets (Yamanishi 08 / Hetionet / Activation / Inhibition),
10/10/5/5-fold CV, 3 split methods (warm / drug_cold / target_cold), 1:10
negative sampling, 360 fold-runs per encoder.

Protein side is fixed: ESM-2 650M (1280-d) mean representation. We only swap
the molecule encoder.
"""
from .config import METHODS, SUBSETS, SUBSET_TO_FOLDS, ESM2_DIM, PROT_EMB_COLS

__all__ = ["METHODS", "SUBSETS", "SUBSET_TO_FOLDS", "ESM2_DIM", "PROT_EMB_COLS"]
