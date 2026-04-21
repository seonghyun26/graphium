"""Cell-bioactivity benchmark (Fredinh et al., Nat. Commun. 2024).

Binary per-assay classification across 29 ChEMBL assays overlapping with JUMP
Cell Painting ``source_11``. Butina-clustered 6-fold CV on ECFP4 distance.

Key differences from ``gram_dti`` / ``tdc_dti_regression``:
  - Multi-label output (29 assays); labels are -1/0/+1 with 0 = unknown.
  - Paper-faithful PyTorch MLP + ``FocalBCEMaskedLoss`` head
    (``downstream.tasks.bioactivity.head``). sklearn MLP doesn't support
    masked-multi-label loss natively.
  - Primary metric: macro-AUROC (and macro-AUPRC) averaged across assays.
"""
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MOL_CACHE_DIR,
    DEFAULT_RESULTS_CSV,
    EXPECTED_ASSAY_IDS,
    N_FOLDS,
)

__all__ = [
    "DEFAULT_DATA_DIR", "DEFAULT_MOL_CACHE_DIR", "DEFAULT_RESULTS_CSV",
    "EXPECTED_ASSAY_IDS", "N_FOLDS",
]
