"""Classification + regression metrics for downstream-task eval.

Classification: AUROC, AUPRC (primary under 1:10 class imbalance), F1,
sensitivity (recall on positives), accuracy. Matches GRAM-DTI Table 13.

Regression: MAE, MSE, R^2, Pearson r, Spearman rho, concordance index (CI).
CI is the DTI-literature standard (DeepDTA / GraphDTA / MolTrans).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5,
) -> Dict[str, float]:
    """Five-metric classification score (GRAM-DTI paper, Table 13)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred_proba = np.asarray(y_pred_proba).astype(np.float64)
    y_pred = (y_pred_proba >= threshold).astype(np.int64)
    # AUROC / AUPRC need both classes present; degenerate folds fall back to NaN.
    try:
        auroc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        auprc = float("nan")
    return {
        "auroc":       auroc,
        "auprc":       auprc,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
    }


def _concordance_index(y: np.ndarray, p: np.ndarray) -> float:
    """Pairwise CI = P(p_i > p_j | y_i > y_j). Ties counted as 0.5."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) < 2:
        return float("nan")
    dy = y[:, None] - y[None, :]
    dp = p[:, None] - p[None, :]
    m = dy > 0
    n_pairs = int(m.sum())
    if n_pairs == 0:
        return float("nan")
    concord = int((dp[m] > 0).sum()) + 0.5 * int((dp[m] == 0).sum())
    return float(concord / n_pairs)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Regression metrics; NaNs when too few samples for correlation stats."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    pear = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    spear = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    return {
        "mae":               float(mean_absolute_error(y_true, y_pred)),
        "mse":               float(mean_squared_error(y_true, y_pred)),
        "r2_score":          float(r2_score(y_true, y_pred)),
        "pearsonr":          float(pear) if pear is not None else float("nan"),
        "spearmanr":         float(spear) if spear is not None else float("nan"),
        "concordance_index": _concordance_index(y_true, y_pred),
        "mean_pred":         float(np.mean(y_pred)),
        "std_pred":          float(np.std(y_pred)),
        "mean_target":       float(np.mean(y_true)),
        "std_target":        float(np.std(y_true)),
    }
