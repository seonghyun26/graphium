"""Shared utilities for GRAM-DTI benchmark eval scripts (``eval_{pairmixer,minimol,mole}.py``).

The per-encoder scripts differ only in how they extract molecule embeddings
from SMILES. Data loading, split resolution, head training, metric computation,
and results logging all live here.

Benchmark spec (frozen — see ``memory/project_gram_dti_benchmark.md``):
    Yamanishi 08   DTI  10-fold CV   {warm, drug_cold, target_cold}
    Hetionet       DTI  10-fold CV   {warm, drug_cold, target_cold}
    Activation     MoA  5-fold  CV   {warm, drug_cold, target_cold}
    Inhibition     MoA  5-fold  CV   {warm, drug_cold, target_cold}
    Negative sampling: 1:10. Metrics: AUROC, AUPRC (primary), F1, Sensitivity, Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Benchmark layout ─────────────────────────────────────────────────────────

SUBSETS: tuple[str, ...] = ("yamanishi_08", "hetionet", "activation", "inhibition")
METHODS: tuple[str, ...] = ("warm", "drug_cold", "target_cold")
SUBSET_TO_FOLDS: dict[str, int] = {
    "yamanishi_08": 10,
    "hetionet":     10,
    "activation":   5,
    "inhibition":   5,
}

ESMC_DIM = 1152
PROT_EMB_COLS: list[str] = [f"prot_emb_{i}" for i in range(ESMC_DIM)]


# ── Data loading ─────────────────────────────────────────────────────────────

@dataclass
class SubsetArrays:
    """Dense arrays for one benchmark subset, loaded once and reused across folds."""
    smiles:    list[str]                    # len N
    label:     np.ndarray                   # (N,) int8 {0, 1}
    prot_emb:  np.ndarray                   # (N, 1152) float32 — z-scored at prep time
    drug_id:   np.ndarray                   # (N,) object — for debugging / error messages
    target_id: np.ndarray                   # (N,) object


def load_subset(parquet_dir: str | Path, subset: str) -> SubsetArrays:
    """Load a per-subset parquet produced by ``02_prepare_dti_classif.py``."""
    parquet_path = Path(parquet_dir) / f"{subset}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} missing. Run scripts/data/dti_classif_eval/02_prepare_dti_classif.py first."
        )
    df = pd.read_parquet(parquet_path)
    missing = set(PROT_EMB_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} missing protein embedding columns: {sorted(missing)[:5]}...")
    return SubsetArrays(
        smiles=df["SMILES_nometa"].tolist(),
        label=df["label"].values.astype(np.int8),
        prot_emb=df[PROT_EMB_COLS].values.astype(np.float32),
        drug_id=df["drug_id"].values,
        target_id=df["target_id"].values,
    )


def load_split(parquet_dir: str | Path, subset: str, method: str, fold: int) -> dict[str, list[int]]:
    """Load a {train, val, test} index dict for one (subset, method, fold)."""
    split_path = Path(parquet_dir) / "splits" / f"{subset}_{method}_fold{fold}.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} missing.")
    splits = torch.load(split_path, weights_only=False)
    for key in ("train", "val", "test"):
        if key not in splits:
            raise ValueError(f"{split_path} missing key {key!r}")
    return splits


# ── Molecule-feature caching (per encoder, per subset) ───────────────────────

def cached_molecule_features(
    cache_dir: str | Path,
    encoder_tag: str,
    smiles: list[str],
    extract_fn: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Extract (or load from cache) per-molecule features, keyed per-SMILES.

    ``extract_fn(unique_smiles_list) -> (M, D)`` is called only on cache miss
    with the deduplicated list of unseen SMILES. The cache is shared across
    subsets within one encoder, so a drug appearing in both Activation and
    Inhibition is only embedded once.

    Mirrors the per-SMILES cache pattern in scripts/{minimol,mole}/{...}_dti_eval.py.
    """
    cache_path = Path(cache_dir) / f"{encoder_tag}_mol_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = (
        torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    )
    unique = list(dict.fromkeys(smiles))
    missing = [s for s in unique if s not in cache]
    if missing:
        print(f"  [{encoder_tag}] embedding {len(missing):,} / {len(unique):,} new SMILES...", flush=True)
        new_feats = extract_fn(missing)
        new_feats = np.asarray(new_feats, dtype=np.float32)
        if new_feats.ndim != 2 or new_feats.shape[0] != len(missing):
            raise ValueError(f"extract_fn returned shape {new_feats.shape}; expected ({len(missing)}, D)")
        for s, f in zip(missing, new_feats):
            cache[s] = f.astype(np.float32)
        torch.save(cache, cache_path)
    return np.stack([cache[s] for s in smiles], axis=0)


# ── Training + metrics ───────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All five paper metrics (Table 13 of GRAM-DTI). Primary: AUPRC (class imbalance 1:10)."""
    y_pred = (y_pred_proba >= threshold).astype(np.int64)
    return {
        "auroc":       float(roc_auc_score(y_true, y_pred_proba)),
        "auprc":       float(average_precision_score(y_true, y_pred_proba)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),  # recall on positive class
        "accuracy":    float(accuracy_score(y_true, y_pred)),
    }


def train_head(
    X_tr: np.ndarray, y_tr: np.ndarray,
    head_type: str,
    *,
    random_state: int = 0,
    mlp_hidden: tuple[int, ...] = (256, 256),
    mlp_max_iter: int = 200,
    logreg_C: float = 1.0,
    logreg_max_iter: int = 2000,
):
    """Fit ``StandardScaler → classifier`` on pre-extracted features.

    Molecule embeddings (raw Minimol/MolE/PairMixer) and ESM-C protein
    embeddings live on different scales — prepping z-scores the protein half
    but leaves the mol half raw. Without this scaler, lbfgs fails to converge
    and MLP training is needlessly slow. The pipeline applies the fit on train
    only, then transforms val/test via the same mean/std.

    ``head_type``: ``"logreg"`` (default) or ``"mlp"``. LogReg is what the
    existing minimol/mole baselines use at the top of the stack; MLP is an
    optional stronger head consistent with GRAM-DTI's learnable projection.
    Returns a ``sklearn.pipeline.Pipeline`` — ``predict_proba`` composes scaler
    and classifier automatically, and ``.classes_`` delegates to the final step.
    """
    if head_type == "logreg":
        estimator = LogisticRegression(
            C=logreg_C,
            max_iter=logreg_max_iter,
            class_weight="balanced",   # matches the 1:10 imbalance under BCE
            random_state=random_state,
            n_jobs=-1,
        )
    elif head_type == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=mlp_hidden,
            max_iter=mlp_max_iter,
            early_stopping=True,        # uses internal val split; we still pass an external val set for reporting
            validation_fraction=0.1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown head_type {head_type!r}")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    estimator),
    ])
    pipeline.fit(X_tr, y_tr)
    return pipeline


def predict_proba(clf, X: np.ndarray) -> np.ndarray:
    """Return shape-(N,) prob of the positive class, robust to single-class training."""
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=np.float32)  # degenerate fold — report zeros


# ── Results logging ──────────────────────────────────────────────────────────

def append_result_row(csv_path: str | Path, row: dict) -> None:
    """Append a single row to a CSV, creating the file and expanding columns as needed.

    Matches the open-schema approach of ``graphium/cli/train_finetune_test.py::_save_results_csv``
    so the downstream dashboard notebook can ingest these results the same way.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        merged = pd.concat([prev, new_df], ignore_index=True, sort=False)
        merged.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)


# ── Main eval loop (shared across per-encoder scripts) ───────────────────────

def run_eval_sweep(
    *,
    encoder_tag: str,
    extract_fn: Callable[[list[str]], np.ndarray],
    parquet_dir: str | Path,
    results_csv: str | Path,
    mol_cache_dir: str | Path,
    subsets: Iterable[str],
    methods: Iterable[str],
    folds: dict[str, Iterable[int]] | None = None,
    head_type: str = "logreg",
    random_state: int = 0,
    extra_row_meta: dict | None = None,
) -> None:
    """Run a full subset × method × fold sweep and append each fold's metrics to ``results_csv``.

    ``folds`` maps subset → iterable of fold indices. If ``None``, uses the full
    paper-parity range (10 for DTI, 5 for MoA).
    """
    folds = folds or {s: range(SUBSET_TO_FOLDS[s]) for s in subsets}
    extra_row_meta = dict(extra_row_meta or {})

    for subset in subsets:
        arrays = load_subset(parquet_dir, subset)
        mol_feats = cached_molecule_features(
            mol_cache_dir, encoder_tag, arrays.smiles, extract_fn,
        )
        X_all = np.concatenate([mol_feats, arrays.prot_emb], axis=1).astype(np.float32)
        y_all = arrays.label.astype(np.int64)

        for method in methods:
            for fold in folds[subset]:
                splits = load_split(parquet_dir, subset, method, fold)
                tr_idx = np.asarray(splits["train"], dtype=np.int64)
                va_idx = np.asarray(splits["val"],   dtype=np.int64)
                te_idx = np.asarray(splits["test"],  dtype=np.int64)

                clf = train_head(X_all[tr_idx], y_all[tr_idx], head_type,
                                 random_state=random_state)
                proba_val  = predict_proba(clf, X_all[va_idx])
                proba_test = predict_proba(clf, X_all[te_idx])

                val_metrics  = compute_metrics(y_all[va_idx],  proba_val)
                test_metrics = compute_metrics(y_all[te_idx], proba_test)

                row = {
                    "encoder":    encoder_tag,
                    "head":       head_type,
                    "subset":     subset,
                    "method":     method,
                    "fold":       int(fold),
                    "n_train":    len(tr_idx),
                    "n_val":      len(va_idx),
                    "n_test":     len(te_idx),
                    **{f"val_{k}":  v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **extra_row_meta,
                }
                append_result_row(results_csv, row)
                print(
                    f"  [{encoder_tag}/{subset}/{method}/fold={fold}] "
                    f"test AUPRC={test_metrics['auprc']:.4f} "
                    f"AUROC={test_metrics['auroc']:.4f} "
                    f"F1={test_metrics['f1']:.4f}"
                )
