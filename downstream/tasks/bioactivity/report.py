"""Reporting helpers for the Cell Bioactivity benchmark.

The eval CLI writes one row per (encoder, fold) to
``results/downstream/bioactivity.csv``. This module provides:

- per-assay metric flattening for CSV rows,
- AUROC threshold-count summaries for each fold,
- enrichment-factor summaries for early hit retrieval,
- per-assay win/rank summaries and assay-level AUROC distributions,
- aggregated tables for notebooks / paper-ready reporting.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


AUROC_THRESHOLDS: tuple[float, ...] = (0.70, 0.80, 0.90)
EF_TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)
PER_ASSAY_AUROC_PREFIX = "test_auroc__"
PER_ASSAY_AUPRC_PREFIX = "test_auprc__"

DEFAULT_DATASET_LABEL = "ChEMBL/JUMP-CP (29 assays)"
DEFAULT_ENCODER_LABELS = {
    "ecfp": "ECFP4 (paper)",
    "cpcnn": "CPCNN (paper cell-painting)",
    "minimol": "MiniMol",
    "mole": "MolE",
    "kpgt": "KPGT",
    "pairmixer": "PairMixer 12M",
}
DEFAULT_MODEL_ORDER = [
    "ECFP4 (paper)",
    "CPCNN (paper cell-painting)",
    "MiniMol",
    "MolE",
    "KPGT",
    "PairMixer 12M",
]
DEFAULT_METRIC_COLS = [
    ("test_auroc_macro", "AUROC ↑"),
    ("test_auprc_macro", "AUPRC ↑"),
]


def _threshold_slug(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _fraction_slug(top_fraction: float) -> str:
    pct = 100.0 * top_fraction
    if float(pct).is_integer():
        return f"{int(pct)}p"
    return f"{pct:.2f}".replace(".", "p")


def format_threshold_label(threshold: float) -> str:
    return f"AUROC ≥ {threshold:.1f}"


def format_ef_label(top_fraction: float) -> str:
    return f"EF@{100.0 * top_fraction:.0f}%"


def ef_column_prefix(top_fraction: float) -> str:
    return f"test_ef_{_fraction_slug(top_fraction)}__"


def ef_macro_column(top_fraction: float) -> str:
    return f"test_ef_{_fraction_slug(top_fraction)}_macro"


def enrichment_factor(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    top_fraction: float,
) -> float:
    """Compute EF@k% for a binary ranked list."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(top_fraction * n)))
    order = np.argsort(-y_score, kind="stable")
    top_hits = y_true[order[:n_top]].sum()
    top_hit_rate = float(top_hits / n_top)
    base_hit_rate = float(y_true.mean())
    return float(top_hit_rate / base_hit_rate) if base_hit_rate > 0 else float("nan")


def assay_threshold_summary(
    per_assay_auroc: Mapping[str, float],
    *,
    thresholds: Sequence[float] = AUROC_THRESHOLDS,
    denominator: int | None = None,
) -> dict[str, float]:
    """Count assays whose AUROC exceeds each threshold."""
    values = np.array(
        [float(v) for v in per_assay_auroc.values() if pd.notna(v)],
        dtype=float,
    )
    denom = int(denominator if denominator is not None else len(values))
    out: dict[str, float] = {}
    for threshold in thresholds:
        slug = _threshold_slug(threshold)
        count = int(np.sum(values >= threshold))
        out[f"test_auroc_ge_{slug}_count"] = count
        out[f"test_auroc_ge_{slug}_pct"] = (100.0 * count / denom) if denom > 0 else float("nan")
    return out


def flatten_per_assay_metrics(
    *,
    per_assay_auroc: Mapping[str, float],
    per_assay_auprc: Mapping[str, float],
    per_assay_ef: Mapping[float, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Flatten per-assay scores into CSV-safe columns."""
    row: dict[str, float] = {}
    for assay, value in sorted(per_assay_auroc.items()):
        row[f"{PER_ASSAY_AUROC_PREFIX}{assay}"] = float(value)
    for assay, value in sorted(per_assay_auprc.items()):
        row[f"{PER_ASSAY_AUPRC_PREFIX}{assay}"] = float(value)
    for top_fraction, assay_scores in sorted((per_assay_ef or {}).items()):
        prefix = ef_column_prefix(top_fraction)
        for assay, value in sorted(assay_scores.items()):
            row[f"{prefix}{assay}"] = float(value)
    return row


def per_assay_metric_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(col for col in df.columns if col.startswith(prefix))


def _sorted_model_labels(
    work: pd.DataFrame,
    *,
    model_order: Sequence[str],
) -> list[str]:
    return sorted(
        {
            label
            for label in work["model_label"].unique()
            if isinstance(label, str) and label
        },
        key=lambda model: model_order.index(model) if model in model_order else len(model_order),
    )


def build_macro_summary_table(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[tuple[str, str]] = DEFAULT_METRIC_COLS,
    encoder_labels: Mapping[str, str] = DEFAULT_ENCODER_LABELS,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    dataset_label: str = DEFAULT_DATASET_LABEL,
) -> pd.DataFrame:
    """Return the current mean ± std summary table used in the dashboard."""
    work = df.copy()
    work["model_label"] = work["encoder"].map(lambda e: encoder_labels.get(str(e), str(e)))

    metric_names = [metric for metric, _ in metric_cols if metric in work.columns]
    if not metric_names:
        return pd.DataFrame(columns=["Dataset", "Split", "Model", *[disp for _, disp in metric_cols]])

    agg = work.groupby(["model_label", "split"])[metric_names].agg(["mean", "std", "count"])
    rows = []
    models_sorted = _sorted_model_labels(work, model_order=model_order)
    for split in sorted(work["split"].dropna().unique()):
        for model_label in models_sorted:
            row = {
                "Dataset": dataset_label,
                "Split": split,
                "Model": model_label,
            }
            try:
                stats = agg.loc[(model_label, split)]
            except KeyError:
                for _, disp in metric_cols:
                    row[disp] = "-"
                rows.append(row)
                continue
            for metric, disp in metric_cols:
                if metric not in metric_names:
                    row[disp] = "-"
                    continue
                mean = stats[(metric, "mean")]
                std = stats[(metric, "std")]
                count = stats[(metric, "count")]
                if pd.notna(mean):
                    row[disp] = f"{mean:.4f} ± {std:.4f}" if pd.notna(std) and count > 1 else f"{mean:.4f}"
                else:
                    row[disp] = "-"
            rows.append(row)
    return pd.DataFrame(rows)


def build_auroc_threshold_table(
    df: pd.DataFrame,
    *,
    thresholds: Sequence[float] = AUROC_THRESHOLDS,
    encoder_labels: Mapping[str, str] = DEFAULT_ENCODER_LABELS,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    dataset_label: str = DEFAULT_DATASET_LABEL,
) -> pd.DataFrame:
    """Aggregate mean assay AUROC across folds, then count assays above thresholds."""
    assay_cols = per_assay_metric_columns(df, PER_ASSAY_AUROC_PREFIX)
    if not assay_cols:
        return pd.DataFrame()

    work = df.copy()
    work["model_label"] = work["encoder"].map(lambda e: encoder_labels.get(str(e), str(e)))

    if "n_assays_total" in work.columns and work["n_assays_total"].notna().any():
        denominator = int(work["n_assays_total"].dropna().iloc[0])
    else:
        denominator = len(assay_cols)

    assay_means = work.groupby(["model_label", "split"])[assay_cols].mean(numeric_only=True)
    rows = []
    split_order = sorted(work["split"].dropna().unique())
    for split in split_order:
        for model_label in model_order:
            if (model_label, split) not in assay_means.index:
                continue
            scores = assay_means.loc[(model_label, split)]
            row = {
                "Dataset": dataset_label,
                "Split": split,
                "Model": model_label,
            }
            valid_scores = scores.dropna().to_numpy(dtype=float)
            for threshold in thresholds:
                count = int(np.sum(valid_scores >= threshold))
                row[format_threshold_label(threshold)] = f"{count} / {denominator} ({100.0 * count / denominator:.1f}%)"
            rows.append(row)

    # Include any non-canonical model names after the standard order.
    seen = {(row["Model"], row["Split"]) for row in rows}
    extra_models = sorted(
        set(work["model_label"].dropna().unique()) - set(model_order),
    )
    for split in split_order:
        for model_label in extra_models:
            if (model_label, split) not in assay_means.index or (model_label, split) in seen:
                continue
            scores = assay_means.loc[(model_label, split)]
            row = {
                "Dataset": dataset_label,
                "Split": split,
                "Model": model_label,
            }
            valid_scores = scores.dropna().to_numpy(dtype=float)
            for threshold in thresholds:
                count = int(np.sum(valid_scores >= threshold))
                row[format_threshold_label(threshold)] = f"{count} / {denominator} ({100.0 * count / denominator:.1f}%)"
            rows.append(row)

    return pd.DataFrame(rows)


def build_enrichment_factor_table(
    df: pd.DataFrame,
    *,
    top_fractions: Sequence[float] = EF_TOP_FRACTIONS,
    encoder_labels: Mapping[str, str] = DEFAULT_ENCODER_LABELS,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    dataset_label: str = DEFAULT_DATASET_LABEL,
) -> pd.DataFrame:
    """Aggregate mean per-assay EF across folds, then summarize across assays."""
    work = df.copy()
    work["model_label"] = work["encoder"].map(lambda e: encoder_labels.get(str(e), str(e)))
    models_sorted = _sorted_model_labels(work, model_order=model_order)
    split_order = sorted(work["split"].dropna().unique())

    rows: list[dict[str, str]] = []
    for split in split_order:
        for model_label in models_sorted:
            row = {"Dataset": dataset_label, "Split": split, "Model": model_label}
            missing_all = True
            for top_fraction in top_fractions:
                assay_cols = per_assay_metric_columns(work, ef_column_prefix(top_fraction))
                if not assay_cols:
                    row[format_ef_label(top_fraction)] = "-"
                    continue
                assay_means = (
                    work.groupby(["model_label", "split"])[assay_cols]
                    .mean(numeric_only=True)
                )
                if (model_label, split) not in assay_means.index:
                    row[format_ef_label(top_fraction)] = "-"
                    continue
                scores = assay_means.loc[(model_label, split)].dropna().to_numpy(dtype=float)
                if len(scores) == 0:
                    row[format_ef_label(top_fraction)] = "-"
                    continue
                missing_all = False
                row[format_ef_label(top_fraction)] = f"{scores.mean():.2f} ± {scores.std(ddof=0):.2f}"
            if not missing_all:
                rows.append(row)
    return pd.DataFrame(rows)


def build_per_assay_win_rank_table(
    df: pd.DataFrame,
    *,
    encoder_labels: Mapping[str, str] = DEFAULT_ENCODER_LABELS,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    dataset_label: str = DEFAULT_DATASET_LABEL,
) -> pd.DataFrame:
    """Count assay-level AUROC wins and average rank per model."""
    assay_cols = per_assay_metric_columns(df, PER_ASSAY_AUROC_PREFIX)
    if not assay_cols:
        return pd.DataFrame()

    work = df.copy()
    work["model_label"] = work["encoder"].map(lambda e: encoder_labels.get(str(e), str(e)))
    assay_means = work.groupby(["model_label", "split"])[assay_cols].mean(numeric_only=True)
    rows: list[dict[str, str]] = []

    for split in sorted(work["split"].dropna().unique()):
        try:
            split_df = assay_means.xs(split, level="split")
        except KeyError:
            continue
        present_models = [m for m in model_order if m in split_df.index]
        present_models += sorted(set(split_df.index) - set(present_models))
        if split_df.empty:
            continue

        rank_df = split_df[assay_cols].rank(axis=0, ascending=False, method="average")
        best_scores = split_df[assay_cols].max(axis=0)
        win_mask = split_df[assay_cols].eq(best_scores, axis=1)
        for model_label in present_models:
            if model_label not in split_df.index:
                continue
            model_scores = split_df.loc[model_label]
            valid_mask = model_scores.notna()
            if int(valid_mask.sum()) == 0:
                continue
            win_count = int(win_mask.loc[model_label, valid_mask].sum())
            avg_rank = float(rank_df.loc[model_label, valid_mask].mean())
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Split": split,
                    "Model": model_label,
                    "Assay wins": win_count,
                    "Avg rank": f"{avg_rank:.2f}",
                }
            )
    return pd.DataFrame(rows)


def build_assay_auroc_boxplot_frame(
    df: pd.DataFrame,
    *,
    encoder_labels: Mapping[str, str] = DEFAULT_ENCODER_LABELS,
    dataset_label: str = DEFAULT_DATASET_LABEL,
) -> pd.DataFrame:
    """Return one mean AUROC-per-assay point per model for boxplotting."""
    assay_cols = per_assay_metric_columns(df, PER_ASSAY_AUROC_PREFIX)
    if not assay_cols:
        return pd.DataFrame()

    work = df.copy()
    work["model_label"] = work["encoder"].map(lambda e: encoder_labels.get(str(e), str(e)))
    assay_means = work.groupby(["model_label", "split"])[assay_cols].mean(numeric_only=True)
    long_df = (
        assay_means.stack(dropna=True)
        .rename("AUROC")
        .reset_index()
        .rename(columns={"model_label": "Model", "split": "Split", "level_2": "Assay"})
    )
    long_df["Dataset"] = dataset_label
    long_df["Assay"] = long_df["Assay"].str.removeprefix(PER_ASSAY_AUROC_PREFIX)
    return long_df[["Dataset", "Split", "Model", "Assay", "AUROC"]]
