from __future__ import annotations

import pandas as pd

from downstream.tasks.bioactivity.report import (
    assay_threshold_summary,
    build_auroc_threshold_table,
    build_enrichment_factor_table,
    build_per_assay_win_rank_table,
    enrichment_factor,
    flatten_per_assay_metrics,
)


def test_assay_threshold_summary_uses_fixed_denominator() -> None:
    summary = assay_threshold_summary(
        {
            "assay_a": 0.71,
            "assay_b": 0.81,
            "assay_c": 0.91,
            "assay_d": 0.69,
        },
        denominator=29,
    )

    assert summary["test_auroc_ge_0p70_count"] == 3
    assert summary["test_auroc_ge_0p80_count"] == 2
    assert summary["test_auroc_ge_0p90_count"] == 1
    assert summary["test_auroc_ge_0p70_pct"] == 100.0 * 3 / 29
    assert summary["test_auroc_ge_0p80_pct"] == 100.0 * 2 / 29
    assert summary["test_auroc_ge_0p90_pct"] == 100.0 * 1 / 29


def test_flatten_per_assay_metrics_uses_stable_column_prefixes() -> None:
    flat = flatten_per_assay_metrics(
        per_assay_auroc={"assay_b": 0.8, "assay_a": 0.7},
        per_assay_auprc={"assay_b": 0.6, "assay_a": 0.5},
        per_assay_ef={0.01: {"assay_b": 3.0, "assay_a": 2.0}},
    )

    assert flat == {
        "test_auroc__assay_a": 0.7,
        "test_auroc__assay_b": 0.8,
        "test_auprc__assay_a": 0.5,
        "test_auprc__assay_b": 0.6,
        "test_ef_1p__assay_a": 2.0,
        "test_ef_1p__assay_b": 3.0,
    }


def test_build_auroc_threshold_table_averages_assays_across_folds() -> None:
    df = pd.DataFrame(
        [
            {
                "encoder": "ecfp",
                "split": "cv6",
                "n_assays_total": 4,
                "test_auroc__assay_a": 0.60,
                "test_auroc__assay_b": 0.81,
                "test_auroc__assay_c": 0.91,
                "test_auroc__assay_d": 0.89,
            },
            {
                "encoder": "ecfp",
                "split": "cv6",
                "n_assays_total": 4,
                "test_auroc__assay_a": 0.80,
                "test_auroc__assay_b": 0.79,
                "test_auroc__assay_c": 0.92,
                "test_auroc__assay_d": 0.90,
            },
            {
                "encoder": "minimol",
                "split": "cv6",
                "n_assays_total": 4,
                "test_auroc__assay_a": 0.50,
                "test_auroc__assay_b": 0.70,
                "test_auroc__assay_c": 0.82,
                "test_auroc__assay_d": 0.95,
            },
            {
                "encoder": "minimol",
                "split": "cv6",
                "n_assays_total": 4,
                "test_auroc__assay_a": 0.60,
                "test_auroc__assay_b": 0.68,
                "test_auroc__assay_c": 0.78,
                "test_auroc__assay_d": 0.96,
            },
        ]
    )

    table = build_auroc_threshold_table(df)
    ecfp = table.loc[table["Model"] == "ECFP4 (paper)"].iloc[0]
    minimol = table.loc[table["Model"] == "MiniMol"].iloc[0]

    # ECFP assay means = [0.70, 0.80, 0.915, 0.895]
    assert ecfp["AUROC ≥ 0.7"] == "4 / 4 (100.0%)"
    assert ecfp["AUROC ≥ 0.8"] == "3 / 4 (75.0%)"
    assert ecfp["AUROC ≥ 0.9"] == "1 / 4 (25.0%)"

    # MiniMol assay means = [0.55, 0.69, 0.80, 0.955]
    assert minimol["AUROC ≥ 0.7"] == "2 / 4 (50.0%)"
    assert minimol["AUROC ≥ 0.8"] == "2 / 4 (50.0%)"
    assert minimol["AUROC ≥ 0.9"] == "1 / 4 (25.0%)"


def test_enrichment_factor_matches_ranked_hit_rate_definition() -> None:
    y_true = [1, 0, 0, 1, 0]
    y_score = [0.95, 0.90, 0.80, 0.10, 0.05]
    # Top 40% => top 2 compounds, containing 1 active. Hit rate = 0.5.
    # Overall hit rate = 2 / 5 = 0.4, so EF@40% = 1.25.
    assert enrichment_factor(y_true, y_score, top_fraction=0.40) == 1.25


def test_build_enrichment_factor_table_summarizes_assay_mean_and_std() -> None:
    df = pd.DataFrame(
        [
            {
                "encoder": "ecfp",
                "split": "cv6",
                "test_ef_1p__assay_a": 2.0,
                "test_ef_1p__assay_b": 4.0,
                "test_ef_5p__assay_a": 1.5,
                "test_ef_5p__assay_b": 2.5,
                "test_ef_10p__assay_a": 1.2,
                "test_ef_10p__assay_b": 2.2,
            },
            {
                "encoder": "ecfp",
                "split": "cv6",
                "test_ef_1p__assay_a": 4.0,
                "test_ef_1p__assay_b": 6.0,
                "test_ef_5p__assay_a": 2.5,
                "test_ef_5p__assay_b": 3.5,
                "test_ef_10p__assay_a": 1.6,
                "test_ef_10p__assay_b": 2.6,
            },
        ]
    )

    table = build_enrichment_factor_table(df)
    row = table.iloc[0]
    # assay means for EF@1% = [3.0, 5.0] => mean 4.00 std 1.00
    assert row["EF@1%"] == "4.00 ± 1.00"
    assert row["EF@5%"] == "2.50 ± 0.50"
    assert row["EF@10%"] == "1.90 ± 0.50"


def test_build_per_assay_win_rank_table_counts_wins_and_average_rank() -> None:
    df = pd.DataFrame(
        [
            {
                "encoder": "ecfp",
                "split": "cv6",
                "test_auroc__assay_a": 0.80,
                "test_auroc__assay_b": 0.70,
            },
            {
                "encoder": "minimol",
                "split": "cv6",
                "test_auroc__assay_a": 0.90,
                "test_auroc__assay_b": 0.60,
            },
            {
                "encoder": "mole",
                "split": "cv6",
                "test_auroc__assay_a": 0.70,
                "test_auroc__assay_b": 0.95,
            },
        ]
    )

    table = build_per_assay_win_rank_table(df)
    minimol = table.loc[table["Model"] == "MiniMol"].iloc[0]
    mole = table.loc[table["Model"] == "MolE"].iloc[0]
    ecfp = table.loc[table["Model"] == "ECFP4 (paper)"].iloc[0]

    assert minimol["Assay wins"] == 1
    assert minimol["Avg rank"] == "2.00"
    assert mole["Assay wins"] == 1
    assert mole["Avg rank"] == "2.00"
    assert ecfp["Assay wins"] == 0
    assert ecfp["Avg rank"] == "2.00"
