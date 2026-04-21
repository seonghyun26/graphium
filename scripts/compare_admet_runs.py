"""Compare scratch TDC ADMET results between two model tags.

Reads ``results/experiment_results.csv`` and compares the latest row per task
for each model/tag combination. The metric used for each task matches the
benchmark primary metric documented in the repo notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


PRIMARY_METRICS: Dict[str, Tuple[str, str]] = {
    "caco2_wang": ("graph_caco2_wang/mae/test", "lower"),
    "hia_hou": ("graph_hia_hou/auroc/test", "higher"),
    "pgp_broccatelli": ("graph_pgp_broccatelli/auroc/test", "higher"),
    "bioavailability_ma": ("graph_bioavailability_ma/auroc/test", "higher"),
    "lipophilicity_astrazeneca": ("graph_lipophilicity_astrazeneca/mae/test", "lower"),
    "solubility_aqsoldb": ("graph_solubility_aqsoldb/mae/test", "lower"),
    "bbb_martins": ("graph_bbb_martins/auroc/test", "higher"),
    "ppbr_az": ("graph_ppbr_az/mae/test", "lower"),
    "vdss_lombardo": ("graph_vdss_lombardo/spearman/test", "higher"),
    "cyp2d6_veith": ("graph_cyp2d6_veith/auprc/test", "higher"),
    "cyp3a4_veith": ("graph_cyp3a4_veith/auprc/test", "higher"),
    "cyp2c9_veith": ("graph_cyp2c9_veith/auprc/test", "higher"),
    "cyp2c9_substrate_carbonmangels": ("graph_cyp2c9_substrate_carbonmangels/auprc/test", "higher"),
    "cyp2d6_substrate_carbonmangels": ("graph_cyp2d6_substrate_carbonmangels/auprc/test", "higher"),
    "cyp3a4_substrate_carbonmangels": ("graph_cyp3a4_substrate_carbonmangels/auprc/test", "higher"),
    "half_life_obach": ("graph_half_life_obach/spearman/test", "higher"),
    "clearance_hepatocyte_az": ("graph_clearance_hepatocyte_az/spearman/test", "higher"),
    "clearance_microsome_az": ("graph_clearance_microsome_az/spearman/test", "higher"),
    "ld50_zhu": ("graph_ld50_zhu/mae/test", "lower"),
    "herg": ("graph_herg/auroc/test", "higher"),
    "ames": ("graph_ames/auroc/test", "higher"),
    "dili": ("graph_dili/auroc/test", "higher"),
}


def latest_rows(df: pd.DataFrame, model_tag: str) -> pd.DataFrame:
    tags = df["wandb_tags"].astype(str)
    mask = tags.str.contains(model_tag) & tags.str.contains("scratch") & tags.str.contains("admet")
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.sort_values("timestamp").drop_duplicates("task", keep="last")
    return sub


def fmt(v):
    return "NA" if pd.isna(v) else f"{float(v):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/experiment_results.csv")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    cand = latest_rows(df, args.candidate).set_index("task")
    base = latest_rows(df, args.baseline).set_index("task")

    lines = [
        f"# TDC ADMET — {args.candidate} vs {args.baseline}",
        "",
        f"- Candidate tasks finished: **{cand.index.nunique()} / {len(PRIMARY_METRICS)}**",
        f"- Baseline tasks available: **{base.index.nunique()} / {len(PRIMARY_METRICS)}**",
        "",
        "| Task | Metric | Direction | Candidate | Baseline | Delta | Winner |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    wins = 0
    losses = 0
    ties = 0
    for task, (metric, direction) in PRIMARY_METRICS.items():
        c = cand[metric].get(task) if metric in cand.columns else float("nan")
        b = base[metric].get(task) if metric in base.columns else float("nan")
        if pd.isna(c) or pd.isna(b):
            winner = "NA"
            delta = "NA"
        else:
            if direction == "higher":
                d = float(c) - float(b)
                if d > 0:
                    wins += 1
                    winner = args.candidate
                elif d < 0:
                    losses += 1
                    winner = args.baseline
                else:
                    ties += 1
                    winner = "tie"
            else:
                d = float(b) - float(c)
                if d > 0:
                    wins += 1
                    winner = args.candidate
                elif d < 0:
                    losses += 1
                    winner = args.baseline
                else:
                    ties += 1
                    winner = "tie"
            delta = f"{d:+.4f}"
        lines.append(
            f"| {task} | `{metric}` | {direction} | {fmt(c)} | {fmt(b)} | {delta} | {winner} |"
        )

    lines += [
        "",
        f"Summary: **{wins} wins**, **{losses} losses**, **{ties} ties** for `{args.candidate}` on tasks where both models have results.",
        "",
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
