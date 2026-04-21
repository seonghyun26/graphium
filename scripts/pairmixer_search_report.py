"""Aggregate pairmixer_12M_search sweep results and write an MD report.

Reads results/experiment_results.csv, filters runs tagged with one of
the search phases, computes a per-task normalised "helpfulness" score
(higher-is-better within the comparison set), and reports the best
(config, pair_init) combination along with per-task/per-variant tables.

Usage (from graphium/ directory):
    python scripts/pairmixer_search_report.py \
        --phase arch        # reads cfg_B..cfg_E, pair_init=opm
    python scripts/pairmixer_search_report.py \
        --phase pairinit --best-cfg C
    python scripts/pairmixer_search_report.py --phase all    # both
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
from notebook_utils import TASK_METRICS, find_metric_column  # noqa: E402

TASKS = [
    "caco2_wang",
    "bbb_martins",
    "cyp2d6_veith",
    "clearance_microsome_az",
    "ld50_zhu",
]


def _extract_cfg_pi(tags: str) -> tuple[str | None, str | None]:
    cfg = re.search(r"cfg_([A-Za-z0-9]+)", tags or "")
    pi = re.search(r"pi_([A-Za-z0-9_]+)", tags or "")
    return (cfg.group(1) if cfg else None, pi.group(1) if pi else None)


def load_search(csv: Path, phase: str) -> pd.DataFrame:
    df = pd.read_csv(csv, low_memory=False)
    tags = df.get("wandb_tags", "").astype(str)
    mask = tags.str.contains("pairmixer_auto") & tags.str.contains(f"{phase}_search")
    sub = df[mask].copy()
    sub["cfg"], sub["pi"] = zip(*sub["wandb_tags"].astype(str).map(_extract_cfg_pi))
    return sub


def primary_metric(row, df_cols) -> float | None:
    task = row["task"]
    spec = TASK_METRICS.get(task)
    if spec is None:
        return None
    metric_name, _ = spec
    col = find_metric_column(pd.DataFrame(columns=df_cols), task, metric_name)
    return row.get(col) if col else None


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["task"] not in TASKS:
            continue
        spec = TASK_METRICS.get(r["task"])
        if spec is None:
            continue
        metric, direction = spec
        col = find_metric_column(df, r["task"], metric)
        if col is None or col not in df.columns:
            continue
        val = r.get(col)
        if pd.isna(val):
            continue
        rows.append({
            "cfg": r.get("cfg"),
            "pi": r.get("pi"),
            "task": r["task"],
            "metric": metric,
            "value": float(val),
            "direction": direction,
            "timestamp": r.get("timestamp"),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # Dedup latest run per (cfg, pi, task)
    res["timestamp"] = pd.to_datetime(res["timestamp"], errors="coerce")
    res = res.sort_values("timestamp").drop_duplicates(
        subset=["cfg", "pi", "task"], keep="last"
    ).reset_index(drop=True)
    return res


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Within each task, map primary metric to [0, 1] where 1 = best."""
    parts = []
    for task, grp in df.groupby("task"):
        vals = grp["value"].astype(float)
        if grp.iloc[0]["direction"] == "lower":
            vals = -vals
        vmin, vmax = vals.min(), vals.max()
        norm = pd.Series(0.5, index=grp.index) if vmax == vmin else (vals - vmin) / (vmax - vmin)
        parts.append(norm)
    df = df.copy()
    df["norm"] = pd.concat(parts).astype(float)
    return df


def summarise(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Per-(cfg,pi) summary: helpfulness = mean of normalised score across tasks."""
    # Per-task pivot
    pivot = df.pivot_table(
        index=group_cols, columns="task", values="value", aggfunc="last"
    )
    norm_pivot = df.pivot_table(
        index=group_cols, columns="task", values="norm", aggfunc="last"
    )
    summary = pd.DataFrame({
        "n_tasks": norm_pivot.count(axis=1),
        "helpfulness": norm_pivot.mean(axis=1),
    }).sort_values("helpfulness", ascending=False)
    return summary, pivot, norm_pivot


def render_md(arch_sum, arch_pivot, arch_norm, pi_sum, pi_pivot, pi_norm, best_cfg):
    out = []
    out.append("# PairMixer ~12M Architecture + Pair-Init Search\n")
    out.append("*Generated from `results/experiment_results.csv` by "
               "`scripts/pairmixer_search_report.py`.*\n")
    out.append("\n## Setup\n")
    out.append("- Model: `pairmixer_auto` (scratch, seed=0, 100 epochs, GPU 0)\n")
    out.append("- Tasks (5, one per ADMET category): "
               + ", ".join(f"`{t}`" for t in TASKS) + "\n")
    out.append("- Virtual node: `logsum`; `pre_nn_edges.out_dim=64`; no `torch.compile`.\n")
    out.append("- Helpfulness = mean of per-task [0,1]-normalised score "
               "(higher-is-better, sign-flipped for MAE).\n")

    if arch_sum is not None and not arch_sum.empty:
        out.append("\n## Phase 1 — Architecture sweep\n")
        out.append(
            "| cfg | D_s | D_z | depth | params | helpfulness | tasks_ok |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        arch_meta = {
            "B": (128, 160, 16, "12.0 M"),
            "C": (128, 192, 12, "12.7 M"),
            "D": (128, 256,  7, "13.2 M"),
            "E": (192, 192, 10, "11.3 M"),
        }
        for cfg, row in arch_sum.iterrows():
            ds, dz, dp, pm = arch_meta.get(cfg, ("", "", "", ""))
            out.append(
                f"| **{cfg}** | {ds} | {dz} | {dp} | {pm} | "
                f"{row['helpfulness']:.3f} | {int(row['n_tasks'])}/5 |\n"
            )
        out.append("\n### Per-task raw scores (phase 1)\n\n```\n")
        out.append(arch_pivot.round(4).to_string() + "\n```\n")

    if pi_sum is not None and not pi_sum.empty:
        out.append(f"\n## Phase 2 — Pair-Init sweep (on best arch: `{best_cfg}`)\n")
        out.append(
            "| pair_init | helpfulness | tasks_ok |\n"
            "|---|---|---|\n"
        )
        pi_desc = {
            "opm": "OPM-only (baseline)",
            "struct": "+ graph_distance + adjacency",
            "struct_ef": "+ graph_dist + adj + edge_feat",
            "struct_path": "+ graph_dist + adj + path_edge",
        }
        for pi, row in pi_sum.iterrows():
            out.append(
                f"| **{pi}** — {pi_desc.get(pi, '')} | "
                f"{row['helpfulness']:.3f} | {int(row['n_tasks'])}/5 |\n"
            )
        out.append("\n### Per-task raw scores (phase 2)\n\n```\n")
        out.append(pi_pivot.round(4).to_string() + "\n```\n")

    # Best combo
    if arch_sum is not None and not arch_sum.empty:
        best = arch_sum.index[0]
        out.append(f"\n## Best architecture: `{best}`\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["arch", "pairinit", "arch2", "all"], default="all")
    ap.add_argument("--best-cfg", default=None,
                    help="Override best cfg for the phase-2 header.")
    ap.add_argument("--out", default=str(ROOT / "results" / "pairmixer_12M_search.md"))
    ap.add_argument("--csv", default=str(ROOT / "results" / "experiment_results.csv"))
    args = ap.parse_args()

    csv_path = Path(args.csv)

    arch_sum = arch_pivot = arch_norm = None
    pi_sum = pi_pivot = pi_norm = None
    arch2_sum = arch2_pivot = None
    best_cfg = args.best_cfg

    if args.phase in ("arch", "all"):
        arch_df = load_search(csv_path, "arch")
        t = build_table(arch_df)
        if not t.empty:
            t = normalise(t)
            arch_sum, arch_pivot, arch_norm = summarise(t, ["cfg"])
            if best_cfg is None and not arch_sum.empty:
                best_cfg = arch_sum.index[0]

    if args.phase in ("pairinit", "all"):
        pi_df = load_search(csv_path, "pairinit")
        t = build_table(pi_df)
        if not t.empty:
            t = normalise(t)
            # Group by (cfg, pi) to separate cfg=D and cfg=E runs
            pi_sum, pi_pivot, pi_norm = summarise(t, ["cfg", "pi"])

    if args.phase in ("arch2", "all"):
        arch2_df = load_search(csv_path, "arch2")
        t = build_table(arch2_df)
        if not t.empty:
            t = normalise(t)
            arch2_sum, arch2_pivot, _ = summarise(t, ["cfg"])

    md = render_md(arch_sum, arch_pivot, arch_norm,
                   pi_sum, pi_pivot, pi_norm, best_cfg or "?")
    # Append phase-3 section if present
    if arch2_sum is not None and not arch2_sum.empty:
        md += "\n## Phase 3 — Refined arch (≤12M) + expanded features\n\n"
        md += "Features added over baseline (toymix.yaml): atom `hybridization, chirality` (onehot); "
        md += "`mass, electronegativity, vdw-radius, num-ring` (float); edge `conjugated`.\n\n"
        md += "| cfg | D_s | D_z | depth | params | helpfulness | tasks_ok |\n|---|---|---|---|---|---|---|\n"
        meta = {
            "F1": (96, 256, 6, "11.5 M"),
            "F2": (128, 256, 6, "11.6 M"),
            "F3": (96, 224, 8, "11.6 M"),
            "F4": (192, 192, 11, "~11.8 M"),
        }
        for cfg, row in arch2_sum.iterrows():
            ds, dz, dp, pm = meta.get(cfg, ("", "", "", ""))
            md += f"| **{cfg}** | {ds} | {dz} | {dp} | {pm} | {row['helpfulness']:.3f} | {int(row['n_tasks'])}/5 |\n"
        md += "\n### Per-task raw scores (phase 3)\n\n```\n"
        md += arch2_pivot.round(4).to_string() + "\n```\n"

    Path(args.out).write_text(md)
    print(f"Wrote {args.out}")
    if arch_sum is not None:
        print("\nPhase 1 (architecture):")
        print(arch_sum.to_string())
    if pi_sum is not None:
        print("\nPhase 2 (pair_init, by cfg):")
        print(pi_sum.to_string())
        print("\nPhase 2 raw pivot:")
        print(pi_pivot.round(4).to_string())
    if arch2_sum is not None:
        print("\nPhase 3 (arch refined):")
        print(arch2_sum.to_string())
        print("\nPhase 3 raw pivot:")
        print(arch2_pivot.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
