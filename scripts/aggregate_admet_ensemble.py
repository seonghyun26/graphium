#!/usr/bin/env python
"""Post-hoc ensemble aggregation for the ADMET N-seed sweep.

Reads ``predictions.pt`` files produced by ``scripts/00_finetune_admet_ensemble.sh``
(via the predictor patch keyed by ``GRAPHIUM_DUMP_PREDS=1``), groups them by
(model, pretrain, task), averages predictions across seeds (the MiniMol /
GRAM-DTI style ensemble), recomputes metrics on the averaged scores, and
writes one row per (model, pretrain, task) to
``results/experiment_results_ensemble.csv``.

Inputs
------
Each Hydra run dir contains:
    predictions.pt       ``{task: {preds: (N, *), targets: (N, *)}}``
    .hydra/config.yaml   records ``model``, ``constants.seed``, ``finetuning.pretrained_model``
    test_results.yaml    per-seed metrics (only used to detect classification vs regression)

Outputs
-------
One row per (model, pretrain, task) with columns:
    model, pretrain, task, n_seeds, primary_metric, <metric>_ensemble, <metric>_mean, <metric>_std, ...

Metric selection
----------------
Per-task classification vs regression detected from the target tensor's values:
    binary target {0, 1} → classification → AUROC / AUPRC primary
    continuous target    → regression     → Pearson / MAE primary

Usage
-----
    python scripts/aggregate_admet_ensemble.py \\
        --model pairmixer_12M --pretrain scratch

    # Restrict to a task subset
    python scripts/aggregate_admet_ensemble.py \\
        --model pairmixer_12M --pretrain toymix \\
        --tasks caco2_wang hia_hou

    # Custom outputs dir / result path
    python scripts/aggregate_admet_ensemble.py \\
        --model pairmixer_12M --pretrain scratch \\
        --runs-glob 'outputs/2026-04-*_ft_*' \\
        --out-csv results/custom_ensemble.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunMeta:
    """Parsed config + prediction for one per-seed graphium-train run."""
    run_dir:  Path
    model:    Optional[str]
    pretrain: Optional[str]
    seed:     Optional[int]
    task:     Optional[str]
    preds_path: Path


# ── Config parsing ──────────────────────────────────────────────────────────


def _infer_pretrain(cfg: dict, ckpt_path: Optional[str]) -> Optional[str]:
    """Extract a pretrain label from the Hydra config.

    Priority:
      1. An explicit ``method`` wandb tag (format ``'method:<name>'``) if the
         base scripts set one — they don't yet, so fall back to:
      2. ``finetuning.pretrained_model`` path — parent-of-parent dir name is the
         pretrain corpus (e.g. ``toymix-dti-esmc-v2``).
      3. Absence of ``finetuning`` section → "scratch".
    """
    if ckpt_path:
        p = Path(ckpt_path)
        # Typical: models_checkpoints/<pretrain>/<model>/<ts>/last.ckpt  → parts[-4]
        parts = p.parts
        for marker in ("models_checkpoints",):
            if marker in parts:
                i = parts.index(marker)
                if i + 1 < len(parts):
                    return parts[i + 1]
    if "finetuning" not in cfg:
        return "scratch"
    return None


def _read_hydra_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text())
    except yaml.YAMLError:
        return {}


def _extract_meta(run_dir: Path) -> RunMeta:
    preds_path = run_dir / "predictions.pt"
    cfg = _read_hydra_cfg(run_dir)

    model = cfg.get("model") if isinstance(cfg, dict) else None
    seed = None
    constants = cfg.get("constants", {}) if isinstance(cfg, dict) else {}
    if isinstance(constants, dict):
        seed = constants.get("seed")
        task = constants.get("task")
    else:
        task = None

    ft = cfg.get("finetuning", {}) if isinstance(cfg, dict) else {}
    ckpt_path = None
    if isinstance(ft, dict):
        ckpt_path = ft.get("pretrained_model")
    pretrain = _infer_pretrain(cfg, ckpt_path)

    return RunMeta(
        run_dir=run_dir, model=model, pretrain=pretrain, seed=seed,
        task=task, preds_path=preds_path,
    )


# ── Metrics ─────────────────────────────────────────────────────────────────


def _is_classification(targets: np.ndarray) -> bool:
    """Binary {0,1} (after NaN removal) → classification; else regression."""
    finite = targets[np.isfinite(targets)]
    if finite.size == 0:
        return False
    uniq = np.unique(finite)
    return uniq.size <= 2 and set(uniq.tolist()).issubset({0.0, 1.0})


def _metrics_classification(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    proba = 1.0 / (1.0 + np.exp(-logits))  # sigmoid on pre-averaged logits
    try:
        auroc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, proba))
    except ValueError:
        auprc = float("nan")
    return {"auroc": auroc, "auprc": auprc}


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pear = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    spear = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    return {
        "mae":       float(mean_absolute_error(y_true, y_pred)),
        "mse":       float(mean_squared_error(y_true, y_pred)),
        "pearsonr":  float(pear) if pear is not None else float("nan"),
        "spearmanr": float(spear) if spear is not None else float("nan"),
    }


# ── Ensemble aggregation ────────────────────────────────────────────────────


def _load_preds(meta: RunMeta) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Return ``{task: {preds: (N,), targets: (N,)}}`` for this run.

    The ADMET finetune pipeline trains one task per run, so the .pt is
    expected to have a single key. Multi-task dumps are handled too.
    """
    if not meta.preds_path.exists():
        return None
    try:
        data = torch.load(meta.preds_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  WARN: could not load {meta.preds_path}: {exc}", file=sys.stderr)
        return None
    out = {}
    for task, td in data.items():
        preds = td["preds"].detach().cpu().float().numpy().reshape(-1)
        targets = td["targets"].detach().cpu().float().numpy().reshape(-1)
        # NaN-target masking (graphium keeps missing labels as NaN in multi-task)
        m = np.isfinite(targets)
        out[task] = {"preds": preds[m], "targets": targets[m]}
    return out


def _compute_ensemble_row(
    model: str, pretrain: str, task: str, runs: List[RunMeta],
) -> Optional[Dict[str, object]]:
    """Average preds across seeds, compute ensemble metrics + per-seed stats."""
    per_seed_preds: List[np.ndarray] = []
    per_seed_targets: List[np.ndarray] = []
    seeds_seen: List[int] = []
    for r in runs:
        loaded = _load_preds(r)
        if loaded is None or task not in loaded:
            continue
        pt = loaded[task]
        per_seed_preds.append(pt["preds"])
        per_seed_targets.append(pt["targets"])
        if r.seed is not None:
            seeds_seen.append(r.seed)

    if not per_seed_preds:
        return None

    # Align shapes — graphium's test set is deterministic per task, so all seeds
    # should produce the same length / order.
    if len({len(p) for p in per_seed_preds}) != 1:
        print(
            f"  WARN: {model}/{pretrain}/{task}: mismatched prediction lengths "
            f"{[len(p) for p in per_seed_preds]}; skipping.",
            file=sys.stderr,
        )
        return None

    y_true = per_seed_targets[0]
    preds_stack = np.stack(per_seed_preds, axis=0)  # (n_seeds, N)
    ensemble_pred = preds_stack.mean(axis=0)

    is_clf = _is_classification(y_true)
    if is_clf:
        ens = _metrics_classification(y_true, ensemble_pred)
        per_seed = [_metrics_classification(y_true, p) for p in per_seed_preds]
        primary = "auroc"
    else:
        ens = _metrics_regression(y_true, ensemble_pred)
        per_seed = [_metrics_regression(y_true, p) for p in per_seed_preds]
        primary = "pearsonr"

    row = {
        "model":    model,
        "pretrain": pretrain,
        "task":     task,
        "n_seeds":  int(preds_stack.shape[0]),
        "n_test":   int(len(y_true)),
        "task_type": "classification" if is_clf else "regression",
        "primary_metric": primary,
    }
    for k, v in ens.items():
        row[f"{k}_ensemble"] = v
    # Per-seed mean ± std
    for k in ens.keys():
        vals = np.array([m[k] for m in per_seed if np.isfinite(m[k])], dtype=np.float64)
        row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        row[f"{k}_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
    row["seeds"] = ",".join(str(s) for s in sorted(seeds_seen))
    return row


def _append_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        merged = pd.concat([prev, new_df], ignore_index=True, sort=False)
        merged.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)


# ── Discovery + main ────────────────────────────────────────────────────────


def discover_runs(runs_glob: str) -> List[RunMeta]:
    run_dirs = sorted(Path().glob(runs_glob))
    metas: List[RunMeta] = []
    for d in run_dirs:
        if not d.is_dir():
            continue
        if not (d / "predictions.pt").exists():
            continue
        metas.append(_extract_meta(d))
    return metas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Filter: Hydra ``model`` config name")
    p.add_argument("--pretrain", required=True,
                   help="Filter: 'scratch' OR a pretrain label parsed from ckpt path")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Optional task subset. Default: every task encountered.")
    p.add_argument("--runs-glob", default="outputs/**/*_ft_*",
                   help="Glob for per-seed run directories (default: ``outputs/**/*_ft_*``).")
    p.add_argument("--out-csv", default="results/experiment_results_ensemble.csv")
    args = p.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        all_runs = discover_runs(args.runs_glob)

    # Filter by model + pretrain first.
    filtered = [r for r in all_runs if r.model == args.model and r.pretrain == args.pretrain]
    if not filtered:
        print(f"No runs matched model={args.model!r} pretrain={args.pretrain!r}.", file=sys.stderr)
        print(f"Candidate dirs scanned: {len(all_runs)}", file=sys.stderr)
        sys.exit(1)

    # Group by task. The predictions.pt has the authoritative task; the Hydra
    # cfg's ``constants.task`` is a convenient pre-filter.
    by_task: Dict[str, List[RunMeta]] = {}
    for r in filtered:
        # Load once here only to discover tasks; re-loaded inside _compute_ensemble_row.
        loaded = _load_preds(r)
        if loaded is None:
            continue
        for t in loaded.keys():
            if args.tasks and t not in args.tasks:
                continue
            by_task.setdefault(t, []).append(r)

    if not by_task:
        print(f"No tasks discovered in {len(filtered)} matching runs.", file=sys.stderr)
        sys.exit(1)

    out_csv = ROOT / args.out_csv if not Path(args.out_csv).is_absolute() else Path(args.out_csv)

    print(f"== aggregating model={args.model} pretrain={args.pretrain}, {len(by_task)} tasks")
    n_written = 0
    for task, runs in sorted(by_task.items()):
        row = _compute_ensemble_row(args.model, args.pretrain, task, runs)
        if row is None:
            continue
        _append_row(out_csv, row)
        primary = row["primary_metric"]
        ens = row[f"{primary}_ensemble"]
        mean = row[f"{primary}_mean"]
        std = row[f"{primary}_std"]
        print(
            f"  {task:40s}  n_seeds={row['n_seeds']:2d}  "
            f"ensemble-{primary}={ens:.4f}  mean-{primary}={mean:.4f} ± {std:.4f}"
        )
        n_written += 1

    print(f"\nWrote {n_written} ensemble rows to {out_csv}")


if __name__ == "__main__":
    main()
