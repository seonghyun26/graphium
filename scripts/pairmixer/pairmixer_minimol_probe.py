#!/usr/bin/env python
"""MiniMol-faithful probe on top of frozen PairMixer fingerprints.

Replicates the recipe from MiniMol's leaderboard submission script
(``graphcore-research/minimol/tdc_leaderboard_submission.py``).

Two modes:

  ``--sweep``     run MiniMol's 18-candidate HP grid per task, pick the
                  HP with the smallest mean val loss across K folds,
                  and write the chosen HPs to a JSON file. No test
                  evaluation in this mode.

  default (eval)  run the MiniMol final-eval recipe with HPs sourced
                  from (in priority order): ``--sweep-results`` JSON,
                  the auto-detected sibling JSON
                  ``results/pairmixer_minimol_probe_sweep_<label>.json``,
                  or the verbatim MiniMol ``SWEEP_RESULTS`` table. The
                  output is a row per task in the eval CSV.

Final-eval pipeline per task (from MiniMol Algorithm 1):
  1. Compute PairMixer fingerprint for every (train_val ∪ test) SMILES
     once and cache to disk.
  2. For rep in 1..REPETITIONS (=5):
       For fold in 1..ENSEMBLE_SIZE (=5):
         seed = cantor_pairing(rep, REPETITIONS+fold)
         (train, val) = group.get_train_valid_split(seed=seed)
         Train MiniMol-style TaskHead for EPOCHS, save best by val loss.
       Logit-average the K fold-models on the test set → this rep's
       test predictions.
  3. group.evaluate_many(predictions_list) → mean ± std across reps,
     using TDC's per-task primary metric (AUROC, MAE, Spearman, etc.).

Sweep pipeline per task (Algorithm 1, outer HP loop):
  1. For each of 18 candidates {hidden_dim, depth, lr} × combine=True:
       For fold in 1..K (=5):
         (train, val) = get_train_valid_split(seed=cantor_pairing(1, K+fold))
         Train, save best val loss (best epoch).
       Mean of the K best-val-losses → this HP's score.
  2. Argmin over the 18 mean scores → ``best`` HP for the task.

The probe trains 25 small MLPs (5 reps × 5 folds) per task in eval mode
and 90 (18 HPs × 5 folds × 1 rep) in sweep mode. Each MLP is ≤10 M params
(hidden_dim=2048 case) and trains in seconds on GPU since the 512-d
fingerprints are pre-computed.

To compare two pretraining schemes, run twice with different ``--ckpt``
and ``--pretrain-label`` (different ``--device`` lets you parallelise on
different GPUs). The CSVs at ``--output-csv`` and ``--sweep-csv`` are
append-only so successive runs accumulate rows.

Usage:
    # arm A: PairMixer 12M pretrained on toymix only
    python scripts/pairmixer/pairmixer_minimol_probe.py \\
        --ckpt models_checkpoints/small-dataset/pairmixer_12M/.../last.ckpt \\
        --pretrain-label toymix-only \\
        --tasks caco2_wang \\
        --device cuda:5 &

    # arm B: PairMixer 12M pretrained on toymix + DTI-ESMC v2
    python scripts/pairmixer/pairmixer_minimol_probe.py \\
        --ckpt models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/.../last.ckpt \\
        --pretrain-label toymix-dti-esmc-v2 \\
        --tasks caco2_wang \\
        --device cuda:7 &
    wait
"""
from __future__ import annotations

# ---- CPU thread caps (must come before numpy / torch import) ----
import os
import sys as _sys


def _cap_cpu_threads() -> int:
    n = int(os.environ.get("PAIRMIXER_PROBE_CPU_THREADS", "8"))
    argv = _sys.argv
    for i, tok in enumerate(argv):
        if tok == "--cpu-threads" and i + 1 < len(argv):
            n = int(argv[i + 1]); break
        if tok.startswith("--cpu-threads="):
            n = int(tok.split("=", 1)[1]); break
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
    return n


_CPU_THREADS = _cap_cpu_threads()

# ---- standard imports ----
import argparse
import json
import math
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

try:
    torch.set_num_threads(_CPU_THREADS)
    torch.set_num_interop_threads(max(1, _CPU_THREADS // 2))
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

from downstream.model import load_encoder  # noqa: E402


# ── SWEEP_RESULTS — verbatim from minimol/tdc_leaderboard_submission.py L143-167
# 18 candidates ≡ {hidden_dim ∈ (512, 1024, 2048)} × {depth ∈ (3, 4)} × {lr ∈ (1e-4, 3e-4, 5e-4)}
# combine=True won across every task in their offline sweep.
SWEEP_RESULTS: Dict[str, Dict[str, object]] = {
    "caco2_wang":                       {"hidden_dim": 2048, "depth": 4, "combine": True, "lr": 5e-4},
    "hia_hou":                          {"hidden_dim": 2048, "depth": 4, "combine": True, "lr": 3e-4},
    "pgp_broccatelli":                  {"hidden_dim": 512,  "depth": 4, "combine": True, "lr": 3e-4},
    "bioavailability_ma":               {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 3e-4},
    "lipophilicity_astrazeneca":        {"hidden_dim": 2048, "depth": 4, "combine": True, "lr": 5e-4},
    "solubility_aqsoldb":               {"hidden_dim": 1024, "depth": 4, "combine": True, "lr": 5e-4},
    "bbb_martins":                      {"hidden_dim": 2048, "depth": 3, "combine": True, "lr": 1e-4},
    "ppbr_az":                          {"hidden_dim": 2048, "depth": 4, "combine": True, "lr": 3e-4},
    "vdss_lombardo":                    {"hidden_dim": 1024, "depth": 4, "combine": True, "lr": 1e-4},
    "cyp2d6_veith":                     {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 1e-4},
    "cyp3a4_veith":                     {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 1e-4},
    "cyp2c9_veith":                     {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 1e-4},
    "cyp2d6_substrate_carbonmangels":   {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 1e-4},
    "cyp3a4_substrate_carbonmangels":   {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 1e-4},
    "cyp2c9_substrate_carbonmangels":   {"hidden_dim": 1024, "depth": 3, "combine": True, "lr": 5e-4},
    "half_life_obach":                  {"hidden_dim": 1024, "depth": 3, "combine": True, "lr": 3e-4},
    "clearance_microsome_az":           {"hidden_dim": 1024, "depth": 4, "combine": True, "lr": 5e-4},
    "clearance_hepatocyte_az":          {"hidden_dim": 2048, "depth": 4, "combine": True, "lr": 5e-4},
    "herg":                             {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 3e-4},
    "ames":                             {"hidden_dim": 512,  "depth": 3, "combine": True, "lr": 1e-4},
    "dili":                             {"hidden_dim": 512,  "depth": 4, "combine": True, "lr": 5e-4},
    "ld50_zhu":                         {"hidden_dim": 1024, "depth": 4, "combine": True, "lr": 1e-4},
}


# ── 18-candidate HP grid (sweep mode) ────────────────────────────────────────
# {hidden_dim ∈ (512, 1024, 2048)} × {depth ∈ (3, 4)} × {lr ∈ (1e-4, 3e-4, 5e-4)}
# combine=True is fixed (every task in MiniMol's SWEEP_RESULTS picked combine=True).
# This is exactly MiniMol's Algorithm-1 candidate set ([hp_1, ..., hp_18]).
SWEEP_GRID: List[Dict[str, object]] = [
    {"hidden_dim": h, "depth": d, "combine": True, "lr": lr}
    for h in (512, 1024, 2048)
    for d in (3, 4)
    for lr in (1e-4, 3e-4, 5e-4)
]
assert len(SWEEP_GRID) == 18, f"expected 18 candidates, got {len(SWEEP_GRID)}"


class TaskHead(nn.Module):
    """Verbatim port of MiniMol's TaskHead (BatchNorm + ReLU + Dropout, optional
    skip-concat with the input fingerprint before the final linear).

    Supports ``depth ∈ {3, 4}`` (MiniMol's original grid):
      depth=3 → 2 hidden blocks
      depth=4 → 3 hidden blocks
    """

    def __init__(self, hidden_dim: int = 512, input_dim: int = 512,
                 dropout: float = 0.1, depth: int = 3, combine: bool = True):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.final_dense = nn.Linear(input_dim + hidden_dim, 1) if combine else nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        x = self.dense1(x); x = self.bn1(x); x = F.relu(x); x = self.dropout(x)
        x = self.dense2(x); x = self.bn2(x); x = F.relu(x); x = self.dropout(x)
        if self.depth == 4:
            x = self.dense3(x); x = self.bn3(x); x = F.relu(x); x = self.dropout(x)
        x = torch.cat((x, original_x), dim=1) if self.combine else x
        x = self.final_dense(x)
        return x


def cantor_pairing(a: int, b: int) -> int:
    """Unique combined seed for nested (rep, fold) loops — same as MiniMol."""
    return (a + b) * (a + b + 1) // 2 + b


def model_factory(hidden_dim: int, depth: int, combine: bool, task: str, lr: float,
                  *, input_dim: int, epochs: int, warmup: int = 5,
                  weight_decay: float = 1e-4):
    """Builds (TaskHead, Adam, LambdaLR cosine-with-warmup, loss_fn)."""
    model = TaskHead(hidden_dim=hidden_dim, input_dim=input_dim, depth=depth, combine=combine)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss() if task == "classification" else nn.MSELoss()

    def lr_fn(epoch: int) -> float:
        if epoch < warmup:
            return epoch / warmup
        return (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2

    scheduler = LambdaLR(optimiser, lr_lambda=lr_fn)
    return model, optimiser, scheduler, loss_fn


class FpDataset(Dataset):
    def __init__(self, fps: np.ndarray, ys: np.ndarray):
        self.fps = torch.from_numpy(fps).float()
        self.ys = torch.from_numpy(ys).float()

    def __len__(self) -> int:
        return self.fps.shape[0]

    def __getitem__(self, idx: int):
        return self.fps[idx], self.ys[idx]


def train_one_epoch(model: nn.Module, loader: DataLoader, optimiser, scheduler,
                    loss_fn, task: str, device: torch.device, epoch: int) -> nn.Module:
    model.train()
    scheduler.step(epoch)
    for fps, ys in loader:
        fps, ys = fps.to(device), ys.to(device)
        optimiser.zero_grad()
        logits = model(fps).squeeze(-1)
        if task == "classification":
            loss = loss_fn(torch.sigmoid(logits), ys)
        else:
            loss = loss_fn(logits, ys)
        loss.backward()
        optimiser.step()
    return model


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, loss_fn, task: str,
              device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for fps, ys in loader:
        fps, ys = fps.to(device), ys.to(device)
        logits = model(fps).squeeze(-1)
        if task == "classification":
            loss = loss_fn(torch.sigmoid(logits), ys)
        else:
            loss = loss_fn(logits, ys)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_ensemble(predictors: List[nn.Module], loader: DataLoader,
                      task: str, device: torch.device) -> np.ndarray:
    """Logit-average across the K fold-models, then sigmoid for classification."""
    chunks: List[torch.Tensor] = []
    for fps, _ in loader:
        fps = fps.to(device)
        ensemble_logits = torch.stack([p(fps).squeeze(-1) for p in predictors], dim=0)
        avg = ensemble_logits.mean(dim=0)
        if task == "classification":
            chunks.append(torch.sigmoid(avg).detach().cpu())
        else:
            chunks.append(avg.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


@torch.no_grad()
def ensemble_val_loss(predictors: List[nn.Module], loader: DataLoader, loss_fn,
                      task: str, device: torch.device) -> float:
    """Mean loss of the K-model ensemble on a single val loader.

    Mirrors MiniMol's ``evaluate(predictor, loader, loss_fn, task)`` but with
    ``predictor`` replaced by the logit-mean of K fold-models. Used during the
    HP sweep to score a candidate by 'the ensemble's mean validation metric',
    per Algorithm 1.
    """
    total, n = 0.0, 0
    for fps, ys in loader:
        fps, ys = fps.to(device), ys.to(device)
        ensemble_logits = torch.stack([p(fps).squeeze(-1) for p in predictors], dim=0)
        avg = ensemble_logits.mean(dim=0)
        if task == "classification":
            loss = loss_fn(torch.sigmoid(avg), ys)
        else:
            loss = loss_fn(avg, ys)
        total += loss.item(); n += 1
    return total / max(n, 1)


def _df_to_xy(df: pd.DataFrame, smi_to_fp: Dict[str, np.ndarray],
              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Look up fingerprints by SMILES; drop rows whose SMILES failed featurisation."""
    fps, ys = [], []
    for s, y in zip(df["Drug"].tolist(), df["Y"].tolist()):
        if s in smi_to_fp:
            fps.append(smi_to_fp[s]); ys.append(float(y))
    dropped = len(df) - len(fps)
    if not fps:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), dropped
    return np.stack(fps).astype(np.float32), np.asarray(ys, dtype=np.float32), dropped


def _train_one_fold(
    tr_loader: DataLoader, va_loader: DataLoader, hp: Dict[str, object],
    task_type: str, input_dim: int, *, args, device: torch.device, seed: int,
) -> Tuple[nn.Module, float]:
    """Train one fold-model for ``args.epochs`` and keep the best by val loss."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, optimiser, scheduler, loss_fn = model_factory(
        hp["hidden_dim"], hp["depth"], hp["combine"], task_type, hp["lr"],
        input_dim=input_dim, epochs=args.epochs,
    )
    model = model.to(device)
    best_state, best_val = None, float("inf")
    for epoch in range(args.epochs):
        model = train_one_epoch(model, tr_loader, optimiser, scheduler,
                                loss_fn, task_type, device, epoch)
        v = eval_loss(model, va_loader, loss_fn, task_type, device)
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval(), best_val


def _build_split_loaders(
    name: str, group, seed: int, smi_to_fp: Dict[str, np.ndarray], args,
) -> Tuple[DataLoader, DataLoader]:
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        tr_df, va_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed,
        )
    X_tr, y_tr, _ = _df_to_xy(tr_df, smi_to_fp)
    X_va, y_va, _ = _df_to_xy(va_df, smi_to_fp)
    tr_loader = DataLoader(FpDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(FpDataset(X_va, y_va), batch_size=128, shuffle=False)
    return tr_loader, va_loader


def run_one_task_sweep(
    task: str, group, smi_to_fp: Dict[str, np.ndarray], input_dim: int,
    *, args, device: torch.device,
) -> Dict[str, object]:
    """MiniMol Algorithm 1, sweep half: triple-nested loop hp → rep → fold.

    Per HP candidate:
      - For each ``sweep_rep`` (default 1, matching 'Each configuration is run
        on the same random seed'): train K fold-models, build the K-model
        ensemble, and score it by:
            * val:  mean across the K fold val sets of the ensemble's val loss.
            * test: ensemble's prediction on the fixed TDC test set, scored by
              ``group.evaluate_many`` (TDC's official primary metric).
      - Aggregate across reps: HP score = mean of per-rep ensemble val losses.

    Selection: argmin over the 'ensemble's mean validation metric' across reps,
    as described in Section 4.2 ("the ensemble's mean validation metric is
    used") and Appendix A.2 ("the model with the smallest validation loss").
    """
    benchmark = group.get(task)
    name = benchmark["name"]
    is_clf = benchmark["test"]["Y"].nunique() == 2
    task_type = "classification" if is_clf else "regression"
    loss_fn = nn.BCELoss() if task_type == "classification" else nn.MSELoss()
    sweep_reps = max(1, int(args.sweep_reps))

    # Test loader stays fixed across HPs.
    X_te, y_te, _ = _df_to_xy(benchmark["test"], smi_to_fp)
    test_loader = DataLoader(FpDataset(X_te, y_te), batch_size=128, shuffle=False)

    candidates = SWEEP_GRID
    results: List[Dict[str, object]] = []
    t0 = time.time()
    for hp_idx, hp in enumerate(candidates, start=1):
        per_rep: List[Dict[str, object]] = []
        for rep_i, sweep_seed_outer in enumerate(range(1, sweep_reps + 1), start=1):
            fold_models: List[nn.Module] = []
            fold_val_loaders: List[DataLoader] = []
            per_fold_indep_val: List[float] = []
            for fold_i, seed2 in enumerate(
                range(args.reps + 1, args.reps + args.ensemble_size + 1), start=1
            ):
                seed = cantor_pairing(sweep_seed_outer, seed2)
                tr_loader, va_loader = _build_split_loaders(name, group, seed, smi_to_fp, args)
                model, best_val = _train_one_fold(
                    tr_loader, va_loader, hp, task_type, input_dim,
                    args=args, device=device, seed=seed,
                )
                fold_models.append(model)
                fold_val_loaders.append(va_loader)
                per_fold_indep_val.append(float(best_val))

            # Algorithm 1: 'build ensemble of numfolds models; evaluate on
            # ensemble; save mean and std of val and test scores'.
            #   val:  mean across the K val splits of the ensemble's val loss.
            #   test: TDC's official primary metric on the ensemble's test pred.
            ens_val_per_fold = [
                ensemble_val_loss(fold_models, vl, loss_fn, task_type, device)
                for vl in fold_val_loaders
            ]
            ens_val_mean = float(np.mean(ens_val_per_fold))
            ens_val_std = float(np.std(ens_val_per_fold))
            test_pred = evaluate_ensemble(fold_models, test_loader, task_type, device)
            ens_test_metric = _compute_primary_metric(
                y_te, test_pred, _PRIMARY_METRIC.get(task, "mae"),
            )

            per_rep.append({
                "rep": rep_i,
                "sweep_seed_outer": sweep_seed_outer,
                "ensemble_val_loss_mean": ens_val_mean,    # selection-relevant
                "ensemble_val_loss_std": ens_val_std,      # std across K folds
                "ensemble_val_loss_per_fold": ens_val_per_fold,
                "ensemble_test_metric": ens_test_metric,
                "fold_indep_val_loss_mean": float(np.mean(per_fold_indep_val)),
            })
            del fold_models, fold_val_loaders
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rep_val = [r["ensemble_val_loss_mean"] for r in per_rep]
        rep_test = [r["ensemble_test_metric"] for r in per_rep]
        hp_val_mean = float(np.mean(rep_val))
        hp_val_std = float(np.std(rep_val))
        hp_test_mean = float(np.mean(rep_test))
        hp_test_std = float(np.std(rep_test))
        results.append({
            "hp": dict(hp),
            "val_loss_mean": hp_val_mean,    # selection metric (kept name for backward compat)
            "val_loss_std": hp_val_std,
            "test_metric_mean": hp_test_mean,
            "test_metric_std": hp_test_std,
            "n_reps": sweep_reps,
            "per_rep": per_rep,
        })
        print(
            f"  [{task}] hp {hp_idx:>2}/{len(candidates)}  "
            f"hidden={hp['hidden_dim']:>4}  depth={hp['depth']}  lr={hp['lr']:>.0e}  "
            f"ens_val={hp_val_mean:.4f} ± {hp_val_std:.4f}  "
            f"ens_test={hp_test_mean:.4f}",
            flush=True,
        )

    best_idx = int(np.argmin([r["val_loss_mean"] for r in results]))
    best_hp = dict(results[best_idx]["hp"])
    elapsed = time.time() - t0
    print(
        f"  [{task}] SWEEP DONE  best={best_hp}  "
        f"ens_val={results[best_idx]['val_loss_mean']:.4f}  "
        f"ens_test={results[best_idx]['test_metric_mean']:.4f}  ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        "best": best_hp,
        "best_val_loss_mean": float(results[best_idx]["val_loss_mean"]),
        "best_val_loss_std": float(results[best_idx]["val_loss_std"]),
        "best_test_metric_mean": float(results[best_idx]["test_metric_mean"]),
        "best_test_metric_std": float(results[best_idx]["test_metric_std"]),
        "best_index": best_idx,
        "task_type": task_type,
        "n_folds": args.ensemble_size,
        "n_sweep_reps": sweep_reps,
        "all_candidates": results,
        "elapsed_sec": round(elapsed, 1),
    }


def run_one_task(
    task: str, group, encoder, smi_to_fp: Dict[str, np.ndarray], input_dim: int,
    *, args, device: torch.device, hp_override: Optional[Dict[str, object]] = None,
    hp_source: str = "minimol",
) -> Optional[Dict[str, object]]:
    benchmark = group.get(task)
    name = benchmark["name"]
    is_clf = benchmark["test"]["Y"].nunique() == 2
    task_type = "classification" if is_clf else "regression"
    hp = dict(hp_override) if hp_override is not None else SWEEP_RESULTS[task]

    X_te, y_te, dropped_te = _df_to_xy(benchmark["test"], smi_to_fp)
    if dropped_te:
        print(f"  [{task}] WARN: dropped {dropped_te} test rows (failed featurisation)", flush=True)
    if X_te.shape[0] == 0:
        print(f"  [{task}] no usable test rows; skipping.", flush=True)
        return None
    test_loader = DataLoader(FpDataset(X_te, y_te), batch_size=128, shuffle=False)

    predictions_per_rep: List[Dict[str, np.ndarray]] = []
    t0 = time.time()
    for rep_i, seed1 in enumerate(range(1, args.reps + 1), start=1):
        best_models: List[nn.Module] = []
        for fold_i, seed2 in enumerate(range(args.reps + 1, args.reps + args.ensemble_size + 1), start=1):
            seed = cantor_pairing(seed1, seed2)
            with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
                tr_df, va_df = group.get_train_valid_split(
                    benchmark=name, split_type="default", seed=seed,
                )
            X_tr, y_tr, _ = _df_to_xy(tr_df, smi_to_fp)
            X_va, y_va, _ = _df_to_xy(va_df, smi_to_fp)
            tr_loader = DataLoader(FpDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
            va_loader = DataLoader(FpDataset(X_va, y_va), batch_size=128, shuffle=False)

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model, optimiser, scheduler, loss_fn = model_factory(
                hp["hidden_dim"], hp["depth"], hp["combine"], task_type, hp["lr"],
                input_dim=input_dim, epochs=args.epochs,
            )
            model = model.to(device)

            best_state, best_val = None, float("inf")
            for epoch in range(args.epochs):
                model = train_one_epoch(model, tr_loader, optimiser, scheduler,
                                        loss_fn, task_type, device, epoch)
                v = eval_loss(model, va_loader, loss_fn, task_type, device)
                if v < best_val:
                    best_val = v
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            assert best_state is not None
            model.load_state_dict(best_state)
            best_models.append(model.eval())
            print(f"  [{task}] rep {rep_i}/{args.reps}  fold {fold_i}/{args.ensemble_size}  "
                  f"best val_loss={best_val:.4f}", flush=True)

        preds = evaluate_ensemble(best_models, test_loader, task_type, device)
        predictions_per_rep.append({name: preds})
        del best_models  # free GPU mem
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute the primary metric per rep directly (sklearn / scipy) instead
    # of TDC's evaluate_many. evaluate_many compares against the *original*
    # test DataFrame whose length is the unfiltered TDC test split — but
    # ``y_te`` here is shorter when some SMILES fail featurisation, so we'd
    # get ``ValueError: inconsistent numbers of samples``. The direct path
    # operates on the surviving (y_te, preds) pairs exactly like the sweep
    # stage already does.
    primary_metric = _PRIMARY_METRIC.get(task, "mae")
    per_rep_metrics = [
        _compute_primary_metric(y_te, rep_dict[name], primary_metric)
        for rep_dict in predictions_per_rep
    ]
    m_mean = float(np.mean(per_rep_metrics))
    m_std  = float(np.std(per_rep_metrics)) if len(per_rep_metrics) > 1 else 0.0
    elapsed = time.time() - t0

    primary_metric_label = _benchmark_metric_name(group, task)
    print(f"  [{task}] DONE  {primary_metric_label}={m_mean:.4f} ± {m_std:.4f}  ({elapsed:.1f}s)",
          flush=True)

    return {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model":          args.model_name,
        "pretrain":       args.pretrain_label,
        "task":           task,
        "task_type":      task_type,
        "primary_metric": primary_metric,
        "metric_mean":    float(m_mean),
        "metric_std":     float(m_std),
        "n_reps":         args.reps,
        "n_ensemble":     args.ensemble_size,
        "n_epochs":       args.epochs,
        "hp_source":      hp_source,
        "hidden_dim":     hp["hidden_dim"],
        "depth":          hp["depth"],
        "combine":        hp["combine"],
        "lr":             hp["lr"],
        "input_dim":      input_dim,
        "ckpt":           args.ckpt,
        "ckpt_tag":       args.ckpt_tag or Path(args.ckpt).stem,
        "elapsed_sec":    round(elapsed, 1),
    }


# Hard-coded leaderboard primary-metric per task (TDC ADMET overview page) so
# the row labels are useful without depending on a TDC-private attribute name.
_PRIMARY_METRIC = {
    "caco2_wang":                       "mae",
    "hia_hou":                          "auroc",
    "pgp_broccatelli":                  "auroc",
    "bioavailability_ma":               "auroc",
    "lipophilicity_astrazeneca":        "mae",
    "solubility_aqsoldb":               "mae",
    "bbb_martins":                      "auroc",
    "ppbr_az":                          "mae",
    "vdss_lombardo":                    "spearman",
    "cyp2d6_veith":                     "auprc",
    "cyp3a4_veith":                     "auprc",
    "cyp2c9_veith":                     "auprc",
    "cyp2d6_substrate_carbonmangels":   "auprc",
    "cyp3a4_substrate_carbonmangels":   "auroc",
    "cyp2c9_substrate_carbonmangels":   "auprc",
    "half_life_obach":                  "spearman",
    "clearance_microsome_az":           "spearman",
    "clearance_hepatocyte_az":          "spearman",
    "herg":                             "auroc",
    "ames":                             "auroc",
    "dili":                             "auroc",
    "ld50_zhu":                         "mae",
}


def _benchmark_metric_name(group, task: str) -> str:
    """Best-effort primary-metric name for the row label."""
    for attr in ("benchmark_metric", "metric"):
        v = getattr(group, attr, None)
        if isinstance(v, dict) and task in v:
            return str(v[task])
    return _PRIMARY_METRIC.get(task, "metric")


def _compute_primary_metric(y_true: np.ndarray, y_pred: np.ndarray,
                            primary_metric: str) -> float:
    """Direct sklearn/scipy primary-metric computation. Used during the sweep
    stage so we don't depend on TDC's ``evaluate_many``, which raises with
    single-rep inputs.
    """
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    try:
        m = primary_metric.lower()
        if m == "mae":
            from sklearn.metrics import mean_absolute_error
            return float(mean_absolute_error(yt, yp))
        if m == "auroc":
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(yt, yp))
        if m == "auprc":
            from sklearn.metrics import average_precision_score
            return float(average_precision_score(yt, yp))
        if m == "spearman":
            from scipy.stats import spearmanr
            r = spearmanr(yt, yp).correlation
            return float(r) if r is not None else float("nan")
        if m == "pearson":
            from scipy.stats import pearsonr
            return float(pearsonr(yt, yp).statistic)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan")


def _append_rows(out_csv: Path, rows: List[Dict[str, object]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        merged = pd.concat([prev, new_df], ignore_index=True, sort=False)
        merged.to_csv(out_csv, index=False)
    else:
        new_df.to_csv(out_csv, index=False)


def _sweep_to_csv_rows(args, task: str, sweep: Dict[str, object],
                       input_dim: int) -> List[Dict[str, object]]:
    """Flatten a sweep result into one row per HP candidate for inspection."""
    out = []
    best_idx = sweep["best_index"]
    for i, cand in enumerate(sweep["all_candidates"]):
        hp = cand["hp"]
        out.append({
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model":             args.model_name,
            "pretrain":          args.pretrain_label,
            "task":              task,
            "task_type":         sweep["task_type"],
            "n_folds":           sweep["n_folds"],
            "n_sweep_reps":      sweep.get("n_sweep_reps", 1),
            "hp_index":          i,
            "is_best":           int(i == best_idx),
            "hidden_dim":        hp["hidden_dim"],
            "depth":             hp["depth"],
            "combine":           hp["combine"],
            "lr":                hp["lr"],
            "val_loss_mean":     cand["val_loss_mean"],     # ensemble val loss (selection)
            "val_loss_std":      cand["val_loss_std"],
            "test_metric_mean":  cand.get("test_metric_mean"),
            "test_metric_std":   cand.get("test_metric_std"),
            "input_dim":         input_dim,
            "ckpt":              args.ckpt,
            "ckpt_tag":          args.ckpt_tag or Path(args.ckpt).stem,
        })
    return out


def _resolve_hp_table(args) -> Tuple[Dict[str, Dict[str, object]], str]:
    """Pick the HP table for eval mode.

    Priority:
      1. ``--sweep-results`` JSON (explicit).
      2. Auto-detected sibling JSON
         ``<output-csv parent>/pairmixer_minimol_probe_sweep_<label>.json``.
      3. Built-in MiniMol verbatim ``SWEEP_RESULTS``.
    """
    if args.sweep_results:
        path = Path(args.sweep_results)
        # Sentinel: ``/dev/null`` (or any empty / non-regular file) means
        # "force the verbatim MiniMol HPs"; useful to add a minimol-baseline
        # row alongside an existing sweep-mode row.
        is_empty = (
            str(path) == "/dev/null"
            or not path.exists()
            or not path.is_file()
            or path.stat().st_size == 0
        )
        if is_empty:
            return SWEEP_RESULTS, "minimol"
        try:
            with open(path) as f:
                blob = json.load(f)
        except json.JSONDecodeError:
            return SWEEP_RESULTS, "minimol"
        return ({t: blob["tasks"][t]["best"] for t in blob["tasks"]},
                f"sweep:{path.name}")

    auto = Path(args.output_csv).parent / f"pairmixer_minimol_probe_sweep_{args.pretrain_label}.json"
    if auto.exists():
        with open(auto) as f:
            blob = json.load(f)
        return ({t: blob["tasks"][t]["best"] for t in blob["tasks"]},
                f"sweep:{auto.name}")

    return SWEEP_RESULTS, "minimol"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="PairMixer checkpoint (.ckpt)")
    p.add_argument("--model-name", default="pairmixer_12M",
                   help="Hydra model name (default: pairmixer_12M)")
    p.add_argument("--ckpt-tag", default=None,
                   help="Short label in the results CSV (default: ckpt filename stem)")
    p.add_argument("--pretrain-label", required=True,
                   help="Free-form label for this checkpoint's pretraining recipe, e.g. "
                        "'toymix-only' or 'toymix-dti-esmc-v2'")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="TDC ADMET tasks to run (default: all 22)")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device for the head trainer + encoder (default: cuda:0)")
    p.add_argument("--reps", type=int, default=5,
                   help="Outer repetitions (MiniMol default: 5)")
    p.add_argument("--ensemble-size", type=int, default=5,
                   help="Inner fold-models per ensemble (MiniMol default: 5)")
    p.add_argument("--epochs", type=int, default=25,
                   help="Head training epochs (MiniMol default: 25)")
    p.add_argument("--batch-size", type=int, default=32, help="Head training batch size")
    p.add_argument("--embed-batch-size", type=int, default=32,
                   help="GNN forward batch size for fingerprint extraction")
    p.add_argument("--featurize-n-jobs", type=int, default=8,
                   help="joblib workers for graphium featurisation")
    p.add_argument("--admet-cache",
                   default=str(ROOT / "expts" / "data" / "admet"),
                   help="TDC admet_group cache dir")
    p.add_argument("--mol-cache-dir",
                   default=str(ROOT / "datacache" / "pairmixer_probe_fps"),
                   help="Per-(ckpt, task) fingerprint cache dir")
    p.add_argument("--output-csv",
                   default=str(ROOT / "results" / "pairmixer_minimol_probe_ensemble.csv"),
                   help="Append-only eval-mode results CSV (one row per task per run; "
                        "kept separate from results/experiment_results.csv and from the "
                        "per-candidate --sweep-csv).")
    p.add_argument("--sweep", action="store_true",
                   help="Run the 18-candidate HP sweep instead of the final eval. "
                        "Writes the chosen-per-task HPs to --sweep-out and per-candidate "
                        "rows to --sweep-csv.")
    p.add_argument("--sweep-out", default=None,
                   help="(sweep mode) path to write the chosen-HP JSON. Defaults to "
                        "<output-csv parent>/pairmixer_minimol_probe_sweep_<label>.json. "
                        "Eval mode auto-loads this file when present unless "
                        "--sweep-results is set.")
    p.add_argument("--sweep-csv",
                   default=str(ROOT / "results" / "pairmixer_minimol_probe_sweep.csv"),
                   help="(sweep mode) per-candidate sweep CSV (append-only).")
    p.add_argument("--sweep-results", default=None,
                   help="(eval mode) JSON file produced by a previous --sweep run. "
                        "Overrides the built-in MiniMol HPs.")
    p.add_argument("--no-resume-sweep", action="store_true",
                   help="(sweep mode) start from a fresh sweep JSON; do not merge "
                        "with the existing file at --sweep-out.")
    p.add_argument("--sweep-reps", type=int, default=1,
                   help="(sweep mode) number of outer reps per HP candidate "
                        "(MiniMol Algorithm 1's `numreps`). Default 1 — matches "
                        "Appendix A.2's 'Each configuration is run on the same "
                        "random seed'. Set higher to mean across reps.")
    p.add_argument("--cpu-threads", type=int, default=_CPU_THREADS,
                   help="OMP/MKL thread cap (already applied at import time)")
    args = p.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"ERROR: ckpt not found: {args.ckpt}")
    Path(args.admet_cache).mkdir(parents=True, exist_ok=True)

    # ---- TDC group ----
    from tdc.benchmark_group import admet_group  # noqa: E402
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        group = admet_group(path=args.admet_cache)

    tasks = list(args.tasks) if args.tasks else sorted(SWEEP_RESULTS.keys())
    unknown = [t for t in tasks if t not in SWEEP_RESULTS]
    if unknown:
        sys.exit(f"ERROR: unknown task(s) (no MiniMol HP): {unknown}")

    # ---- Encoder (frozen PairMixer) ----
    encoder = load_encoder(
        "pairmixer", ckpt_path=args.ckpt, model_name=args.model_name,
        device=args.device, batch_size=args.embed_batch_size,
        featurize_n_jobs=args.featurize_n_jobs, ckpt_tag=args.ckpt_tag,
    )

    # Cache key includes the ckpt's parent dir + filename so different ckpts
    # don't collide. Per-task cache file keeps reuse cheap for multi-task runs.
    Path(args.mol_cache_dir).mkdir(parents=True, exist_ok=True)
    ckpt_p = Path(args.ckpt)
    ckpt_stem = f"{ckpt_p.parent.name}__{ckpt_p.stem}"

    device = torch.device(args.device)

    # Resolve eval-mode HP table once (a no-op in sweep mode).
    if args.sweep:
        hp_table, hp_source = (None, "sweep-runtime")
    else:
        hp_table, hp_source = _resolve_hp_table(args)
        print(f"[hp] using {hp_source}", flush=True)

    # In sweep mode, write the chosen-HP JSON at the end.
    sweep_blob: Dict[str, object] = {
        "model":          args.model_name,
        "ckpt":           args.ckpt,
        "ckpt_tag":       args.ckpt_tag or Path(args.ckpt).stem,
        "pretrain_label": args.pretrain_label,
        "n_folds":        args.ensemble_size,
        "n_epochs":       args.epochs,
        "tasks":          {},
    }
    sweep_out_path = (
        Path(args.sweep_out)
        if args.sweep_out
        else (Path(args.output_csv).parent
              / f"pairmixer_minimol_probe_sweep_{args.pretrain_label}.json")
    )
    # Merge with any pre-existing sweep JSON so a multi-stage workflow
    # (sweep caco2_wang first, then sweep the remaining 21 tasks separately)
    # accumulates HPs instead of clobbering them. Disable with --no-resume-sweep.
    if args.sweep and sweep_out_path.exists() and not args.no_resume_sweep:
        try:
            with open(sweep_out_path) as f:
                _existing = json.load(f) or {}
            for _t, _v in (_existing.get("tasks") or {}).items():
                if _t not in tasks:  # never replace a task we're about to re-sweep
                    sweep_blob["tasks"][_t] = _v
            if sweep_blob["tasks"]:
                print(
                    f"[sweep] merged {len(sweep_blob['tasks'])} task(s) from "
                    f"existing {sweep_out_path.name}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[sweep] WARN: could not merge existing {sweep_out_path}: {exc}",
                file=sys.stderr, flush=True,
            )

    for task in tasks:
        bench = group.get(task)
        all_smiles = sorted(set(bench["train_val"]["Drug"].tolist())
                            | set(bench["test"]["Drug"].tolist()))
        cache_path = Path(args.mol_cache_dir) / f"{ckpt_stem}__{task}.pt"
        X, mask = encoder.extract_cached(all_smiles, cache_path)
        if encoder.out_dim <= 0:
            sys.exit("ERROR: encoder.out_dim not set after extract_cached")
        survivors = [s for s, ok in zip(all_smiles, mask) if ok]
        smi_to_fp = {s: X[i] for i, s in enumerate(survivors)}
        n_drop = len(all_smiles) - len(survivors)
        print(f"[{task}] fp dim={int(encoder.out_dim)}  smi={len(survivors):,}/{len(all_smiles):,}"
              + (f"  (dropped {n_drop} bad SMILES)" if n_drop else ""), flush=True)

        if args.sweep:
            try:
                sw = run_one_task_sweep(
                    task, group, smi_to_fp, int(encoder.out_dim),
                    args=args, device=device,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{task}] SWEEP FAIL: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                continue
            sweep_blob["tasks"][task] = sw
            # Persist after every task so a crash mid-sweep doesn't lose work.
            sweep_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sweep_out_path, "w") as f:
                json.dump(sweep_blob, f, indent=2, default=float)
            _append_rows(
                Path(args.sweep_csv),
                _sweep_to_csv_rows(args, task, sw, int(encoder.out_dim)),
            )
        else:
            hp_override = (hp_table or SWEEP_RESULTS).get(task)
            try:
                row = run_one_task(
                    task, group, encoder, smi_to_fp, int(encoder.out_dim),
                    args=args, device=device,
                    hp_override=hp_override, hp_source=hp_source,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{task}] FAIL: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                row = None
            if row is not None:
                _append_rows(Path(args.output_csv), [row])

    if args.sweep:
        print(f"\nSweep done. {len(sweep_blob['tasks'])} task(s) → {sweep_out_path}")
        print(f"Per-candidate rows appended to {args.sweep_csv}")
    else:
        print(f"\nDone. Rows appended to {args.output_csv}")


if __name__ == "__main__":
    main()
