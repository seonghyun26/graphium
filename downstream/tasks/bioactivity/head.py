"""Paper-faithful PyTorch MLP + focal-BCE-masked loss.

Ported from ``cfredinh/bioactive/MLP_predictor.py`` via
``scripts/dep_baseline_mlp_cell_bioactivity.py``. Kept in its own module
(instead of ``downstream/tasks/common/heads.py``) because the masked
multi-label loss semantics don't fit sklearn's API.

Labels are ``{-1, 0, +1}`` where ``0`` means "unknown" (masked out of the
loss) — the existing 0/1/NaN CSV gets remapped by ``data.load_dataset``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .report import EF_TOP_FRACTIONS, ef_macro_column, enrichment_factor


class _FpDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


class MLP(nn.Module):
    """3-layer MLP with ReLU + dropout, matching the upstream paper code."""

    def __init__(
        self, in_dim: int, hidden: int, num_hidden: int, out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        layers: "OrderedDict[str, nn.Module]" = OrderedDict()
        layers["fc_in"] = nn.Linear(in_dim, hidden)
        layers["relu_in"] = nn.ReLU()
        layers["drop_in"] = nn.Dropout(dropout)
        for k in range(1, num_hidden - 1):
            layers[f"fc_{k}"] = nn.Linear(hidden, hidden)
            layers[f"relu_{k}"] = nn.ReLU()
            layers[f"drop_{k}"] = nn.Dropout(dropout)
        layers["fc_out"] = nn.Linear(hidden, out_dim)
        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalBCEMaskedLoss(nn.Module):
    """Focal BCE-with-logits, masking out ``y == 0`` (unknown) entries.

    Identical to upstream ``BCEMASKEDLoss`` (γ=2). Targets are in ``{-1, 0, +1}``;
    mask := y² (1 when labeled, 0 otherwise); internally remaps y to {0, 1}.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target ** 2
        tgt = (target + mask) / 2.0
        sel_logits = logits[mask > 0]
        sel_tgt = tgt[mask > 0]
        if sel_logits.numel() == 0:
            # Empty batch (rare) — keep an autograd connection so .backward()
            # doesn't explode on "element 0 of tensors does not require grad".
            return (logits * 0.0).sum()
        basic = self.bce(sel_logits, sel_tgt)
        focal = (sel_tgt - torch.sigmoid(sel_logits)).abs() ** self.gamma
        return (focal * basic).mean()


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    assay_cols: List[str],
) -> Dict[str, object]:
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(device)).cpu().numpy())
            all_targets.append(y.numpy())
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    aurocs, auprcs = [], []
    per_assay_auroc: Dict[str, float] = {}
    per_assay_auprc: Dict[str, float] = {}
    per_assay_ef: Dict[float, Dict[str, float]] = {top_fraction: {} for top_fraction in EF_TOP_FRACTIONS}
    for j, col in enumerate(assay_cols):
        m = targets[:, j] != 0
        if m.sum() < 2:
            continue
        y_true = (targets[m, j] > 0).astype(int)
        if y_true.min() == y_true.max():
            continue
        try:
            au = float(roc_auc_score(y_true, probs[m, j]))
            ap = float(average_precision_score(y_true, probs[m, j]))
        except ValueError:
            continue
        aurocs.append(au)
        auprcs.append(ap)
        per_assay_auroc[col] = au
        per_assay_auprc[col] = ap
        for top_fraction in EF_TOP_FRACTIONS:
            per_assay_ef[top_fraction][col] = enrichment_factor(
                y_true,
                probs[m, j],
                top_fraction=top_fraction,
            )
    ef_macro = {
        ef_macro_column(top_fraction): (
            float(np.mean(list(per_assay_ef[top_fraction].values())))
            if per_assay_ef[top_fraction]
            else float("nan")
        )
        for top_fraction in EF_TOP_FRACTIONS
    }
    return {
        "auroc_macro": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auprc_macro": float(np.mean(auprcs)) if auprcs else float("nan"),
        "n_assays_scored": len(aurocs),
        "per_assay_auroc": per_assay_auroc,
        "per_assay_auprc": per_assay_auprc,
        "per_assay_ef": per_assay_ef,
        **ef_macro,
    }


def train_and_eval_fold(
    X: np.ndarray, Y: np.ndarray, splits: Dict[str, np.ndarray],
    assay_cols: List[str],
    *,
    standardize: bool,
    hidden: int = 512, num_hidden: int = 3, dropout: float = 0.5,
    epochs: int = 200, batch_size: int = 64,
    optimizer: str = "sgd", lr: float = 2.0, weight_decay: float = 0.0,
    lr_patience: int = 8, min_lr: float = 1e-5,
    seed: int = 0, device: str = "cpu", tag: str = "",
    log_every: int = 10,
) -> Dict[str, object]:
    """Train one fold; early-stop + LR-halve on macro val AUROC; evaluate on test.

    ``standardize=True`` z-scores dense inputs (CPCNN / Minimol / MolE / PairMixer)
    using only this fold's train split — matches the paper's MLP recipe and
    prevents val/test stats from leaking into training. Binary ECFP stays raw.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = X.astype(np.float32, copy=False)
    if standardize:
        mu = X[splits["train"]].mean(axis=0, keepdims=True)
        sd = X[splits["train"]].std(axis=0, keepdims=True) + 1e-6
        X = ((X - mu) / sd).astype(np.float32)

    dev = torch.device(device)

    def _loader(name: str, shuffle: bool) -> DataLoader:
        idx = splits[name]
        return DataLoader(
            _FpDataset(X[idx], Y[idx]), batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(dev.type == "cuda"),
        )

    tr_loader, va_loader, te_loader = _loader("train", True), _loader("val", False), _loader("test", False)

    model = MLP(X.shape[1], hidden, num_hidden, len(assay_cols), dropout=dropout).to(dev)
    criterion = FocalBCEMaskedLoss(gamma=2.0)
    opt = (
        torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer == "sgd"
        else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    print(
        f"  [{tag}] in_dim={X.shape[1]} optim={optimizer} lr={lr} "
        f"sizes(train/val/test)={len(splits['train']):,}/{len(splits['val']):,}/{len(splits['test']):,}"
    )

    best_auroc, best_state, patience = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = _evaluate(model, va_loader, dev, assay_cols)
        if epoch == 1 or epoch % log_every == 0:
            print(
                f"  [{tag}] ep {epoch:3d} | train_loss={np.mean(losses):.4f} "
                f"| val_auroc={val['auroc_macro']:.4f} val_auprc={val['auprc_macro']:.4f}"
            )
        if val["auroc_macro"] > best_auroc:
            best_auroc = val["auroc_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > lr_patience:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                patience = 0
                if opt.param_groups[0]["lr"] < min_lr:
                    print(f"  [{tag}] lr below {min_lr} at epoch {epoch}; early stop")
                    break

    model.load_state_dict(best_state)
    test = _evaluate(model, te_loader, dev, assay_cols)
    test["best_val_auroc"] = best_auroc
    print(
        f"  [{tag}] TEST auroc={test['auroc_macro']:.4f} auprc={test['auprc_macro']:.4f} "
        f"(best val={best_auroc:.4f}, {test['n_assays_scored']}/{len(assay_cols)} assays scored)"
    )
    return test


def train_and_eval_fold_ensemble(
    X: np.ndarray, Y: np.ndarray, splits: Dict[str, np.ndarray],
    assay_cols: List[str],
    *,
    standardize: bool,
    hidden: int = 512, num_hidden: int = 3, dropout: float = 0.5,
    epochs: int = 200, batch_size: int = 64,
    optimizer: str = "sgd", lr: float = 2.0, weight_decay: float = 0.0,
    lr_patience: int = 8, min_lr: float = 1e-5,
    seed: int = 0, device: str = "cpu", tag: str = "",
    log_every: int = 10,
    ensemble_size: int = 5,
) -> Dict[str, object]:
    """MiniMol-style ensemble over ``ensemble_size`` head members.

    Each member retrains the MLP from a fresh init on the same train/val/test
    split with seed = ``seed + 1000 * m``, picks its best-val checkpoint, then
    we sigmoid+average the per-member test probabilities and recompute the
    macro/per-assay metrics from the average. Mirrors the inner-fold ensemble
    used in ``scripts/pairmixer/pairmixer_minimol_probe.py``.
    """
    X = X.astype(np.float32, copy=False)
    if standardize:
        mu = X[splits["train"]].mean(axis=0, keepdims=True)
        sd = X[splits["train"]].std(axis=0, keepdims=True) + 1e-6
        X = ((X - mu) / sd).astype(np.float32)

    dev = torch.device(device)

    def _loader(name: str, shuffle: bool) -> DataLoader:
        idx = splits[name]
        return DataLoader(
            _FpDataset(X[idx], Y[idx]), batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(dev.type == "cuda"),
        )

    tr_loader = _loader("train", True)
    va_loader = _loader("val", False)
    te_loader = _loader("test", False)

    member_test_probs: List[np.ndarray] = []
    member_best_val_aurocs: List[float] = []
    member_test_metrics: List[Dict[str, object]] = []
    test_targets: np.ndarray | None = None

    print(
        f"  [{tag}] in_dim={X.shape[1]} optim={optimizer} lr={lr} ensemble_size={ensemble_size} "
        f"sizes(train/val/test)={len(splits['train']):,}/{len(splits['val']):,}/{len(splits['test']):,}"
    )

    for m in range(ensemble_size):
        member_seed = seed + 1000 * m
        member_tag = f"{tag}/m{m+1}"
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)

        model = MLP(X.shape[1], hidden, num_hidden, len(assay_cols), dropout=dropout).to(dev)
        criterion = FocalBCEMaskedLoss(gamma=2.0)
        opt = (
            torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            if optimizer == "sgd"
            else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        )

        best_auroc, best_state, patience = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for x, y in tr_loader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            val = _evaluate(model, va_loader, dev, assay_cols)
            if epoch == 1 or epoch % log_every == 0:
                print(
                    f"  [{member_tag}] ep {epoch:3d} | train_loss={np.mean(losses):.4f} "
                    f"| val_auroc={val['auroc_macro']:.4f} val_auprc={val['auprc_macro']:.4f}"
                )
            if val["auroc_macro"] > best_auroc:
                best_auroc = val["auroc_macro"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience > lr_patience:
                    for g in opt.param_groups:
                        g["lr"] *= 0.5
                    patience = 0
                    if opt.param_groups[0]["lr"] < min_lr:
                        print(f"  [{member_tag}] lr below {min_lr} at epoch {epoch}; early stop")
                        break

        model.load_state_dict(best_state)
        # Pull this member's per-sample test logits + targets so we can both
        # ensemble and report per-member metrics.
        model.eval()
        te_logits, te_targets = [], []
        with torch.no_grad():
            for x, y in te_loader:
                te_logits.append(model(x.to(dev)).cpu().numpy())
                te_targets.append(y.numpy())
        te_logits = np.concatenate(te_logits)
        te_targets = np.concatenate(te_targets)
        if test_targets is None:
            test_targets = te_targets

        te_probs = 1.0 / (1.0 + np.exp(-np.clip(te_logits, -60.0, 60.0)))
        member_test_probs.append(te_probs)
        member_best_val_aurocs.append(best_auroc)

        # Per-member metrics use a tmp model whose forward returns the captured
        # logits; cheaper to inline a one-shot eval-from-arrays.
        m_metrics = _eval_metrics_from_probs(te_probs, te_targets, assay_cols)
        member_test_metrics.append(m_metrics)
        print(
            f"  [{member_tag}] member-test auroc={m_metrics['auroc_macro']:.4f} "
            f"auprc={m_metrics['auprc_macro']:.4f} (best val={best_auroc:.4f})"
        )

    avg_probs = np.mean(np.stack(member_test_probs, axis=0), axis=0)
    test = _eval_metrics_from_probs(avg_probs, test_targets, assay_cols)
    test["best_val_auroc"] = float(np.mean(member_best_val_aurocs))
    test["ensemble_size"] = int(ensemble_size)
    test["member_auroc_mean"] = float(np.mean([m["auroc_macro"] for m in member_test_metrics]))
    test["member_auroc_std"] = float(np.std([m["auroc_macro"] for m in member_test_metrics], ddof=0))
    test["member_auprc_mean"] = float(np.mean([m["auprc_macro"] for m in member_test_metrics]))
    test["member_auprc_std"] = float(np.std([m["auprc_macro"] for m in member_test_metrics], ddof=0))
    print(
        f"  [{tag}] ENSEMBLE TEST (n={ensemble_size}) "
        f"auroc={test['auroc_macro']:.4f} (members {test['member_auroc_mean']:.4f}±{test['member_auroc_std']:.4f})  "
        f"auprc={test['auprc_macro']:.4f} (members {test['member_auprc_mean']:.4f}±{test['member_auprc_std']:.4f})"
    )
    return test


def _eval_metrics_from_probs(
    probs: np.ndarray, targets: np.ndarray, assay_cols: List[str],
) -> Dict[str, object]:
    """Mirror of ``_evaluate``'s metric-computation half, taking ready-made
    sigmoid probabilities so the ensemble path can average across members
    without re-running a forward pass."""
    aurocs, auprcs = [], []
    per_assay_auroc: Dict[str, float] = {}
    per_assay_auprc: Dict[str, float] = {}
    per_assay_ef: Dict[float, Dict[str, float]] = {top_fraction: {} for top_fraction in EF_TOP_FRACTIONS}
    for j, col in enumerate(assay_cols):
        m = targets[:, j] != 0
        if m.sum() < 2:
            continue
        y_true = (targets[m, j] > 0).astype(int)
        if y_true.min() == y_true.max():
            continue
        try:
            au = float(roc_auc_score(y_true, probs[m, j]))
            ap = float(average_precision_score(y_true, probs[m, j]))
        except ValueError:
            continue
        aurocs.append(au)
        auprcs.append(ap)
        per_assay_auroc[col] = au
        per_assay_auprc[col] = ap
        for top_fraction in EF_TOP_FRACTIONS:
            per_assay_ef[top_fraction][col] = enrichment_factor(
                y_true, probs[m, j], top_fraction=top_fraction,
            )
    ef_macro = {
        ef_macro_column(top_fraction): (
            float(np.mean(list(per_assay_ef[top_fraction].values())))
            if per_assay_ef[top_fraction]
            else float("nan")
        )
        for top_fraction in EF_TOP_FRACTIONS
    }
    return {
        "auroc_macro": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auprc_macro": float(np.mean(auprcs)) if auprcs else float("nan"),
        "n_assays_scored": len(aurocs),
        "per_assay_auroc": per_assay_auroc,
        "per_assay_auprc": per_assay_auprc,
        "per_assay_ef": per_assay_ef,
        **ef_macro,
    }
