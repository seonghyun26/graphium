"""
MLP baseline for the cell_bioactivity benchmark (Fredinh et al. 2024).

Mirrors `MLP_predictor.py` from github.com/cfredinh/bioactive (3-layer MLP,
focal-BCE masked loss, SGD); the upstream script is generic over per-compound
feature vectors and the paper runs it twice — once with chemistry features
(ECFP) and once with cell-painting features (CPCNN embeddings). Pick which
via `--features {ecfp,cpcnn}`.

Reads graphium's CSV outputs from `scripts/prep_cell_bioactivity.py` and
appends one row to `results/experiment_results.csv` (model name
`ecfp_mlp` or `cpcnn_mlp`) so the dashboard notebook picks it up alongside
GNN runs.

Feature modes
-------------
- `ecfp`  : ECFP4 (r=2, 1024 bits) computed on-the-fly from the `smiles`
            column. Compound-only baseline.
- `cpcnn` : 672-dim CPCNN cell-painting embeddings, mean-aggregated per
            SMILES from `--cpcnn-csv`. Compounds without a CPCNN profile
            (no Cell Painting wells) are dropped from the splits.

Loss is the upstream `BCEMASKEDLoss` (focal BCE-with-logits, gamma=2) with
the ±1/0 label encoding produced internally by remapping graphium's 0/1/NaN.

Usage
-----
    # Chemistry baseline
    python scripts/baseline_mlp_cell_bioactivity.py --features ecfp

    # Cell-painting baseline
    python scripts/baseline_mlp_cell_bioactivity.py --features cpcnn
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


# ─── Featurizers ───────────────────────────────────────────────────────────────


def ecfp4(smiles: str, n_bits: int = 1024) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_ecfp(smiles_series: pd.Series, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N x n_bits], valid_mask [N])."""
    fps = np.zeros((len(smiles_series), n_bits), dtype=np.float32)
    valid = np.ones(len(smiles_series), dtype=bool)
    for i, smi in enumerate(smiles_series):
        f = ecfp4(smi, n_bits)
        if f is None:
            valid[i] = False
        else:
            fps[i] = f
    return fps, valid


def featurize_cpcnn(
    smiles_series: pd.Series, cpcnn_csv: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-aggregate 672-dim CPCNN embeddings per SMILES, then look up.

    Compounds without a CPCNN profile become invalid (their split rows are
    dropped downstream). This mirrors the paper's "cell-feature MLP" baseline,
    which only scores compounds that have Cell Painting profiles.
    """
    logger.info(f"loading CPCNN embeddings from {cpcnn_csv}")
    cp = pd.read_csv(cpcnn_csv)
    feat_cols = [c for c in cp.columns if c.startswith("feature_")]
    if "SMILES_nometa" not in cp.columns:
        raise RuntimeError(
            f"CPCNN CSV missing SMILES_nometa column; got {list(cp.columns)[:8]}..."
        )
    logger.info(f"CPCNN: {len(cp):,} wells, {len(feat_cols)} feature dims, "
                f"{cp.SMILES_nometa.nunique():,} unique SMILES")
    agg = cp.groupby("SMILES_nometa", sort=False)[feat_cols].mean()
    matrix = np.zeros((len(smiles_series), len(feat_cols)), dtype=np.float32)
    valid = np.zeros(len(smiles_series), dtype=bool)
    lookup = agg.to_dict("index")
    for i, smi in enumerate(smiles_series):
        row = lookup.get(smi)
        if row is None:
            continue
        matrix[i] = np.fromiter((row[c] for c in feat_cols), dtype=np.float32,
                                count=len(feat_cols))
        valid[i] = True
    logger.info(f"CPCNN match: {valid.sum():,} / {len(smiles_series):,} compounds "
                f"have profiles ({100*valid.mean():.1f}%)")
    return matrix, valid


# ─── Dataset / loaders ─────────────────────────────────────────────────────────


class FpDataset(Dataset):
    def __init__(self, fps: np.ndarray, labels: np.ndarray):
        # labels in {-1, 0, +1} where 0 == unknown, mirroring upstream BCEMASKEDLoss.
        self.fps = torch.from_numpy(fps).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return self.fps.shape[0]

    def __getitem__(self, i: int):
        return self.fps[i], self.labels[i]


def load_dataset(args: argparse.Namespace):
    """Load prep'd CSVs, featurize per `--features`, return per-split tensors."""
    df = pd.read_csv(args.csv)
    splits = pd.read_csv(args.splits)
    label_cols = [c for c in df.columns if c.startswith("assay_")]
    logger.info(f"loaded {len(df):,} compounds x {len(label_cols)} assays")

    if args.features == "ecfp":
        feats, valid = featurize_ecfp(df.smiles, n_bits=args.n_bits)
    elif args.features == "cpcnn":
        feats, valid = featurize_cpcnn(df.smiles, Path(args.cpcnn_csv))
    else:
        raise ValueError(f"unknown --features {args.features!r}")

    if (~valid).any():
        logger.warning(
            f"{(~valid).sum()} compounds invalid for --features={args.features} "
            "(unparseable SMILES or missing profile); dropping from splits"
        )

    # Labels: 0/1/NaN -> -1/+1/0 (BCEMASKEDLoss convention).
    raw = df[label_cols].values.astype(np.float32)
    labels = np.where(np.isnan(raw), 0.0, np.where(raw > 0.5, 1.0, -1.0)).astype(np.float32)

    splits_idx = {}
    for split in ("train", "val", "test"):
        idx = splits[split].dropna().astype(int).values
        idx = idx[valid[idx]]
        splits_idx[split] = idx
    logger.info(
        "split sizes (post-filter): "
        f"train={len(splits_idx['train']):,} "
        f"val={len(splits_idx['val']):,} "
        f"test={len(splits_idx['test']):,}"
    )

    # Dense real-valued features (CPCNN) need standardization before an MLP
    # trained with SGD/Adam — without it, lr tuning becomes brittle and the
    # gradient magnitude blows up. Binary ECFP bits are left alone.
    if args.features == "cpcnn":
        mu = feats[splits_idx["train"]].mean(axis=0, keepdims=True)
        sd = feats[splits_idx["train"]].std(axis=0, keepdims=True) + 1e-6
        feats = ((feats - mu) / sd).astype(np.float32)
        logger.info(f"standardized CPCNN features using train-split stats "
                    f"(mean abs after: {np.abs(feats[splits_idx['train']]).mean():.3f})")

    return feats, labels, splits_idx, label_cols


# ─── Model + loss ──────────────────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_hidden: int, out_dim: int,
                 dropout: float = 0.5):
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
    """Verbatim port of `BCEMASKEDLoss` from cfredinh/bioactive/MLP_predictor.py."""

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
            return logits.new_zeros(())
        basic = self.bce(sel_logits, sel_tgt)
        focal = (sel_tgt - torch.sigmoid(sel_logits)).abs() ** self.gamma
        return (focal * basic).mean()


# ─── Eval ──────────────────────────────────────────────────────────────────────


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             label_cols: List[str]) -> dict:
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(device)).cpu().numpy())
            all_targets.append(y.numpy())
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs = 1.0 / (1.0 + np.exp(-logits))

    aurocs, auprcs = [], []
    per_assay = {}
    for j, col in enumerate(label_cols):
        mask = targets[:, j] != 0
        if mask.sum() < 2:
            continue
        y_true = (targets[mask, j] > 0).astype(int)
        if y_true.min() == y_true.max():
            continue
        try:
            au = roc_auc_score(y_true, probs[mask, j])
            ap = average_precision_score(y_true, probs[mask, j])
        except ValueError:
            continue
        aurocs.append(au)
        auprcs.append(ap)
        per_assay[col] = (au, ap)

    return {
        "auroc_macro": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auprc_macro": float(np.mean(auprcs)) if auprcs else float("nan"),
        "n_assays_scored": len(aurocs),
        "per_assay": per_assay,
    }


# ─── Train ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"features={args.features} device={device}")

    feats, labels, splits_idx, label_cols = load_dataset(args)
    in_dim = feats.shape[1]

    def make_loader(name: str, shuffle: bool) -> DataLoader:
        idx = splits_idx[name]
        return DataLoader(
            FpDataset(feats[idx], labels[idx]),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    model = MLP(in_dim, args.hidden_dim, args.num_hidden, len(label_cols)).to(device)
    criterion = FocalBCEMaskedLoss(gamma=2.0)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    logger.info(
        f"in_dim={in_dim} | optim={args.optimizer} lr={args.lr} "
        f"| model params: {sum(p.numel() for p in model.parameters()):,}"
    )

    best_val_auroc, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_metrics = evaluate(model, val_loader, device, label_cols)
        logger.info(
            f"epoch {epoch:3d} | train_loss={np.mean(train_losses):.4f} "
            f"| val_auroc={val_metrics['auroc_macro']:.4f} "
            f"val_auprc={val_metrics['auprc_macro']:.4f}"
        )
        if val_metrics["auroc_macro"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience > args.lr_patience:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
                logger.info(f"plateau: dropped lr to {g['lr']:.4g}")
            patience = 0
            if optimizer.param_groups[0]["lr"] < args.min_lr:
                logger.info(f"lr below {args.min_lr}; early stop")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, label_cols)
    logger.info(
        f"TEST | features={args.features} "
        f"auroc={test_metrics['auroc_macro']:.4f} "
        f"auprc={test_metrics['auprc_macro']:.4f} "
        f"({test_metrics['n_assays_scored']} assays scored)"
    )
    return test_metrics


# ─── Results CSV append (graphium-compatible columns) ──────────────────────────


def append_results_row(test_metrics: dict, args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "experiment_results.csv"

    model_name = f"{args.features}_mlp"  # ecfp_mlp or cpcnn_mlp
    row: dict = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "task": "cell_bioactivity",
        "seed": args.seed,
        "pretrain_dataset": "scratch",
        "is_finetuning": False,
        "unfreeze_depth": "N/A",
        "epoch_unfreeze_all": "N/A",
        "hidden_dim": args.hidden_dim,
        "gnn_depth": "N/A",
        "output_dir": str(Path.cwd()),
        "wandb_tags": f"['{model_name}','baseline','cell_bioactivity']",
        "graph_cell_bioactivity/auroc/test": test_metrics["auroc_macro"],
        "graph_cell_bioactivity/auprc/test": test_metrics["auprc_macro"],
    }

    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            existing = list(csv.DictReader(f).fieldnames or [])
        new_cols = [k for k in row if k not in existing]
        all_cols = existing + new_cols
        if new_cols:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    else:
        all_cols = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        if not csv_path.stat().st_size:
            w.writeheader()
        w.writerow(row)
    logger.info(f"appended row to {csv_path}  (model={model_name})")


# ─── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/home/shpark/prj-molrepr/datacache/cell_bioactivity")
    p.add_argument("--features", choices=("ecfp", "cpcnn"), default="ecfp",
                   help="Feature mode: ecfp (chemistry) or cpcnn (cell-painting).")
    p.add_argument("--csv", default=str(base / "cell_bioactivity.csv"))
    p.add_argument("--splits", default=str(base / "cell_bioactivity_split.csv"))
    p.add_argument("--cpcnn-csv",
                   default="/home/shpark/prj-molrepr/datacache/jump_cpcnn/jump_cpcnn_smiles_embeddings.csv",
                   help="CPCNN embeddings CSV (used when --features=cpcnn).")
    p.add_argument("--results-dir", default="/home/shpark/prj-molrepr/graphium/results")
    p.add_argument("--n-bits", type=int, default=1024,
                   help="ECFP bit count (only used when --features=ecfp).")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-hidden", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", choices=("sgd", "adam", "auto"), default="auto",
                   help="'auto' picks sgd for ecfp (paper default) and adam for cpcnn.")
    p.add_argument("--lr", type=float, default=None,
                   help="Defaults to 2.0 (sgd/ecfp) or 1e-3 (adam/cpcnn) when unset.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:5" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-results-csv", action="store_true",
                   help="Skip appending to results/experiment_results.csv.")
    args = p.parse_args()

    # Resolve per-feature-mode defaults for optimizer + lr when left on 'auto'.
    # ECFP inputs are sparse binary → paper's SGD+lr=2.0 works; dense CPCNN
    # embeddings need Adam+lr=1e-3 to converge (SGD+2.0 explodes gradients).
    if args.optimizer == "auto":
        args.optimizer = "sgd" if args.features == "ecfp" else "adam"
    if args.lr is None:
        args.lr = 2.0 if args.optimizer == "sgd" else 1e-3
    return args


if __name__ == "__main__":
    args = parse_args()
    test_metrics = run(args)
    if not args.no_results_csv:
        append_results_row(test_metrics, args)
