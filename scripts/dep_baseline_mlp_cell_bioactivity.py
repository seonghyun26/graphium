"""
MLP baseline for the cell_bioactivity benchmark (Fredinh et al. 2024).

Mirrors `MLP_predictor.py` from github.com/cfredinh/bioactive (3-layer MLP,
focal-BCE masked loss, SGD); the upstream script is generic over per-compound
feature vectors. The paper runs it with chemistry features (ECFP) and with
cell-painting features (CPCNN embeddings); we additionally support Minimol
and MolE pretrained molecule embeddings to benchmark learned chemistry
representations against the paper's ECFP baseline.

Pick which via `--features {ecfp,cpcnn,minimol,mole}`.

Reads graphium's CSV outputs from `scripts/prep_cell_bioactivity.py` and
appends one row to `results/experiment_results.csv` (model name
`{features}_mlp`, e.g. `minimol_mlp`) so the dashboard notebook picks it up
alongside GNN runs.

Feature modes
-------------
- `ecfp`    : ECFP4 (r=2, 1024 bits) computed on-the-fly from the `smiles`
              column. Paper's compound-only chemistry baseline.
- `cpcnn`   : 672-dim CPCNN cell-painting embeddings, mean-aggregated per
              SMILES from `--cpcnn-csv`. Compounds without a CPCNN profile
              (no Cell Painting wells) are dropped from the splits.
- `minimol` : 512-dim Minimol (Kläser et al., 2024) embeddings from the
              frozen pretrained encoder (`pip install minimol`). Cached
              per-SMILES at `datacache/minimol_embeddings/cell_bioactivity.pt`
              and reused across folds/seeds.
- `mole`    : 1000-dim MolE (gin_concat_R1000_E8000_lambda0.0001; Méndez-Lucio
              et al., 2025) embeddings from the frozen pretrained encoder,
              using the Zenodo checkpoint fetched by
              `scripts/mole/download_mole_baseline.sh`. Cached at
              `datacache/mole_embeddings/cell_bioactivity.pt`.

Loss is the upstream `BCEMASKEDLoss` (focal BCE-with-logits, gamma=2) with
the ±1/0 label encoding produced internally by remapping graphium's 0/1/NaN.

Usage
-----
    # Chemistry baseline
    python scripts/baseline_mlp_cell_bioactivity.py --features ecfp

    # Cell-painting baseline
    python scripts/baseline_mlp_cell_bioactivity.py --features cpcnn

    # Pretrained-chemistry baselines
    python scripts/baseline_mlp_cell_bioactivity.py --features minimol --cv
    python scripts/baseline_mlp_cell_bioactivity.py --features mole --cv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


# Repo root for loading sibling scripts/caches; baseline lives in scripts/.
ROOT = Path(__file__).resolve().parents[1]
MINIMOL_CACHE_DIR = ROOT / "datacache" / "minimol_embeddings"
MOLE_CACHE_DIR = ROOT / "datacache" / "mole_embeddings"


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


def _embed_with_bisect(
    smiles_list: List[str],
    embed_chunk: "Callable[[List[str]], Dict[str, np.ndarray]]",
    *, chunk_size: int, label: str,
) -> Dict[str, np.ndarray]:
    """Embed a list of SMILES in chunks; bisect failing chunks down to singletons
    so one poisonous SMILES (e.g. kekulize errors, exotic bond types) doesn't
    crash the whole featurization. Returns {smi -> vector} for successes only.
    """
    from tqdm import tqdm
    results: Dict[str, np.ndarray] = {}
    stack: List[List[str]] = []
    for i in range(0, len(smiles_list), chunk_size):
        stack.append(smiles_list[i : i + chunk_size])
    pbar = tqdm(total=len(smiles_list), desc=f"{label} embedding",
                unit="mol", smoothing=0.05)
    while stack:
        chunk = stack.pop()
        if not chunk:
            continue
        try:
            results.update(embed_chunk(chunk))
            pbar.update(len(chunk))
        except Exception:
            if len(chunk) == 1:
                pbar.update(1)
                continue
            mid = len(chunk) // 2
            stack.append(chunk[mid:])
            stack.append(chunk[:mid])
    pbar.close()
    return results


def featurize_minimol(smiles_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Frozen Minimol 512-d embeddings, cached per-SMILES across folds/seeds.

    Drops SMILES the upstream featurizer can't handle (rare kekulize / valence
    issues); those rows become invalid so split_indices() excludes them.
    """
    MINIMOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MINIMOL_CACHE_DIR / "cell_bioactivity.pt"
    cache: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        raw_cache = torch.load(cache_path, weights_only=False)
        for s, v in raw_cache.items():
            cache[s] = v.detach().cpu().float().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)

    unique_smiles = list(dict.fromkeys(smiles_series.tolist()))
    missing = [s for s in unique_smiles if s not in cache]
    if missing:
        logger.info(f"minimol: embedding {len(missing):,} / {len(unique_smiles):,} new SMILES")
        # Imported lazily: the `minimol` package spins up a torch model at import time.
        from minimol import Minimol
        model = Minimol()

        def _embed(chunk: List[str]) -> Dict[str, np.ndarray]:
            embs = model(chunk)
            return {s: e.detach().cpu().float().numpy() for s, e in zip(chunk, embs)}

        new_embs = _embed_with_bisect(missing, _embed, chunk_size=128, label="minimol")
        cache.update(new_embs)
        # Save as torch tensors to match existing DTI-eval cache format.
        torch.save({s: torch.from_numpy(v) for s, v in cache.items()}, cache_path)

    return _materialize_cache(smiles_series, cache, label="minimol")


def featurize_mole(smiles_series: pd.Series, device: str,
                   batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Frozen MolE 1000-d (gin_concat_R1000) embeddings, cached per-SMILES.

    Uses the Zenodo checkpoint + upstream inference path from
    scripts/mole/mole_eval.py. Drops SMILES the upstream featurizer can't
    handle (DATIVE bonds, etc.) — those rows become invalid.
    """
    MOLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MOLE_CACHE_DIR / "cell_bioactivity.pt"
    cache: Dict[str, np.ndarray] = (
        torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    )

    unique_smiles = list(dict.fromkeys(smiles_series.tolist()))
    missing = [s for s in unique_smiles if s not in cache]
    if missing:
        logger.info(f"mole: embedding {len(missing):,} / {len(unique_smiles):,} new SMILES")
        sys.path.insert(0, str(ROOT / "scripts" / "mole"))
        from mole_eval import ensure_upstream_checkpoint_layout, MOLE_MODEL_NAME
        repo_dir = ensure_upstream_checkpoint_layout()
        sys.path.insert(0, str(repo_dir))
        from dataset.dataset_representation import batch_representation, load_pretrained_model

        with contextlib.redirect_stdout(io.StringIO()):
            model = load_pretrained_model(
                pretrain_architecture="gin_concat",
                pretrained_model=MOLE_MODEL_NAME,
                pretrained_dir=str(repo_dir / "ckpt"),
                device=device,
            )

        def _embed(chunk: List[str]) -> Dict[str, np.ndarray]:
            smile_df = pd.DataFrame({"chem_id": chunk, "smiles": chunk})
            embs_df = batch_representation(smile_df, model, batch_size=batch_size, device=device)
            return {str(cid): row.to_numpy(dtype=np.float32) for cid, row in embs_df.iterrows()}

        new_embs = _embed_with_bisect(missing, _embed, chunk_size=256, label="mole")
        cache.update(new_embs)
        torch.save(cache, cache_path)

    return _materialize_cache(smiles_series, cache, label="mole")


def _materialize_cache(
    smiles_series: pd.Series, cache: Dict[str, np.ndarray], *, label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack cached embeddings in `smiles_series` order; mark missing as invalid."""
    first_vec = next(iter(cache.values())) if cache else None
    if first_vec is None:
        raise RuntimeError(f"{label}: no embeddings produced (all SMILES failed)")
    dim = int(first_vec.shape[0])
    matrix = np.zeros((len(smiles_series), dim), dtype=np.float32)
    valid = np.zeros(len(smiles_series), dtype=bool)
    for i, smi in enumerate(smiles_series):
        vec = cache.get(smi)
        if vec is None:
            continue
        matrix[i] = vec
        valid[i] = True
    dropped = int((~valid).sum())
    if dropped:
        logger.warning(f"{label}: {dropped:,} SMILES couldn't be embedded; rows will be dropped")
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
    """Load prep'd CSV, featurize per `--features`, return (feats, labels,
    folds, valid_mask, label_cols). Split rotation is done per-fold in run_cv."""
    df = pd.read_csv(args.csv)
    if "fold" not in df.columns:
        raise RuntimeError(
            "cell_bioactivity.csv is missing the `fold` column; re-run "
            "scripts/prep_cell_bioactivity.py to regenerate it."
        )
    label_cols = [c for c in df.columns if c.startswith("assay_")]
    logger.info(f"loaded {len(df):,} compounds x {len(label_cols)} assays")

    if args.features == "ecfp":
        feats, valid = featurize_ecfp(df.smiles, n_bits=args.n_bits)
    elif args.features == "cpcnn":
        feats, valid = featurize_cpcnn(df.smiles, Path(args.cpcnn_csv))
    elif args.features == "minimol":
        feats, valid = featurize_minimol(df.smiles)
    elif args.features == "mole":
        feats, valid = featurize_mole(df.smiles, args.embed_device, args.embed_batch_size)
    else:
        raise ValueError(f"unknown --features {args.features!r}")

    if (~valid).any():
        logger.warning(
            f"{(~valid).sum()} compounds invalid for --features={args.features} "
            "(unparseable SMILES or missing profile); will be dropped per-fold"
        )

    # Labels: 0/1/NaN -> -1/+1/0 (BCEMASKEDLoss convention).
    raw = df[label_cols].values.astype(np.float32)
    labels = np.where(np.isnan(raw), 0.0, np.where(raw > 0.5, 1.0, -1.0)).astype(np.float32)

    folds = df.fold.values.astype(np.int8)
    return feats, labels, folds, valid, label_cols


def split_indices(folds: np.ndarray, valid: np.ndarray, test_fold: int,
                  n_folds: int = 6) -> dict:
    """For 6-fold CV rotation: test={test_fold}, val={(test_fold+1)%6}, train=rest."""
    val_fold = (test_fold + 1) % n_folds
    idx_all = np.arange(len(folds))
    test_idx = idx_all[(folds == test_fold) & valid]
    val_idx = idx_all[(folds == val_fold) & valid]
    train_idx = idx_all[(~np.isin(folds, [test_fold, val_fold])) & valid]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


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
            # Empty-mask batch: keep a connection to the graph so .backward()
            # doesn't crash with "element 0 of tensors does not require grad".
            return (logits * 0.0).sum()
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


def _train_one_fold(
    feats: np.ndarray, labels: np.ndarray, splits_idx: dict,
    label_cols: List[str], args: argparse.Namespace, device: torch.device,
    tag: str = "",
) -> dict:
    """Train once on the given train/val/test index split. Dense features
    (CPCNN) are standardized using the *train-split* stats of this fold so CV
    folds don't leak test stats back into training.
    """
    # Per-fold standardization of dense features (no-op for binary ECFP).
    # Dense pretrained molecule embeddings (Minimol/MolE) benefit from the same
    # standardization as CPCNN — stats are computed on the current fold's train
    # split only to avoid leaking test statistics.
    if args.features in {"cpcnn", "minimol", "mole"}:
        mu = feats[splits_idx["train"]].mean(axis=0, keepdims=True)
        sd = feats[splits_idx["train"]].std(axis=0, keepdims=True) + 1e-6
        feats = ((feats - mu) / sd).astype(np.float32)

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

    in_dim = feats.shape[1]
    model = MLP(in_dim, args.hidden_dim, args.num_hidden, len(label_cols)).to(device)
    criterion = FocalBCEMaskedLoss(gamma=2.0)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    logger.info(
        f"[{tag}] in_dim={in_dim} optim={args.optimizer} lr={args.lr} "
        f"params={sum(p.numel() for p in model.parameters()):,} "
        f"sizes(train/val/test)={len(splits_idx['train']):,}/"
        f"{len(splits_idx['val']):,}/{len(splits_idx['test']):,}"
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
        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                f"[{tag}] epoch {epoch:3d} | train_loss={np.mean(train_losses):.4f} "
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
            patience = 0
            if optimizer.param_groups[0]["lr"] < args.min_lr:
                logger.info(f"[{tag}] lr below {args.min_lr} at epoch {epoch}; early stop")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, label_cols)
    logger.info(
        f"[{tag}] TEST auroc={test_metrics['auroc_macro']:.4f} "
        f"auprc={test_metrics['auprc_macro']:.4f} "
        f"(best val_auroc={best_val_auroc:.4f}, "
        f"{test_metrics['n_assays_scored']} assays scored)"
    )
    return test_metrics


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"features={args.features} device={device} cv={args.cv}")
    feats, labels, folds, valid, label_cols = load_dataset(args)

    # Default test fold (matches the original pre-packed splits CSV: test=5, val=4).
    test_folds = list(range(6)) if args.cv else [5]

    per_fold = []
    for k in test_folds:
        splits_idx = split_indices(folds, valid, test_fold=k)
        tag = f"fold{k}" if args.cv else "single"
        # Reset seeds per fold so fold N isn't influenced by fold N-1 training.
        torch.manual_seed(args.seed + k)
        np.random.seed(args.seed + k)
        per_fold.append(_train_one_fold(feats, labels, splits_idx, label_cols, args, device, tag))

    aurocs = np.array([m["auroc_macro"] for m in per_fold])
    auprcs = np.array([m["auprc_macro"] for m in per_fold])
    summary = {
        "auroc_mean": float(aurocs.mean()),
        "auroc_std": float(aurocs.std(ddof=0)),
        "auprc_mean": float(auprcs.mean()),
        "auprc_std": float(auprcs.std(ddof=0)),
        "n_folds": len(per_fold),
        "per_fold_auroc": aurocs.tolist(),
        "per_fold_auprc": auprcs.tolist(),
    }
    if args.cv:
        logger.info(
            f"CV SUMMARY ({len(per_fold)} folds) | "
            f"auroc={summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f} | "
            f"auprc={summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}"
        )
        logger.info(f"per-fold auroc: "
                    f"{', '.join(f'{a:.4f}' for a in aurocs)}")
    return summary


# ─── Results CSV append (graphium-compatible columns) ──────────────────────────


def append_results_row(summary: dict, args: argparse.Namespace) -> None:
    """Append one row to results/experiment_results.csv. For CV runs the
    `test` columns are the fold-mean; fold-std lands in `_std` sibling columns.
    """
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "experiment_results.csv"

    model_name = f"{args.features}_mlp"  # ecfp_mlp or cpcnn_mlp
    tag_list = [model_name, "baseline", "cell_bioactivity"]
    if args.cv:
        tag_list.append("cv6")
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
        "wandb_tags": "[" + ",".join(f"'{t}'" for t in tag_list) + "]",
        "n_folds": summary["n_folds"],
        "graph_cell_bioactivity/auroc/test": summary["auroc_mean"],
        "graph_cell_bioactivity/auroc_std/test": summary["auroc_std"],
        "graph_cell_bioactivity/auprc/test": summary["auprc_mean"],
        "graph_cell_bioactivity/auprc_std/test": summary["auprc_std"],
    }
    # Per-fold AUROC for downstream analysis (only meaningful when --cv).
    for i, au in enumerate(summary["per_fold_auroc"]):
        row[f"graph_cell_bioactivity/auroc_fold{i}/test"] = au

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
    p.add_argument("--features", choices=("ecfp", "cpcnn", "minimol", "mole"), default="ecfp",
                   help="Feature mode: ecfp (chemistry), cpcnn (cell-painting), "
                        "minimol (512-d pretrained), or mole (256-d pretrained).")
    p.add_argument("--csv", default=str(base / "cell_bioactivity.csv"))
    p.add_argument("--splits", default=str(base / "cell_bioactivity_split.csv"))
    p.add_argument("--cpcnn-csv",
                   default="/home/shpark/prj-molrepr/datacache/jump_cpcnn/jump_cpcnn_smiles_embeddings.csv",
                   help="CPCNN embeddings CSV (used when --features=cpcnn).")
    p.add_argument("--embed-device", default=None,
                   help="Device for minimol/mole embedding pass "
                        "(default: cuda:0 if available else cpu; only affects embedding).")
    p.add_argument("--embed-batch-size", type=int, default=2048,
                   help="Batch size for minimol/mole upstream embedding call.")
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
    p.add_argument("--cv", action="store_true",
                   help="Run 6-fold CV (test rotated across folds 0..5). "
                        "Default: single split with test=fold 5, val=fold 4.")
    p.add_argument("--no-results-csv", action="store_true",
                   help="Skip appending to results/experiment_results.csv.")
    args = p.parse_args()

    # Resolve per-feature-mode defaults for optimizer + lr when left on 'auto'.
    # ECFP inputs are sparse binary → paper's SGD+lr=2.0 works; dense learned
    # embeddings (CPCNN / Minimol / MolE) need Adam+lr=1e-3 to converge
    # (SGD+2.0 explodes gradients when inputs are continuous).
    if args.optimizer == "auto":
        args.optimizer = "sgd" if args.features == "ecfp" else "adam"
    if args.lr is None:
        args.lr = 2.0 if args.optimizer == "sgd" else 1e-3
    if args.embed_device is None:
        args.embed_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    args = parse_args()
    summary = run(args)
    if not args.no_results_csv:
        append_results_row(summary, args)
