"""Shared helpers for ADMET pre-training monitors.

Used by :class:`graphium.finetuning.linear_probe.ADMETLinearProbeCallback` and
:class:`graphium.finetuning.admet_finetune.ADMETFinetuneCallback` to avoid
duplicating TDC featurization, caching, and backbone-forward logic.
"""
from __future__ import annotations

import hashlib
import pickle
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from graphium.utils import fs


ALLOWED_KINDS = {"regression", "classification"}
ALLOWED_METRICS = {"mae", "spearman", "auroc"}


def featurizer_hash(smiles_transformer) -> str:
    """Stable short hash of the featurizer partial's function + keywords.

    Avoids ``repr(partial)`` because that includes the function's memory
    address, which changes every process.
    """
    fn = getattr(smiles_transformer, "func", smiles_transformer)
    fn_key = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}"
    kwargs = getattr(smiles_transformer, "keywords", {}) or {}
    kw_key = repr(sorted((str(k), repr(v)) for k, v in kwargs.items()))
    key = (fn_key + "|" + kw_key).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def load_admet_group(cache_dir: str):
    try:
        from tdc.benchmark_group import admet_group
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "ADMET monitor callbacks require PyTDC. Install with `pip install PyTDC`."
        ) from err
    # TDC prints aggressively; mute stdout/stderr (same pattern as datamodule).
    with tempfile.TemporaryFile("w") as f:
        with redirect_stderr(f), redirect_stdout(f):
            return admet_group(path=cache_dir)


def featurize_split(df, smiles_transformer, kind: str) -> Tuple[List[Any], torch.Tensor]:
    graphs: List[Any] = []
    labels: List[float] = []
    for smiles, y in zip(df["Drug"].tolist(), df["Y"].tolist()):
        try:
            g = smiles_transformer(smiles, mask_nan=0.0)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(g, (tuple, list)) and g is None:
            continue
        # Featurizer can return a string on error (on_error="ignore" path).
        if isinstance(g, str):
            continue
        graphs.append(g)
        labels.append(float(y))
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    if kind == "classification":
        y_tensor = y_tensor.clamp(min=0.0, max=1.0)
    return graphs, y_tensor


def load_and_featurize_task(
    group,
    name: str,
    smiles_transformer,
    kind: str,
    features_cache: Optional[str] = None,
) -> Tuple[List[Any], torch.Tensor, List[Any], torch.Tensor]:
    cache_path = None
    if features_cache is not None:
        cache_path = fs.join(features_cache, f"{name}.pkl")
        if fs.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    blob = pickle.load(f)
                return (
                    blob["train_graphs"],
                    blob["train_y"],
                    blob["test_graphs"],
                    blob["test_y"],
                )
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "admet_common/{}: cache load failed ({}), re-featurizing.", name, err
                )

    benchmark = group.get(name)
    train_df = benchmark["train_val"] if "train_val" in benchmark else benchmark["train"]
    test_df = benchmark["test"]

    train_graphs, train_y = featurize_split(train_df, smiles_transformer, kind)
    test_graphs, test_y = featurize_split(test_df, smiles_transformer, kind)
    if len(train_graphs) == 0 or len(test_graphs) == 0:
        raise RuntimeError(f"admet_common/{name}: featurization produced empty split")

    if cache_path is not None:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "train_graphs": train_graphs,
                        "train_y": train_y,
                        "test_graphs": test_graphs,
                        "test_y": test_y,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as err:  # noqa: BLE001
            logger.warning("admet_common/{}: cache write failed ({}).", name, err)

    return train_graphs, train_y, test_graphs, test_y


def forward_graph_embedding_batch(
    backbone: nn.Module,
    batch,
    embedding_level: str,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Run a single batch through pre_nn + gnn + graph_output_nn[level].

    Returns the graph-level embedding tensor. Caller manages grad context
    (``torch.no_grad()`` for the linear probe, plain forward for fine-tune).
    Does NOT detach or move to CPU.
    """
    _keys = batch.keys() if callable(getattr(batch, "keys", None)) else list(batch.keys)
    for _key in list(_keys):
        _val = batch[_key]
        if isinstance(_val, torch.Tensor) and _val.is_floating_point():
            batch[_key] = _val.to(target_dtype)

    g = backbone.encoder_manager(batch)
    if backbone.pre_nn is not None:
        g["feat"] = backbone.pre_nn.forward(g["feat"])
    if backbone.pre_nn_edges is not None:
        e = g["edge_feat"]
        if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
            e = torch.zeros(
                list(e.shape[:-1]) + [backbone.pre_nn_edges.out_dim],
                device=e.device,
                dtype=e.dtype,
            )
        else:
            e = backbone.pre_nn_edges.forward(e)
        g["edge_feat"] = e
    g = backbone.gnn.forward(g)
    if backbone.gnn_layer_pooling is not None:
        g["feat"] = backbone.gnn_layer_pooling(backbone.gnn._readout_cache)
    graph_output_nn = backbone.task_heads.graph_output_nn[embedding_level]
    return graph_output_nn(g)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    def _rank(a: torch.Tensor) -> torch.Tensor:
        arr = a.numpy()
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr), dtype=np.float64)
        _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
        sums = np.zeros_like(counts, dtype=np.float64)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        return torch.from_numpy(avg[inv])

    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).item()
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom)


def score_predictions(
    pred: torch.Tensor, y: torch.Tensor, *, kind: str, metric: str
) -> float:
    """Compute mae / spearman / auroc from final predictions.

    ``pred`` is the scalar-logit-per-sample tensor (pre-sigmoid for classification).
    """
    pred = pred.detach().float().cpu().view(-1)
    y_cpu = y.detach().cpu().float().view(-1)
    if metric == "mae":
        return float(torch.mean(torch.abs(pred - y_cpu)).item())
    if metric == "spearman":
        return spearman(pred, y_cpu)
    if metric == "auroc":
        from torchmetrics.functional.classification import binary_auroc

        prob = torch.sigmoid(pred)
        return float(binary_auroc(prob, y_cpu.to(torch.int)).item())
    raise ValueError(f"Unsupported metric: {metric}")
