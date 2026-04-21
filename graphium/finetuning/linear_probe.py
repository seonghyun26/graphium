"""
ADMET linear-probe Lightning callback for pre-training.

Hooks into ``on_validation_epoch_end`` and, for each configured ADMET task,
extracts graph-level embeddings from the live backbone, fits a fresh 1-layer
linear head on the TDC train_val split, and logs the metric on the TDC test
split via ``pl_module.log("probe/{task}/{metric}", ...)``.

Enabled via Hydra group ``+probe=admet_linear``. Feature is off by default, so
runs that don't set the probe config are byte-identical to previous behavior.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from graphium.data.collate import graphium_collate_fn
from graphium.finetuning.admet_common import (
    ALLOWED_KINDS as _ALLOWED_KINDS,
    ALLOWED_METRICS as _ALLOWED_METRICS,
    featurizer_hash as _featurizer_hash,
    forward_graph_embedding_batch,
    load_admet_group as _load_admet_group,
    load_and_featurize_task as _load_and_featurize_task,
    score_predictions,
)
from graphium.utils import fs


class ADMETLinearProbeCallback(pl.Callback):
    """Run per-epoch ADMET linear probes alongside pretrain validation."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        # OmegaConf DictConfig is dict-like; coerce to plain dict for safety.
        self.cfg = dict(cfg) if not isinstance(cfg, dict) else cfg
        self.tasks: List[Dict[str, str]] = [dict(t) for t in self.cfg["tasks"]]
        for t in self.tasks:
            if t["kind"] not in _ALLOWED_KINDS:
                raise ValueError(f"probe.tasks[*].kind must be in {_ALLOWED_KINDS}, got {t['kind']}")
            if t["metric"] not in _ALLOWED_METRICS:
                raise ValueError(
                    f"probe.tasks[*].metric must be in {_ALLOWED_METRICS}, got {t['metric']}"
                )
        self.max_epochs = int(self.cfg.get("max_epochs", 50))
        self.batch_size = int(self.cfg.get("batch_size", 256))
        self.lr = float(self.cfg.get("lr", 1e-3))
        self.weight_decay = float(self.cfg.get("weight_decay", 1e-5))
        self.embedding_level = str(self.cfg.get("embedding_level", "graph"))
        self.every_n_val_epochs = int(self.cfg.get("every_n_val_epochs", 1))
        self.tdc_cache_dir: Optional[str] = self.cfg.get("tdc_cache_dir") or None
        self.probe_seed = int(self.cfg.get("probe_seed", 0))
        self.skip_epoch_zero = bool(self.cfg.get("skip_epoch_zero", True))

        self._datasets: Optional[Dict[str, Dict[str, Any]]] = None
        self._collate_fn = None
        self._val_call_idx = 0
        self._last_probed_epoch = -1
        self._last_probe_time = 0.0

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage != "fit" or self._datasets is not None:
            return
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "smiles_transformer"):
            raise RuntimeError(
                "ADMETLinearProbeCallback requires trainer.datamodule to expose "
                "`smiles_transformer` (a graphium featurizer partial)."
            )
        smiles_transformer = datamodule.smiles_transformer

        # Collate: mask NaNs to 0 to match datamodule behaviour. We intentionally
        # skip batch_size_per_pack (packing is disabled in collage_pyg_graph).
        self._collate_fn = partial(graphium_collate_fn, mask_nan=0)

        tdc_cache = self.tdc_cache_dir
        if tdc_cache is None:
            tdc_cache = fs.join(fs.get_cache_dir("tdc"), "ADMET_Benchmark")
        fs.mkdir(tdc_cache, exist_ok=True)

        # Persistent feature cache keyed on featurizer config. Subsequent runs
        # with the same smiles_transformer skip the ~1-min TDC featurization.
        feat_hash = _featurizer_hash(smiles_transformer)
        features_cache = fs.join(tdc_cache, "probe_features", feat_hash)
        fs.mkdir(features_cache, exist_ok=True)

        logger.info(
            "ADMETLinearProbeCallback: featurizing {} ADMET tasks (cache={}, feat_cache={})",
            len(self.tasks),
            tdc_cache,
            features_cache,
        )
        group = _load_admet_group(tdc_cache)

        datasets: Dict[str, Dict[str, Any]] = {}
        for task_cfg in self.tasks:
            name = task_cfg["name"]
            train_graphs, train_y, test_graphs, test_y = _load_and_featurize_task(
                group=group,
                name=name,
                smiles_transformer=smiles_transformer,
                kind=task_cfg["kind"],
                features_cache=features_cache,
            )
            logger.info(
                "  probe/{} — train={}  test={}  kind={}  metric={}",
                name,
                len(train_graphs),
                len(test_graphs),
                task_cfg["kind"],
                task_cfg["metric"],
            )
            datasets[name] = {
                "train_graphs": train_graphs,
                "train_y": train_y,
                "test_graphs": test_graphs,
                "test_y": test_y,
                "kind": task_cfg["kind"],
                "metric": task_cfg["metric"],
            }
        self._datasets = datasets

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or self._datasets is None:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return
        # Guard against duplicate calls within the same epoch. Lightning may
        # fire on_validation_epoch_end more than once per validation pass and
        # increment current_epoch between calls. Use both epoch tracking and a
        # time-based debounce (probes take ~10s, epochs take minutes).
        import time

        current_epoch = trainer.current_epoch
        if self.skip_epoch_zero and current_epoch == 0:
            return
        now = time.monotonic()
        if current_epoch == self._last_probed_epoch or (now - self._last_probe_time) < 60:
            return
        self._last_probed_epoch = current_epoch
        self._last_probe_time = now

        self._val_call_idx += 1
        if self._val_call_idx % self.every_n_val_epochs != 0:
            return

        backbone = pl_module.model
        if not hasattr(backbone, "task_heads") or backbone.task_heads is None:
            logger.warning("ADMETLinearProbeCallback: backbone has no task_heads; skipping probe.")
            return
        if self.embedding_level not in backbone.task_heads.graph_output_nn:
            logger.warning(
                "ADMETLinearProbeCallback: embedding_level={} not found in task_heads.graph_output_nn "
                "(available={}); skipping probe.",
                self.embedding_level,
                list(backbone.task_heads.graph_output_nn.keys()),
            )
            return

        device = pl_module.device
        was_training = backbone.training
        # Isolate RNG so probe doesn't perturb pretrain dropout / ordering.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.probe_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.probe_seed)

            backbone.eval()
            for task_cfg in self.tasks:
                name = task_cfg["name"]
                ds = self._datasets[name]
                try:
                    metric_value = self._probe_one_task(backbone, ds, device)
                except Exception as err:  # noqa: BLE001
                    logger.warning("ADMETLinearProbeCallback: probe/{} failed: {}", name, err)
                    continue
                metric_key = f"probe/{name}/{ds['metric']}"
                pl_module.log(
                    metric_key,
                    float(metric_value),
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                logger.info(
                    "ADMETLinearProbeCallback: {}={:.4f} (epoch={})",
                    metric_key,
                    float(metric_value),
                    trainer.current_epoch,
                )
        finally:
            if was_training:
                backbone.train()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    # ------------------------------------------------------------------
    # Probe core
    # ------------------------------------------------------------------

    def _probe_one_task(
        self,
        backbone: nn.Module,
        ds: Dict[str, Any],
        device: torch.device,
    ) -> float:
        with torch.no_grad():
            z_train = self._extract_embeddings(backbone, ds["train_graphs"], device)
            z_test = self._extract_embeddings(backbone, ds["test_graphs"], device)
        # Embeddings live on CPU (we detach+move after extraction) so the
        # probe training is cheap and doesn't fight for GPU memory.
        y_train = ds["train_y"]
        y_test = ds["test_y"]

        kind = ds["kind"]
        in_dim = z_train.shape[1]
        out_dim = 1  # both regression and binary classification use scalar logit
        head = nn.Linear(in_dim, out_dim).to(device)
        # Lightning's validation loop wraps callbacks in torch.no_grad(); the
        # linear head needs grad enabled to backprop on its own parameters.
        with torch.enable_grad():
            _fit_linear_head(
                head=head,
                z=z_train,
                y=y_train,
                kind=kind,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                device=device,
            )
        return _score_linear_head(
            head=head,
            z=z_test,
            y=y_test,
            kind=kind,
            metric=ds["metric"],
            batch_size=self.batch_size,
            device=device,
        )

    def _extract_embeddings(
        self,
        backbone: nn.Module,
        graphs: List[Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Forward-pass graphs through pre_nn + gnn + graph_output_nn[level]."""
        loader = DataLoader(
            graphs,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
        )
        target_dtype = next(backbone.parameters()).dtype
        chunks: List[torch.Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            z = forward_graph_embedding_batch(
                backbone, batch, self.embedding_level, target_dtype
            )
            chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)


# ----------------------------------------------------------------------
# Helpers (linear-head training lives here since it's probe-specific)
# ----------------------------------------------------------------------

def _fit_linear_head(
    head: nn.Linear,
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    kind: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    z = z.to(device)
    y = y.to(device).float().view(-1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn: nn.Module = nn.MSELoss() if kind == "regression" else nn.BCEWithLogitsLoss()

    n = z.shape[0]
    head.train()
    for _ in range(max_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = head(z[idx]).squeeze(-1)
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
    head.eval()


def _score_linear_head(
    head: nn.Linear,
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    kind: str,
    metric: str,
    batch_size: int,
    device: torch.device,
) -> float:
    z = z.to(device)
    with torch.no_grad():
        preds: List[torch.Tensor] = []
        for i in range(0, z.shape[0], batch_size):
            preds.append(head(z[i : i + batch_size]).squeeze(-1).detach().cpu())
    pred = torch.cat(preds, dim=0)
    return score_predictions(pred, y, kind=kind, metric=metric)
