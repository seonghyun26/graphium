"""
ADMET fine-tune Lightning callback for pre-training.

Runs alongside :class:`graphium.finetuning.linear_probe.ADMETLinearProbeCallback`
so you can watch the correlation between linear-probe and fine-tune performance
on the same 5 representative ADMET tasks as pre-training progresses.

Mechanics per validation-epoch call:
  1. Snapshot the backbone state_dict (cheap; restored at the end).
  2. For each configured task, build a fresh MLP head on top of the live
     backbone, fine-tune end-to-end for ``max_epochs`` with Adam, evaluate
     on the TDC test split, and log ``finetune/{task}/{metric}``.
  3. Restore the backbone state_dict and clear any gradients left over, so
     the next pre-training step is numerically identical to a run without
     this callback.

Enabled via Hydra group ``+ft_monitor=admet_mlp``. Opt-in (default disabled).
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from loguru import logger

from graphium.data.collate import graphium_collate_fn
from graphium.finetuning.admet_common import (
    ALLOWED_KINDS,
    ALLOWED_METRICS,
    featurizer_hash,
    forward_graph_embedding_batch,
    load_admet_group,
    load_and_featurize_task,
    score_predictions,
)
from graphium.utils import fs


class ADMETFinetuneCallback(pl.Callback):
    """End-to-end fine-tune monitor for pre-training runs.

    Companion to the linear-probe callback: same task list, same TDC splits,
    same featurization cache — but trains the backbone jointly with a small
    MLP head instead of keeping the backbone frozen.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = dict(cfg) if not isinstance(cfg, dict) else cfg
        self.tasks: List[Dict[str, str]] = [dict(t) for t in self.cfg["tasks"]]
        for t in self.tasks:
            if t["kind"] not in ALLOWED_KINDS:
                raise ValueError(f"ft_monitor.tasks[*].kind must be in {ALLOWED_KINDS}, got {t['kind']}")
            if t["metric"] not in ALLOWED_METRICS:
                raise ValueError(
                    f"ft_monitor.tasks[*].metric must be in {ALLOWED_METRICS}, got {t['metric']}"
                )
        self.max_epochs = int(self.cfg.get("max_epochs", 20))
        self.batch_size = int(self.cfg.get("batch_size", 64))
        self.lr_backbone = float(self.cfg.get("lr_backbone", 1e-4))
        self.lr_head = float(self.cfg.get("lr_head", 1e-3))
        self.weight_decay = float(self.cfg.get("weight_decay", 0.0))
        self.hidden_dim = int(self.cfg.get("hidden_dim", 256))
        self.depth = int(self.cfg.get("depth", 2))
        self.dropout = float(self.cfg.get("dropout", 0.0))
        self.embedding_level = str(self.cfg.get("embedding_level", "graph"))
        self.every_n_val_epochs = int(self.cfg.get("every_n_val_epochs", 1))
        self.tdc_cache_dir: Optional[str] = self.cfg.get("tdc_cache_dir") or None
        self.finetune_seed = int(self.cfg.get("finetune_seed", 0))
        self.skip_epoch_zero = bool(self.cfg.get("skip_epoch_zero", True))

        self._datasets: Optional[Dict[str, Dict[str, Any]]] = None
        self._collate_fn = None
        self._val_call_idx = 0
        self._last_called_epoch = -1
        self._last_called_time = 0.0

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage != "fit" or self._datasets is not None:
            return
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "smiles_transformer"):
            raise RuntimeError(
                "ADMETFinetuneCallback requires trainer.datamodule to expose "
                "`smiles_transformer` (a graphium featurizer partial)."
            )
        smiles_transformer = datamodule.smiles_transformer
        self._collate_fn = partial(graphium_collate_fn, mask_nan=0)

        tdc_cache = self.tdc_cache_dir
        if tdc_cache is None:
            tdc_cache = fs.join(fs.get_cache_dir("tdc"), "ADMET_Benchmark")
        fs.mkdir(tdc_cache, exist_ok=True)

        # Share the feature cache with the linear-probe callback: same hash.
        feat_hash = featurizer_hash(smiles_transformer)
        features_cache = fs.join(tdc_cache, "probe_features", feat_hash)
        fs.mkdir(features_cache, exist_ok=True)

        logger.info(
            "ADMETFinetuneCallback: featurizing {} ADMET tasks (cache={}, feat_cache={})",
            len(self.tasks),
            tdc_cache,
            features_cache,
        )
        group = load_admet_group(tdc_cache)

        datasets: Dict[str, Dict[str, Any]] = {}
        for task_cfg in self.tasks:
            name = task_cfg["name"]
            train_graphs, train_y, test_graphs, test_y = load_and_featurize_task(
                group=group,
                name=name,
                smiles_transformer=smiles_transformer,
                kind=task_cfg["kind"],
                features_cache=features_cache,
            )
            logger.info(
                "  finetune/{} — train={}  test={}  kind={}  metric={}",
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
        import time

        current_epoch = trainer.current_epoch
        if self.skip_epoch_zero and current_epoch == 0:
            return
        now = time.monotonic()
        # De-dup: Lightning may fire this hook more than once per val pass;
        # guard by epoch and a coarse wall-clock debounce (fine-tune is slow).
        if current_epoch == self._last_called_epoch or (now - self._last_called_time) < 60:
            return
        self._last_called_epoch = current_epoch
        self._last_called_time = now

        self._val_call_idx += 1
        if self._val_call_idx % self.every_n_val_epochs != 0:
            return

        backbone = pl_module.model
        if not hasattr(backbone, "task_heads") or backbone.task_heads is None:
            logger.warning("ADMETFinetuneCallback: backbone has no task_heads; skipping.")
            return
        if self.embedding_level not in backbone.task_heads.graph_output_nn:
            logger.warning(
                "ADMETFinetuneCallback: embedding_level={} not in task_heads.graph_output_nn "
                "(available={}); skipping.",
                self.embedding_level,
                list(backbone.task_heads.graph_output_nn.keys()),
            )
            return

        device = pl_module.device
        was_training = backbone.training

        # Snapshot backbone weights on CPU (keeps GPU memory headroom). Restored
        # before returning, so pre-training is numerically identical to a run
        # without this callback.
        initial_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}

        # Isolate RNG so fine-tune doesn't perturb pretrain dropout / ordering.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.finetune_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.finetune_seed)

            for task_cfg in self.tasks:
                name = task_cfg["name"]
                ds = self._datasets[name]
                try:
                    metric_value = self._finetune_one_task(
                        backbone=backbone,
                        initial_state=initial_state,
                        ds=ds,
                        device=device,
                    )
                except Exception as err:  # noqa: BLE001
                    logger.warning("ADMETFinetuneCallback: finetune/{} failed: {}", name, err)
                    continue
                metric_key = f"finetune/{name}/{ds['metric']}"
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
                    "ADMETFinetuneCallback: {}={:.4f} (epoch={})",
                    metric_key,
                    float(metric_value),
                    trainer.current_epoch,
                )
        finally:
            # Restore backbone and clear any grad state we left on its params.
            backbone.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
            for p in backbone.parameters():
                p.grad = None
            if was_training:
                backbone.train()
            else:
                backbone.eval()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    # ------------------------------------------------------------------
    # Fine-tune one ADMET task
    # ------------------------------------------------------------------

    def _finetune_one_task(
        self,
        backbone: nn.Module,
        initial_state: Dict[str, torch.Tensor],
        ds: Dict[str, Any],
        device: torch.device,
    ) -> float:
        # Always start each task from the same pretrain snapshot.
        backbone.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
        backbone.train()

        kind = ds["kind"]
        train_graphs: List[Any] = ds["train_graphs"]
        train_y: torch.Tensor = ds["train_y"].to(device).float().view(-1)
        test_graphs: List[Any] = ds["test_graphs"]
        test_y: torch.Tensor = ds["test_y"]

        # Infer embedding dim from one forward pass.
        target_dtype = next(backbone.parameters()).dtype
        with torch.no_grad():
            warmup_batch = self._collate_fn(train_graphs[: min(4, len(train_graphs))]).to(device)
            z_probe = forward_graph_embedding_batch(
                backbone, warmup_batch, self.embedding_level, target_dtype
            )
            in_dim = int(z_probe.shape[1])

        head = _build_mlp_head(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        ).to(device)

        # Optimize backbone (minus per-task pretrain heads, which aren't used
        # in the forward path) + the new MLP head. Two LR groups because a
        # freshly-initialized head needs a larger step size than an already-
        # trained backbone.
        backbone_params = _backbone_finetune_params(backbone)
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": head.parameters(), "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )
        loss_fn: nn.Module = nn.MSELoss() if kind == "regression" else nn.BCEWithLogitsLoss()

        n = len(train_graphs)
        with torch.enable_grad():
            for _ in range(self.max_epochs):
                perm = torch.randperm(n, device="cpu").tolist()
                for i in range(0, n, self.batch_size):
                    idx = perm[i : i + self.batch_size]
                    batch_graphs = [train_graphs[j] for j in idx]
                    batch_y = train_y[torch.tensor(idx, device=device)]
                    batch = self._collate_fn(batch_graphs).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    z = forward_graph_embedding_batch(
                        backbone, batch, self.embedding_level, target_dtype
                    )
                    pred = head(z).squeeze(-1)
                    loss = loss_fn(pred, batch_y)
                    loss.backward()
                    optimizer.step()

        # Evaluate.
        backbone.eval()
        head.eval()
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(test_graphs), self.batch_size):
                chunk = test_graphs[i : i + self.batch_size]
                batch = self._collate_fn(chunk).to(device)
                z = forward_graph_embedding_batch(
                    backbone, batch, self.embedding_level, target_dtype
                )
                preds.append(head(z).squeeze(-1).detach().cpu())
        pred = torch.cat(preds, dim=0)
        return score_predictions(pred, test_y, kind=kind, metric=ds["metric"])


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _build_mlp_head(in_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Module:
    """Fresh MLP: (Linear, ReLU, Dropout) × (depth-1) → Linear(1).

    ``depth=1`` yields a plain linear head (matches the linear probe's head
    for a fair ablation).
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _backbone_finetune_params(backbone: nn.Module) -> List[nn.Parameter]:
    """All backbone params EXCEPT the per-task pretrain heads.

    ``task_heads.graph_output_nn`` IS used in the fine-tune forward path and
    stays in the optimizer, while ``task_heads.task_heads`` (the per-pretrain-
    task MLPs) does not receive gradients and is excluded to avoid spurious
    weight-decay drift on unused parameters.
    """
    skip_prefix = "task_heads.task_heads."
    return [
        p for name, p in backbone.named_parameters()
        if not name.startswith(skip_prefix) and p.requires_grad
    ]
