"""Exponential moving average of model weights, as a Lightning callback.

Maintains a shadow copy of `pl_module.model` parameters updated as

    ema = decay * ema + (1 - decay) * param

with a bias-corrected effective decay during early steps (timm-style),

    effective = min(decay, (1 + n) / (10 + n))

so the shadow doesn't stay anchored at initialization for the first ~100 updates.

Default behaviour:
- Update shadow once per real optimizer step (`on_before_zero_grad`), so grad
  accumulation is handled correctly without extra book-keeping.
- Validation, test, and predict run on the EMA weights; training resumes from
  the raw weights afterwards.
- The shadow is persisted in the regular Lightning checkpoint under the
  callback's namespace, so resumption preserves it.
- At the end of training, write a sibling ``*_ema.ckpt`` with EMA values baked
  into ``state_dict["model.*"]``. That file is a drop-in replacement for the
  raw last-checkpoint when used as a pretrained backbone for finetuning.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.999,
        use_for_validation: bool = True,
        use_for_test: bool = True,
        save_ema_checkpoint: bool = True,
    ):
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.use_for_validation = bool(use_for_validation)
        self.use_for_test = bool(use_for_test)
        self.save_ema_checkpoint = bool(save_ema_checkpoint)

        self._shadow: Dict[str, torch.Tensor] = {}
        self._saved_raw: Optional[Dict[str, torch.Tensor]] = None
        self._n_updates: int = 0

    # --- shadow init / state ---------------------------------------------------

    def _init_shadow(self, pl_module: pl.LightningModule) -> None:
        if self._shadow:
            return
        for name, p in pl_module.model.named_parameters():
            if p.requires_grad:
                self._shadow[name] = p.detach().clone()

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._init_shadow(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "n_updates": self._n_updates,
            "shadow": {k: v.detach().cpu() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self._n_updates = int(state_dict.get("n_updates", 0))
        self._shadow = {k: v.clone() for k, v in state_dict.get("shadow", {}).items()}

    # --- update ----------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self, pl_module: pl.LightningModule) -> None:
        if not self._shadow:
            self._init_shadow(pl_module)

        # Bias-corrected decay: (1+n)/(10+n) ramps from ~0.09 → self.decay over ~100 steps.
        eff = min(self.decay, (1.0 + self._n_updates) / (10.0 + self._n_updates))

        for name, p in pl_module.model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self._shadow.get(name)
            if shadow is None:
                self._shadow[name] = p.detach().clone()
                continue
            if shadow.device != p.device:
                shadow = shadow.to(p.device)
                self._shadow[name] = shadow
            shadow.mul_(eff).add_(p.detach(), alpha=1.0 - eff)

        self._n_updates += 1

    def on_before_zero_grad(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        # Fires once per real optimizer step (post-step, pre zero_grad), so
        # gradient accumulation is automatically respected.
        self._update_ema(pl_module)

    # --- swap for evaluation ---------------------------------------------------

    @torch.no_grad()
    def _swap_in_ema(self, pl_module: pl.LightningModule) -> None:
        if not self._shadow:
            return
        self._saved_raw = {}
        for name, p in pl_module.model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            self._saved_raw[name] = p.detach().clone()
            p.data.copy_(shadow.to(p.device, non_blocking=True))

    @torch.no_grad()
    def _restore_raw(self, pl_module: pl.LightningModule) -> None:
        if self._saved_raw is None:
            return
        for name, p in pl_module.model.named_parameters():
            saved = self._saved_raw.get(name)
            if saved is None:
                continue
            p.data.copy_(saved.to(p.device, non_blocking=True))
        self._saved_raw = None

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_validation and self._shadow:
            self._swap_in_ema(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_validation and self._saved_raw is not None:
            self._restore_raw(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_test and self._shadow:
            self._swap_in_ema(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_test and self._saved_raw is not None:
            self._restore_raw(pl_module)

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_test and self._shadow:
            self._swap_in_ema(pl_module)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_test and self._saved_raw is not None:
            self._restore_raw(pl_module)

    # --- write *_ema.ckpt for downstream finetuning ----------------------------

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.save_ema_checkpoint or not trainer.is_global_zero:
            return
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb is None or getattr(ckpt_cb, "dirpath", None) is None:
            logger.warning("EMACallback: no ModelCheckpoint dirpath; skipping EMA ckpt save")
            return

        dirpath = Path(ckpt_cb.dirpath)
        last = dirpath / "last.ckpt"
        if last.exists():
            base = last
        else:
            ckpts = sorted(dirpath.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpts:
                logger.warning(f"EMACallback: no .ckpt found in {dirpath}")
                return
            base = ckpts[0]

        out = base.with_name(base.stem + "_ema.ckpt")
        try:
            ckpt = torch.load(base, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", {})
            n_replaced = 0
            for shadow_name, shadow_val in self._shadow.items():
                key = f"model.{shadow_name}"
                if key in sd:
                    sd[key] = shadow_val.detach().cpu().clone()
                    n_replaced += 1
            torch.save(ckpt, out)
            logger.info(
                f"EMACallback: wrote {out.name} "
                f"({n_replaced}/{len(self._shadow)} params replaced, "
                f"{self._n_updates} EMA updates, decay={self.decay})"
            )
        except Exception as e:
            logger.warning(f"EMACallback: failed to save EMA ckpt: {e}")


def export_ema_checkpoint(src_ckpt: str, dst_ckpt: str) -> None:
    """Bake the stored EMA shadow into ``state_dict["model.*"]`` and write a new ckpt.

    Useful for converting a mid-training checkpoint into an EMA-based checkpoint
    after the fact, without restarting training.
    """
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    callbacks = ckpt.get("callbacks", {}) or {}
    ema_state = None
    for k, v in callbacks.items():
        if isinstance(v, dict) and "shadow" in v and "decay" in v:
            ema_state = v
            break
    if ema_state is None:
        raise RuntimeError(f"No EMACallback state found in {src_ckpt}")
    sd = ckpt.setdefault("state_dict", {})
    n_replaced = 0
    for shadow_name, shadow_val in ema_state["shadow"].items():
        key = f"model.{shadow_name}"
        if key in sd:
            sd[key] = shadow_val.cpu().clone()
            n_replaced += 1
    torch.save(ckpt, dst_ckpt)
    logger.info(
        f"export_ema_checkpoint: {n_replaced}/{len(ema_state['shadow'])} params "
        f"baked into {dst_ckpt}"
    )
