"""Measure parameter count of a Graphium model config.

Usage (from the graphium/ directory):
    python scripts/count_params.py model=pairmixer_12M tasks=admet
    python scripts/count_params.py model=pairmixer_10M tasks=admet
    python scripts/count_params.py model=pairmixer_40M tasks=admet

Any Hydra override is forwarded (e.g. architecture overrides, task=..., etc.)
"""
import os
import sys
from os.path import dirname, abspath

import hydra
from omegaconf import DictConfig, OmegaConf

import graphium
from graphium.config._loader import load_architecture, load_datamodule, load_accelerator

os.chdir(dirname(dirname(abspath(graphium.__file__))))


def _count(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@hydra.main(version_base=None, config_path="../expts/hydra-configs", config_name="main")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg, accel = load_accelerator(cfg)

    datamodule = load_datamodule(cfg, accel)
    model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)
    model = model_class(**model_kwargs)

    total, trainable = _count(model)
    print(f"\n=== {cfg.get('model', {}).get('name', 'model')} ===")
    print(f"Total params     : {total:>12,d}  ({total/1e6:6.2f} M)")
    print(f"Trainable params : {trainable:>12,d}  ({trainable/1e6:6.2f} M)\n")

    print("Per-submodule breakdown:")
    for name, child in model.named_children():
        t, _ = _count(child)
        print(f"  {name:<24s} {t:>12,d}  ({t/1e6:6.2f} M)")


if __name__ == "__main__":
    main()
