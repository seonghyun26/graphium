"""Molecule-encoder adapters for downstream tasks.

All encoders implement ``MoleculeEncoder`` from ``.base``. Use ``load_encoder``
to instantiate by name from task drivers so adding a new encoder is one-line.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .base import MoleculeEncoder, bisect_embed

if TYPE_CHECKING:  # pragma: no cover
    from .minimol import MinimolEncoder
    from .mole import MolEEncoder
    from .pairmixer import PairMixerEncoder


_REGISTRY: dict[str, tuple[str, str]] = {
    "pairmixer": ("downstream.model.pairmixer", "PairMixerEncoder"),
    "minimol":   ("downstream.model.minimol",   "MinimolEncoder"),
    "mole":      ("downstream.model.mole",      "MolEEncoder"),
    "ecfp":      ("downstream.model.ecfp",      "ECFPEncoder"),
    "cpcnn":     ("downstream.model.cpcnn",     "CPCNNEncoder"),
}


def load_encoder(name: str, **kwargs) -> MoleculeEncoder:
    """Instantiate an encoder by registry name (``pairmixer`` / ``minimol`` / ``mole``)."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown encoder {name!r}. Known: {sorted(_REGISTRY)}")
    module_path, cls_name = _REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)(**kwargs)


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["MoleculeEncoder", "bisect_embed", "load_encoder", "available_encoders"]
