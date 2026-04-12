"""Utility for loading pretrained backbone weights into a fresh predictor.

Used by the continual pre-training flow: take a checkpoint trained on one task
mix (e.g. LargeMix) and copy its backbone parameters into a freshly instantiated
``FullGraphMultiTaskNetwork`` whose task heads target a different mix
(e.g. BBBC047 + DTI pActivity). Parameters that exist only in the checkpoint
(old task heads) or only in the new model (new task heads) are skipped, leaving
the new heads randomly initialized while the backbone starts from the
pretrained weights.

Unlike the existing ``FullGraphFinetuningNetwork`` path, this keeps the model
as a standard ``FullGraphMultiTaskNetwork`` so checkpoints stay flat and remain
directly consumable by downstream finetuning.
"""

from typing import Dict, List, Optional, Set

from loguru import logger

from graphium.trainer.predictor import PredictorModule


def load_pretrained_backbone(
    predictor: PredictorModule,
    checkpoint_path: str,
    strict: bool = False,
    exclude_prefixes: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """Copy matching parameters from a checkpoint into ``predictor`` in-place.

    A parameter is copied iff its key exists in both state_dicts AND the shapes
    match. Mismatches (different task heads, different output dims) are skipped
    so the new model retains its random initialization for those tensors.

    Args:
        predictor: The freshly instantiated ``PredictorModule`` to populate.
        checkpoint_path: Path to a ``.ckpt`` file (or a registered name in
            ``GRAPHIUM_PRETRAINED_MODELS_DICT``).
        strict: If ``True``, raise on shape mismatches for shared keys instead
            of skipping them. Useful as a sanity check during config bring-up.
        exclude_prefixes: Optional set of key prefixes to skip explicitly,
            e.g. ``{"model.task_heads.graph_output_nn."}`` to force re-init of
            the graph output MLP even when shapes happen to match.

    Returns:
        A dict with keys ``loaded``, ``skipped_shape_mismatch``,
        ``skipped_not_in_new``, ``skipped_not_in_old``, ``skipped_excluded``,
        each mapping to the list of parameter names in that category.
    """
    exclude_prefixes = set(exclude_prefixes or set())

    logger.info(f"Continual pre-training: loading backbone from {checkpoint_path}")
    pretrained_predictor = PredictorModule.load_pretrained_model(
        checkpoint_path, device="cpu"
    )
    old_sd = pretrained_predictor.state_dict()
    new_sd = predictor.state_dict()

    loaded: List[str] = []
    skipped_shape: List[str] = []
    skipped_not_in_new: List[str] = []
    skipped_excluded: List[str] = []

    for key, old_param in old_sd.items():
        if any(key.startswith(p) for p in exclude_prefixes):
            skipped_excluded.append(key)
            continue
        if key not in new_sd:
            skipped_not_in_new.append(key)
            continue
        if old_param.shape != new_sd[key].shape:
            if strict:
                raise ValueError(
                    f"Shape mismatch for parameter '{key}': "
                    f"checkpoint={tuple(old_param.shape)}, "
                    f"model={tuple(new_sd[key].shape)}"
                )
            skipped_shape.append(key)
            continue
        new_sd[key] = old_param
        loaded.append(key)

    skipped_not_in_old = [k for k in new_sd.keys() if k not in old_sd]

    predictor.load_state_dict(new_sd)

    logger.info(
        f"Continual pre-training: loaded {len(loaded)} params from checkpoint, "
        f"left {len(skipped_not_in_old)} params at random init "
        f"(typically new task heads)."
    )
    if skipped_not_in_new:
        logger.info(
            f"Continual pre-training: skipped {len(skipped_not_in_new)} checkpoint "
            f"params not present in new model (typically old task heads / output levels)."
        )
    if skipped_shape:
        logger.warning(
            f"Continual pre-training: skipped {len(skipped_shape)} params due to "
            f"shape mismatch: {skipped_shape}"
        )
    if skipped_excluded:
        logger.info(
            f"Continual pre-training: skipped {len(skipped_excluded)} params via "
            f"exclude_prefixes."
        )

    # Free the duplicate model from CPU memory.
    del pretrained_predictor, old_sd

    return {
        "loaded": loaded,
        "skipped_shape_mismatch": skipped_shape,
        "skipped_not_in_new": skipped_not_in_new,
        "skipped_not_in_old": skipped_not_in_old,
        "skipped_excluded": skipped_excluded,
    }
