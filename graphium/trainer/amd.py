"""
Adaptive Modality Dropout (AMD) from GRAM-DTI (arxiv.org/abs/2509.21971).

At each training step, AMD measures the gradient norm of each modality's loss
with respect to the shared encoder parameters, then drops modalities whose
norm lies outside ``[mean / threshold, mean * threshold]`` — i.e., modalities
whose gradient either dominates or is unusually small. The remaining (active)
modalities are summed and returned as the effective training loss.

Notes
-----
Gradient norms are measured via ``torch.autograd.grad(..., retain_graph=True)``
so that the aggregated loss can still be backpropagated by the outer training
loop. The measurement itself does not populate ``.grad`` on any parameter.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def _flatten_grad_norm(grads: List[Optional[Tensor]]) -> Tensor:
    nonnull = [g.detach().flatten() for g in grads if g is not None]
    if not nonnull:
        return torch.zeros(())
    return torch.cat(nonnull).norm(p=2)


def group_losses_by_modality(
    task_losses: Dict[str, Tensor],
    task_to_modality: Optional[Dict[str, str]] = None,
) -> Dict[str, Tensor]:
    """Aggregate per-task losses into per-modality losses (sum within group).

    When ``task_to_modality`` is empty, each task is treated as its own
    modality. Lookup tolerates both prefixed task keys (e.g. ``graph_lpm24``)
    and their unprefixed forms (e.g. ``lpm24``).
    """
    if not task_to_modality:
        return dict(task_losses)

    def resolve(key: str) -> str:
        if key in task_to_modality:
            return task_to_modality[key]
        unprefixed = key.split("_", 1)[-1]
        if unprefixed in task_to_modality:
            return task_to_modality[unprefixed]
        return key

    per_modality: Dict[str, Tensor] = {}
    for task, loss in task_losses.items():
        mod = resolve(task)
        per_modality[mod] = per_modality[mod] + loss if mod in per_modality else loss
    return per_modality


def compute_modality_grad_norms(
    modality_losses: Dict[str, Tensor],
    shared_params: List[torch.nn.Parameter],
) -> Dict[str, Tensor]:
    """L2 norm of d(loss)/d(shared_params) for each modality, graph-preserving."""
    norms: Dict[str, Tensor] = {}
    for name, loss in modality_losses.items():
        grads = torch.autograd.grad(
            loss,
            shared_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        norms[name] = _flatten_grad_norm(list(grads))
    return norms


def select_active_modalities(
    grad_norms: Dict[str, Tensor], threshold: float
) -> Tuple[List[str], Tensor]:
    """Keep modalities with ``mean/threshold <= norm <= mean*threshold``.

    If the filter would empty the set (all modalities outside the band), all
    modalities are kept to avoid a zero-loss step.
    """
    vals = torch.stack(list(grad_norms.values()))
    mean_norm = vals.mean()
    lo = mean_norm / threshold
    hi = mean_norm * threshold
    active = [name for name, gn in grad_norms.items() if (gn >= lo) and (gn <= hi)]
    if not active:
        active = list(grad_norms.keys())
    return active, mean_norm
