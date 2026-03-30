"""
PairMixer layer for Graphium, based on the PairMixer architecture.

Reference:
    Gao et al., "Triangle Multiplication Is All You Need for Biomolecular
    Structure Representations", arXiv:2510.18870, 2025.

PairMixer simplifies the Pairformer by removing:
    - Triangle attention (both starting and ending)
    - Attention with pair bias on the single track

The backbone operates exclusively on the pair representation z via:
    1. Triangle multiplication (outgoing)
    2. Triangle multiplication (incoming)
    3. Transition MLP on z

The single (node) track receives only a simple transition MLP update
(no attention). This yields significant speedups on large molecules due
to eliminating the O(N^2 * H * D) triangle attention operations while
retaining the O(N^3) triangle multiplication that captures the essential
triangular geometric constraints.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint
from torch_geometric.data import Batch

from graphium.nn.base_graph_layer import BaseGraphModule
from graphium.nn.base_layers import MLP, MoELayer
from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from graphium.utils.decorators import classproperty

from .pairformer_pyg import (
    OuterProductMean,
    Transition,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    _get_dropout_mask,
)


class PairMixerLayerPyg(BaseGraphModule):
    """A single PairMixer layer adapted for Graphium.

    Like PairformerLayerPyg, this layer maintains two representation tracks:
        - **Single (node) track** ``s`` of shape ``(B, N, D_s)``
        - **Pairwise track** ``z`` of shape ``(B, N, N, D_z)``

    Each layer applies:
        1. Triangle multiplication (outgoing + incoming) on z
        2. Transition MLP on z
        3. Transition MLP on s (no attention — single track is lightweight)

    Compared to Pairformer, this removes:
        - Triangle attention (starting + ending) on the pair track
        - Attention with pair bias on the single track

    Parameters
    ----------
    in_dim : int
        Node feature dimension (D_s).
    out_dim : int
        Output node feature dimension.
    pair_dim : int
        Pairwise representation dimension (D_z).
    pair_dropout : float
        Dropout rate applied row/column-wise on the pair track.
    hidden_dim_scaling : float
        Factor to scale hidden dim in transition MLPs (default 4).
    opm_hidden : int
        Hidden dimension of the outer-product-mean layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pair_dim: int = 64,
        in_dim_edges: Optional[int] = None,
        out_dim_edges: Optional[int] = None,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        pair_dropout: float = 0.25,
        hidden_dim_scaling: float = 4.0,
        opm_hidden: int = 32,
        use_checkpoint: bool = True,
        # ---- Speed optimizations ----
        compile_mode: str = "none",
        tri_mul_mode: str = "einsum",
        force_float32_einsums: bool = True,
        parallel_pair_ops: bool = False,
        # ---- Mixture-of-Experts ----
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_aux_loss_coeff: float = 0.01,
        # Accept and ignore pairformer-specific kwargs for config compat
        num_heads: int = 8,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        use_sdpa_attn_pair_bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            **kwargs,
        )

        self.pair_dim = pair_dim
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges
        self.pair_dropout = pair_dropout
        self.use_checkpoint = use_checkpoint
        self.parallel_pair_ops = parallel_pair_ops

        # ---- Pair track: outer-product mean initialization ----
        self.opm = OuterProductMean(
            in_dim, opm_hidden, pair_dim,
            force_float32=force_float32_einsums,
        )

        # ---- Pair track: triangle multiplication only (no attention) ----
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pair_dim, tri_mul_mode=tri_mul_mode,
            force_float32=force_float32_einsums,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pair_dim, tri_mul_mode=tri_mul_mode,
            force_float32=force_float32_einsums,
        )

        # ---- Pair track: transition MLP ----
        self.transition_z = Transition(pair_dim, int(pair_dim * hidden_dim_scaling))

        # ---- Single track: transition MLP only (no attention) ----
        s_hidden = int(in_dim * hidden_dim_scaling)
        if moe_num_experts > 0:
            self.transition_s = MoELayer(
                expert_cls=Transition,
                expert_kwargs={"dim": in_dim, "hidden": s_hidden},
                router_dim=in_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                aux_loss_coeff=moe_aux_loss_coeff,
            )
            self.use_moe = True
        else:
            self.transition_s = Transition(in_dim, s_hidden)
            self.use_moe = False

        # ---- Output projection (if in_dim != out_dim) ----
        if in_dim != out_dim:
            self.out_proj = nn.Linear(in_dim, out_dim)
        else:
            self.out_proj = nn.Identity()

        # ---- Apply torch.compile if requested ----
        if compile_mode == "pair_track":
            self._pair_track_ops = torch.compile(self._pair_track_ops, mode="reduce-overhead")
        elif compile_mode == "full":
            self.forward = torch.compile(self.forward, mode="reduce-overhead")

    def _pair_track_ops(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        """Pair-track operations: triangle multiplication + transition only.

        No triangle attention — this is the key difference from Pairformer.
        """
        if self.parallel_pair_ops:
            dmask1 = _get_dropout_mask(self.pair_dropout, z, self.training)
            dmask2 = _get_dropout_mask(self.pair_dropout, z, self.training)
            z = z + dmask1 * self.tri_mul_out(z, mask=pair_mask) \
                  + dmask2 * self.tri_mul_in(z, mask=pair_mask)
        else:
            dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
            z = z + dmask * self.tri_mul_out(z, mask=pair_mask)

            dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
            z = z + dmask * self.tri_mul_in(z, mask=pair_mask)

        z = z + self.transition_z(z)
        return z

    def forward(self, batch: Batch) -> Batch:
        """Forward pass operating on a PyG Batch.

        Parameters
        ----------
        batch : torch_geometric.data.Batch

        Returns
        -------
        torch_geometric.data.Batch
            Updated batch with new ``feat`` and ``pair_feat``.
        """
        feat = batch.feat  # (total_nodes, D_s) – sparse

        # --- Convert to dense (B, N_max, D_s) ---
        s_dense, key_padding_mask, idx = to_dense_batch(
            feat, batch=batch.batch,
        )
        node_mask = key_padding_mask.float()

        # --- Pair representation ---
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)

        if hasattr(batch, "pair_feat") and batch.pair_feat is not None:
            z = batch.pair_feat
        else:
            z = self.opm(s_dense, node_mask)

        # ---- Pair track updates (optionally checkpointed) ----
        if self.use_checkpoint and self.training:
            z = _checkpoint(
                self._pair_track_ops, z, pair_mask,
                use_reentrant=False,
            )
        else:
            z = self._pair_track_ops(z, pair_mask)

        # ---- Single track: transition only (no attention) ----
        if self.use_moe:
            s_dense = s_dense + self.transition_s(s_dense, node_mask=node_mask)
        else:
            s_dense = s_dense + self.transition_s(s_dense)

        # ---- Output projection ----
        s_dense = self.out_proj(s_dense)

        # --- Convert back to sparse ---
        feat_out = to_sparse_batch(s_dense, mask_idx=idx)

        batch.feat = feat_out
        batch.pair_feat = z
        return batch

    # ---- BaseGraphModule interface ----

    @classproperty
    def layer_supports_edges(cls) -> bool:
        return True

    @property
    def layer_inputs_edges(self) -> bool:
        return False

    @property
    def layer_outputs_edges(self) -> bool:
        return False

    @property
    def out_dim_factor(self) -> int:
        return 1
