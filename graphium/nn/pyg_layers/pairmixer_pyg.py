"""
PairMixer layer for Graphium, based on the PairMixer architecture.

Reference:
    Ouyang-Zhang et al., "Triangle Multiplication Is All You Need for
    Biomolecular Structure Representations", arXiv:2510.18870, 2025.
    Code: https://github.com/genesistherapeutics/pairmixer

PairMixer simplifies the Pairformer by removing:
    - Triangle attention (both starting and ending) on the pair track
    - Attention with pair bias on the single track
    - Sequence (single-track) updates entirely

The backbone operates exclusively on the pair representation z via:
    1. Triangle multiplication (outgoing)
    2. Triangle multiplication (incoming)
    3. Transition MLP (FFN) on z

The single representation s is left unchanged (s_backbone = s_init),
matching the reference implementation. Final predictions are made by
pooling pair_feat directly in GraphOutputNN (see ``pair_dim`` kwarg
in the graph_output_nn config).
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint
from torch_geometric.data import Batch

from graphium.nn.base_graph_layer import BaseGraphModule
from graphium.nn.base_layers import MoELayer
from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from graphium.utils.decorators import classproperty

from .pairformer_pyg import (
    OuterProductMean,
    Transition,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    _get_dropout_mask,
    _lecun_normal_init,
)


class PairMixerLayerPyg(BaseGraphModule):
    """A single PairMixer layer adapted for Graphium.

    Following the reference implementation, only the pair representation
    z is updated.  The single (node) track s passes through unchanged
    (s_backbone = s_init).

    Each layer applies:
        1. Triangle multiplication (outgoing) on z
        2. Triangle multiplication (incoming) on z
        3. Transition MLP (FFN) on z

    The pair representation is stored on the batch as ``pair_feat`` and
    ``pair_mask``.  Downstream, ``GraphOutputNN`` pools ``pair_feat``
    directly for graph-level predictions when its ``pair_dim`` kwarg
    is set.

    Parameters
    ----------
    in_dim : int
        Node feature dimension (D_s).
    out_dim : int
        Output node feature dimension.
    pair_dim : int
        Pairwise representation dimension (D_z).
    pair_dropout : float
        Dropout rate applied row-wise on the pair track.
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
        # ---- Mixture-of-Experts on the pair-track transition ----
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
        # Only the first layer needs OPM (subsequent layers reuse pair_feat).
        # This saves ~25% of parameters that would otherwise be dead.
        if self.layer_idx is None or self.layer_idx == 0:
            self.opm = OuterProductMean(
                in_dim, opm_hidden, pair_dim,
                force_float32=force_float32_einsums,
            )
            # Override zero-init on proj_o: PairMixer has no single-track
            # updates to bootstrap the pair representation, so OPM must
            # produce non-zero initial features (unlike Pairformer where
            # the attention track provides the initial learning signal).
            _lecun_normal_init(self.opm.proj_o.weight)
        else:
            self.opm = None

        # ---- Pair track: triangle multiplication only (no attention) ----
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pair_dim, tri_mul_mode=tri_mul_mode,
            force_float32=force_float32_einsums,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pair_dim, tri_mul_mode=tri_mul_mode,
            force_float32=force_float32_einsums,
        )

        # ---- Pair track: transition MLP (or MoE) ----
        z_hidden = int(pair_dim * hidden_dim_scaling)
        if moe_num_experts > 0:
            # Graph-level routing on a pooled pair signal (mirrors
            # _pool_pair_feat in global_architectures.py): the flattened
            # (B, N*N, D_z) view is masked-mean-pooled by MoELayer._forward_dense,
            # producing the same graph-level vector as averaging over (i, j).
            self.transition_z = MoELayer(
                expert_cls=Transition,
                expert_kwargs={"dim": pair_dim, "hidden": z_hidden},
                router_dim=pair_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                aux_loss_coeff=moe_aux_loss_coeff,
            )
            self.use_moe = True
        else:
            self.transition_z = Transition(pair_dim, z_hidden)
            self.use_moe = False

        # ---- Output projection on node features (if in_dim != out_dim) ----
        # Node features pass through unchanged (s_backbone = s_init),
        # but FeedForwardGraph may require in_dim == out_dim across layers.
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

        Follows the reference implementation (genesistherapeutics/pairmixer):
        TriMulOutgoing → TriMulIncoming → FFN.
        (Note: the paper's Algorithm 1 writes Incoming→Outgoing, but the
        code that produced the published results uses Outgoing→Incoming,
        consistent with the Pairformer/Boltz convention.)

        When MoE is enabled, the transition is applied *outside* this
        function (see ``forward``) so that its aux_loss gradient path
        survives the gradient-checkpointing boundary.
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

        if not self.use_moe:
            z = z + self.transition_z(z)

        # Re-mask padding: LayerNorm bias in triangle multiplication and
        # transition MLP produce non-zero values at padding positions that
        # would otherwise accumulate across layers via residual connections.
        z = z * pair_mask.unsqueeze(-1)
        return z

    def forward(self, batch: Batch) -> Batch:
        """Forward pass operating on a PyG Batch.

        Only the pair representation z is updated.  Node features pass
        through unchanged (s_backbone = s_init), matching the reference
        PairMixer implementation.

        Parameters
        ----------
        batch : torch_geometric.data.Batch

        Returns
        -------
        torch_geometric.data.Batch
            Updated batch with ``pair_feat``, ``pair_mask``, and
            (pass-through) ``feat``.
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
        elif self.opm is not None:
            z = self.opm(s_dense, node_mask)
        else:
            raise RuntimeError(
                f"PairMixerLayerPyg (layer_idx={self.layer_idx}): "
                "pair_feat not found on batch and no OPM available. "
                "Only layer 0 has OPM; pair_feat must be set by a preceding layer."
            )

        # ---- Pair track updates (optionally checkpointed) ----
        if self.use_checkpoint and self.training:
            z = _checkpoint(
                self._pair_track_ops, z, pair_mask,
                use_reentrant=False,
            )
        else:
            z = self._pair_track_ops(z, pair_mask)

        # ---- MoE transition on the pair track (outside checkpoint) ----
        # Kept out of ``_pair_track_ops`` so the MoE ``aux_loss`` autograd
        # graph is not torn down by ``torch.utils.checkpoint`` — mirrors
        # how Pairformer keeps its MoE ``transition_s`` outside the
        # checkpointed pair-track block.
        if self.use_moe:
            B, N, _, D = z.shape
            z_flat = z.reshape(B, N * N, D)
            mask_flat = pair_mask.reshape(B, N * N)
            z = z + self.transition_z(z_flat, node_mask=mask_flat).reshape(B, N, N, D)
            z = z * pair_mask.unsqueeze(-1)  # re-mask after MoE transition

        # ---- Node features: pass through unchanged ----
        s_dense = self.out_proj(s_dense)
        feat_out = to_sparse_batch(s_dense, mask_idx=idx)

        batch.feat = feat_out
        batch.pair_feat = z
        batch.pair_mask = pair_mask        # stored for downstream pooling
        batch._pair_dense_idx = idx        # dense→sparse mapping for node-level pooling
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
