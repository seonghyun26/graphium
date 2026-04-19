"""
PairMixer++ layer for Graphium.

This is a lightweight hybrid between PairMixer and a single-track GNN:
    - the single/node representation is updated every layer with one GINE block
    - the pair representation is updated with PairMixer-style triangle ops
    - the updated single representation is injected into the pair track through
      a cheap low-rank single-to-pair coupling path

Compared with Pairformer, this keeps the coupling intentionally light:
    - no pair-biased attention on the single track
    - no triangle attention on the pair track
    - no full outer-product refresh at every layer

The pair track remains the main relational backbone; the single track provides
fresh local semantics each layer so ``pair_feat`` does not drift away from the
node state after initialization.
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

from .gin_pyg import GINEConvPyg
from .pair_init import PairFeatureInitializer, _build_dense_edge_feat
from .pairformer_pyg import (
    Transition,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    _LinearNoMup,
    _gating_init,
    _get_dropout_mask,
    _lecun_normal_init,
)


class PairMixerPPLayerPyg(BaseGraphModule):
    """A lightweight two-track PairMixer++ layer.

    The layer updates the node track with one residual GINE block, injects the
    refreshed node state into the pair track through a low-rank broadcasted
    Hadamard interaction, and then applies the standard PairMixer pair-track
    updates:

        1. Single track: GINE + residual + norm
        2. Pair track: optional pair init / edge injection
        3. Single -> pair low-rank coupling
        4. Triangle multiplication (outgoing)
        5. Triangle multiplication (incoming)
        6. Transition MLP (or MoE) on the pair track

    This preserves the inexpensive PairMixer backbone while ensuring the pair
    representation keeps seeing the updated single representation every layer.
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
        pair_init: Optional[dict] = None,
        compile_mode: str = "none",
        tri_mul_mode: str = "einsum",
        force_float32_einsums: bool = True,
        parallel_pair_ops: bool = False,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_aux_loss_coeff: float = 0.01,
        edge_injection: bool = False,
        single_to_pair_rank: int = 16,
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

        if single_to_pair_rank <= 0:
            raise ValueError(f"single_to_pair_rank must be > 0, got {single_to_pair_rank}.")
        if in_dim_edges is None:
            raise ValueError("PairMixer++ requires in_dim_edges because the single-track update uses GINE.")

        self.pair_dim = pair_dim
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges
        self.pair_dropout = pair_dropout
        self.use_checkpoint = use_checkpoint
        self.parallel_pair_ops = parallel_pair_ops
        self.single_to_pair_rank = single_to_pair_rank

        # ---- Single track: one lightweight GINE block every layer ----
        self.single_update = GINEConvPyg(
            in_dim=in_dim,
            out_dim=in_dim,
            in_dim_edges=in_dim_edges,
            activation=activation,
            dropout=0.0,
            normalization="none",
        )
        self.single_residual_dropout = self._parse_dropout(dropout)
        self.single_residual_norm = self._parse_norm(normalization, dim=in_dim)

        # ---- Per-layer edge injection ----
        self.edge_proj = None
        if edge_injection and in_dim_edges > 0:
            self.edge_proj = _LinearNoMup(in_dim_edges, pair_dim, bias=False)

        # ---- Pair representation initialization (layer 0 only) ----
        if self.layer_idx is None or self.layer_idx == 0:
            self.pair_init = PairFeatureInitializer(
                in_dim=in_dim,
                pair_dim=pair_dim,
                opm_hidden=opm_hidden,
                force_float32=force_float32_einsums,
                pair_init_kwargs=pair_init,
                edge_feat_in_dim=in_dim_edges,
            )
        else:
            self.pair_init = None

        # ---- Single -> pair lightweight coupling ----
        self.norm_s2z = nn.LayerNorm(in_dim, eps=1e-5)
        self.proj_left = _LinearNoMup(in_dim, single_to_pair_rank, bias=False)
        self.proj_right = _LinearNoMup(in_dim, single_to_pair_rank, bias=False)
        self.proj_pair = _LinearNoMup(single_to_pair_rank, pair_dim, bias=False)
        self.norm_pair_gate = nn.LayerNorm(pair_dim, eps=1e-5)
        self.proj_gate = nn.Linear(pair_dim, pair_dim, bias=False)

        _lecun_normal_init(self.proj_left.weight)
        _lecun_normal_init(self.proj_right.weight)
        _lecun_normal_init(self.proj_pair.weight)
        _gating_init(self.proj_gate.weight)

        # ---- Pair track: triangle multiplication + transition ----
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pair_dim, tri_mul_mode=tri_mul_mode, force_float32=force_float32_einsums
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pair_dim, tri_mul_mode=tri_mul_mode, force_float32=force_float32_einsums
        )

        z_hidden = int(pair_dim * hidden_dim_scaling)
        if moe_num_experts > 0:
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

        if in_dim != out_dim:
            self.out_proj = nn.Linear(in_dim, out_dim)
        else:
            self.out_proj = nn.Identity()

        if compile_mode == "pair_track":
            self._pair_track_ops = torch.compile(self._pair_track_ops, mode="default", dynamic=True)
        elif compile_mode == "full":
            self.forward = torch.compile(self.forward, mode="default", dynamic=True)

    def _inject_edges(self, batch: Batch, pair_mask: Tensor) -> Tensor:
        """Project dense edge features into pair_dim and mask padding pairs."""
        if not hasattr(batch, "_edge_feat_dense") or batch._edge_feat_dense is None:
            B, N = pair_mask.shape[:2]
            batch._edge_feat_dense = _build_dense_edge_feat(batch, B, N)
        proj = self.edge_proj(batch._edge_feat_dense)
        return proj * pair_mask.unsqueeze(-1)

    def _single_to_pair_update(self, s_dense: Tensor, z: Tensor, pair_mask: Tensor) -> Tensor:
        """Inject updated single features into the pair track via a low-rank map."""
        s_norm = self.norm_s2z(s_dense)
        left = self.proj_left(s_norm)
        right = self.proj_right(s_norm)
        pair_rank = left.unsqueeze(2) * right.unsqueeze(1)
        delta_z = self.proj_pair(pair_rank)
        gate = torch.sigmoid(self.proj_gate(self.norm_pair_gate(z)))
        z = z + gate * delta_z
        return z * pair_mask.unsqueeze(-1)

    def _pair_track_ops(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        """Pair-track operations: PairMixer triangle updates + transition."""
        if self.parallel_pair_ops:
            dmask1 = _get_dropout_mask(self.pair_dropout, z, self.training)
            dmask2 = _get_dropout_mask(self.pair_dropout, z, self.training)
            z = z + dmask1 * self.tri_mul_out(z, mask=pair_mask) + dmask2 * self.tri_mul_in(
                z, mask=pair_mask
            )
        else:
            dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
            z = z + dmask * self.tri_mul_out(z, mask=pair_mask)

            dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
            z = z + dmask * self.tri_mul_in(z, mask=pair_mask)

        if not self.use_moe:
            z = z + self.transition_z(z)

        return z * pair_mask.unsqueeze(-1)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass on a PyG batch with coupled single/pair updates."""
        feat_in = batch.feat

        single_batch = batch.clone()
        single_batch.feat = feat_in
        single_batch = self.single_update(single_batch)

        feat_delta = single_batch.feat
        if self.single_residual_dropout is not None:
            feat_delta = self.single_residual_dropout(feat_delta)
        feat = feat_in + feat_delta
        if self.single_residual_norm is not None:
            feat = self.single_residual_norm(feat)

        s_dense, key_padding_mask, idx = to_dense_batch(feat, batch=batch.batch)
        node_mask = key_padding_mask.float()
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)

        if hasattr(batch, "pair_feat") and batch.pair_feat is not None:
            z = batch.pair_feat
        elif self.pair_init is not None:
            z = self.pair_init(s_dense, node_mask, batch)
        else:
            raise RuntimeError(
                f"PairMixerPPLayerPyg (layer_idx={self.layer_idx}): pair_feat not found on batch and "
                "no pair_init available. Only layer 0 has pair_init; pair_feat must be set by a "
                "preceding layer."
            )

        if self.edge_proj is not None:
            z = z + self._inject_edges(batch, pair_mask)

        z = self._single_to_pair_update(s_dense, z, pair_mask)

        if self.use_checkpoint and self.training:
            z = _checkpoint(self._pair_track_ops, z, pair_mask, use_reentrant=False)
        else:
            z = self._pair_track_ops(z, pair_mask)

        if self.use_moe:
            B, N, _, D = z.shape
            z_flat = z.reshape(B, N * N, D)
            mask_flat = pair_mask.reshape(B, N * N)
            z = z + self.transition_z(z_flat, node_mask=mask_flat).reshape(B, N, N, D)
            z = z * pair_mask.unsqueeze(-1)

        feat_out = to_sparse_batch(self.out_proj(s_dense), mask_idx=idx)

        batch.feat = feat_out
        batch.pair_feat = z
        batch.pair_mask = pair_mask
        batch._pair_dense_idx = idx
        return batch

    @classproperty
    def layer_supports_edges(cls) -> bool:
        return True

    @property
    def layer_inputs_edges(self) -> bool:
        return True

    @property
    def layer_outputs_edges(self) -> bool:
        return False

    @property
    def out_dim_factor(self) -> int:
        return 1
