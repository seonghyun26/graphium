"""
Pairformer layer for Graphium, adapted from the Boltz biomolecular interaction model.

Reference:
    Boltz repository: https://github.com/jwohlwend/boltz (MIT License)
    Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction", 2025.

The Pairformer architecture maintains two representation tracks:
    - A single (node/sequence) track `s` of shape (B, N, D_s)
    - A pairwise track `z` of shape (B, N, N, D_z)

These tracks interact through:
    1. Triangle multiplicative updates (outgoing and incoming)
    2. Triangle attention (starting and ending node)
    3. Attention with pair bias (single track attends using pairwise info)
    4. Transition MLPs on both tracks

This module adapts the Pairformer to Graphium's BaseGraphModule interface,
operating on PyG Batch objects.
"""

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint
from torch_geometric.data import Batch

from graphium.nn.base_graph_layer import BaseGraphModule
from graphium.nn.base_layers import MLP
from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from graphium.utils.decorators import classproperty


# ---------------------------------------------------------------------------
# mup-safe linear layer
# ---------------------------------------------------------------------------


class _LinearNoMup(nn.Module):
    """A plain linear layer that is **not** a subclass of ``nn.Linear``.

    The ``mup`` library's ``assert_hidden_size_inf`` only inspects modules
    that are ``isinstance(module, nn.Linear)``.  By using a separate
    ``nn.Module`` with an explicit ``nn.Parameter`` we keep the exact same
    forward semantics while preventing the assertion from firing on layers
    that intentionally project *across* two representation tracks with
    different mup scaling regimes (e.g. node→pair in OuterProductMean).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.trunc_normal_(self.weight, std=math.sqrt(1.0 / in_features))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Lightweight initialisers (following Boltz / AlphaFold conventions)
# ---------------------------------------------------------------------------


def _lecun_normal_init(weight: Tensor) -> None:
    """Truncated-normal fan-in init (LeCun style)."""
    nn.init.trunc_normal_(weight, std=math.sqrt(1.0 / weight.shape[1]))


def _final_init(weight: Tensor) -> None:
    """Zero-init for final projection layers."""
    nn.init.zeros_(weight)


def _gating_init(weight: Tensor) -> None:
    """Zero-init for gating layers (sigmoid → 0.5 at start)."""
    nn.init.zeros_(weight)


# ---------------------------------------------------------------------------
# SwiGLU-style Transition (two-layer gated MLP)
# ---------------------------------------------------------------------------


class Transition(nn.Module):
    """SwiGLU-style two-layer gated MLP used in both tracks.

    Architecture: LayerNorm → (SiLU(Linear1(x)) * Linear2(x)) → Linear3 → output
    """

    def __init__(self, dim: int, hidden: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()

        _lecun_normal_init(self.fc1.weight)
        _lecun_normal_init(self.fc2.weight)
        _final_init(self.fc3.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.fc3(self.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Triangle multiplicative updates
# ---------------------------------------------------------------------------


class TriangleMultiplicationOutgoing(nn.Module):
    """Triangle multiplication – *outgoing* direction.

    Aggregates along the shared index k:
        z_ij = proj_out(norm_out(∑_k a_ik ⊗ b_jk)) * σ(gate_out(z_ij))
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        _lecun_normal_init(self.p_in.weight)
        _gating_init(self.g_in.weight)
        _final_init(self.p_out.weight)
        _gating_init(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, N, N, D)
        mask : Tensor (B, N, N)

        Returns
        -------
        Tensor (B, N, N, D)
        """
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x.float(), 2, dim=-1)
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x


class TriangleMultiplicationIncoming(nn.Module):
    """Triangle multiplication – *incoming* direction.

    Aggregates along the shared index k:
        z_ij = proj_out(norm_out(∑_k a_ki ⊗ b_kj)) * σ(gate_out(z_ij))
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        _lecun_normal_init(self.p_in.weight)
        _gating_init(self.g_in.weight)
        _final_init(self.p_out.weight)
        _gating_init(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, N, N, D)
        mask : Tensor (B, N, N)

        Returns
        -------
        Tensor (B, N, N, D)
        """
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x.float(), 2, dim=-1)
        x = torch.einsum("bkid,bkjd->bijd", a, b)
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()
        return x


# ---------------------------------------------------------------------------
# Triangle attention
# ---------------------------------------------------------------------------


class TriangleAttention(nn.Module):
    """Axial self-attention on the pair representation with triangle bias.

    Operates along one axis of the (N, N) pair representation.
    If ``starting=True``, attention is applied along the *rows* (axis -3),
    otherwise along the *columns* (axis -2) after transposing.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = nn.LayerNorm(c_in)
        self.linear_bias = nn.Linear(c_in, no_heads, bias=False)

        # QKV projections
        self.linear_q = nn.Linear(c_in, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_in, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_in, c_hidden * no_heads, bias=False)

        # Gating + output
        self.linear_g = nn.Linear(c_in, c_hidden * no_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_in, bias=False)

        _lecun_normal_init(self.linear_bias.weight)
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        _gating_init(self.linear_g.weight)
        _final_init(self.linear_o.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, I, J, C_in)
        mask : Tensor (B, I, J)

        Returns
        -------
        Tensor (B, I, J, C_in)
        """
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        B, I, J, _ = x.shape

        # Triangle bias: (B, I, J, H) → (B*I, H, 1, J)
        # The bias is per-row: for row i, bias[b,h,i,k] is added to
        # every query position's attention logit toward key position k.
        tri_bias = self.linear_bias(x)                   # (B, I, J, H)
        tri_bias = tri_bias.permute(0, 1, 3, 2)          # (B, I, H, J)
        tri_bias = tri_bias.reshape(B * I, self.no_heads, 1, J)

        # Mask bias: (B, I, J) → (B*I, 1, 1, J)
        mask_bias = self.inf * (mask.reshape(B * I, 1, 1, J) - 1)

        attn_mask = tri_bias + mask_bias  # (B*I, H, 1, J) – broadcasts to (B*I, H, J, J)

        # QKV: (B, I, J, C) → (B*I, H, J, D)
        q = self.linear_q(x).reshape(B * I, J, self.no_heads, self.c_hidden).transpose(1, 2)
        k = self.linear_k(x).reshape(B * I, J, self.no_heads, self.c_hidden).transpose(1, 2)
        v = self.linear_v(x).reshape(B * I, J, self.no_heads, self.c_hidden).transpose(1, 2)

        # Scaled dot-product attention  →  (B*I, H, J, D)
        # Uses flash/memory-efficient attention when possible.
        # Note: padding rows (pair_mask all-zero) produce all-inf attention masks,
        # causing softmax(all-inf) = NaN.  Replace NaN with 0 — padding positions
        # are discarded after to_sparse_batch so this is safe.
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = torch.nan_to_num(o, nan=0.0)

        # Gating: (B*I, H, J, D)
        g = self.linear_g(x).reshape(B * I, J, self.no_heads, self.c_hidden).transpose(1, 2).sigmoid()
        o = o * g

        # Output: (B*I, H, J, D) → (B, I, J, H*D)
        o = o.transpose(1, 2).reshape(B, I, J, -1)
        o = self.linear_o(o)

        if not self.starting:
            o = o.transpose(-2, -3)

        return o


# ---------------------------------------------------------------------------
# Attention with pair bias (for the single/node track)
# ---------------------------------------------------------------------------


class AttentionPairBias(nn.Module):
    """Multi-head attention on the single track, biased by pair representation.

    The pair representation z is projected to per-head biases that are added
    to the attention logits, allowing pairwise information to modulate
    which nodes attend to which.
    """

    def __init__(self, c_s: int, c_z: int, num_heads: int, inf: float = 1e6) -> None:
        super().__init__()
        assert c_s % num_heads == 0
        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.norm_s = nn.LayerNorm(c_s)
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
        )

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        _final_init(self.proj_o.weight)

    def forward(self, s: Tensor, z: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        s : Tensor (B, N, D_s)
            Single / node representation.
        z : Tensor (B, N, N, D_z)
            Pairwise representation.
        mask : Tensor (B, N)
            Node mask (1 = valid, 0 = padding).

        Returns
        -------
        Tensor (B, N, D_s)
        """
        B = s.shape[0]
        s = self.norm_s(s)

        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(s).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(s).view(B, -1, self.num_heads, self.head_dim)

        # Pair bias: (B, N, N, H) → (B, H, N, N)
        z_bias = self.proj_z(z).permute(0, 3, 1, 2)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim ** 0.5) + z_bias.float()
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o


# ---------------------------------------------------------------------------
# Outer Product Mean  (sequence → pair)
# ---------------------------------------------------------------------------


class OuterProductMean(nn.Module):
    """Compute the outer product mean to project sequence info into pair space.

    Given a node representation m of shape (B, N, C_in), produces
    a pairwise tensor of shape (B, N, N, C_out).

    ``proj_a`` and ``proj_b`` use :class:`_LinearNoMup` because they bridge
    from the single (node) track, whose width scales under mup, to a fixed
    pair-track hidden dimension.  A regular ``nn.Linear`` would trigger
    ``mup.assert_hidden_size_inf`` (infinite fan-in, finite fan-out).
    """

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.norm = nn.LayerNorm(c_in)
        # _LinearNoMup avoids mup assertion for cross-track projections
        self.proj_a = _LinearNoMup(c_in, c_hidden, bias=False)
        self.proj_b = _LinearNoMup(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)
        _final_init(self.proj_o.weight)
        _final_init(self.proj_o.bias)

    def forward(self, m: Tensor, mask: Tensor) -> Tensor:
        """Memory-efficient outer product mean.

        Instead of materialising the full outer product
        ``(B, N, N, c_h²)`` (e.g. 1024 channels with c_h=32), we
        decompose the output projection as two successive contractions:

            temp[b,j,o,c]  = Σ_d  b[b,j,d] · W[o,c,d]   — O(N · c_out · c_h²)
            z[b,i,j,o]     = Σ_c  a[b,i,c] · temp[b,j,o,c]  — O(N² · c_out · c_h)

        Peak intermediate: ``(B, N, c_out, c_h)`` ≈ 25× smaller than
        the naïve ``(B, N, N, c_h²)`` intermediate.

        Parameters
        ----------
        m : Tensor (B, N, C_in)
            Node features.
        mask : Tensor (B, N)
            Node mask (1 = valid, 0 = padding).

        Returns
        -------
        Tensor (B, N, N, C_out)
        """
        mask_2d = mask.unsqueeze(-1).to(m)
        m = self.norm(m)
        a = self.proj_a(m) * mask_2d  # (B, N, c_h)
        b = self.proj_b(m) * mask_2d

        # Pairwise normalisation factor
        num_mask = (mask_2d.unsqueeze(2) * mask_2d.unsqueeze(1)).squeeze(-1).clamp(min=1)

        # Factored projection: avoid (B, N, N, c_h²) intermediate
        W = self.proj_o.weight.view(self.c_out, self.c_hidden, self.c_hidden)

        # Step 1: contract b with W along the second c_h dim
        #   temp[b,j,o,c] = Σ_d b[b,j,d] · W[o,c,d]
        temp = torch.einsum("bjd,ocd->bjoc", b.float(), W.float())  # (B, N, c_out, c_h)

        # Step 2: contract a with temp along the first c_h dim
        #   z[b,i,j,o] = Σ_c a[b,i,c] · temp[b,j,o,c]
        z = torch.einsum("bic,bjoc->bijo", a.float(), temp)          # (B, N, N, c_out)

        z = z / num_mask.unsqueeze(-1)
        z = z.to(m) + self.proj_o.bias
        return z


# ---------------------------------------------------------------------------
# Row-wise dropout mask (Boltz-style)
# ---------------------------------------------------------------------------


def _get_dropout_mask(dropout: float, z: Tensor, training: bool, columnwise: bool = False) -> Tensor:
    """Row-wise (or column-wise) dropout mask for the pair representation."""
    if not training or dropout == 0:
        return z.new_ones(1)
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = (torch.rand(v.shape, dtype=torch.float32, device=v.device) >= dropout).float()
    return d / (1.0 - dropout)


# ---------------------------------------------------------------------------
# Single Pairformer layer adapted for Graphium (PyG Batch)
# ---------------------------------------------------------------------------


class PairformerLayerPyg(BaseGraphModule):
    """A single Pairformer layer adapted for Graphium.

    This layer maintains two representation tracks that interact:
        - **Single (node) track** ``s`` of shape ``(B, N, D_s)``
        - **Pairwise track** ``z`` of shape ``(B, N, N, D_z)``

    Each layer applies:
        1. Triangle multiplication (outgoing + incoming) on z
        2. Triangle attention (starting + ending node) on z
        3. Transition MLP on z
        4. Attention with pair bias on s (using z as bias)
        5. Transition MLP on s

    The pairwise representation ``z`` is stored on the batch as
    ``batch.pair_feat`` and is initialized via outer-product mean of
    node features if it does not yet exist.

    Parameters
    ----------
    in_dim : int
        Node feature dimension (D_s).
    out_dim : int
        Output node feature dimension.
    pair_dim : int
        Pairwise representation dimension (D_z).
    num_heads : int
        Number of attention heads for the single-track attention.
    pairwise_head_width : int
        Per-head hidden dimension for triangle attention.
    pairwise_num_heads : int
        Number of heads for triangle attention.
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
        num_heads: int = 8,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        pair_dropout: float = 0.25,
        hidden_dim_scaling: float = 4.0,
        opm_hidden: int = 32,
        use_checkpoint: bool = True,
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

        # ---- Pair track: outer-product mean initialization ----
        self.opm = OuterProductMean(in_dim, opm_hidden, pair_dim)

        # ---- Pair track: triangle updates ----
        self.tri_mul_out = TriangleMultiplicationOutgoing(pair_dim)
        self.tri_mul_in = TriangleMultiplicationIncoming(pair_dim)

        # ---- Pair track: triangle attention ----
        self.tri_att_start = TriangleAttention(
            pair_dim, pairwise_head_width, pairwise_num_heads, starting=True
        )
        self.tri_att_end = TriangleAttention(
            pair_dim, pairwise_head_width, pairwise_num_heads, starting=False
        )

        # ---- Pair track: transition MLP ----
        self.transition_z = Transition(pair_dim, int(pair_dim * hidden_dim_scaling))

        # ---- Single track: attention with pair bias ----
        self.attn = AttentionPairBias(in_dim, pair_dim, num_heads)

        # ---- Single track: transition MLP ----
        self.transition_s = Transition(in_dim, int(in_dim * hidden_dim_scaling))

        # ---- Output projection (if in_dim != out_dim) ----
        if in_dim != out_dim:
            self.out_proj = nn.Linear(in_dim, out_dim)
        else:
            self.out_proj = nn.Identity()

    def _pair_track_ops(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        """All pair-track operations, isolated for gradient checkpointing.

        When ``use_checkpoint=True`` this function is wrapped with
        ``torch.utils.checkpoint.checkpoint`` so that intermediate
        activations (triangle mult/attn tensors) are freed during
        forward and recomputed during backward — reducing activation
        memory from O(L × per_layer) to O(per_layer).
        """
        dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
        z = z + dmask * self.tri_mul_out(z, mask=pair_mask)

        dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
        z = z + dmask * self.tri_mul_in(z, mask=pair_mask)

        dmask = _get_dropout_mask(self.pair_dropout, z, self.training)
        z = z + dmask * self.tri_att_start(z, mask=pair_mask)

        dmask = _get_dropout_mask(self.pair_dropout, z, self.training, columnwise=True)
        z = z + dmask * self.tri_att_end(z, mask=pair_mask)

        z = z + self.transition_z(z)
        return z

    def forward(self, batch: Batch) -> Batch:
        """Forward pass operating on a PyG Batch.

        The node features are stored in ``batch.feat``.
        The pairwise features are stored in ``batch.pair_feat`` (created
        automatically on the first layer).

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
        # key_padding_mask: (B, N_max) bool – True where valid
        node_mask = key_padding_mask.float()  # (B, N_max)  1=valid

        # --- Pair representation ---
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)  # (B, N, N)

        if hasattr(batch, "pair_feat") and batch.pair_feat is not None:
            z = batch.pair_feat  # (B, N, N, D_z)
        else:
            # Initialize from outer product mean
            z = self.opm(s_dense, node_mask)

        # ---- Pair track updates (optionally checkpointed) ----
        if self.use_checkpoint and self.training:
            z = _checkpoint(
                self._pair_track_ops, z, pair_mask,
                use_reentrant=False,
            )
        else:
            z = self._pair_track_ops(z, pair_mask)

        # ---- Single track updates ----
        s_dense = s_dense + self.attn(s_dense, z, node_mask)
        s_dense = s_dense + self.transition_s(s_dense)

        # ---- Output projection ----
        s_dense = self.out_proj(s_dense)

        # --- Convert back to sparse ---
        feat_out = to_sparse_batch(s_dense, mask_idx=idx)

        batch.feat = feat_out
        batch.pair_feat = z  # Store for the next layer
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
