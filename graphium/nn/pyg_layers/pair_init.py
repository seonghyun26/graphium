"""
Pair representation initialization for PairMixer.

Rather than initializing the pair track exclusively via
:class:`OuterProductMean` (OPM), :class:`PairFeatureInitializer` combines
multiple sources additively::

    z_0 = z_pair_prior + z_OPM

where ``z_pair_prior`` is produced by :class:`PairPositionalEncoder` from
graph-structural features (shortest-path hop distance, adjacency, edge
features) and ``z_OPM`` is produced by the existing OPM from node
features.

Motivation
----------
PairMixer has no single-track updates to bootstrap the pair
representation.  If OPM output is weak at initialization, ``z ≈ 0`` and
the triangle-multiplication backbone has no alternative signal path —
leading to optimization failure (e.g., NaN collapse under bf16).

AlphaFold-style architectures initialize the pair representation from
multiple sources precisely to avoid this single-point-of-failure.  Here
we apply the same principle by pre-computing cheap, non-zero, and
guaranteed-informative graph-structural priors and folding them into
``z_0`` additively.

Backwards compatibility
-----------------------
Default kwargs reproduce the previous PairMixer behavior exactly
(OPM with LeCun init, ``use_pair_positional=False``).  Priors are
enabled explicitly via :attr:`pair_init_kwargs`.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj

from .pairformer_pyg import (
    OuterProductMean,
    _LinearNoMup,
    _final_init,
    _lecun_normal_init,
)


_VALID_MODES = ("additive", "opm_only", "prior_only")
_VALID_POSITIONAL_SOURCES = (
    "graph_distance",
    "adjacency",
    "edge_feat",
    "path_edge",
)


# ---------------------------------------------------------------------------
# Low-level helpers: dense adjacency, shortest-path, and edge-feat scatter
# ---------------------------------------------------------------------------


def _compute_fw_shortest_path(
    adj: Tensor, max_dist: int,
) -> Tuple[Tensor, Tensor]:
    """Floyd-Warshall all-pairs shortest path with predecessor tracking.

    Computes both the shortest-path distance matrix and a "next-hop"
    predecessor matrix in a single pass.  The next-hop matrix is
    required by :class:`PathEdgeEncoder` for walking shortest paths,
    which is not possible from distances alone.

    Parameters
    ----------
    adj : Tensor (B, N, N)
        Binary adjacency (float-valued).  Graphium's featurizer emits
        both directions of every bond, so this is expected to be
        symmetric for molecular graphs.
    max_dist : int
        Distances strictly greater than ``max_dist`` are clamped to
        the "unreachable" bin at index ``max_dist + 1`` in the
        returned distance tensor.  The predecessor matrix is **not**
        clamped by ``max_dist`` — it stays ``-1`` only for pairs that
        are truly disconnected (no path at all).

    Returns
    -------
    dist : Tensor (B, N, N) long
        Shortest-path hop distances in
        ``{0, 1, ..., max_dist, max_dist + 1}``.  0 on the diagonal,
        ``max_dist + 1`` for pairs with no path or more than
        ``max_dist`` hops apart.
    next_hop : Tensor (B, N, N) long
        ``next_hop[b, i, j]`` is the index of the next node to visit
        from ``i`` when heading toward ``j`` along some shortest path.
        Equal to ``i`` when ``i == j`` (self-loop; no walking needed)
        and ``-1`` when ``j`` is unreachable from ``i`` (truly
        disconnected components).  For reachable-but-far pairs
        (``dist > max_dist``), the next-hop is still a valid index
        pointing along a real shortest path.

    Notes
    -----
    Executed in float32 under an autocast-disabled context so that
    running under mixed precision does not silently corrupt the
    comparisons in the FW inner loop.  Uses ``float('inf')`` as the
    unreachable sentinel during the computation, which is finite-safe
    because IEEE 754 guarantees ``inf + x = inf`` and
    ``inf < finite = False``.
    """
    B, N, _ = adj.shape
    device = adj.device

    ctx = (
        torch.autocast(device.type, enabled=False)
        if device.type in ("cuda", "cpu")
        else nullcontext()
    )

    with ctx:
        adj_f = adj.float()
        inf_f = float("inf")

        # Distance matrix: 0 on diagonal, 1 for direct edges, inf otherwise.
        dist = torch.where(
            adj_f > 0,
            torch.ones_like(adj_f),
            torch.full_like(adj_f, inf_f),
        )
        diag_idx = torch.arange(N, device=device)
        dist[:, diag_idx, diag_idx] = 0.0

        # Index scaffolding for next_hop initialization.
        j_idx = (
            torch.arange(N, device=device)
            .view(1, 1, N).expand(B, N, N).contiguous()
        )
        i_idx = (
            torch.arange(N, device=device)
            .view(1, N, 1).expand(B, N, N).contiguous()
        )
        eye_mask = (
            torch.eye(N, device=device, dtype=torch.bool)
            .unsqueeze(0).expand(B, N, N)
        )
        edge_mask = adj_f > 0

        # next_hop initialization:
        #   * direct edges: next_hop[i, j] = j
        #   * self-loops:   next_hop[i, i] = i  (never actually used)
        #   * otherwise:    -1 (unreachable until FW updates it)
        next_hop = torch.full(
            (B, N, N), -1, dtype=torch.long, device=device,
        )
        next_hop = torch.where(edge_mask, j_idx, next_hop)
        next_hop = torch.where(eye_mask, i_idx, next_hop)

        # Floyd-Warshall inner loop.
        for k in range(N):
            # dist_ik + dist_kj, with broadcasting: (B, N, 1) + (B, 1, N).
            dist_ik = dist[:, :, k:k + 1]
            dist_kj = dist[:, k:k + 1, :]
            through_k = dist_ik + dist_kj                 # (B, N, N)
            update_mask = through_k < dist                # bool
            dist = torch.where(update_mask, through_k, dist)

            # If going through k is better, the first step from i toward j
            # is the same as the first step from i toward k.
            next_hop_via_k = next_hop[:, :, k:k + 1].expand(-1, -1, N)
            next_hop = torch.where(update_mask, next_hop_via_k, next_hop)

    # Clamp distance to the "unreachable" bin for return. The next_hop
    # matrix is intentionally NOT clamped — for reachable-but-far pairs
    # we still want valid next-hop pointers so that ``path_edge`` can
    # truncate the walk while following a real path.
    inf_bin = max_dist + 1
    dist_clamped = torch.where(
        dist > float(max_dist),
        torch.full_like(dist, float(inf_bin)),
        dist,
    )

    return dist_clamped.long(), next_hop


def _compute_graph_distance(adj: Tensor, max_dist: int) -> Tensor:
    """Shortest-path hop distance via Floyd-Warshall.

    Thin wrapper over :func:`_compute_fw_shortest_path` that discards
    the predecessor matrix.  Kept as a standalone function because the
    ``graph_distance`` source only needs the distances.

    Parameters
    ----------
    adj : Tensor (B, N, N), binary adjacency (float).
    max_dist : int, distances greater than this are clamped to
        ``max_dist + 1`` ("unreachable" bin).

    Returns
    -------
    Tensor (B, N, N) long, with integer hop distances in
    ``{0, 1, ..., max_dist, max_dist + 1}``.
    """
    dist, _ = _compute_fw_shortest_path(adj, max_dist=max_dist)
    return dist


def _compute_local_edge_indices(
    batch_vec: Tensor, edge_index: Tensor, B: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert sparse global edge indices to ``(graph, local_i, local_j)``.

    PyG's dense batching reorders nodes contiguously per graph, so the
    dense "local" index within a graph is ``global_idx - graph_offset``.
    This helper computes that mapping once so it can be shared between
    the ``edge_feat`` and ``path_edge`` sources.

    Returns
    -------
    graph_of_edge : Tensor (E,) long
    local_i : Tensor (E,) long — per-graph local index of each edge source.
    local_j : Tensor (E,) long — per-graph local index of each edge target.
    """
    e_i, e_j = edge_index[0], edge_index[1]
    counts = torch.bincount(batch_vec, minlength=B)
    offsets = torch.cumsum(counts, dim=0) - counts
    graph_of_edge = batch_vec[e_i]
    local_i = e_i - offsets[graph_of_edge]
    local_j = e_j - offsets[batch_vec[e_j]]
    return graph_of_edge, local_i, local_j


def _build_dense_edge_feat(batch: Batch, B: int, N: int) -> Tensor:
    """Scatter sparse ``batch.edge_feat`` into a dense ``(B, N, N, d_edge)``.

    Pair positions without an edge are zero-filled.  Graphium's
    featurizer emits both directions of every bond, so the dense
    tensor is symmetric in practice (but the scatter does not enforce
    symmetry — the original sparse edges are preserved exactly).
    """
    edge_feat = getattr(batch, "edge_feat", None)
    if edge_feat is None:
        raise RuntimeError(
            "_build_dense_edge_feat requires batch.edge_feat to be set."
        )
    d_edge = edge_feat.shape[-1]
    device = edge_feat.device

    graph_of_edge, local_i, local_j = _compute_local_edge_indices(
        batch.batch, batch.edge_index, B,
    )
    dense = torch.zeros(
        B, N, N, d_edge, device=device, dtype=edge_feat.dtype,
    )
    dense[graph_of_edge, local_i, local_j] = edge_feat
    return dense


# ---------------------------------------------------------------------------
# Graphormer-style path-summary edge encoding
# ---------------------------------------------------------------------------


class PathEdgeEncoder(nn.Module):
    """Graphormer-style path-summary edge encoding.

    Reference
    ---------
    Ying et al., "Do Transformers Really Perform Bad for Graph
    Representation?", NeurIPS 2021, Section 3.1.3 "Edge Encoding in
    the Attention".

    For each pair ``(i, j)``, walks a shortest path from ``i`` to
    ``j`` and averages the edge features along the path using a
    learnable per-position projection::

        c_{ij} = (1 / N_{ij}) * Σ_{n=1..N_{ij}} W_n * x_{e_n}

    where:

    * ``N_{ij}`` is the path length in edges (clamped to
      ``max_path_length``),
    * ``x_{e_n}`` is the feature vector of the ``n``-th edge on a
      shortest path,
    * ``W_n ∈ R^{embedding_dim × d_edge}`` is a learnable projection
      specific to path-position ``n``.

    Graphormer uses this as a scalar bias on attention logits; here
    we adapt it to the pair representation by producing a
    ``(B, N, N, embedding_dim)`` vector that is added to the pair
    track.

    Unlike the ``edge_feat`` source — which only populates directly
    bonded (1-hop) pair positions and leaves everything else zero —
    ``path_edge`` gives non-zero, chemically meaningful features at
    every reachable pair position by summarising the bonds traversed
    along the shortest path.

    For pairs with true distance greater than ``max_path_length``,
    the walk is truncated at ``max_path_length`` edges and the
    divisor is clamped correspondingly; the encoding therefore always
    reflects the *first* few bonds on the path.

    Parameters
    ----------
    edge_feat_in_dim : int
        Input dimension of ``batch.edge_feat``.
    embedding_dim : int
        Output dimension per pair position (before the final
        :class:`PairPositionalEncoder` projection).
    max_path_length : int
        Maximum number of edges to walk per pair.  Graphormer used 5.
    """

    def __init__(
        self,
        edge_feat_in_dim: int,
        embedding_dim: int,
        max_path_length: int,
    ) -> None:
        super().__init__()
        if int(max_path_length) < 1:
            raise ValueError(
                f"path_edge.max_path_length must be ≥ 1, got "
                f"{max_path_length}."
            )
        if int(embedding_dim) < 1:
            raise ValueError(
                f"path_edge.embedding_dim must be ≥ 1, got {embedding_dim}."
            )
        if int(edge_feat_in_dim) < 1:
            raise ValueError(
                f"path_edge requires edge_feat_in_dim ≥ 1, got "
                f"{edge_feat_in_dim}."
            )
        self.edge_feat_in_dim = int(edge_feat_in_dim)
        self.embedding_dim = int(embedding_dim)
        self.max_path_length = int(max_path_length)
        # Position-specific projection W_n.  ``_LinearNoMup`` keeps
        # these mup-safe because structural features should not scale
        # with model width.
        self.proj_per_step = nn.ModuleList([
            _LinearNoMup(
                self.edge_feat_in_dim, self.embedding_dim, bias=False,
            )
            for _ in range(self.max_path_length)
        ])

    def forward(
        self,
        batch: Batch,
        B: int,
        N: int,
        pair_mask: Tensor,
        dist: Tensor,
        next_hop: Tensor,
    ) -> Tensor:
        """Walk shortest paths and accumulate position-weighted edge features.

        Parameters
        ----------
        batch : PyG Batch
            Must carry ``edge_index``, ``batch``, and ``edge_feat``.
        B, N : int
            Batch size and max nodes per graph.
        pair_mask : Tensor (B, N, N)
        dist : Tensor (B, N, N) long
            Shortest-path distances (as produced by
            :func:`_compute_fw_shortest_path`).  Pairs beyond
            ``max_dist`` are in the ``max_dist + 1`` bin; pairs at
            distance > ``max_path_length`` are truncated by the walk.
        next_hop : Tensor (B, N, N) long
            Predecessor matrix from FW; ``-1`` for truly unreachable
            pairs, otherwise a valid next-step index.

        Returns
        -------
        Tensor (B, N, N, embedding_dim)
            Path-summary features, masked at padding and untouched at
            self / unreachable pairs.
        """
        device = pair_mask.device
        dense_edge = _build_dense_edge_feat(batch, B, N)  # (B, N, N, d_edge)

        # Index scaffolding for gathers.
        i_range = torch.arange(N, device=device)
        b_idx = (
            torch.arange(B, device=device)
            .view(B, 1, 1).expand(B, N, N)
        )
        j_idx = i_range.view(1, 1, N).expand(B, N, N)
        curr = i_range.view(1, N, 1).expand(B, N, N).contiguous()  # start at i

        out = torch.zeros(
            B, N, N, self.embedding_dim,
            device=device, dtype=dense_edge.dtype,
        )

        for step in range(self.max_path_length):
            # nxt[b, i, j] = next_hop[b, curr[b, i, j], j]
            nxt = next_hop[b_idx, curr, j_idx]                # (B, N, N)

            # Pairs that still have an edge to walk at this step.
            # ``dist > step`` is True while we have not yet reached j;
            # ``nxt >= 0`` filters out truly unreachable pairs (where
            # next_hop is -1).
            alive = (dist > step) & (nxt >= 0)                # bool

            # Safe index: replace -1 with 0 to avoid OOB.  Masked out
            # immediately via ``alive_f`` so the spurious gather has no
            # effect on the output.
            safe_nxt = torch.where(alive, nxt, torch.zeros_like(nxt))

            # Edge feature of (curr → safe_nxt).
            edge_x = dense_edge[b_idx, curr, safe_nxt]        # (B, N, N, d_edge)
            proj = self.proj_per_step[step](edge_x)            # (B, N, N, d_emb)

            alive_f = alive.to(proj.dtype).unsqueeze(-1)
            out = out + proj * alive_f

            # Advance position (only for alive pairs).
            curr = torch.where(alive, nxt, curr)

        # Normalise by the *effective* path length: min(dist, max_path_length),
        # floored at 1 to avoid division-by-zero on self-pairs (their
        # accumulator is 0 anyway, so the quotient is 0).  Unreachable
        # pairs also have out == 0, so any finite divisor works.
        effective_length = (
            torch.clamp(dist, max=self.max_path_length)
            .clamp(min=1)
            .to(out.dtype)
            .unsqueeze(-1)
        )
        out = out / effective_length

        out = out * pair_mask.unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# Pair positional encoder
# ---------------------------------------------------------------------------


class PairPositionalEncoder(nn.Module):
    """Builds a dense pair representation from graph-structural features.

    Each enabled source is embedded independently and the resulting
    tensors are concatenated along the feature dimension, then projected
    to ``pair_dim`` by a mup-safe linear.

    Supported sources (enabled via ``sources``):
        * ``graph_distance`` — shortest-path hop distance, embedded via
          a learned :class:`~torch.nn.Embedding` table with
          ``max_dist + 2`` entries.  The extra bin absorbs distances
          greater than ``max_dist`` and pairs in disconnected components.
        * ``adjacency`` — binary bond-existence indicator, embedded via
          a 2-entry :class:`~torch.nn.Embedding`.  Subsumed by
          ``graph_distance`` but kept as an opt-in source for ablation
          because it isolates the 1-hop signal.
        * ``edge_feat`` — learned linear projection of
          ``batch.edge_feat`` (the raw per-edge feature tensor produced
          by :mod:`graphium.features.featurizer`) scattered into its
          ``(i, j)`` pair positions.  Bond-type, conjugation, and stereo
          information flow in through this source.  Populates only
          directly bonded pairs (1-hop); other positions stay zero.
        * ``path_edge`` — Graphormer-style path-summary encoding
          (:class:`PathEdgeEncoder`).  Walks a shortest path between
          every reachable pair and averages the edge features along
          the path with learnable per-position projections.  Unlike
          ``edge_feat``, it populates *every* reachable pair position,
          not only direct bonds.  Complements ``edge_feat``; use
          either or both.  Requires ``batch.edge_feat``.

    Parameters
    ----------
    pair_dim : int
        Target pair representation dimension (matches the PairMixer
        pair track).
    sources : list[str]
        Ordered list of source names to enable.
    graph_distance_kwargs, adjacency_kwargs, edge_feat_kwargs,
    path_edge_kwargs : dict
        Per-source configuration.  See the source-specific sections
        below for recognized keys.
    edge_feat_in_dim : int, optional
        Dimensionality of ``batch.edge_feat``.  Required when either
        ``edge_feat`` or ``path_edge`` is in ``sources``.
    """

    def __init__(
        self,
        pair_dim: int,
        sources: List[str],
        graph_distance_kwargs: Optional[Dict[str, Any]] = None,
        adjacency_kwargs: Optional[Dict[str, Any]] = None,
        edge_feat_kwargs: Optional[Dict[str, Any]] = None,
        path_edge_kwargs: Optional[Dict[str, Any]] = None,
        edge_feat_in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        normalized_sources: List[str] = []
        for src in sources:
            if src not in _VALID_POSITIONAL_SOURCES:
                raise ValueError(
                    f"Unknown pair positional source {src!r}; expected one of "
                    f"{_VALID_POSITIONAL_SOURCES}."
                )
            if src in normalized_sources:
                continue
            normalized_sources.append(src)

        if len(normalized_sources) == 0:
            raise ValueError(
                "PairPositionalEncoder requires at least one source; got an "
                "empty source list."
            )

        self.sources = normalized_sources
        self.source_modules: nn.ModuleDict = nn.ModuleDict()
        self.source_cfg: Dict[str, Dict[str, Any]] = {}
        total_embed_dim = 0

        if "graph_distance" in self.sources:
            cfg = dict(graph_distance_kwargs or {})
            max_dist = int(cfg.get("max_dist", 8))
            emb_dim = int(cfg.get("embedding_dim", 32))
            if max_dist < 1:
                raise ValueError(
                    f"graph_distance.max_dist must be ≥ 1, got {max_dist}."
                )
            if emb_dim < 1:
                raise ValueError(
                    f"graph_distance.embedding_dim must be ≥ 1, got {emb_dim}."
                )
            # max_dist + 2 entries: {0 (self), 1..max_dist, max_dist+1 (far)}.
            emb = nn.Embedding(max_dist + 2, emb_dim)
            nn.init.trunc_normal_(emb.weight, std=0.02)
            self.source_modules["graph_distance"] = emb
            self.source_cfg["graph_distance"] = {
                "max_dist": max_dist, "embedding_dim": emb_dim,
            }
            total_embed_dim += emb_dim

        if "adjacency" in self.sources:
            cfg = dict(adjacency_kwargs or {})
            emb_dim = int(cfg.get("embedding_dim", 8))
            if emb_dim < 1:
                raise ValueError(
                    f"adjacency.embedding_dim must be ≥ 1, got {emb_dim}."
                )
            emb = nn.Embedding(2, emb_dim)
            nn.init.trunc_normal_(emb.weight, std=0.02)
            self.source_modules["adjacency"] = emb
            self.source_cfg["adjacency"] = {"embedding_dim": emb_dim}
            total_embed_dim += emb_dim

        if "edge_feat" in self.sources:
            if edge_feat_in_dim is None or int(edge_feat_in_dim) <= 0:
                raise ValueError(
                    "pair_init.pair_positional sources includes 'edge_feat' "
                    "but edge_feat_in_dim is not set (or is ≤ 0). Either set "
                    "layer_kwargs.in_dim_edges, or remove 'edge_feat' from "
                    "pair_positional.sources."
                )
            cfg = dict(edge_feat_kwargs or {})
            emb_dim = int(cfg.get("embedding_dim", 16))
            if emb_dim < 1:
                raise ValueError(
                    f"edge_feat.embedding_dim must be ≥ 1, got {emb_dim}."
                )
            # _LinearNoMup: structural features are not scaled by model width.
            proj = _LinearNoMup(int(edge_feat_in_dim), emb_dim, bias=True)
            self.source_modules["edge_feat"] = proj
            self.source_cfg["edge_feat"] = {
                "embedding_dim": emb_dim,
                "in_dim": int(edge_feat_in_dim),
            }
            total_embed_dim += emb_dim

        if "path_edge" in self.sources:
            if edge_feat_in_dim is None or int(edge_feat_in_dim) <= 0:
                raise ValueError(
                    "pair_init.pair_positional sources includes 'path_edge' "
                    "but edge_feat_in_dim is not set (or is ≤ 0). Either set "
                    "layer_kwargs.in_dim_edges, or remove 'path_edge' from "
                    "pair_positional.sources."
                )
            cfg = dict(path_edge_kwargs or {})
            emb_dim = int(cfg.get("embedding_dim", 16))
            max_path_length = int(cfg.get("max_path_length", 5))
            encoder = PathEdgeEncoder(
                edge_feat_in_dim=int(edge_feat_in_dim),
                embedding_dim=emb_dim,
                max_path_length=max_path_length,
            )
            self.source_modules["path_edge"] = encoder
            self.source_cfg["path_edge"] = {
                "embedding_dim": emb_dim,
                "max_path_length": max_path_length,
                "in_dim": int(edge_feat_in_dim),
            }
            total_embed_dim += emb_dim

        # Final projection to pair_dim. ``_LinearNoMup`` already performs a
        # LeCun-style fan-in init internally; no extra init call needed.
        self.out_proj = _LinearNoMup(total_embed_dim, pair_dim, bias=True)

    def forward(
        self,
        batch: Batch,
        B: int,
        N: int,
        pair_mask: Tensor,
    ) -> Tensor:
        """Compute pair positional features.

        Parameters
        ----------
        batch : PyG Batch
            Needs ``edge_index`` and ``batch`` attributes, and
            ``edge_feat`` when the ``edge_feat`` source is enabled.
        B : int
            Batch size (number of graphs).
        N : int
            Max number of nodes per graph after dense padding.
        pair_mask : Tensor (B, N, N)
            1 where both endpoints are valid, 0 at padding.

        Returns
        -------
        Tensor (B, N, N, pair_dim)
            Dense pair features, masked at padding positions.
        """
        device = pair_mask.device
        parts: List[Tensor] = []

        needs_adj = any(
            src in self.source_modules
            for src in ("graph_distance", "adjacency", "path_edge")
        )
        adj: Optional[Tensor] = None
        if needs_adj:
            # to_dense_adj returns (B, N, N) with ``edge_feat`` accumulated as
            # 1 per directed edge; multi-edges collapse to the same binary bin
            # via the ``> 0`` clamp below. Graphium's featurizer emits both
            # directions of every bond, so the adjacency is symmetric.
            adj = to_dense_adj(
                batch.edge_index, batch=batch.batch, max_num_nodes=N,
            )
            adj = (adj > 0).to(pair_mask.dtype) * pair_mask

        # Shortest-path cache: computed once if either ``graph_distance``
        # or ``path_edge`` is enabled, and shared between them.  We
        # request FW at the largest max_dist needed so both sources can
        # consume it without re-running the inner loop.
        dist_cache: Optional[Tensor] = None
        next_hop_cache: Optional[Tensor] = None
        if "path_edge" in self.source_modules:
            pe_cfg = self.source_cfg["path_edge"]
            max_dist_for_fw = pe_cfg["max_path_length"]
            if "graph_distance" in self.source_modules:
                gd_cfg = self.source_cfg["graph_distance"]
                max_dist_for_fw = max(max_dist_for_fw, gd_cfg["max_dist"])
            dist_cache, next_hop_cache = _compute_fw_shortest_path(
                adj, max_dist=max_dist_for_fw,
            )

        if "graph_distance" in self.source_modules:
            cfg = self.source_cfg["graph_distance"]
            gd_max = cfg["max_dist"]
            if dist_cache is not None:
                # Re-clamp the shared distance tensor to the graph_distance
                # source's own ``max_dist`` bin (the FW tensor may have
                # been computed at a larger bound to serve path_edge).
                gd_inf = gd_max + 1
                dist = torch.where(
                    dist_cache > gd_max,
                    torch.full_like(dist_cache, gd_inf),
                    dist_cache,
                )
            else:
                dist = _compute_graph_distance(adj, max_dist=gd_max)
            # Zero out padding positions: embedding index 0 is a valid bin,
            # so rely on the final pair_mask multiply to discard padding
            # contributions (the embeddings are cheap to compute anyway).
            emb = self.source_modules["graph_distance"](dist)  # (B, N, N, D)
            parts.append(emb)

        if "adjacency" in self.source_modules:
            adj_long = (adj > 0).long()
            emb = self.source_modules["adjacency"](adj_long)   # (B, N, N, D)
            parts.append(emb)

        if "edge_feat" in self.source_modules:
            edge_feat = getattr(batch, "edge_feat", None)
            if edge_feat is None:
                raise RuntimeError(
                    "pair_init edge_feat source is enabled but "
                    "batch.edge_feat is None."
                )
            proj = self.source_modules["edge_feat"](edge_feat)  # (E, D_embed)
            D_embed = proj.shape[-1]
            # Scatter (E, D) → (B, N, N, D) via edge_index. The dense "local"
            # index within a graph is ``(global_idx - graph_offset)``, which
            # matches to_dense_batch's reordering (it preserves per-graph
            # order).
            graph_of_edge, local_i, local_j = _compute_local_edge_indices(
                batch.batch, batch.edge_index, B,
            )
            scatter = torch.zeros(
                B, N, N, D_embed, device=device, dtype=proj.dtype,
            )
            scatter[graph_of_edge, local_i, local_j] = proj
            parts.append(scatter)

        if "path_edge" in self.source_modules:
            assert dist_cache is not None and next_hop_cache is not None
            emb = self.source_modules["path_edge"](
                batch, B=B, N=N, pair_mask=pair_mask,
                dist=dist_cache, next_hop=next_hop_cache,
            )  # (B, N, N, embedding_dim)
            parts.append(emb)

        x = torch.cat(parts, dim=-1)    # (B, N, N, sum_embed_dim)
        x = self.out_proj(x)            # (B, N, N, pair_dim)
        x = x * pair_mask.unsqueeze(-1)
        return x


# ---------------------------------------------------------------------------
# Pair representation initializer (OPM + pair positional prior)
# ---------------------------------------------------------------------------


class PairFeatureInitializer(nn.Module):
    """Constructs the initial pair representation ``z_0`` for PairMixer.

    Combines multiple sources::

        additive mode (default):   z_0 = z_pair_prior + z_OPM
        opm_only mode:             z_0 = z_OPM                (legacy)
        prior_only mode:           z_0 = z_pair_prior

    The goal is to guarantee that ``z_0`` is non-zero and informative
    at initialization so that the PairMixer triangle-multiplication
    backbone always has a meaningful signal to refine.

    Parameters
    ----------
    in_dim : int
        Node feature dimension ``D_s`` (for OPM).
    pair_dim : int
        Pair representation dimension ``D_z``.
    opm_hidden : int
        Hidden dim of the outer-product-mean layer.
    force_float32 : bool
        Force OPM einsums in float32 for numerical stability under
        mixed precision.
    pair_init_kwargs : dict, optional
        Configuration dict. Recognized keys (all optional):

        * ``mode`` : ``"additive"`` (default) | ``"opm_only"`` |
          ``"prior_only"``.
        * ``use_opm`` : bool (default ``True``).  Ignored when
          ``mode == "prior_only"``.
        * ``opm_init`` : ``"lecun"`` (default) | ``"zero"``.  The
          legacy Pairformer init is ``"zero"``; the PairMixer
          default is ``"lecun"`` because PairMixer has no
          single-track updates to bootstrap a zero-init pair
          representation.
        * ``use_pair_positional`` : bool (default ``False``).  Must
          be ``True`` when ``mode == "prior_only"``.
        * ``pair_positional`` : dict with keys ``sources``,
          ``graph_distance``, ``adjacency``, ``edge_feat``,
          ``path_edge`` — see :class:`PairPositionalEncoder`.
        * ``use_smiles_pair_prior`` : reserved for future use, must be
          ``False`` (a ``NotImplementedError`` is raised otherwise).

        When ``pair_init_kwargs`` is ``None`` (or empty), defaults
        reproduce the previous PairMixer behavior exactly: OPM with
        LeCun init, no pair positional prior.
    edge_feat_in_dim : int, optional
        Edge feature dimension, forwarded to
        :class:`PairPositionalEncoder` for the ``edge_feat`` source.
    """

    def __init__(
        self,
        in_dim: int,
        pair_dim: int,
        opm_hidden: int = 32,
        force_float32: bool = True,
        pair_init_kwargs: Optional[Dict[str, Any]] = None,
        edge_feat_in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        cfg: Dict[str, Any] = dict(pair_init_kwargs or {})

        mode = str(cfg.get("mode", "additive")).lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"pair_init.mode must be one of {_VALID_MODES}, got {mode!r}."
            )

        if cfg.get("use_smiles_pair_prior", False):
            raise NotImplementedError(
                "pair_init.use_smiles_pair_prior is reserved for a future "
                "implementation and must be False."
            )

        use_opm = bool(cfg.get("use_opm", True))
        use_pair_positional = bool(cfg.get("use_pair_positional", False))

        # Mode-driven overrides.
        if mode == "opm_only":
            use_opm = True
            use_pair_positional = False
        elif mode == "prior_only":
            use_opm = False
            if not use_pair_positional:
                raise ValueError(
                    "pair_init.mode='prior_only' requires "
                    "use_pair_positional=True."
                )

        if not (use_opm or use_pair_positional):
            raise ValueError(
                "PairFeatureInitializer: at least one of `use_opm` or "
                "`use_pair_positional` must be True."
            )

        self.mode = mode
        self.use_opm = use_opm
        self.use_pair_positional = use_pair_positional

        # ---- Outer-product-mean ----
        if self.use_opm:
            self.opm = OuterProductMean(
                in_dim, opm_hidden, pair_dim,
                force_float32=force_float32,
            )
            opm_init = str(cfg.get("opm_init", "lecun")).lower()
            if opm_init == "lecun":
                _lecun_normal_init(self.opm.proj_o.weight)
            elif opm_init == "zero":
                _final_init(self.opm.proj_o.weight)
            else:
                raise ValueError(
                    f"pair_init.opm_init must be 'lecun' or 'zero', got "
                    f"{opm_init!r}."
                )
            self.opm_init = opm_init
        else:
            self.opm = None
            self.opm_init = None

        # ---- Pair positional prior ----
        if self.use_pair_positional:
            pp_cfg: Dict[str, Any] = dict(cfg.get("pair_positional", {}) or {})
            sources = list(pp_cfg.get("sources", ["graph_distance"]))
            self.pair_positional = PairPositionalEncoder(
                pair_dim=pair_dim,
                sources=sources,
                graph_distance_kwargs=pp_cfg.get("graph_distance"),
                adjacency_kwargs=pp_cfg.get("adjacency"),
                edge_feat_kwargs=pp_cfg.get("edge_feat"),
                path_edge_kwargs=pp_cfg.get("path_edge"),
                edge_feat_in_dim=edge_feat_in_dim,
            )
        else:
            self.pair_positional = None

    def forward(
        self,
        s_dense: Tensor,
        node_mask: Tensor,
        batch: Batch,
    ) -> Tensor:
        """Build the initial pair representation ``z_0``.

        Parameters
        ----------
        s_dense : Tensor (B, N, D_s)
            Dense node features.
        node_mask : Tensor (B, N)
            1 at valid nodes, 0 at padding.
        batch : PyG Batch
            Original batch carrying ``edge_index``, ``batch``, and
            (when required) ``edge_feat``.

        Returns
        -------
        Tensor (B, N, N, pair_dim)
            Initial pair representation, masked at padding positions.
        """
        B, N, _ = s_dense.shape
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)

        z: Optional[Tensor] = None

        if self.use_opm:
            z_opm = self.opm(s_dense, node_mask)
            z = z_opm if z is None else z + z_opm

        if self.use_pair_positional:
            z_prior = self.pair_positional(
                batch, B=B, N=N, pair_mask=pair_mask,
            )
            z = z_prior if z is None else z + z_prior

        # Defensive re-mask: makes this function idempotent with respect to
        # the downstream ``pair_feat * pair_mask`` multiplies inside
        # PairMixer layers and the GraphOutputNN pooling path.
        z = z * pair_mask.unsqueeze(-1)
        return z
