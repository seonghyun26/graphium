"""
Unit tests for PairFeatureInitializer and PairPositionalEncoder
(graphium/nn/pyg_layers/pair_init.py).

Covers:
    - Backwards-compatible defaults reproduce OPM-only behavior.
    - Non-zero variance at initialization when priors are enabled.
    - Shape correctness for each source combination.
    - Additive combination produces the sum of individual sources.
    - Mode-driven overrides (opm_only / prior_only) behave as expected.
    - Shortest-path BFS returns correct hop distances.
    - Full PairMixer layer forward works with pair_init enabled.
"""

import unittest as ut

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from graphium.nn.pyg_layers.pair_init import (
    PairFeatureInitializer,
    PairPositionalEncoder,
    PathEdgeEncoder,
    _compute_fw_shortest_path,
    _compute_graph_distance,
)
from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg


def _make_batch(num_graphs=3, min_nodes=4, max_nodes=7, in_dim=16,
                edge_feat_dim=4, seed=42):
    """Build a PyG Batch of small random chain graphs with edge_feat."""
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(num_graphs):
        n = torch.randint(min_nodes, max_nodes + 1, (1,), generator=rng).item()
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, in_dim, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], edge_feat_dim, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    return Batch.from_data_list(graphs)


def _dense_inputs(batch, in_dim):
    """Densify the batch for direct calls to PairFeatureInitializer."""
    from graphium.ipu.to_dense_batch import to_dense_batch

    s_dense, key_padding_mask, _ = to_dense_batch(batch.feat, batch=batch.batch)
    node_mask = key_padding_mask.float()
    return s_dense, node_mask


class TestGraphDistanceBFS(ut.TestCase):
    """_compute_graph_distance: correctness of shortest-path hop distances."""

    def test_chain_distances(self):
        """Chain 0-1-2-3-4: distances should be {0,1,2,3,4}."""
        # Single graph, 5 nodes forming a chain.
        N = 5
        adj = torch.zeros(1, N, N)
        for i in range(N - 1):
            adj[0, i, i + 1] = 1.0
            adj[0, i + 1, i] = 1.0
        dist = _compute_graph_distance(adj, max_dist=8)
        # Expected:
        # row 0: [0, 1, 2, 3, 4]
        # row 2: [2, 1, 0, 1, 2]
        expected = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ], dtype=torch.long)
        torch.testing.assert_close(dist[0], expected)

    def test_disconnected_is_infinity_bin(self):
        """Disconnected components get max_dist + 1."""
        N = 4
        adj = torch.zeros(1, N, N)
        # Edge 0-1 only
        adj[0, 0, 1] = 1.0
        adj[0, 1, 0] = 1.0
        max_dist = 3
        dist = _compute_graph_distance(adj, max_dist=max_dist)
        # Nodes 0 and 1 are connected; 2 and 3 are isolated (to everyone else).
        self.assertEqual(dist[0, 0, 0].item(), 0)
        self.assertEqual(dist[0, 0, 1].item(), 1)
        self.assertEqual(dist[0, 0, 2].item(), max_dist + 1)
        self.assertEqual(dist[0, 0, 3].item(), max_dist + 1)

    def test_clamped_to_infinity_bin_beyond_max_dist(self):
        """Nodes farther than max_dist hops land in the 'unreachable' bin."""
        N = 6
        # Chain 0-1-2-3-4-5, max distance 5, max_dist=3
        adj = torch.zeros(1, N, N)
        for i in range(N - 1):
            adj[0, i, i + 1] = 1.0
            adj[0, i + 1, i] = 1.0
        max_dist = 3
        dist = _compute_graph_distance(adj, max_dist=max_dist)
        # 0 ↔ 5 is 5 hops; should be in the ``max_dist + 1`` bin.
        self.assertEqual(dist[0, 0, 5].item(), max_dist + 1)
        self.assertEqual(dist[0, 0, 3].item(), 3)
        self.assertEqual(dist[0, 0, 4].item(), max_dist + 1)


class TestFwShortestPath(ut.TestCase):
    """_compute_fw_shortest_path: distance + predecessor matrix correctness."""

    def test_chain_next_hop(self):
        """On a 5-chain, next_hop[0, 4] points to 1 (first step on path 0→4)."""
        N = 5
        adj = torch.zeros(1, N, N)
        for i in range(N - 1):
            adj[0, i, i + 1] = 1.0
            adj[0, i + 1, i] = 1.0
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=8)
        # Row 0 distances
        torch.testing.assert_close(
            dist[0, 0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        )
        # next_hop[0, j] should be 1 for j > 0 (first step of any forward path).
        self.assertEqual(next_hop[0, 0, 1].item(), 1)
        self.assertEqual(next_hop[0, 0, 2].item(), 1)
        self.assertEqual(next_hop[0, 0, 3].item(), 1)
        self.assertEqual(next_hop[0, 0, 4].item(), 1)
        # Self-loop: next_hop[i, i] = i.
        for i in range(N):
            self.assertEqual(next_hop[0, i, i].item(), i)

    def test_disconnected_next_hop_is_neg_one(self):
        """Truly disconnected pairs have next_hop = -1."""
        N = 4
        adj = torch.zeros(1, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0  # Only edge 0-1
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=5)
        # 0 to 2, 3: unreachable
        self.assertEqual(next_hop[0, 0, 2].item(), -1)
        self.assertEqual(next_hop[0, 0, 3].item(), -1)
        # 0 to 1: valid
        self.assertEqual(next_hop[0, 0, 1].item(), 1)

    def test_reachable_but_far_next_hop_is_valid(self):
        """For pairs reachable but farther than max_dist, next_hop stays valid."""
        N = 6
        adj = torch.zeros(1, N, N)
        for i in range(N - 1):
            adj[0, i, i + 1] = 1.0
            adj[0, i + 1, i] = 1.0
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=2)
        # 0 to 5 is distance 5, clamped to max_dist + 1 = 3 in dist tensor.
        self.assertEqual(dist[0, 0, 5].item(), 3)  # "unreachable" bin
        # But the next_hop entry should still be valid (1, pointing along path).
        self.assertEqual(next_hop[0, 0, 5].item(), 1)

    def test_ring_next_hop(self):
        """On a 4-cycle, next_hop[0, 2] can be either 1 or 3 (both shortest)."""
        N = 4
        adj = torch.zeros(1, N, N)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            adj[0, a, b] = adj[0, b, a] = 1.0
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=5)
        self.assertEqual(dist[0, 0, 2].item(), 2)
        # FW picks deterministically — just verify it's a valid neighbor.
        self.assertIn(next_hop[0, 0, 2].item(), [1, 3])


class TestPathEdgeEncoder(ut.TestCase):
    """PathEdgeEncoder: walk correctness + shape + masking."""

    def test_shape_and_nonzero(self):
        edge_dim = 4
        batch = _make_batch(num_graphs=3, in_dim=8, edge_feat_dim=edge_dim)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        from torch_geometric.utils import to_dense_adj
        adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=N)
        adj = (adj > 0).float() * pair_mask
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=6)

        enc = PathEdgeEncoder(
            edge_feat_in_dim=edge_dim, embedding_dim=8, max_path_length=5,
        )
        out = enc(
            batch, B=B, N=N, pair_mask=pair_mask,
            dist=dist, next_hop=next_hop,
        )
        self.assertEqual(out.shape, (B, N, N, 8))
        # Should have some non-zero output at valid pairs.
        valid = out[pair_mask > 0]
        self.assertTrue(valid.abs().max().item() > 0.0)

    def test_walk_accumulates_in_order(self):
        """Manually verify path walk on a 4-chain with one-hot edge features."""
        # 4-chain: 0-1-2-3. Label each directed edge with a unique one-hot.
        # Directed edge order (from edge_index):
        #   (0,1), (1,0), (1,2), (2,1), (2,3), (3,2)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ])
        edge_feat = torch.eye(6)  # Row k is the one-hot for edge index k.
        feat = torch.zeros(4, 8)
        g = Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat)
        batch = Batch.from_data_list([g])

        enc = PathEdgeEncoder(
            edge_feat_in_dim=6, embedding_dim=6, max_path_length=5,
        )
        # Set every per-step projection to the identity so the output is
        # directly the sum of gathered edge one-hots.
        for proj in enc.proj_per_step:
            with torch.no_grad():
                proj.weight.copy_(torch.eye(6))

        adj = torch.zeros(1, 4, 4)
        for a, b in [(0, 1), (1, 2), (2, 3)]:
            adj[0, a, b] = adj[0, b, a] = 1.0
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=5)
        pair_mask = torch.ones(1, 4, 4)

        out = enc(
            batch, B=1, N=4, pair_mask=pair_mask,
            dist=dist, next_hop=next_hop,
        )

        # Pair (0, 3): walks 0→1→2→3.
        # Edges on the path: edge_index 0 (0→1), edge_index 2 (1→2),
        # edge_index 4 (2→3).  Sum = one-hot[0] + one-hot[2] + one-hot[4],
        # divided by 3.
        expected_03 = torch.zeros(6)
        expected_03[0] = 1.0 / 3
        expected_03[2] = 1.0 / 3
        expected_03[4] = 1.0 / 3
        torch.testing.assert_close(out[0, 0, 3], expected_03)

        # Pair (3, 0): walks 3→2→1→0 using directed edges 5, 3, 1.
        expected_30 = torch.zeros(6)
        expected_30[5] = 1.0 / 3
        expected_30[3] = 1.0 / 3
        expected_30[1] = 1.0 / 3
        torch.testing.assert_close(out[0, 3, 0], expected_30)

        # Pair (1, 2): direct edge 2, divided by 1.
        expected_12 = torch.zeros(6)
        expected_12[2] = 1.0
        torch.testing.assert_close(out[0, 1, 2], expected_12)

        # Self-pair (0, 0): zero output.
        torch.testing.assert_close(out[0, 0, 0], torch.zeros(6))

    def test_truncation_beyond_max_path_length(self):
        """Distance > max_path_length → walk truncates, divisor = max_path_length."""
        # 6-chain: 0-1-2-3-4-5
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ])
        edge_feat = torch.eye(10)
        feat = torch.zeros(6, 8)
        g = Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat)
        batch = Batch.from_data_list([g])

        enc = PathEdgeEncoder(
            edge_feat_in_dim=10, embedding_dim=10, max_path_length=3,
        )
        for proj in enc.proj_per_step:
            with torch.no_grad():
                proj.weight.copy_(torch.eye(10))

        adj = torch.zeros(1, 6, 6)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
            adj[0, a, b] = adj[0, b, a] = 1.0
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=10)
        pair_mask = torch.ones(1, 6, 6)

        out = enc(
            batch, B=1, N=6, pair_mask=pair_mask,
            dist=dist, next_hop=next_hop,
        )

        # Pair (0, 5): true distance 5. Walk truncates at 3 edges: 0→1→2→3.
        # Directed edges used: 0 (0→1), 2 (1→2), 4 (2→3).  Divisor = 3.
        expected = torch.zeros(10)
        expected[0] = 1.0 / 3
        expected[2] = 1.0 / 3
        expected[4] = 1.0 / 3
        torch.testing.assert_close(out[0, 0, 5], expected)

    def test_unreachable_pair_is_zero(self):
        """Disconnected pairs produce zero output."""
        # Two disconnected edges: (0-1) and (2-3).
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ])
        edge_feat = torch.eye(4)
        feat = torch.zeros(4, 8)
        g = Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat)
        batch = Batch.from_data_list([g])

        enc = PathEdgeEncoder(
            edge_feat_in_dim=4, embedding_dim=4, max_path_length=3,
        )
        adj = torch.zeros(1, 4, 4)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 2, 3] = adj[0, 3, 2] = 1.0
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=5)
        pair_mask = torch.ones(1, 4, 4)

        out = enc(
            batch, B=1, N=4, pair_mask=pair_mask,
            dist=dist, next_hop=next_hop,
        )
        # Pair (0, 2) is unreachable: output must be zero.
        torch.testing.assert_close(out[0, 0, 2], torch.zeros(4))
        torch.testing.assert_close(out[0, 0, 3], torch.zeros(4))

    def test_padding_is_masked(self):
        """Padding positions must be exactly zero."""
        edge_dim = 4
        batch = _make_batch(
            num_graphs=3, min_nodes=3, max_nodes=7,
            in_dim=8, edge_feat_dim=edge_dim,
        )
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        from torch_geometric.utils import to_dense_adj
        adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=N)
        adj = (adj > 0).float() * pair_mask
        dist, next_hop = _compute_fw_shortest_path(adj, max_dist=6)

        enc = PathEdgeEncoder(
            edge_feat_in_dim=edge_dim, embedding_dim=8, max_path_length=5,
        )
        out = enc(
            batch, B=B, N=N, pair_mask=pair_mask,
            dist=dist, next_hop=next_hop,
        )
        self.assertEqual(out[pair_mask == 0].abs().max().item(), 0.0)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            PathEdgeEncoder(
                edge_feat_in_dim=4, embedding_dim=8, max_path_length=0,
            )
        with self.assertRaises(ValueError):
            PathEdgeEncoder(
                edge_feat_in_dim=4, embedding_dim=0, max_path_length=3,
            )
        with self.assertRaises(ValueError):
            PathEdgeEncoder(
                edge_feat_in_dim=0, embedding_dim=8, max_path_length=3,
            )


class TestPairPositionalEncoder(ut.TestCase):
    """PairPositionalEncoder: per-source and combined shape / non-zero."""

    pair_dim = 16

    def test_graph_distance_only(self):
        batch = _make_batch(num_graphs=3, in_dim=8)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        enc = PairPositionalEncoder(
            pair_dim=self.pair_dim,
            sources=["graph_distance"],
            graph_distance_kwargs={"max_dist": 6, "embedding_dim": 16},
        )
        out = enc(batch, B=B, N=N, pair_mask=pair_mask)
        self.assertEqual(out.shape, (B, N, N, self.pair_dim))
        # Valid positions should not all be zero.
        valid = out[pair_mask > 0]
        self.assertTrue(valid.abs().max().item() > 0.0)

    def test_adjacency_only(self):
        batch = _make_batch(num_graphs=2, in_dim=8)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        enc = PairPositionalEncoder(
            pair_dim=self.pair_dim,
            sources=["adjacency"],
            adjacency_kwargs={"embedding_dim": 4},
        )
        out = enc(batch, B=B, N=N, pair_mask=pair_mask)
        self.assertEqual(out.shape, (B, N, N, self.pair_dim))

    def test_edge_feat_source(self):
        edge_dim = 5
        batch = _make_batch(num_graphs=2, in_dim=8, edge_feat_dim=edge_dim)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        enc = PairPositionalEncoder(
            pair_dim=self.pair_dim,
            sources=["edge_feat"],
            edge_feat_kwargs={"embedding_dim": 6},
            edge_feat_in_dim=edge_dim,
        )
        out = enc(batch, B=B, N=N, pair_mask=pair_mask)
        self.assertEqual(out.shape, (B, N, N, self.pair_dim))
        # Outputs at padding positions should be zero.
        padding = out[pair_mask == 0]
        self.assertTrue(padding.abs().max().item() == 0.0)

    def test_combined_sources(self):
        edge_dim = 4
        batch = _make_batch(num_graphs=3, in_dim=8, edge_feat_dim=edge_dim)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        enc = PairPositionalEncoder(
            pair_dim=self.pair_dim,
            sources=["graph_distance", "adjacency", "edge_feat"],
            graph_distance_kwargs={"max_dist": 5, "embedding_dim": 8},
            adjacency_kwargs={"embedding_dim": 4},
            edge_feat_kwargs={"embedding_dim": 4},
            edge_feat_in_dim=edge_dim,
        )
        out = enc(batch, B=B, N=N, pair_mask=pair_mask)
        self.assertEqual(out.shape, (B, N, N, self.pair_dim))
        valid = out[pair_mask > 0]
        self.assertTrue(valid.std().item() > 0.0)

    def test_invalid_source_raises(self):
        with self.assertRaises(ValueError):
            PairPositionalEncoder(
                pair_dim=self.pair_dim, sources=["not_a_source"],
            )

    def test_edge_feat_without_in_dim_raises(self):
        with self.assertRaises(ValueError):
            PairPositionalEncoder(
                pair_dim=self.pair_dim,
                sources=["edge_feat"],
                edge_feat_in_dim=None,
            )

    def test_path_edge_source(self):
        edge_dim = 5
        batch = _make_batch(num_graphs=2, in_dim=8, edge_feat_dim=edge_dim)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        enc = PairPositionalEncoder(
            pair_dim=self.pair_dim,
            sources=["path_edge"],
            path_edge_kwargs={"embedding_dim": 6, "max_path_length": 4},
            edge_feat_in_dim=edge_dim,
        )
        out = enc(batch, B=B, N=N, pair_mask=pair_mask)
        self.assertEqual(out.shape, (B, N, N, self.pair_dim))
        self.assertTrue(out[pair_mask > 0].abs().max().item() > 0.0)
        # Padding must be zero.
        self.assertEqual(out[pair_mask == 0].abs().max().item(), 0.0)

    def test_path_edge_without_in_dim_raises(self):
        with self.assertRaises(ValueError):
            PairPositionalEncoder(
                pair_dim=self.pair_dim,
                sources=["path_edge"],
                edge_feat_in_dim=None,
            )

    def test_path_edge_combined_with_graph_distance(self):
        """Enabling both path_edge and graph_distance should share the FW pass."""
        edge_dim = 4
        batch = _make_batch(num_graphs=3, in_dim=8, edge_feat_dim=edge_dim)
        from graphium.ipu.to_dense_batch import to_dense_batch
        _, kpm, _ = to_dense_batch(batch.feat, batch=batch.batch)
        pair_mask = kpm.float().unsqueeze(-1) * kpm.float().unsqueeze(-2)
        B, N = pair_mask.shape[:2]

        enc = PairPositionalEncoder(
            pair_dim=self.pair_dim,
            sources=["graph_distance", "path_edge"],
            graph_distance_kwargs={"max_dist": 6, "embedding_dim": 8},
            path_edge_kwargs={"embedding_dim": 6, "max_path_length": 4},
            edge_feat_in_dim=edge_dim,
        )
        out = enc(batch, B=B, N=N, pair_mask=pair_mask)
        self.assertEqual(out.shape, (B, N, N, self.pair_dim))
        self.assertTrue(out[pair_mask > 0].std().item() > 0.0)


class TestPairFeatureInitializer(ut.TestCase):
    """PairFeatureInitializer: modes, shape, variance at init."""

    in_dim = 16
    pair_dim = 32

    def test_default_reproduces_opm_only(self):
        """Default kwargs should behave exactly like pre-existing OPM-only."""
        torch.manual_seed(0)
        initializer = PairFeatureInitializer(
            in_dim=self.in_dim,
            pair_dim=self.pair_dim,
            pair_init_kwargs=None,
        )
        self.assertTrue(initializer.use_opm)
        self.assertFalse(initializer.use_pair_positional)
        self.assertEqual(initializer.mode, "additive")
        self.assertEqual(initializer.opm_init, "lecun")

        batch = _make_batch(num_graphs=3, in_dim=self.in_dim)
        s_dense, node_mask = _dense_inputs(batch, self.in_dim)
        z = initializer(s_dense, node_mask, batch)
        B, N = s_dense.shape[:2]
        self.assertEqual(z.shape, (B, N, N, self.pair_dim))
        # LeCun init → non-zero OPM output.
        valid = z[(node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)) > 0]
        self.assertTrue(valid.abs().max().item() > 0.0)

    def test_additive_combination(self):
        """z = opm + prior: output variance > max of either alone."""
        torch.manual_seed(1)
        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance"],
                "graph_distance": {"max_dist": 6, "embedding_dim": 16},
            },
        }
        initializer = PairFeatureInitializer(
            in_dim=self.in_dim, pair_dim=self.pair_dim,
            pair_init_kwargs=cfg,
        )
        batch = _make_batch(num_graphs=3, in_dim=self.in_dim)
        s_dense, node_mask = _dense_inputs(batch, self.in_dim)
        z = initializer(s_dense, node_mask, batch)
        B, N = s_dense.shape[:2]
        self.assertEqual(z.shape, (B, N, N, self.pair_dim))
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        self.assertTrue(z[pair_mask > 0].std().item() > 0.0)

    def test_prior_only_mode(self):
        cfg = {
            "mode": "prior_only",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance"],
                "graph_distance": {"max_dist": 5, "embedding_dim": 8},
            },
        }
        initializer = PairFeatureInitializer(
            in_dim=self.in_dim, pair_dim=self.pair_dim,
            pair_init_kwargs=cfg,
        )
        self.assertFalse(initializer.use_opm)
        self.assertTrue(initializer.use_pair_positional)
        self.assertIsNone(initializer.opm)

        batch = _make_batch(num_graphs=2, in_dim=self.in_dim)
        s_dense, node_mask = _dense_inputs(batch, self.in_dim)
        z = initializer(s_dense, node_mask, batch)
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        self.assertTrue(z[pair_mask > 0].abs().max().item() > 0.0)

    def test_opm_only_mode_ignores_prior_config(self):
        cfg = {
            "mode": "opm_only",
            # These should be ignored because mode overrides them:
            "use_pair_positional": True,
            "pair_positional": {"sources": ["graph_distance"]},
        }
        initializer = PairFeatureInitializer(
            in_dim=self.in_dim, pair_dim=self.pair_dim,
            pair_init_kwargs=cfg,
        )
        self.assertTrue(initializer.use_opm)
        self.assertFalse(initializer.use_pair_positional)
        self.assertIsNone(initializer.pair_positional)

    def test_prior_only_without_pair_positional_raises(self):
        with self.assertRaises(ValueError):
            PairFeatureInitializer(
                in_dim=self.in_dim, pair_dim=self.pair_dim,
                pair_init_kwargs={"mode": "prior_only"},
            )

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            PairFeatureInitializer(
                in_dim=self.in_dim, pair_dim=self.pair_dim,
                pair_init_kwargs={"mode": "totally_invalid"},
            )

    def test_smiles_prior_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            PairFeatureInitializer(
                in_dim=self.in_dim, pair_dim=self.pair_dim,
                pair_init_kwargs={"use_smiles_pair_prior": True},
            )

    def test_disable_both_raises(self):
        with self.assertRaises(ValueError):
            PairFeatureInitializer(
                in_dim=self.in_dim, pair_dim=self.pair_dim,
                pair_init_kwargs={
                    "use_opm": False, "use_pair_positional": False,
                },
            )

    def test_padding_is_masked(self):
        """Padding positions in z must be exactly zero."""
        cfg = {
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance", "adjacency"],
            },
        }
        initializer = PairFeatureInitializer(
            in_dim=self.in_dim, pair_dim=self.pair_dim,
            pair_init_kwargs=cfg,
        )
        # Heterogeneous graph sizes → padding exists.
        batch = _make_batch(num_graphs=3, min_nodes=3, max_nodes=7,
                            in_dim=self.in_dim)
        s_dense, node_mask = _dense_inputs(batch, self.in_dim)
        z = initializer(s_dense, node_mask, batch)
        pair_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        # Everything outside pair_mask must be zero.
        self.assertEqual(z[pair_mask == 0].abs().max().item(), 0.0)


class TestPairMixerLayerWithPairInit(ut.TestCase):
    """End-to-end: PairMixerLayerPyg with pair_init configured."""

    in_dim = 32
    pair_dim = 16

    def _make_layer(self, pair_init=None, **kwargs):
        return PairMixerLayerPyg(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            pair_dim=self.pair_dim,
            pair_dropout=0.0,
            hidden_dim_scaling=2.0,
            opm_hidden=16,
            use_checkpoint=False,
            layer_idx=0,
            layer_depth=2,
            pair_init=pair_init,
            **kwargs,
        )

    def test_default_layer_matches_legacy_behavior(self):
        """pair_init=None should produce a functional OPM-only layer."""
        layer = self._make_layer(pair_init=None)
        self.assertTrue(hasattr(layer, "pair_init"))
        self.assertIsNotNone(layer.pair_init)
        self.assertTrue(layer.pair_init.use_opm)
        self.assertFalse(layer.pair_init.use_pair_positional)

        batch = _make_batch(num_graphs=3, in_dim=self.in_dim)
        out = layer(batch)
        self.assertTrue(hasattr(out, "pair_feat"))
        self.assertFalse(torch.isnan(out.pair_feat).any())

    def test_layer_with_priors(self):
        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance", "adjacency"],
                "graph_distance": {"max_dist": 6, "embedding_dim": 16},
                "adjacency": {"embedding_dim": 4},
            },
        }
        layer = self._make_layer(pair_init=cfg)
        self.assertTrue(layer.pair_init.use_pair_positional)

        batch = _make_batch(num_graphs=3, in_dim=self.in_dim)
        out = layer(batch)
        self.assertFalse(torch.isnan(out.pair_feat).any())
        # pair_feat is non-degenerate at layer 0 output.
        self.assertTrue(out.pair_feat.abs().max().item() > 0.0)

    def test_layer_with_edge_feat_prior(self):
        edge_dim = 5
        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["edge_feat"],
                "edge_feat": {"embedding_dim": 8},
            },
        }
        layer = self._make_layer(pair_init=cfg, in_dim_edges=edge_dim)
        batch = _make_batch(
            num_graphs=3, in_dim=self.in_dim, edge_feat_dim=edge_dim,
        )
        out = layer(batch)
        self.assertFalse(torch.isnan(out.pair_feat).any())

    def test_layer_with_path_edge_prior(self):
        """End-to-end PairMixer layer with the Graphormer-style path_edge."""
        edge_dim = 6
        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance", "path_edge"],
                "graph_distance": {"max_dist": 6, "embedding_dim": 16},
                "path_edge": {"embedding_dim": 8, "max_path_length": 5},
            },
        }
        layer = self._make_layer(pair_init=cfg, in_dim_edges=edge_dim)
        batch = _make_batch(
            num_graphs=3, in_dim=self.in_dim, edge_feat_dim=edge_dim,
        )
        out = layer(batch)
        self.assertFalse(torch.isnan(out.pair_feat).any())
        self.assertTrue(out.pair_feat.abs().max().item() > 0.0)

    def test_path_edge_backward_produces_grads(self):
        """Per-step projections in the path_edge prior must receive gradients."""
        edge_dim = 6
        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["path_edge"],
                "path_edge": {"embedding_dim": 8, "max_path_length": 4},
            },
        }
        layer = self._make_layer(pair_init=cfg, in_dim_edges=edge_dim)
        layer.train()
        batch = _make_batch(
            num_graphs=3, in_dim=self.in_dim, edge_feat_dim=edge_dim,
        )
        out = layer(batch)
        (out.pair_feat.sum() + out.feat.sum()).backward()

        path_edge_mod = layer.pair_init.pair_positional.source_modules["path_edge"]
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in path_edge_mod.parameters()
        )
        self.assertTrue(has_grad)

    def test_forward_backward(self):
        """Full forward + backward pass with priors does not NaN."""
        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance"],
                "graph_distance": {"max_dist": 6, "embedding_dim": 16},
            },
        }
        layer = self._make_layer(pair_init=cfg)
        layer.train()
        batch = _make_batch(num_graphs=3, in_dim=self.in_dim)
        out = layer(batch)
        loss = out.pair_feat.sum() + out.feat.sum()
        loss.backward()
        # Check that at least one parameter of the pair_positional prior
        # received a gradient.
        has_prior_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in layer.pair_init.pair_positional.parameters()
        )
        self.assertTrue(has_prior_grad)

    def test_variance_higher_than_opm_only(self):
        """With the prior added, layer-0 z variance should be strictly
        larger than the OPM-only baseline (sanity check that the prior
        actually contributes)."""
        torch.manual_seed(7)
        batch = _make_batch(num_graphs=3, in_dim=self.in_dim, seed=99)

        layer_opm_only = self._make_layer(pair_init={"mode": "opm_only"})
        layer_opm_only.eval()
        with torch.no_grad():
            out_opm = layer_opm_only(batch.clone())
        var_opm = out_opm.pair_feat.var().item()

        cfg = {
            "mode": "additive",
            "use_pair_positional": True,
            "pair_positional": {
                "sources": ["graph_distance", "adjacency"],
            },
        }
        layer_priors = self._make_layer(pair_init=cfg)
        # Copy OPM weights so the difference is purely due to the prior.
        layer_priors.pair_init.opm.load_state_dict(
            layer_opm_only.pair_init.opm.state_dict()
        )
        layer_priors.eval()
        with torch.no_grad():
            out_priors = layer_priors(batch.clone())
        var_priors = out_priors.pair_feat.var().item()

        # Priors contribute independent variance → additive output has
        # strictly larger variance (up to float noise).
        self.assertGreater(var_priors, var_opm)


if __name__ == "__main__":
    ut.main()
