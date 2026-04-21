"""
Unit tests for the Pairformer layer (graphium/nn/pyg_layers/pairformer_pyg.py).

Tests numerical stability under different precision modes (fp16 vs bf16)
and verifies correct masking behavior for padded graphs.
"""

import torch
import unittest as ut
from copy import deepcopy
import pytest

from torch_geometric.data import Data, Batch

from graphium.nn.pyg_layers.pairformer_pyg import (
    PairformerLayerPyg,
    Transition,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttention,
    AttentionPairBias,
    OuterProductMean,
)


def _make_batch(num_graphs=4, min_nodes=3, max_nodes=12, in_dim=64, seed=42):
    """Create a PyG Batch of random graphs with varying sizes."""
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(num_graphs):
        n = torch.randint(min_nodes, max_nodes + 1, (1,), generator=rng).item()
        # Simple chain edges
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        feat = torch.randn(n, in_dim, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index))
    return Batch.from_data_list(graphs)


class TestPairformerComponents(ut.TestCase):
    """Test individual Pairformer sub-modules."""

    dim = 64
    pair_dim = 32
    B, N = 3, 8

    def _random_pair_inputs(self, dtype=torch.float32):
        x = torch.randn(self.B, self.N, self.N, self.pair_dim, dtype=dtype)
        mask = torch.ones(self.B, self.N, self.N)
        # Mask out last 2 positions to simulate padding
        mask[:, -2:, :] = 0
        mask[:, :, -2:] = 0
        return x, mask

    def _random_node_inputs(self, dtype=torch.float32):
        s = torch.randn(self.B, self.N, self.dim, dtype=dtype)
        mask = torch.ones(self.B, self.N)
        mask[:, -2:] = 0
        return s, mask

    def test_transition_no_nan(self):
        t = Transition(self.pair_dim, self.pair_dim * 4)
        x = torch.randn(self.B, self.N, self.N, self.pair_dim)
        out = t(x)
        self.assertFalse(torch.isnan(out).any(), "Transition produced NaN")
        self.assertEqual(out.shape, x.shape)

    def test_triangle_mul_outgoing_no_nan(self):
        layer = TriangleMultiplicationOutgoing(self.pair_dim)
        x, mask = self._random_pair_inputs()
        out = layer(x, mask)
        self.assertFalse(torch.isnan(out).any(), "TriangleMultiplicationOutgoing produced NaN")

    def test_triangle_mul_incoming_no_nan(self):
        layer = TriangleMultiplicationIncoming(self.pair_dim)
        x, mask = self._random_pair_inputs()
        out = layer(x, mask)
        self.assertFalse(torch.isnan(out).any(), "TriangleMultiplicationIncoming produced NaN")

    def test_triangle_attention_no_nan(self):
        for starting in [True, False]:
            layer = TriangleAttention(self.pair_dim, c_hidden=16, no_heads=2, starting=starting)
            x, mask = self._random_pair_inputs()
            out = layer(x, mask)
            self.assertFalse(torch.isnan(out).any(), f"TriangleAttention(starting={starting}) produced NaN")

    def test_triangle_attention_all_masked_row(self):
        """When an entire row is masked (padding graph), attention should not produce NaN."""
        layer = TriangleAttention(self.pair_dim, c_hidden=16, no_heads=2, starting=True)
        x = torch.randn(2, 5, 5, self.pair_dim)
        mask = torch.ones(2, 5, 5)
        mask[1, :, :] = 0  # Entire second graph is padding
        out = layer(x, mask)
        self.assertFalse(torch.isnan(out).any(), "TriangleAttention NaN on fully-masked graph")

    def test_attention_pair_bias_no_nan(self):
        layer = AttentionPairBias(self.dim, self.pair_dim, num_heads=4)
        s, node_mask = self._random_node_inputs()
        z = torch.randn(self.B, self.N, self.N, self.pair_dim)
        out = layer(s, z, node_mask)
        self.assertFalse(torch.isnan(out).any(), "AttentionPairBias produced NaN")

    def test_outer_product_mean_no_nan(self):
        layer = OuterProductMean(self.dim, c_hidden=16, c_out=self.pair_dim)
        s, mask = self._random_node_inputs()
        out = layer(s, mask)
        self.assertFalse(torch.isnan(out).any(), "OuterProductMean produced NaN")
        self.assertEqual(out.shape, (self.B, self.N, self.N, self.pair_dim))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPairformerPrecision(ut.TestCase):
    """Test Pairformer layer under different precision modes on GPU."""

    in_dim = 256
    pair_dim = 64

    def _make_layer(self):
        return PairformerLayerPyg(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            pair_dim=self.pair_dim,
            num_heads=8,
            pairwise_head_width=32,
            pairwise_num_heads=4,
            pair_dropout=0.0,  # Disable dropout for deterministic testing
            hidden_dim_scaling=4.0,
            opm_hidden=32,
            use_checkpoint=False,
        )

    def _run_forward_backward(self, layer, batch, precision_context):
        """Run forward + backward and check for NaN in output and gradients."""
        layer.train()
        with precision_context:
            out_batch = layer(batch)
        loss = out_batch.feat.sum()
        loss.backward()

        has_nan_output = torch.isnan(out_batch.feat).any().item()
        has_nan_grad = any(
            torch.isnan(p.grad).any().item()
            for p in layer.parameters()
            if p.grad is not None
        )
        return has_nan_output, has_nan_grad

    def test_bf16_no_nan_single_layer(self):
        """A single Pairformer layer under bf16-mixed should not produce NaN."""
        layer = self._make_layer().cuda()
        batch = _make_batch(num_graphs=8, min_nodes=5, max_nodes=20, in_dim=self.in_dim, seed=123).cuda()

        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        has_nan_output, has_nan_grad = self._run_forward_backward(layer, batch, ctx)
        self.assertFalse(has_nan_output, "bf16: NaN in output")
        self.assertFalse(has_nan_grad, "bf16: NaN in gradients")

    def test_bf16_no_nan_stacked_layers(self):
        """Multiple Pairformer layers under bf16-mixed should not produce NaN."""
        depth = 4
        layers = [self._make_layer().cuda() for _ in range(depth)]
        batch = _make_batch(num_graphs=8, min_nodes=5, max_nodes=20, in_dim=self.in_dim, seed=456).cuda()

        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        with ctx:
            for layer in layers:
                batch = layer(batch)

        loss = batch.feat.sum()
        loss.backward()

        self.assertFalse(torch.isnan(batch.feat).any(), f"bf16 stacked ({depth} layers): NaN in output")

    def test_fp16_nan_expected(self):
        """Demonstrate that fp16 is prone to NaN under the same conditions.

        This test documents the fp16 overflow issue — it passes if NaN IS
        produced (confirming the problem) or if it happens to not overflow
        with these specific random seeds (non-deterministic).
        """
        depth = 4
        layers = [self._make_layer().cuda() for _ in range(depth)]
        batch = _make_batch(num_graphs=8, min_nodes=5, max_nodes=20, in_dim=self.in_dim, seed=789).cuda()

        ctx = torch.autocast("cuda", dtype=torch.float16)
        try:
            with ctx:
                for layer in layers:
                    batch = layer(batch)
            has_nan = torch.isnan(batch.feat).any().item()
        except RuntimeError:
            has_nan = True

        # This is a documentation test — we just print whether fp16 overflowed.
        # Not asserting failure because it depends on random init.
        if has_nan:
            print("fp16 produced NaN as expected (triangle ops overflow)")
        else:
            print("fp16 did not produce NaN with this seed (may still overflow with larger graphs/more layers)")

    def test_bf16_large_molecules(self):
        """Test with larger molecules that stress the triangle N-sum."""
        layer = self._make_layer().cuda()
        batch = _make_batch(num_graphs=4, min_nodes=30, max_nodes=50, in_dim=self.in_dim, seed=999).cuda()

        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        has_nan_output, has_nan_grad = self._run_forward_backward(layer, batch, ctx)
        self.assertFalse(has_nan_output, "bf16 large molecules: NaN in output")
        self.assertFalse(has_nan_grad, "bf16 large molecules: NaN in gradients")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPairformerMasking(ut.TestCase):
    """Test that padding nodes are handled correctly."""

    in_dim = 64
    pair_dim = 32

    def test_padding_does_not_affect_valid_nodes(self):
        """Valid node outputs should be the same regardless of padding size."""
        layer = PairformerLayerPyg(
            in_dim=self.in_dim, out_dim=self.in_dim, pair_dim=self.pair_dim,
            num_heads=4, pairwise_head_width=16, pairwise_num_heads=2,
            pair_dropout=0.0, hidden_dim_scaling=2.0, opm_hidden=16,
            use_checkpoint=False,
        ).cuda().eval()

        # Single graph with 5 nodes
        g = Data(
            feat=torch.randn(5, self.in_dim),
            edge_index=torch.stack([torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])]),
        )

        # Batch with just this graph
        batch1 = Batch.from_data_list([g]).cuda()
        # Batch with this graph + a padding graph (different N_max)
        g_pad = Data(
            feat=torch.randn(10, self.in_dim),
            edge_index=torch.stack([torch.arange(9), torch.arange(1, 10)]),
        )
        batch2 = Batch.from_data_list([g, g_pad]).cuda()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out1 = layer(deepcopy(batch1))
            out2 = layer(deepcopy(batch2))

        # Extract the first graph's features from both batches
        feat1 = out1.feat[:5]
        feat2 = out2.feat[:5]

        # They should be identical (padding shouldn't affect valid graph outputs)
        torch.testing.assert_close(feat1, feat2, atol=1e-2, rtol=1e-2)


class TestGraphOutputNNPairStatsPooling(ut.TestCase):
    """Graph-aware statistics pooling in GraphOutputNN (for PairMixer)."""

    pair_dim = 8
    node_dim = 16

    def _make_batch(self, n_list, seed=0):
        """Build a PyG Batch with synthetic pair_feat, pair_mask, edge_index."""
        rng = torch.Generator().manual_seed(seed)
        graphs = []
        for n in n_list:
            if n > 1:
                src = torch.arange(n - 1)
                dst = torch.arange(1, n)
                edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
            feat = torch.randn(n, self.node_dim, generator=rng)
            graphs.append(Data(feat=feat, edge_index=edge_index))
        batch = Batch.from_data_list(graphs)
        B = len(n_list)
        N = max(n_list)
        pair_feat = torch.randn(B, N, N, self.pair_dim, generator=rng)
        pair_mask = torch.zeros(B, N, N)
        for b, n in enumerate(n_list):
            pair_mask[b, :n, :n] = 1.0
        batch.pair_feat = pair_feat
        batch.pair_mask = pair_mask
        batch._pair_dense_idx = torch.zeros(batch.num_nodes, dtype=torch.bool)
        return batch

    def _make_head(self, pair_pool):
        from graphium.nn.architectures.global_architectures import GraphOutputNN
        return GraphOutputNN(
            in_dim=self.node_dim, in_dim_edges=0, task_level="graph",
            graph_output_nn_kwargs={"graph": {
                "pair_dim": self.pair_dim,
                "pair_pool": pair_pool,
                "hidden_dims": [self.pair_dim * (6 if pair_pool == "stats" else 1)],
                "out_dim": 4,
                "depth": 2,
                "activation": "relu",
            }},
        )

    def test_stats_output_shape(self):
        head = self._make_head("stats")
        out = head(self._make_batch([3, 5, 4], seed=1))
        self.assertEqual(out.shape, (3, 4))

    def test_level_in_dim_is_6x_pair_dim(self):
        head = self._make_head("stats")
        self.assertEqual(head._pair_pool_mult, 6)
        self.assertEqual(head.graph_output_nn.full_dims[0], 6 * self.pair_dim)

    def test_mean_pool_level_in_dim_is_pair_dim(self):
        """Regression: mean mode still uses pair_dim as level_in_dim."""
        head = self._make_head("mean")
        self.assertEqual(head._pair_pool_mult, 1)
        self.assertEqual(head.graph_output_nn.full_dims[0], self.pair_dim)

    def test_stats_global_matches_legacy_mean(self):
        """The 'global' stratum mean slice must equal _pool_pair_feat."""
        head = self._make_head("stats")
        batch = self._make_batch([4, 3, 5], seed=7)
        stats = head._pool_pair_stats(batch)
        legacy = head._pool_pair_feat(batch)
        # Layout: [mean_diag, mean_1hop, mean_global, std_diag, std_1hop, std_global]
        global_mean = stats[:, 2 * self.pair_dim:3 * self.pair_dim]
        torch.testing.assert_close(global_mean, legacy, atol=1e-6, rtol=1e-5)

    def test_empty_1hop_zero(self):
        """A single-atom graph has no 1-hop pairs → mean_1hop = std_1hop = 0."""
        head = self._make_head("stats")
        batch = self._make_batch([1, 4], seed=11)
        stats = head._pool_pair_stats(batch)
        mean_1hop = stats[0, self.pair_dim:2 * self.pair_dim]
        std_1hop = stats[0, 4 * self.pair_dim:5 * self.pair_dim]
        torch.testing.assert_close(mean_1hop, torch.zeros_like(mean_1hop))
        torch.testing.assert_close(std_1hop, torch.zeros_like(std_1hop))

    def test_no_nan_with_padding(self):
        """Huge sentinel values in padded positions must be masked out."""
        head = self._make_head("stats")
        n_list = [2, 6, 4]
        batch = self._make_batch(n_list, seed=19)
        for b, n in enumerate(n_list):
            batch.pair_feat[b, n:, :, :] = 1e6
            batch.pair_feat[b, :, n:, :] = 1e6
        stats = head._pool_pair_stats(batch)
        self.assertFalse(torch.isnan(stats).any())
        self.assertFalse(torch.isinf(stats).any())
        self.assertLess(stats.abs().max().item(), 1e3)

    def test_invalid_pair_pool_raises(self):
        from graphium.nn.architectures.global_architectures import GraphOutputNN
        with self.assertRaises(ValueError):
            GraphOutputNN(
                in_dim=self.node_dim, in_dim_edges=0, task_level="graph",
                graph_output_nn_kwargs={"graph": {
                    "pair_dim": self.pair_dim,
                    "pair_pool": "median",  # invalid
                    "hidden_dims": [8],
                    "out_dim": 4,
                    "depth": 2,
                    "activation": "relu",
                }},
            )


if __name__ == "__main__":
    ut.main()
