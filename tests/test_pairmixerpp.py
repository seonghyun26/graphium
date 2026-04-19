import unittest as ut
from copy import deepcopy

import torch
from torch_geometric.data import Batch, Data

from graphium.nn.pyg_layers import PairMixerPPLayerPyg


def _make_batch(num_graphs=2, in_dim=32, edge_feat_dim=8, seed=123):
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for gidx in range(num_graphs):
        n = 4 + gidx
        src = torch.arange(n - 1)
        dst = src + 1
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])],
            dim=0,
        )
        feat = torch.randn(n, in_dim, generator=rng)
        edge_feat = torch.randn(edge_index.shape[1], edge_feat_dim, generator=rng)
        graphs.append(Data(feat=feat, edge_index=edge_index, edge_feat=edge_feat))
    return Batch.from_data_list(graphs)


class TestPairMixerPP(ut.TestCase):
    in_dim = 32
    pair_dim = 24
    edge_dim = 8

    def _make_layer(self):
        return PairMixerPPLayerPyg(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            pair_dim=self.pair_dim,
            in_dim_edges=self.edge_dim,
            pair_dropout=0.0,
            hidden_dim_scaling=2.0,
            opm_hidden=8,
            use_checkpoint=False,
            edge_injection=True,
            single_to_pair_rank=8,
            layer_idx=0,
            layer_depth=2,
        )

    def test_forward_updates_single_and_pair_tracks(self):
        layer = self._make_layer()
        batch = _make_batch(in_dim=self.in_dim, edge_feat_dim=self.edge_dim)
        feat_in = batch.feat.clone()

        out = layer(deepcopy(batch))

        self.assertTrue(hasattr(out, "pair_feat"))
        self.assertTrue(hasattr(out, "pair_mask"))
        self.assertFalse(torch.isnan(out.feat).any())
        self.assertFalse(torch.isnan(out.pair_feat).any())
        self.assertEqual(out.feat.shape, feat_in.shape)
        self.assertEqual(out.pair_feat.shape[-1], self.pair_dim)
        self.assertFalse(torch.allclose(out.feat, feat_in))

    def test_backward_reaches_single_to_pair_coupling(self):
        layer = self._make_layer()
        layer.train()
        batch = _make_batch(in_dim=self.in_dim, edge_feat_dim=self.edge_dim)

        out = layer(batch)
        loss = out.feat.sum() + out.pair_feat.sum()
        loss.backward()

        self.assertIsNotNone(layer.proj_left.weight.grad)
        self.assertGreater(layer.proj_left.weight.grad.abs().sum().item(), 0.0)
        single_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in layer.single_update.parameters()
        )
        self.assertTrue(single_has_grad)


if __name__ == "__main__":
    ut.main()
