"""
Tests for SizeBucketSampler.
"""

import numpy as np
import pytest

from graphium.data.sampler import SizeBucketSampler


class TestSizeBucketSampler:
    def _make_sampler(self, num_nodes_list, batch_size, shuffle=False, indices=None):
        return SizeBucketSampler(
            num_nodes_list=num_nodes_list,
            batch_size=batch_size,
            shuffle=shuffle,
            indices=indices,
        )

    def test_all_indices_yielded_once(self):
        num_nodes = [5, 20, 3, 15, 8, 12, 1, 30, 10, 25]
        sampler = self._make_sampler(num_nodes, batch_size=3, shuffle=False)
        result = list(sampler)
        assert sorted(result) == list(range(10))
        assert len(result) == len(sampler)

    def test_consecutive_batch_has_similar_sizes(self):
        np.random.seed(42)
        num_nodes = list(np.random.randint(1, 100, size=100))
        batch_size = 10
        sampler = self._make_sampler(num_nodes, batch_size=batch_size, shuffle=False)
        indices = list(sampler)

        # Check that within each batch-sized chunk, the size range is small
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            chunk = indices[start : start + batch_size]
            sizes = [num_nodes[i] for i in chunk]
            size_range = max(sizes) - min(sizes)
            # Sorted chunks should have much tighter range than full dataset
            assert size_range < 20, f"Chunk {start} has range {size_range}: {sizes}"

    def test_shuffle_produces_different_chunk_orders(self):
        num_nodes = list(range(100))  # 0..99 nodes
        batch_size = 10
        sampler = self._make_sampler(num_nodes, batch_size=batch_size, shuffle=True)

        orderings = []
        for _ in range(5):
            orderings.append(list(sampler))

        # At least 2 of the 5 orderings should differ (chunk shuffle is random)
        unique = set(tuple(o) for o in orderings)
        assert len(unique) > 1, "Shuffle should produce different orderings across epochs"

    def test_no_shuffle_is_deterministic(self):
        num_nodes = [5, 20, 3, 15, 8]
        sampler = self._make_sampler(num_nodes, batch_size=2, shuffle=False)
        order1 = list(sampler)
        order2 = list(sampler)
        assert order1 == order2

    def test_indices_subset(self):
        num_nodes = [5, 20, 3, 15, 8, 12, 1, 30]
        subset = [0, 2, 4, 6]  # sizes: 5, 3, 8, 1
        sampler = self._make_sampler(num_nodes, batch_size=2, shuffle=False, indices=subset)
        result = list(sampler)
        assert sorted(result) == sorted(subset)
        assert len(result) == len(sampler)

    def test_len(self):
        num_nodes = [5, 20, 3, 15, 8]
        sampler = self._make_sampler(num_nodes, batch_size=2)
        assert len(sampler) == 5

        sampler_subset = self._make_sampler(num_nodes, batch_size=2, indices=[0, 1, 2])
        assert len(sampler_subset) == 3

    def test_batch_size_larger_than_dataset(self):
        num_nodes = [5, 3, 8]
        sampler = self._make_sampler(num_nodes, batch_size=10, shuffle=False)
        result = list(sampler)
        assert sorted(result) == [0, 1, 2]
