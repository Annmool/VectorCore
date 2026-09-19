"""
Unit tests for distance metrics.
"""

import numpy as np
import pytest
from vectordb.core.distance import (
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    dot_product,
    manhattan_distance,
    compute_distance,
    distance_to_score,
)


def test_cosine_similarity():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    w = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert np.isclose(cosine_similarity(u, v), 1.0)
    assert np.isclose(cosine_similarity(u, w), 0.0)
    assert np.isclose(cosine_distance(u, v), 0.0)
    assert np.isclose(cosine_distance(u, w), 1.0)


def test_batch_distance():
    q = np.array([1.0, 0.0], dtype=np.float32)
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    dists = compute_distance(q, mat, metric="cosine")
    assert len(dists) == 3
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)
    assert np.isclose(dists[2], 2.0)


def test_euclidean_and_manhattan():
    u = np.array([0.0, 0.0], dtype=np.float32)
    v = np.array([3.0, 4.0], dtype=np.float32)

    assert np.isclose(euclidean_distance(u, v), 5.0)
    assert np.isclose(manhattan_distance(u, v), 7.0)
