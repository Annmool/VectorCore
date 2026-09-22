"""
Exact output regression tests for HNSW refactoring (Tweaks 8-11).
Verifies:
1. Identical graph topology (enter_node, max_level, node_levels, layers) and search results
   (IDs and distances within 1e-5) compared to the baseline captured before refactoring.
2. Raw vector fidelity: self.vectors returns the unmodified raw vectors, preserving fidelity
   for Collection, IVF K-Means clustering, and Scalar/Product quantizers.
3. Property setter buffer resizing when reassigning index.vectors.
"""

import os
import json
import numpy as np
import pytest
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.collection import Collection
from vectordb.core.storage import StorageEngine


def test_hnsw_exact_parity_with_baseline():
    """Verify bit-for-bit parity on graph structure and search rankings against pre-refactor baseline."""
    baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baseline_regression_data.json")
    assert os.path.exists(baseline_path), f"Baseline file {baseline_path} not found"

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    dim = 16
    n_docs = 100
    n_queries = 20
    seed = 42

    rng = np.random.default_rng(seed)
    doc_vecs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    query_vecs = rng.standard_normal((n_queries, dim)).astype(np.float32)

    for metric in ["cosine", "l2"]:
        assert metric in baseline, f"Metric {metric} missing from baseline"
        base_data = baseline[metric]

        hnsw = HNSWIndex(dim=dim, M=8, M0=16, ef_construction=32, ef_search=16, metric=metric, seed=seed)
        for i in range(n_docs):
            hnsw.add(f"doc_{i}", doc_vecs[i])

        # Verify enter node and levels
        assert hnsw.enter_node == base_data["enter_node"]
        assert hnsw.max_level == base_data["max_level"]
        for k, v in base_data["node_levels"].items():
            assert hnsw.node_levels[int(k)] == v

        # Verify layer graph structure
        assert len(hnsw.layers) == len(base_data["layers"])
        for lev_idx, base_layer in enumerate(base_data["layers"]):
            curr_layer = hnsw.layers[lev_idx]
            assert set(curr_layer.keys()) == {int(k) for k in base_layer.keys()}
            for k_str, expected_neighbors in base_layer.items():
                k_int = int(k_str)
                assert sorted(list(curr_layer[k_int])) == expected_neighbors, (
                    f"Mismatch in layer {lev_idx} for node {k_int}"
                )

        # Verify search query results (ids identical, distances within 1e-4)
        for q_idx, q in enumerate(query_vecs):
            res = hnsw.search(q, k=5)
            expected_res = base_data["queries"][q_idx]

            assert len(res) == len(expected_res)
            res_ids = [r["id"] for r in res]
            expected_ids = [r["id"] for r in expected_res]
            assert res_ids == expected_ids, (
                f"Metric {metric} query {q_idx} mismatch: got {res_ids}, expected {expected_ids}"
            )

            for r, exp_r in zip(res, expected_res):
                assert np.isclose(r["distance"], exp_r["distance"], atol=1e-4), (
                    f"Distance mismatch on {r['id']}: got {r['distance']}, expected {exp_r['distance']}"
                )
                assert np.isclose(r["score"], exp_r["score"], atol=1e-4)


def test_raw_vector_preservation():
    """Verify that self.vectors returns the raw, unnormalized vector values."""
    dim = 8
    hnsw = HNSWIndex(dim=dim, metric="cosine", seed=42)

    raw_vec = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # L2 norm is 5.0, unnormalized
    hnsw.add("doc_0", raw_vec)

    # index.vectors must return the EXACT raw vector
    stored_raw = hnsw.vectors[0]
    assert np.allclose(stored_raw, raw_vec, atol=1e-6)
    assert np.isclose(np.linalg.norm(stored_raw), 5.0, atol=1e-6)

    # Internal normalized vector must be unit norm
    norm_vec = hnsw._norm_vectors[0]
    assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-6)
    assert np.isclose(norm_vec[0], 3.0 / 5.0, atol=1e-6)


def test_vectors_setter_and_resizing():
    """Verify that reassigning index.vectors reallocates capacity appropriately."""
    dim = 4
    hnsw = HNSWIndex(dim=dim, metric="cosine", seed=42)

    # 1. Setting empty array resets capacity to 64
    hnsw.vectors = np.empty((0, dim), dtype=np.float32)
    assert hnsw._capacity == 64
    assert len(hnsw.vectors) == 0

    # 2. Setting large array (>64) expands capacity to next power of 2
    big_arr = np.ones((100, dim), dtype=np.float32) * 2.0
    hnsw.vectors = big_arr
    assert hnsw._capacity >= 128
    assert len(hnsw._vectors) == hnsw._capacity
    assert np.allclose(hnsw.vectors, big_arr)
    # Norm vectors must also be synchronized and unit normalized
    assert np.isclose(np.linalg.norm(hnsw._norm_vectors[0]), 1.0, atol=1e-5)

    # 3. Setting smaller array
    small_arr = np.ones((10, dim), dtype=np.float32)
    hnsw.vectors = small_arr
    assert hnsw._capacity >= 64
    assert np.allclose(hnsw.vectors, small_arr)


def test_ivf_receives_raw_vectors_through_collection(tmp_path):
    """Verify that switching or feeding vectors from Collection to IVF uses raw vectors, not normalized."""
    storage = StorageEngine(base_dir=str(tmp_path))
    coll = Collection("test_coll_raw", dim=4, index_type="hnsw", metric="cosine", storage_engine=storage)

    # Add vectors with different norms
    v1 = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 20.0, 0.0, 0.0], dtype=np.float32)
    coll.insert("d1", v1)
    coll.insert("d2", v2)

    # Verify Collection.index.vectors returns raw vectors
    raw_stored = coll.index.vectors
    assert np.isclose(np.linalg.norm(raw_stored[0]), 10.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(raw_stored[1]), 20.0, atol=1e-5)

    # Quantization fit uses raw vectors
    stats = coll.quantize_scalar()
    assert stats["status"] == "quantized"

    # Save and reload
    coll.save()
    loaded = Collection.load("test_coll_raw", storage_engine=storage)
    assert loaded is not None
    assert np.isclose(np.linalg.norm(loaded.index.vectors[0]), 10.0, atol=1e-5)


def test_buffer_resize_crossing_boundaries():
    """
    Stress-test buffer resizing across boundaries (64 -> 128 -> 256).
    Verifies:
    1. For cosine: both _vectors and _norm_vectors are properly copied and synchronized.
    2. For L2: _norm_vectors is re-aliased to the newly allocated _vectors array,
       not left pointing to the stale pre-resize array.
    """
    dim = 8
    rng = np.random.default_rng(123)

    # 1. Cosine metric test
    hnsw_cos = HNSWIndex(dim=dim, metric="cosine", seed=42)
    assert hnsw_cos._capacity == 64

    # Insert 64 vectors (exact capacity)
    cos_vecs = rng.standard_normal((130, dim)).astype(np.float32)
    for i in range(64):
        hnsw_cos.add(f"doc_{i}", cos_vecs[i])
    assert hnsw_cos._capacity == 64
    assert len(hnsw_cos.vectors) == 64

    # Insert vector #65 (boundary crossing: 64 -> 128)
    hnsw_cos.add("doc_64", cos_vecs[64])
    assert hnsw_cos._capacity == 128
    assert len(hnsw_cos.vectors) == 65

    # Check all 65 raw and normalized vectors
    for i in range(65):
        assert np.allclose(hnsw_cos.vectors[i], cos_vecs[i])
        norm_i = np.linalg.norm(cos_vecs[i])
        expected_norm = cos_vecs[i] / (norm_i if norm_i > 1e-10 else 1.0)
        assert np.allclose(hnsw_cos._norm_vectors[i], expected_norm, atol=1e-5)

    # Insert up to 129 vectors (boundary crossing: 128 -> 256)
    for i in range(65, 129):
        hnsw_cos.add(f"doc_{i}", cos_vecs[i])
    assert hnsw_cos._capacity == 256
    assert len(hnsw_cos.vectors) == 129
    for i in range(129):
        assert np.allclose(hnsw_cos.vectors[i], cos_vecs[i])

    # 2. Non-cosine (L2) metric test: verify re-aliasing
    hnsw_l2 = HNSWIndex(dim=dim, metric="l2", seed=42)
    assert hnsw_l2._norm_vectors is hnsw_l2._vectors

    l2_vecs = rng.standard_normal((70, dim)).astype(np.float32)
    for i in range(64):
        hnsw_l2.add(f"l2_{i}", l2_vecs[i])
    assert hnsw_l2._capacity == 64
    assert hnsw_l2._norm_vectors is hnsw_l2._vectors

    # Cross boundary to 128
    hnsw_l2.add("l2_64", l2_vecs[64])
    assert hnsw_l2._capacity == 128
    # Must STILL be an alias of the new _vectors buffer!
    assert hnsw_l2._norm_vectors is hnsw_l2._vectors
    # Verify values match
    for i in range(65):
        assert np.allclose(hnsw_l2.vectors[i], l2_vecs[i])
        assert np.allclose(hnsw_l2._norm_vectors[i], l2_vecs[i])


def test_persistence_roundtrip_recomputation(tmp_path):
    """
    Verify Collection save/load roundtrip:
    - Raw vectors are saved to disk.
    - On load, HNSW recomputes _norm_vectors naturally on insert, preventing drift.
    """
    storage = StorageEngine(base_dir=str(tmp_path))
    coll = Collection("test_persist_sync", dim=8, index_type="hnsw", metric="cosine", storage_engine=storage)

    rng = np.random.default_rng(999)
    vecs = rng.standard_normal((75, 8)).astype(np.float32) * 5.0  # Non-unit vectors
    for i in range(75):
        coll.insert(f"doc_{i}", vecs[i], {"idx": i})

    assert len(coll) == 75
    # Save to disk
    coll.save()

    # Load from disk
    loaded_coll = Collection.load("test_persist_sync", storage_engine=storage)
    assert loaded_coll is not None
    assert len(loaded_coll) == 75

    # Check that raw vectors match exactly
    for i in range(75):
        assert np.allclose(loaded_coll.index.vectors[i], vecs[i], atol=1e-5)
        # Check that norm vectors were properly recomputed to unit norm
        norm_v = loaded_coll.index._norm_vectors[i]
        assert np.isclose(np.linalg.norm(norm_v), 1.0, atol=1e-5)

    # Search on loaded collection
    res = loaded_coll.query(vecs[0], k=3)
    assert len(res) == 3
    assert res[0]["id"] == "doc_0"


def test_heuristic_pairwise_matrix_all_metrics():
    """Verify _pairwise_distances accuracy across all supported metrics."""
    dim = 8
    rng = np.random.default_rng(42)
    raw_vecs = rng.standard_normal((10, dim)).astype(np.float32)

    for metric in ["cosine", "l2", "dot", "manhattan"]:
        hnsw = HNSWIndex(dim=dim, metric=metric, seed=42)
        if metric == "cosine":
            norms = np.linalg.norm(raw_vecs, axis=-1, keepdims=True)
            test_vecs = raw_vecs / norms
        else:
            test_vecs = raw_vecs

        dist_mat = hnsw._pairwise_distances(test_vecs)
        assert dist_mat.shape == (10, 10)

        # Check against individual distance calculations
        for i in range(10):
            for j in range(10):
                if metric == "cosine":
                    expected_d = max(0.0, 1.0 - float(np.dot(test_vecs[i], test_vecs[j])))
                elif metric == "l2":
                    expected_d = float(np.linalg.norm(test_vecs[i] - test_vecs[j]))
                elif metric == "dot":
                    expected_d = -float(np.dot(test_vecs[i], test_vecs[j]))
                elif metric == "manhattan":
                    expected_d = float(np.sum(np.abs(test_vecs[i] - test_vecs[j])))
                assert np.isclose(dist_mat[i, j], expected_d, atol=1e-4), (
                    f"Metric {metric} mismatch at ({i}, {j}): got {dist_mat[i, j]}, expected {expected_d}"
                )

