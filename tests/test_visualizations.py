"""
Unit tests for 2D Embedding Space Projection and HNSW Query Traversal Tracing.
"""

import numpy as np
from vectordb.visualization.projector import EmbeddingProjector
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.collection import Collection


def test_embedding_projector_pca():
    projector = EmbeddingProjector()
    dim = 64
    rng = np.random.default_rng(42)

    # Create 3 distinct synthetic clusters
    cluster1 = rng.normal(loc=2.0, scale=0.3, size=(10, dim)).astype(np.float32)
    cluster2 = rng.normal(loc=-2.0, scale=0.3, size=(10, dim)).astype(np.float32)
    cluster3 = rng.normal(loc=0.0, scale=0.3, size=(10, dim)).astype(np.float32)
    vectors = np.vstack([cluster1, cluster2, cluster3])

    ids = [f"doc_{i}" for i in range(30)]
    metadata = [
        {"source": "Paper A" if i < 10 else ("Paper B" if i < 20 else "Paper C"), "title": f"Doc {i}"}
        for i in range(30)
    ]

    res = projector.fit_transform(vectors, ids, metadata)

    assert "points" in res
    assert len(res["points"]) == 30
    assert "clusters" in res
    assert len(res["clusters"]) == 3
    assert "explained_variance" in res
    assert len(res["explained_variance"]) == 2
    assert res["total_points"] == 30

    # Test query projection
    query_vec = rng.normal(loc=2.0, scale=0.3, size=(dim,)).astype(np.float32)
    qx, qy = projector.project_query(query_vec)
    assert isinstance(qx, float)
    assert isinstance(qy, float)


def test_hnsw_search_with_trace():
    dim = 32
    hnsw = HNSWIndex(dim=dim, M=8, ef_construction=32, ef_search=16, metric="cosine", seed=42)
    rng = np.random.default_rng(42)

    # Insert 25 vectors
    for i in range(25):
        vec = rng.standard_normal(dim).astype(np.float32)
        hnsw.add(f"chunk_{i}", vec, {"title": f"Paper {i}", "source": f"paper_{i % 3}.pdf"})

    q = rng.standard_normal(dim).astype(np.float32)
    trace = hnsw.search_with_trace(q, k=3)

    assert "steps" in trace
    assert len(trace["steps"]) > 0
    assert "results" in trace
    assert len(trace["results"]) <= 3
    assert trace["total_hops"] == len(trace["steps"])
    assert trace["enter_node"] is not None

    # First step must be entry point
    assert trace["steps"][0]["type"] == "entry"
    assert trace["steps"][0]["layer"] == hnsw.max_level


def test_collection_query_with_trace():
    dim = 32
    coll = Collection(name="trace_test", dim=dim, index_type="hnsw")
    rng = np.random.default_rng(123)

    ids = [f"item_{i}" for i in range(15)]
    vecs = rng.standard_normal((15, dim)).astype(np.float32)
    metas = [{"source": "Test", "title": f"Doc {i}"} for i in range(15)]

    coll.insert_batch(ids, vecs, metas)
    q = rng.standard_normal(dim).astype(np.float32)

    res = coll.query_with_trace(q, k=4)
    assert "steps" in res
    assert len(res["steps"]) > 0
    assert len(res["results"]) == 4
