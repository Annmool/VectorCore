"""
Unit tests for Flat, IVF, and HNSW indexes with metadata filtering.
"""

import numpy as np
import pytest
from vectordb.core.flat_index import FlatIndex
from vectordb.core.ivf_index import IVFIndex
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.collection import Collection, compile_metadata_filter


def test_flat_index_search():
    dim = 16
    idx = FlatIndex(dim=dim, metric="cosine")
    v1 = np.ones(dim, dtype=np.float32)
    v2 = -np.ones(dim, dtype=np.float32)

    idx.add("doc1", v1, {"category": "nlp"})
    idx.add("doc2", v2, {"category": "cv"})

    # Query with v1
    res = idx.search(v1, k=2)
    assert len(res) == 2
    assert res[0]["id"] == "doc1"
    assert np.isclose(res[0]["distance"], 0.0)


def test_metadata_filter():
    filter_dict = {
        "year": {"$gte": 2020},
        "tags": {"$in": ["nlp", "ml"]},
    }
    filter_fn = compile_metadata_filter(filter_dict)

    doc_pass = {"year": 2022, "tags": "nlp"}
    doc_fail_year = {"year": 2018, "tags": "nlp"}
    doc_fail_tag = {"year": 2023, "tags": "graphics"}

    assert filter_fn(doc_pass) is True
    assert filter_fn(doc_fail_year) is False
    assert filter_fn(doc_fail_tag) is False


def test_ivf_index():
    np.random.seed(42)
    dim = 32
    vectors = np.random.randn(100, dim).astype(np.float32)
    ids = [f"id_{i}" for i in range(100)]

    ivf = IVFIndex(dim=dim, nlist=8, nprobe=4, metric="cosine")
    ivf.add_batch(ids, vectors)

    assert len(ivf) == 100
    res = ivf.search(vectors[0], k=5)
    assert len(res) <= 5
    assert res[0]["id"] == "id_0"


def test_hnsw_index():
    np.random.seed(42)
    dim = 32
    vectors = np.random.randn(100, dim).astype(np.float32)
    ids = [f"id_{i}" for i in range(100)]

    hnsw = HNSWIndex(dim=dim, M=8, ef_construction=32, ef_search=16, metric="cosine")
    hnsw.add_batch(ids, vectors)

    assert len(hnsw) == 100
    res = hnsw.search(vectors[0], k=5)
    assert len(res) == 5
    assert res[0]["id"] == "id_0"


def test_collection_crud_and_persistence(tmp_path):
    from vectordb.core.storage import StorageEngine

    storage = StorageEngine(base_dir=str(tmp_path))
    coll = Collection("test_coll", dim=8, index_type="hnsw", storage_engine=storage)

    v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    coll.insert("doc_1", v1, {"title": "Paper 1", "year": 2024})
    coll.insert("doc_2", v2, {"title": "Paper 2", "year": 2019})

    assert len(coll) == 2

    # Query with filter
    res = coll.query(v1, k=2, filter={"year": {"$gt": 2020}})
    assert len(res) == 1
    assert res[0]["id"] == "doc_1"

    # Save and reload
    coll.save()
    loaded_coll = Collection.load("test_coll", storage_engine=storage)
    assert loaded_coll is not None
    assert len(loaded_coll) == 2
    res_loaded = loaded_coll.query(v1, k=1)
    assert res_loaded[0]["id"] == "doc_1"
