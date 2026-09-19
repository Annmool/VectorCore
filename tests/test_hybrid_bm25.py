"""
Unit tests for BM25 keyword search, Hybrid RRF retrieval, and Cross-Encoder reranker.
"""

import numpy as np
from vectordb.retrieval.bm25 import BM25Index
from vectordb.retrieval.hybrid import HybridRetriever
from vectordb.retrieval.reranker import CrossEncoderReranker
from vectordb.core.collection import Collection
from vectordb.embeddings.embedder import Embedder


def test_bm25_index():
    bm25 = BM25Index()
    bm25.add_document("doc1", "Hierarchical Navigable Small World graphs for vector search", {"category": "graph"})
    bm25.add_document("doc2", "Product Quantization Asymmetric Distance Computation compression", {"category": "pq"})
    bm25.add_document("doc3", "BM25 keyword search ranking function with term frequencies", {"category": "bm25"})

    res = bm25.search("Navigable Small World", k=2)
    assert len(res) > 0
    assert res[0]["id"] == "doc1"

    res_pq = bm25.search("Quantization compression", k=2)
    assert len(res_pq) > 0
    assert res_pq[0]["id"] == "doc2"


def test_hybrid_rrf_retrieval():
    embedder = Embedder()
    coll = Collection("hybrid_test", dim=embedder.dim, index_type="hnsw")
    bm25 = BM25Index()

    docs = [
        ("doc_hnsw", "HNSW builds multi-layer proximity graphs for nearest neighbor search"),
        ("doc_pq", "Product Quantization decomposes vectors into sub-vectors with K-Means"),
        ("doc_bm25", "BM25 uses IDF and term frequency saturation parameters"),
    ]

    for doc_id, text in docs:
        vec = embedder.embed_texts(text)
        coll.insert(doc_id, vec, {"text": text})
        bm25.add_document(doc_id, text, {"text": text})

    retriever = HybridRetriever(coll, bm25, embedder)
    results = retriever.search("proximity graphs HNSW", k=2, fusion_method="rrf")

    assert len(results) >= 1
    assert results[0]["id"] == "doc_hnsw"


def test_reranker():
    reranker = CrossEncoderReranker()
    query = "What is Product Quantization?"
    candidates = [
        {"id": "c1", "text": "HNSW graphs are skip-list proximity graphs for ANN search."},
        {"id": "c2", "text": "Product Quantization is a vector compression technique dividing space into sub-spaces."},
    ]

    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["id"] == "c2"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
