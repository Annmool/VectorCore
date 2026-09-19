"""
Unit tests for RAG generation, Citation Engine, LRU Cache, and Evaluation Metrics.
"""

from vectordb.rag.cache import LRUQueryCache
from vectordb.rag.citation import CitationEngine
from vectordb.rag.generator import RAGGenerator
from vectordb.evaluation.metrics import recall_at_k, reciprocal_rank, evaluate_retrieval_system


def test_lru_cache():
    cache = LRUQueryCache(capacity=2)
    cache.put("q1", {"result": "data1"})
    cache.put("q2", {"result": "data2"})

    assert cache.get("q1")["result"] == "data1"
    cache.put("q3", {"result": "data3"})  # Should evict q2

    assert cache.get("q2") is None
    assert cache.get("q3") is not None
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_citation_engine():
    text = "HNSW graphs use multi-layer skip lists [1]. Product Quantization uses sub-vectors [2]."
    indices = CitationEngine.extract_citation_indices(text)
    assert indices == [1, 2]

    chunks = [
        {"id": "chunk_1", "text": "HNSW graphs use multi-layer skip lists for O(log N) search."},
        {"id": "chunk_2", "text": "Product Quantization decomposes vectors into sub-vectors."},
    ]
    resp, citations, metrics = CitationEngine.build_grounded_citations(text, chunks)
    assert len(citations) == 2
    assert citations[0]["is_cited"] is True
    assert citations[1]["is_cited"] is True
    assert metrics["grounding_faithfulness_ratio"] > 0.5


def test_rag_extractive_generator():
    generator = RAGGenerator(preferred_provider="smart_extractive")
    chunks = [
        {
            "id": "c1",
            "text": "Okapi BM25 uses k1 for term frequency saturation and b for document length penalization.",
            "metadata": {"source": "doc_bm25.pdf", "page": 1},
        }
    ]
    res = generator.generate_answer("Why does BM25 use k1 and b?", chunks)
    assert "answer" in res
    assert "[1]" in res["answer"]
    assert len(res["citations"]) == 1
    assert res["citations"][0]["source"] == "doc_bm25.pdf"


def test_evaluation_metrics():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    gt = {"doc2", "doc5"}

    rec_1 = recall_at_k(retrieved, gt, k=1)
    rec_2 = recall_at_k(retrieved, gt, k=2)
    mrr = reciprocal_rank(retrieved, gt)

    assert rec_1 == 0.0
    assert rec_2 == 0.5
    assert mrr == 0.5  # First match at rank 2 -> 1/2

    eval_data = [
        {"retrieved_ids": ["d1", "d2"], "ground_truth_ids": ["d1"]},
        {"retrieved_ids": ["d3", "d2"], "ground_truth_ids": ["d2"]},
    ]
    summary = evaluate_retrieval_system(eval_data, k_values=[1, 2])
    assert summary["recall@1"] == 0.5
    assert summary["recall@2"] == 1.0
    assert summary["mrr"] == 0.75
