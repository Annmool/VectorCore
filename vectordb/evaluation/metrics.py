"""
Standard Information Retrieval Evaluation Metrics:
Recall@K, Precision@K, MRR (Mean Reciprocal Rank), MAP (Mean Average Precision), and NDCG@K.
"""

import math
from typing import List, Set, Dict, Any
import numpy as np


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Compute Recall@K: proportion of relevant documents retrieved in top-k."""
    if not ground_truth_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return hits / len(ground_truth_ids)


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Compute Precision@K: proportion of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return hits / k


def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Compute Reciprocal Rank: 1 / rank of first relevant retrieved item."""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    """Compute Average Precision (AP) for a single query."""
    if not ground_truth_ids:
        return 1.0

    score = 0.0
    num_hits = 0
    top_k = retrieved_ids[:k]

    for rank, doc_id in enumerate(top_k, 1):
        if doc_id in ground_truth_ids:
            num_hits += 1
            score += num_hits / rank

    return score / min(len(ground_truth_ids), k)


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not ground_truth_ids or k <= 0:
        return 1.0

    dcg = 0.0
    top_k = retrieved_ids[:k]
    for rank, doc_id in enumerate(top_k, 1):
        rel = 1.0 if doc_id in ground_truth_ids else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_system(
    query_results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute aggregate metrics across all evaluation queries.
    query_results format: [{'retrieved_ids': [...], 'ground_truth_ids': set(...)}]
    """
    if not query_results:
        return {}

    num_queries = len(query_results)
    metrics: Dict[str, float] = {}

    for k in k_values:
        recalls = [
            recall_at_k(qr["retrieved_ids"], set(qr["ground_truth_ids"]), k)
            for qr in query_results
        ]
        precisions = [
            precision_at_k(qr["retrieved_ids"], set(qr["ground_truth_ids"]), k)
            for qr in query_results
        ]
        ndcgs = [
            ndcg_at_k(qr["retrieved_ids"], set(qr["ground_truth_ids"]), k)
            for qr in query_results
        ]
        metrics[f"recall@{k}"] = round(float(np.mean(recalls)), 4)
        metrics[f"precision@{k}"] = round(float(np.mean(precisions)), 4)
        metrics[f"ndcg@{k}"] = round(float(np.mean(ndcgs)), 4)

    mrrs = [
        reciprocal_rank(qr["retrieved_ids"], set(qr["ground_truth_ids"]))
        for qr in query_results
    ]
    maps = [
        average_precision(qr["retrieved_ids"], set(qr["ground_truth_ids"]))
        for qr in query_results
    ]

    metrics["mrr"] = round(float(np.mean(mrrs)), 4)
    metrics["map"] = round(float(np.mean(maps)), 4)
    metrics["num_queries_evaluated"] = num_queries

    return metrics
