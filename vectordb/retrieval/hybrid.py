"""
Hybrid Retrieval Engine combining Dense Vector Search and Sparse BM25 Keyword Search.
Implements Reciprocal Rank Fusion (RRF) and Convex Weighted Score Combination.
"""

from typing import List, Dict, Any, Optional, Literal
import numpy as np
from vectordb.core.collection import Collection
from vectordb.retrieval.bm25 import BM25Index
from vectordb.embeddings.embedder import Embedder

FusionMethod = Literal["rrf", "convex"]


class HybridRetriever:
    """
    Orchestrates Dense Vector Search + Sparse BM25 Search with advanced rank fusion.
    """

    def __init__(
        self,
        collection: Collection,
        bm25_index: BM25Index,
        embedder: Embedder,
        default_rrf_k: int = 60,
        default_alpha: float = 0.65,  # Weight for vector vs. BM25 (0.65 dense, 0.35 sparse)
    ):
        self.collection = collection
        self.bm25 = bm25_index
        self.embedder = embedder
        self.rrf_k = default_rrf_k
        self.alpha = default_alpha

    def search(
        self,
        query: str,
        k: int = 10,
        candidate_multiplier: int = 2,
        fusion_method: FusionMethod = "rrf",
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Search:
        1. Query dense vector collection for top candidate_multiplier * k
        2. Query BM25 index for top candidate_multiplier * k
        3. Merge candidate sets via RRF or Convex weighting
        4. Return top-k merged results with detailed rank provenance
        """
        pool_size = max(k * candidate_multiplier, 15)
        use_alpha = alpha if alpha is not None else self.alpha
        use_rrf_k = rrf_k if rrf_k is not None else self.rrf_k

        # 1. Dense Search
        query_vec = self.embedder.embed_query(query)
        dense_results = self.collection.query(query_vec, k=pool_size, filter=filter)

        # 2. BM25 Search
        bm25_results = self.bm25.search(query, k=pool_size)

        # Build candidate pools
        dense_rank_map = {res["id"]: (rank + 1, res) for rank, res in enumerate(dense_results)}
        bm25_rank_map = {res["id"]: (rank + 1, res) for rank, res in enumerate(bm25_results)}

        all_candidate_ids = set(dense_rank_map.keys()).union(set(bm25_rank_map.keys()))

        merged_scores: Dict[str, Dict[str, Any]] = {}

        for doc_id in all_candidate_ids:
            dense_rank, dense_item = dense_rank_map.get(doc_id, (None, None))
            bm25_rank, bm25_item = bm25_rank_map.get(doc_id, (None, None))

            # Retrieve text and metadata from available item
            item_meta = (dense_item["metadata"] if dense_item else bm25_item.get("metadata", {}))
            item_text = (
                item_meta.get("text")
                or (bm25_item["text"] if bm25_item else "")
                or item_meta.get("content", "")
            )

            dense_score = dense_item["score"] if dense_item else 0.0
            bm25_score = bm25_item["score"] if bm25_item else 0.0

            if fusion_method == "rrf":
                # Reciprocal Rank Fusion formula
                rrf_dense = 1.0 / (use_rrf_k + dense_rank) if dense_rank else 0.0
                rrf_bm25 = 1.0 / (use_rrf_k + bm25_rank) if bm25_rank else 0.0
                combined_score = (use_alpha * rrf_dense) + ((1.0 - use_alpha) * rrf_bm25)
            else:
                # Convex Combination
                combined_score = (use_alpha * dense_score) + ((1.0 - use_alpha) * bm25_score)

            merged_scores[doc_id] = {
                "id": doc_id,
                "score": float(combined_score),
                "text": item_text,
                "metadata": item_meta,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_score": float(dense_score),
                "bm25_score": float(bm25_score),
                "fusion_method": fusion_method,
            }

        # Sort combined results descending by score
        sorted_results = sorted(merged_scores.values(), key=lambda x: x["score"], reverse=True)[:k]

        return sorted_results
