"""
Flat (Brute-Force) Vector Index.
Computes exact nearest neighbors across all indexed vectors.
Supports pre-filtering and post-filtering on metadata.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from vectordb.core.distance import compute_distance, MetricType, distance_to_score


class FlatIndex:
    """
    Exact search index using brute-force matrix operations in NumPy.
    Guarantees 100% recall (ground truth baseline).
    """

    def __init__(self, dim: int, metric: MetricType = "cosine"):
        self.dim = dim
        self.metric: MetricType = metric
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.metadata: List[Dict[str, Any]] = []
        self.is_deleted: List[bool] = []

    def __len__(self) -> int:
        return len(self.ids) - sum(self.is_deleted)

    def add(self, id: str, vector: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> int:
        """Add a single vector with id and metadata."""
        if id in self.id_to_idx:
            idx = self.id_to_idx[id]
            self.vectors[idx] = np.asarray(vector, dtype=np.float32)
            self.metadata[idx] = meta or {}
            self.is_deleted[idx] = False
            return idx

        idx = len(self.ids)
        vec = np.asarray(vector, dtype=np.float32).reshape(1, self.dim)
        if len(self.vectors) == 0:
            self.vectors = vec
        else:
            self.vectors = np.vstack([self.vectors, vec])

        self.ids.append(id)
        self.id_to_idx[id] = idx
        self.metadata.append(meta or {})
        self.is_deleted.append(False)
        return idx

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add a batch of vectors and metadata."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]

        indices = []
        new_vecs = []
        new_ids = []
        new_metas = []

        for i, (doc_id, vec, meta) in enumerate(zip(ids, vectors, metadatas)):
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                self.vectors[idx] = vec
                self.metadata[idx] = meta
                self.is_deleted[idx] = False
                indices.append(idx)
            else:
                idx = len(self.ids) + len(new_ids)
                new_ids.append(doc_id)
                new_vecs.append(vec)
                new_metas.append(meta)
                self.id_to_idx[doc_id] = idx
                indices.append(idx)

        if new_vecs:
            new_vecs_arr = np.array(new_vecs, dtype=np.float32)
            if len(self.vectors) == 0:
                self.vectors = new_vecs_arr
            else:
                self.vectors = np.vstack([self.vectors, new_vecs_arr])
            self.ids.extend(new_ids)
            self.metadata.extend(new_metas)
            self.is_deleted.extend([False] * len(new_ids))

        return indices

    def delete(self, id: str) -> bool:
        """Mark a vector as deleted (tombstone)."""
        if id in self.id_to_idx:
            idx = self.id_to_idx[id]
            if not self.is_deleted[idx]:
                self.is_deleted[idx] = True
                return True
        return False

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k nearest neighbors.
        Returns list of dicts: [{'id': ..., 'score': ..., 'distance': ..., 'metadata': ...}]
        """
        if len(self.ids) == 0:
            return []

        query = np.asarray(query, dtype=np.float32).flatten()
        
        # Valid active indices
        active_mask = ~np.array(self.is_deleted, dtype=bool)
        if not np.any(active_mask):
            return []

        # If filter_fn is provided, apply metadata filtering
        if filter_fn is not None:
            filter_mask = np.array([
                filter_fn(self.metadata[i]) if not self.is_deleted[i] else False
                for i in range(len(self.ids))
            ], dtype=bool)
            candidate_indices = np.where(filter_mask)[0]
        else:
            candidate_indices = np.where(active_mask)[0]

        if len(candidate_indices) == 0:
            return []

        candidate_vectors = self.vectors[candidate_indices]
        distances = compute_distance(query, candidate_vectors, metric=self.metric)
        
        if np.isscalar(distances):
            distances = np.array([distances], dtype=np.float32)

        # Sort candidate distances
        top_k_indices = np.argsort(distances)[:k]

        results = []
        for rank_idx in top_k_indices:
            global_idx = candidate_indices[rank_idx]
            dist = float(distances[rank_idx])
            score = float(distance_to_score(dist, metric=self.metric))
            results.append({
                "id": self.ids[global_idx],
                "score": score,
                "distance": dist,
                "metadata": self.metadata[global_idx],
                "index_type": "flat",
            })

        return results

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Retrieve vector by id."""
        if id in self.id_to_idx:
            idx = self.id_to_idx[id]
            if not self.is_deleted[idx]:
                return self.vectors[idx].copy()
        return None
