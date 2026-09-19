"""
Inverted File (IVF) Vector Index implemented from scratch in NumPy.
Partitions the vector space into Voronoi cells via K-Means clustering.
Searches only the top-nprobe nearest clusters for fast approximate retrieval.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from vectordb.core.distance import compute_distance, MetricType, distance_to_score


class IVFIndex:
    """
    IVF-Flat Index:
    1. Clusters training vectors into `nlist` Voronoi centroids via K-Means.
    2. Assigns each vector to its nearest centroid posting list.
    3. At query time, finds `nprobe` closest centroids and scans only those inverted lists.
    """

    def __init__(
        self,
        dim: int,
        nlist: int = 16,
        nprobe: int = 4,
        metric: MetricType = "cosine",
        max_iter: int = 25,
    ):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = min(nprobe, nlist)
        self.metric: MetricType = metric
        self.max_iter = max_iter

        self.centroids: Optional[np.ndarray] = None  # Shape: (nlist, dim)
        self.is_trained: bool = False

        # Inverted lists: cluster_idx -> list of global internal indices
        self.inverted_lists: Dict[int, List[int]] = {i: [] for i in range(nlist)}

        # Global storage
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.metadata: List[Dict[str, Any]] = []
        self.is_deleted: List[bool] = []
        self.cluster_assignments: List[int] = []

    def __len__(self) -> int:
        return len(self.ids) - sum(self.is_deleted)

    def _train_kmeans(self, data: np.ndarray) -> np.ndarray:
        """K-Means clustering in NumPy."""
        n_samples = data.shape[0]
        k = min(self.nlist, n_samples)
        if k <= 0:
            return np.zeros((self.nlist, self.dim), dtype=np.float32)

        # K-Means++ initialization
        centroids = [data[np.random.randint(n_samples)]]
        for _ in range(1, k):
            dists_sq = np.min(
                np.sum((data[:, np.newaxis, :] - np.array(centroids)) ** 2, axis=-1),
                axis=-1,
            )
            probs = dists_sq / (np.sum(dists_sq) + 1e-10)
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(data[next_idx])

        # Fill up if k < self.nlist
        while len(centroids) < self.nlist:
            centroids.append(data[np.random.randint(n_samples)])

        centroids = np.array(centroids, dtype=np.float32)

        for _ in range(self.max_iter):
            # Compute distance of all points to all centroids
            dists = compute_distance(centroids, data, metric=self.metric)  # (nlist, N)
            labels = np.argmin(dists, axis=0)  # (N,)

            new_centroids = np.zeros_like(centroids)
            for cluster_id in range(self.nlist):
                cluster_mask = (labels == cluster_id)
                if np.any(cluster_mask):
                    new_centroids[cluster_id] = np.mean(data[cluster_mask], axis=0)
                else:
                    new_centroids[cluster_id] = centroids[cluster_id]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return centroids

    def train(self, vectors: np.ndarray) -> "IVFIndex":
        """Train centroids on a set of representative vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) < self.nlist:
            # If not enough data, use what we have and pad
            self.centroids = self._train_kmeans(vectors)
        else:
            self.centroids = self._train_kmeans(vectors)
        self.is_trained = True
        return self

    def _assign_cluster(self, vector: np.ndarray) -> int:
        """Find the nearest centroid for a vector."""
        if not self.is_trained or self.centroids is None:
            return 0
        dists = compute_distance(vector, self.centroids, metric=self.metric)
        return int(np.argmin(dists))

    def add(self, id: str, vector: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> int:
        """Add a single vector to the IVF index."""
        vector = np.asarray(vector, dtype=np.float32).flatten()

        if not self.is_trained:
            # Auto-bootstrap training if empty
            if len(self.vectors) + 1 >= self.nlist:
                training_data = np.vstack([self.vectors, vector.reshape(1, -1)])
                self.train(training_data)
                # Re-assign existing
                self.inverted_lists = {i: [] for i in range(self.nlist)}
                for i in range(len(self.ids)):
                    if not self.is_deleted[i]:
                        c_id = self._assign_cluster(self.vectors[i])
                        self.cluster_assignments[i] = c_id
                        self.inverted_lists[c_id].append(i)

        idx = len(self.ids)
        if id in self.id_to_idx:
            # Overwrite existing
            idx = self.id_to_idx[id]
            old_cluster = self.cluster_assignments[idx]
            if idx in self.inverted_lists[old_cluster]:
                self.inverted_lists[old_cluster].remove(idx)
            self.vectors[idx] = vector
            self.metadata[idx] = meta or {}
            self.is_deleted[idx] = False
            cluster_id = self._assign_cluster(vector) if self.is_trained else 0
            self.cluster_assignments[idx] = cluster_id
            self.inverted_lists[cluster_id].append(idx)
            return idx

        vec_2d = vector.reshape(1, self.dim)
        if len(self.vectors) == 0:
            self.vectors = vec_2d
        else:
            self.vectors = np.vstack([self.vectors, vec_2d])

        cluster_id = self._assign_cluster(vector) if self.is_trained else 0
        self.ids.append(id)
        self.id_to_idx[id] = idx
        self.metadata.append(meta or {})
        self.is_deleted.append(False)
        self.cluster_assignments.append(cluster_id)
        self.inverted_lists[cluster_id].append(idx)
        return idx

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add batch of vectors, training centroids if necessary."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]

        if not self.is_trained:
            all_vecs = vectors if len(self.vectors) == 0 else np.vstack([self.vectors, vectors])
            self.train(all_vecs)

        indices = []
        for doc_id, vec, meta in zip(ids, vectors, metadatas):
            idx = self.add(doc_id, vec, meta)
            indices.append(idx)
        return indices

    def delete(self, id: str) -> bool:
        """Mark item as deleted."""
        if id in self.id_to_idx:
            idx = self.id_to_idx[id]
            if not self.is_deleted[idx]:
                self.is_deleted[idx] = True
                cluster_id = self.cluster_assignments[idx]
                if idx in self.inverted_lists[cluster_id]:
                    self.inverted_lists[cluster_id].remove(idx)
                return True
        return False

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search top-k nearest neighbors across the nearest nprobe clusters.
        """
        if len(self.ids) == 0:
            return []

        query = np.asarray(query, dtype=np.float32).flatten()
        actual_nprobe = nprobe if nprobe is not None else self.nprobe
        actual_nprobe = max(1, min(actual_nprobe, self.nlist))

        if not self.is_trained or self.centroids is None:
            # Fall back to scanning all existing
            candidate_indices = [i for i in range(len(self.ids)) if not self.is_deleted[i]]
        else:
            # 1. Find the nearest nprobe centroids to the query
            centroid_dists = compute_distance(query, self.centroids, metric=self.metric)
            nearest_clusters = np.argsort(centroid_dists)[:actual_nprobe]

            # 2. Gather candidates from the selected inverted lists
            candidate_indices = []
            for c_id in nearest_clusters:
                for idx in self.inverted_lists[c_id]:
                    if not self.is_deleted[idx]:
                        candidate_indices.append(idx)

        # Apply metadata filter if specified
        if filter_fn is not None:
            candidate_indices = [
                idx for idx in candidate_indices if filter_fn(self.metadata[idx])
            ]

        if not candidate_indices:
            return []

        candidate_indices = np.array(candidate_indices, dtype=np.int64)
        candidate_vectors = self.vectors[candidate_indices]
        distances = compute_distance(query, candidate_vectors, metric=self.metric)

        if np.isscalar(distances):
            distances = np.array([distances], dtype=np.float32)

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
                "cluster": self.cluster_assignments[global_idx],
                "index_type": "ivf",
            })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return IVF cluster distribution statistics."""
        cluster_sizes = {c_id: len(lst) for c_id, lst in self.inverted_lists.items()}
        return {
            "total_vectors": len(self),
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "is_trained": self.is_trained,
            "cluster_sizes": cluster_sizes,
            "avg_cluster_size": float(np.mean(list(cluster_sizes.values()))) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
        }
