"""
Hierarchical Navigable Small World (HNSW) Vector Index implemented from scratch in NumPy.
Based on the seminal paper by Yu. A. Malkov and D. A. Yashunin (2018):
"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
"""

import math
import heapq
import threading
from collections import deque
from typing import List, Dict, Any, Tuple, Set, Optional, Callable
import numpy as np
from vectordb.core.distance import compute_distance, MetricType, distance_to_score


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) Graph Index.
    Features:
    - Multi-layer skip graph structure
    - Heuristic neighbor selection for graph connectivity & diversity
    - Configurable M, M0, efConstruction, efSearch
    - Dynamic vector insertion & metadata filtering
    - Pre-normalized unit vectors for fast cosine dot products
    - Vectorized batched distance computation in search and descent
    - Precomputed pairwise distance matrix for heuristic selection
    - Preallocated buffer doubling for O(1) amortized insertion
    - Thread-safe operations via RLock
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        M0: Optional[int] = None,
        ef_construction: int = 64,
        ef_search: int = 32,
        metric: MetricType = "cosine",
        heuristic_neighbors: bool = True,
        extend_candidates: bool = False,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.M = M
        self.M0 = M0 if M0 is not None else 2 * M  # Maximum edges for level 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric: MetricType = metric
        self.heuristic_neighbors = heuristic_neighbors
        self.extend_candidates = extend_candidates
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0

        self._lock = threading.RLock()

        # Graph storage: layer -> node_idx -> set of neighbor node_idx
        self.layers: List[Dict[int, Set[int]]] = []
        self.enter_node: Optional[int] = None
        self.max_level: int = -1

        # Node properties & data storage
        self.node_levels: Dict[int, int] = {}
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.metadata: List[Dict[str, Any]] = []
        self.is_deleted: List[bool] = []

        # Vector buffers: capacity doubling buffer for raw vectors + normalized search vectors
        self._capacity: int = 64
        self._num_vectors: int = 0
        self._vectors: np.ndarray = np.empty((self._capacity, dim), dtype=np.float32)
        if self.metric == "cosine":
            self._norm_vectors: np.ndarray = np.empty((self._capacity, dim), dtype=np.float32)
        else:
            self._norm_vectors = self._vectors

    @property
    def vectors(self) -> np.ndarray:
        """Active raw vectors stored in the index."""
        count = max(len(self.ids), self._num_vectors)
        return self._vectors[:count]

    @vectors.setter
    def vectors(self, new_arr: np.ndarray) -> None:
        """
        Handle reassignment to index.vectors (e.g. during reset, compaction, or deserialization).
        Reallocates internal buffer with power-of-2 capacity and keeps normalized vectors synchronized.
        """
        arr = np.asarray(new_arr, dtype=np.float32)
        n = len(arr)
        self._num_vectors = n
        if n == 0:
            self._capacity = 64
            self._vectors = np.empty((self._capacity, self.dim), dtype=np.float32)
            if self.metric == "cosine":
                self._norm_vectors = np.empty((self._capacity, self.dim), dtype=np.float32)
            else:
                self._norm_vectors = self._vectors
        else:
            self._capacity = max(64, int(2 ** math.ceil(math.log2(max(n, 1)))))
            self._vectors = np.empty((self._capacity, self.dim), dtype=np.float32)
            self._vectors[:n] = arr
            if self.metric == "cosine":
                self._norm_vectors = np.empty((self._capacity, self.dim), dtype=np.float32)
                norms = np.linalg.norm(arr, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                self._norm_vectors[:n] = arr / norms
            else:
                self._norm_vectors = self._vectors

    def __len__(self) -> int:
        with self._lock:
            return len(self.ids) - sum(self.is_deleted)

    def _random_level(self) -> int:
        """Sample a level for a new node from exponential decay distribution."""
        unif = self.rng.uniform(1e-9, 1.0)
        return int(-math.log(unif) * self.mL)

    def _normalize_vec(self, vec: np.ndarray) -> np.ndarray:
        """Normalize a single vector to unit L2 norm if metric is cosine."""
        if self.metric == "cosine":
            norm = float(np.linalg.norm(vec))
            if norm > 1e-10:
                return vec / norm
        return vec

    def _dists_to_nodes(self, query: np.ndarray, node_indices: List[int]) -> np.ndarray:
        """
        Compute distances between a query and a list of node indices in one vectorized NumPy call.
        Query and stored vectors must already be normalized for cosine.
        Centralizes metric branching for 1-to-many calculations.
        """
        if not node_indices:
            return np.empty(0, dtype=np.float32)
        vecs = self._norm_vectors[node_indices]
        if self.metric == "cosine":
            # Unit normalized vectors: cosine distance is 1.0 - dot product
            dots = np.dot(vecs, query)
            return np.maximum(0.0, 1.0 - dots)
        elif self.metric == "l2":
            diff = vecs - query
            return np.sqrt(np.sum(diff * diff, axis=-1))
        elif self.metric == "dot":
            return -np.dot(vecs, query)
        elif self.metric == "manhattan":
            return np.sum(np.abs(vecs - query), axis=-1)
        else:
            return compute_distance(query, vecs, metric=self.metric)

    def _pairwise_distances(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix for candidate vectors (shape: C x dim) in one vectorized call.
        Vectors in matrix must already be normalized for cosine.
        Centralizes metric branching for matrix-to-matrix calculations.
        """
        if len(matrix) == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self.metric == "cosine":
            sim = np.dot(matrix, matrix.T)
            np.fill_diagonal(sim, 1.0)
            return np.maximum(0.0, 1.0 - sim)
        elif self.metric == "l2":
            sq_norms = np.sum(matrix * matrix, axis=1, keepdims=True)
            sq_dists = np.maximum(0.0, sq_norms + sq_norms.T - 2.0 * np.dot(matrix, matrix.T))
            np.fill_diagonal(sq_dists, 0.0)
            return np.sqrt(sq_dists)
        elif self.metric == "dot":
            return -np.dot(matrix, matrix.T)
        elif self.metric == "manhattan":
            dists = np.sum(np.abs(matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]), axis=-1)
            np.fill_diagonal(dists, 0.0)
            return dists
        else:
            return compute_distance(matrix, matrix, metric=self.metric)

    def _dist(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute scalar distance between two vectors."""
        if self.metric == "cosine":
            norm_a = float(np.linalg.norm(vec_a))
            norm_b = float(np.linalg.norm(vec_b))
            if abs(norm_a - 1.0) < 1e-4 and abs(norm_b - 1.0) < 1e-4:
                return float(max(0.0, 1.0 - np.dot(vec_a, vec_b)))
            return float(compute_distance(vec_a, vec_b, metric="cosine"))
        return float(compute_distance(vec_a, vec_b, metric=self.metric))

    def _search_layer(
        self,
        query: np.ndarray,
        enter_points: List[int],
        ef: int,
        level: int,
        accept: Optional[Callable[[int], bool]] = None,
    ) -> List[Tuple[float, int]]:
        """
        Search for ef nearest neighbors at a specific graph level.
        query must be normalized if metric == 'cosine'.
        Computes distances for all unvisited neighbors in one vectorized NumPy call.
        Returns list of (distance, node_idx) sorted ascending by distance.
        """
        visited: Set[int] = set(enter_points)
        candidates: List[Tuple[float, int]] = []
        w_best: List[Tuple[float, int]] = []

        if enter_points:
            ep_dists = self._dists_to_nodes(query, enter_points)
            for d, ep in zip(ep_dists, enter_points):
                d_val = float(d)
                heapq.heappush(candidates, (d_val, ep))
                if accept is None or accept(ep):
                    heapq.heappush(w_best, (-d_val, ep))

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            if len(w_best) >= ef and c_dist > -w_best[0][0]:
                break

            neighbors = self.layers[level].get(c_node, set())
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited:
                continue

            for n in unvisited:
                visited.add(n)

            # Batched vectorized distance computation for all unvisited neighbors
            d_neighbors = self._dists_to_nodes(query, unvisited)
            for neighbor, d_neighbor in zip(unvisited, d_neighbors):
                d_val = float(d_neighbor)
                worst = -w_best[0][0] if len(w_best) >= ef else float("inf")

                if d_val < worst or len(w_best) < ef:
                    heapq.heappush(candidates, (d_val, neighbor))
                    if accept is None or accept(neighbor):
                        heapq.heappush(w_best, (-d_val, neighbor))
                        if len(w_best) > ef:
                            heapq.heappop(w_best)

        # Return sorted ascending by distance
        results = [(-neg_d, node) for neg_d, node in w_best]
        results.sort(key=lambda x: x[0])
        return results

    def _select_neighbors_simple(
        self,
        candidates: List[Tuple[float, int]],
        max_m: int,
    ) -> List[int]:
        """Simple neighbor selection: select max_m closest candidates."""
        candidates.sort(key=lambda x: x[0])
        return [node for _, node in candidates[:max_m]]

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        max_m: int,
        level: Optional[int] = None,
        extend_candidates: bool = False,
        keep_pruned_connections: bool = True,
    ) -> List[int]:
        """
        Heuristic neighbor selection (SELECT-NEIGHBORS-HEURISTIC):
        Computes pairwise distance matrix once via _pairwise_distances and evaluates greedy selection.
        """
        # Extend candidates with neighbors of candidates at `level` if requested (Algorithm 4)
        cands_dict = {node: d for d, node in candidates}
        if extend_candidates and level is not None and level < len(self.layers):
            to_query = []
            for _, c_node in list(candidates):
                for adj in self.layers[level].get(c_node, set()):
                    if adj not in cands_dict and not self.is_deleted[adj]:
                        to_query.append(adj)
                        cands_dict[adj] = 0.0
            if to_query:
                to_query_unique = list(dict.fromkeys(to_query))
                dists = self._dists_to_nodes(query, to_query_unique)
                for adj, d in zip(to_query_unique, dists):
                    cands_dict[adj] = float(d)
            candidates = [(d, node) for node, d in cands_dict.items()]

        if len(candidates) <= max_m:
            candidates.sort(key=lambda x: x[0])
            return [node for _, node in candidates]

        # Candidates sorted by distance to query
        candidates_sorted = sorted(candidates, key=lambda x: x[0])
        cand_nodes = [node for _, node in candidates_sorted]
        d_query = [d for d, _ in candidates_sorted]

        # Compute pairwise distance matrix ONCE for all candidates
        cand_matrix = self._norm_vectors[cand_nodes]
        dist_matrix = self._pairwise_distances(cand_matrix)

        selected_indices: List[int] = []
        discarded_indices: List[int] = []

        for i, d_qe in enumerate(d_query):
            is_good = True
            for sel_idx in selected_indices:
                if dist_matrix[i, sel_idx] < d_qe:
                    is_good = False
                    break

            if is_good:
                selected_indices.append(i)
                if len(selected_indices) == max_m:
                    break
            else:
                discarded_indices.append(i)

        # Optionally keep pruned connections up to max_m
        if keep_pruned_connections and len(selected_indices) < max_m:
            for disc_idx in discarded_indices:
                if disc_idx not in selected_indices:
                    selected_indices.append(disc_idx)
                    if len(selected_indices) == max_m:
                        break

        return [cand_nodes[i] for i in selected_indices]

    def add(self, id: str, vector: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> int:
        """Insert a vector into the HNSW graph."""
        with self._lock:
            raw_vector = np.asarray(vector, dtype=np.float32).flatten()
            norm_vector = self._normalize_vec(raw_vector)

            if id in self.id_to_idx:
                # Re-insertion: soft-tombstone the old node so its edges stay intact for routing
                old_idx = self.id_to_idx[id]
                self.is_deleted[old_idx] = True

            idx = len(self.ids)

            # Preallocated buffer growth with capacity doubling
            if idx >= self._capacity:
                new_capacity = max(64, self._capacity * 2)
                new_vectors = np.empty((new_capacity, self.dim), dtype=np.float32)
                if idx > 0:
                    new_vectors[:idx] = self._vectors[:idx]
                self._vectors = new_vectors

                if self.metric == "cosine":
                    new_norm_vectors = np.empty((new_capacity, self.dim), dtype=np.float32)
                    if idx > 0:
                        new_norm_vectors[:idx] = self._norm_vectors[:idx]
                    self._norm_vectors = new_norm_vectors
                else:
                    self._norm_vectors = self._vectors
                self._capacity = new_capacity

            self._vectors[idx] = raw_vector
            if self.metric == "cosine":
                self._norm_vectors[idx] = norm_vector

            self.ids.append(id)
            self._num_vectors = len(self.ids)
            self.id_to_idx[id] = idx
            self.metadata.append(meta or {})
            self.is_deleted.append(False)

            node_level = self._random_level()
            self.node_levels[idx] = node_level

            # Expand layer graphs if needed
            while len(self.layers) <= max(node_level, self.max_level, 0):
                self.layers.append({})

            if self.enter_node is None:
                # First element in graph
                for lev in range(node_level + 1):
                    self.layers[lev][idx] = set()
                self.enter_node = idx
                self.max_level = node_level
                return idx

            # Multi-layer insertion
            curr_obj = self.enter_node
            curr_dist = float(self._dists_to_nodes(norm_vector, [curr_obj])[0])

            # 1. Top-down greedy routing to find closest entry point down to node_level + 1
            for lev in range(self.max_level, node_level, -1):
                visited: Set[int] = {curr_obj}
                while True:
                    unvisited = [n for n in self.layers[lev].get(curr_obj, set()) if n not in visited]
                    if not unvisited:
                        break
                    visited.update(unvisited)
                    dists = self._dists_to_nodes(norm_vector, unvisited)
                    j = int(np.argmin(dists))
                    if dists[j] < curr_dist:
                        curr_dist = float(dists[j])
                        curr_obj = unvisited[j]
                    else:
                        break

            # 2. From min(max_level, node_level) down to 0: search layer and connect neighbors
            enter_points = [curr_obj]
            for lev in range(min(self.max_level, node_level), -1, -1):
                if idx not in self.layers[lev]:
                    self.layers[lev][idx] = set()

                candidates = self._search_layer(norm_vector, enter_points, ef=self.ef_construction, level=lev, accept=None)
                enter_points = [node for _, node in candidates]

                # Select M neighbors for new node at all levels (including level 0)
                if self.heuristic_neighbors:
                    neighbors_to_add = self._select_neighbors_heuristic(
                        norm_vector, candidates, max_m=self.M, level=lev, extend_candidates=self.extend_candidates
                    )
                else:
                    neighbors_to_add = self._select_neighbors_simple(candidates, max_m=self.M)

                # Shrink cap for neighbor connections: M0 for level 0, M for higher levels
                shrink_cap = self.M0 if lev == 0 else self.M

                # Establish bi-directional edges (excluding self-loops)
                for neighbor in neighbors_to_add:
                    if neighbor == idx:
                        continue
                    self.layers[lev][idx].add(neighbor)
                    if neighbor not in self.layers[lev]:
                        self.layers[lev][neighbor] = set()
                    self.layers[lev][neighbor].add(idx)

                    # Shrink neighbor's connections if exceeding shrink_cap
                    if len(self.layers[lev][neighbor]) > shrink_cap:
                        other_nodes = [n for n in self.layers[lev][neighbor] if n != neighbor]
                        neigh_vec = self._norm_vectors[neighbor]
                        dists = self._dists_to_nodes(neigh_vec, other_nodes)
                        neigh_cands = [(float(d), n) for d, n in zip(dists, other_nodes)]

                        if self.heuristic_neighbors:
                            shrunk = self._select_neighbors_heuristic(
                                neigh_vec, neigh_cands, max_m=shrink_cap, level=lev, extend_candidates=self.extend_candidates
                            )
                        else:
                            shrunk = self._select_neighbors_simple(neigh_cands, max_m=shrink_cap)
                        self.layers[lev][neighbor] = set(shrunk)

            if node_level > self.max_level:
                for lev in range(self.max_level + 1, node_level + 1):
                    self.layers[lev][idx] = set()
                self.max_level = node_level
                self.enter_node = idx

            return idx

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add a batch of vectors to HNSW graph."""
        with self._lock:
            vectors = np.asarray(vectors, dtype=np.float32)
            if metadatas is None:
                metadatas = [{} for _ in range(len(ids))]
            indices = []
            for doc_id, vec, meta in zip(ids, vectors, metadatas):
                indices.append(self.add(doc_id, vec, meta))
            return indices

    def delete(self, id: str) -> bool:
        """Mark node as deleted (soft tombstone to preserve graph connectivity)."""
        with self._lock:
            if id in self.id_to_idx:
                idx = self.id_to_idx[id]
                if not self.is_deleted[idx]:
                    self.is_deleted[idx] = True
                    return True
            return False

    def compact(self) -> None:
        """
        Rebuild the HNSW index from scratch using only active (non-deleted) nodes.
        Cleans up accumulated tombstones from re-additions or deletions.
        Sensibly resets capacity to next power of 2 of active count.
        """
        with self._lock:
            active_mask = [not d for d in self.is_deleted]
            if all(active_mask):
                return

            active_ids = [self.ids[i] for i, act in enumerate(active_mask) if act]
            active_vecs = [self._vectors[i] for i, act in enumerate(active_mask) if act]
            active_metas = [self.metadata[i] for i, act in enumerate(active_mask) if act]

            # Reset internal storage with sensible capacity
            active_count = len(active_ids)
            self._capacity = max(64, int(2 ** math.ceil(math.log2(max(active_count, 1)))))
            self._vectors = np.empty((self._capacity, self.dim), dtype=np.float32)
            if self.metric == "cosine":
                self._norm_vectors = np.empty((self._capacity, self.dim), dtype=np.float32)
            else:
                self._norm_vectors = self._vectors

            self.layers = []
            self.enter_node = None
            self.max_level = -1
            self.node_levels = {}
            self.ids = []
            self._num_vectors = 0
            self.id_to_idx = {}
            self.metadata = []
            self.is_deleted = []

            for doc_id, vec, meta in zip(active_ids, active_vecs, active_metas):
                self.add(doc_id, vec, meta)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k approximate nearest neighbors using HNSW graph traversal.
        """
        with self._lock:
            if self.enter_node is None or len(self.ids) == 0:
                return []

            active_indices = [i for i, deleted in enumerate(self.is_deleted) if not deleted]
            active_count = len(active_indices)
            if active_count == 0:
                return []

            query_raw = np.asarray(query, dtype=np.float32).flatten()
            query_norm = self._normalize_vec(query_raw)

            # If filter is provided, check selectivity
            if filter_fn is not None:
                matching_indices = [i for i in active_indices if filter_fn(self.metadata[i])]
                if len(matching_indices) == 0:
                    return []
                # If filter is very selective (< 1% match or <= 50 matching nodes), fall back to exact scan
                if len(matching_indices) <= 50 or (len(matching_indices) / max(active_count, 1)) < 0.01:
                    dists = self._dists_to_nodes(query_norm, matching_indices)
                    results = [(float(d), node_idx) for d, node_idx in zip(dists, matching_indices)]
                    results.sort(key=lambda x: x[0])
                    return [
                        {
                            "id": self.ids[node_idx],
                            "score": float(distance_to_score(dist, metric=self.metric)),
                            "distance": dist,
                            "metadata": self.metadata[node_idx],
                            "layer_level": self.node_levels.get(node_idx, 0),
                            "index_type": "hnsw",
                        }
                        for dist, node_idx in results[:k]
                    ]

            # Standard HNSW traversal
            curr_obj = self.enter_node
            curr_dist = float(self._dists_to_nodes(query_norm, [curr_obj])[0])

            # 1. Greedy top-down traversal from max_level down to 1
            for lev in range(self.max_level, 0, -1):
                visited: Set[int] = {curr_obj}
                while True:
                    unvisited = [n for n in self.layers[lev].get(curr_obj, set()) if n not in visited]
                    if not unvisited:
                        break
                    visited.update(unvisited)
                    dists = self._dists_to_nodes(query_norm, unvisited)
                    j = int(np.argmin(dists))
                    if dists[j] < curr_dist:
                        curr_dist = float(dists[j])
                        curr_obj = unvisited[j]
                    else:
                        break

            # 2. Bottom layer 0: beam search with iterative ef expansion
            accept_pred = (
                lambda idx: (not self.is_deleted[idx]) and (filter_fn is None or filter_fn(self.metadata[idx]))
            )

            curr_ef = max(ef_search or self.ef_search, k)
            while True:
                candidates = self._search_layer(query_norm, [curr_obj], ef=curr_ef, level=0, accept=accept_pred)
                if len(candidates) >= k or curr_ef >= active_count:
                    break
                next_ef = min(curr_ef * 2, active_count)
                if next_ef == curr_ef:
                    break
                curr_ef = next_ef

            # Format top-k results
            results = []
            for dist, node_idx in candidates[:k]:
                score = float(distance_to_score(dist, metric=self.metric))
                results.append({
                    "id": self.ids[node_idx],
                    "score": score,
                    "distance": dist,
                    "metadata": self.metadata[node_idx],
                    "layer_level": self.node_levels.get(node_idx, 0),
                    "index_type": "hnsw",
                })

            return results

    def get_graph_topology(self, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Export graph topology for 2D/3D visual inspection in the dashboard.
        Samples a connected subgraph via BFS from enter_node, always preserving higher-level nodes.
        """
        with self._lock:
            if self.enter_node is None or len(self.ids) == 0:
                return {
                    "num_layers": len(self.layers),
                    "max_level": self.max_level,
                    "enter_node": None,
                    "nodes": [],
                    "edges": [],
                    "M": self.M,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                }

            # 1. Always include all active nodes with max_level >= 1 so higher layers don't vanish
            sampled_nodes: Set[int] = set()
            for idx, lvl in self.node_levels.items():
                if lvl >= 1 and not self.is_deleted[idx]:
                    sampled_nodes.add(idx)

            # 2. BFS from enter_node (traversing through tombstones to preserve connectivity,
            #    but only emitting non-deleted nodes to sampled_nodes)
            queue: deque = deque([self.enter_node])
            visited_bfs: Set[int] = {self.enter_node}
            if not self.is_deleted[self.enter_node]:
                sampled_nodes.add(self.enter_node)

            while queue and len(sampled_nodes) < max_nodes:
                curr = queue.popleft()
                # Traverse neighbors across all layers from high to low
                for lev in range(len(self.layers) - 1, -1, -1):
                    for neighbor in self.layers[lev].get(curr, set()):
                        if neighbor not in visited_bfs:
                            visited_bfs.add(neighbor)
                            queue.append(neighbor)
                            if not self.is_deleted[neighbor]:
                                sampled_nodes.add(neighbor)
                                if len(sampled_nodes) >= max_nodes:
                                    break
                    if len(sampled_nodes) >= max_nodes:
                        break

            # 3. Build node representations keyed by integer idx to avoid collisions with re-added duplicate IDs
            nodes = [
                {
                    "id": self.ids[i],
                    "idx": i,
                    "max_level": self.node_levels.get(i, 0),
                    "meta": self.metadata[i],
                }
                for i in sampled_nodes
            ]

            # 4. Dedupe undirected edges per layer using (min(src, tgt), max(src, tgt), level)
            edges = []
            seen_edges: Set[Tuple[int, int, int]] = set()

            for lev_idx, layer_edges in enumerate(self.layers):
                for src in sampled_nodes:
                    for tgt in layer_edges.get(src, set()):
                        if tgt in sampled_nodes and src != tgt:
                            edge_key = (min(src, tgt), max(src, tgt), lev_idx)
                            if edge_key not in seen_edges:
                                seen_edges.add(edge_key)
                                edges.append({
                                    "source": self.ids[src],
                                    "target": self.ids[tgt],
                                    "source_idx": src,
                                    "target_idx": tgt,
                                    "level": lev_idx,
                                    "layer": lev_idx,
                                })

            enter_id = self.ids[self.enter_node] if self.enter_node is not None else None

            return {
                "num_layers": len(self.layers),
                "max_level": self.max_level,
                "enter_node": enter_id,
                "enter_node_idx": self.enter_node,
                "nodes": nodes,
                "edges": edges,
                "M": self.M,
                "M0": self.M0,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
            }
