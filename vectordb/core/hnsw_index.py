"""
Hierarchical Navigable Small World (HNSW) Vector Index implemented from scratch in NumPy.
Based on the seminal paper by Yu. A. Malkov and D. A. Yashunin (2018):
"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
"""

import math
import heapq
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
    ):
        self.dim = dim
        self.M = M
        self.M0 = M0 if M0 is not None else 2 * M  # Maximum edges for level 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric: MetricType = metric
        self.heuristic_neighbors = heuristic_neighbors
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0

        # Graph storage: layer -> node_idx -> list of neighbor node_idx
        # layers[level][node_idx] = set of neighbor node_idx
        self.layers: List[Dict[int, Set[int]]] = []
        self.enter_node: Optional[int] = None
        self.max_level: int = -1

        # Node properties & data storage
        self.node_levels: Dict[int, int] = {}
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.metadata: List[Dict[str, Any]] = []
        self.is_deleted: List[bool] = []

    def __len__(self) -> int:
        return len(self.ids) - sum(self.is_deleted)

    def _random_level(self) -> int:
        """Sample a level for a new node from exponential decay distribution."""
        unif = np.random.uniform(1e-9, 1.0)
        return int(-math.log(unif) * self.mL)

    def _dist(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute scalar distance between two single vectors."""
        return float(compute_distance(vec_a, vec_b, metric=self.metric))

    def _search_layer(
        self,
        query: np.ndarray,
        enter_points: List[int],
        ef: int,
        level: int,
    ) -> List[Tuple[float, int]]:
        """
        Search for ef nearest neighbors at a specific graph level.
        Returns list of (distance, node_idx) sorted ascending by distance.
        """
        visited: Set[int] = set(enter_points)

        # Candidates min-heap: (dist, node_idx)
        candidates: List[Tuple[float, int]] = []
        
        # Best elements max-heap (to easily pop furthest): (-dist, node_idx)
        w_best: List[Tuple[float, int]] = []

        for ep in enter_points:
            d = self._dist(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w_best, (-d, ep))

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_w_dist = -w_best[0][0]

            if c_dist > furthest_w_dist:
                break

            # Explore neighbors at this level
            neighbors = self.layers[level].get(c_node, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    furthest_w_dist = -w_best[0][0]
                    d_neighbor = self._dist(query, self.vectors[neighbor])

                    if d_neighbor < furthest_w_dist or len(w_best) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor))
                        heapq.heappush(w_best, (-d_neighbor, neighbor))

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
        extend_candidates: bool = True,
        keep_pruned_connections: bool = True,
    ) -> List[int]:
        """
        Heuristic neighbor selection (SELECT-NEIGHBORS-HEURISTIC):
        Selects neighbors that are not only close to query, but also diverse (forming small-world edges).
        """
        if len(candidates) <= max_m:
            return [node for _, node in candidates]

        # Candidates sorted by distance to query
        candidates_sorted = sorted(candidates, key=lambda x: x[0])
        w_selected: List[int] = []
        w_selected_vecs: List[np.ndarray] = []
        discarded: List[Tuple[float, int]] = []

        for d_qe, e_node in candidates_sorted:
            e_vec = self.vectors[e_node]
            # Check if e_node is closer to any already selected neighbor than to query
            is_good = True
            for sel_vec in w_selected_vecs:
                d_cand_sel = self._dist(e_vec, sel_vec)
                if d_cand_sel < d_qe:
                    is_good = False
                    break

            if is_good:
                w_selected.append(e_node)
                w_selected_vecs.append(e_vec)
                if len(w_selected) == max_m:
                    break
            else:
                discarded.append((d_qe, e_node))

        # Optionally keep pruned connections up to max_m
        if keep_pruned_connections and len(w_selected) < max_m:
            for _, disc_node in discarded:
                if disc_node not in w_selected:
                    w_selected.append(disc_node)
                    if len(w_selected) == max_m:
                        break

        return w_selected

    def add(self, id: str, vector: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> int:
        """Insert a vector into the HNSW graph."""
        vector = np.asarray(vector, dtype=np.float32).flatten()

        if id in self.id_to_idx:
            # Re-insertion: soft-tombstone the old node so its edges stay intact for routing,
            # and insert the new node fresh with new edges matching the new vector.
            old_idx = self.id_to_idx[id]
            self.is_deleted[old_idx] = True

        idx = len(self.ids)
        vec_2d = vector.reshape(1, self.dim)
        if len(self.vectors) == 0:
            self.vectors = vec_2d
        else:
            self.vectors = np.vstack([self.vectors, vec_2d])

        self.ids.append(id)
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
        curr_dist = self._dist(vector, self.vectors[curr_obj])

        # 1. Top-down greedy routing to find closest entry point down to node_level + 1
        for lev in range(self.max_level, node_level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[lev].get(curr_obj, set())
                for neighbor in neighbors:
                    d = self._dist(vector, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # 2. From min(max_level, node_level) down to 0: search layer and connect neighbors
        enter_points = [curr_obj]
        for lev in range(min(self.max_level, node_level), -1, -1):
            # Ensure dictionary exists for this level
            if idx not in self.layers[lev]:
                self.layers[lev][idx] = set()

            # Search layer with ef_construction
            candidates = self._search_layer(vector, enter_points, ef=self.ef_construction, level=lev)
            enter_points = [node for _, node in candidates]

            # Select M neighbors for new node at all levels (including level 0)
            if self.heuristic_neighbors:
                neighbors_to_add = self._select_neighbors_heuristic(vector, candidates, self.M)
            else:
                neighbors_to_add = self._select_neighbors_simple(candidates, self.M)

            # Shrink cap for neighbor connections: M0 for level 0, M for higher levels
            shrink_cap = self.M0 if lev == 0 else self.M

            # Establish bi-directional edges
            for neighbor in neighbors_to_add:
                if neighbor == idx:
                    continue
                self.layers[lev][idx].add(neighbor)
                if neighbor not in self.layers[lev]:
                    self.layers[lev][neighbor] = set()
                self.layers[lev][neighbor].add(idx)

                # Shrink neighbor's connections if exceeding shrink_cap
                if len(self.layers[lev][neighbor]) > shrink_cap:
                    neigh_vec = self.vectors[neighbor]
                    neigh_cands = [(self._dist(neigh_vec, self.vectors[n]), n) for n in self.layers[lev][neighbor] if n != neighbor]
                    if self.heuristic_neighbors:
                        shrunk = self._select_neighbors_heuristic(neigh_vec, neigh_cands, shrink_cap)
                    else:
                        shrunk = self._select_neighbors_simple(neigh_cands, shrink_cap)
                    self.layers[lev][neighbor] = set(shrunk)

        if node_level > self.max_level:
            # Update global enter node
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
        vectors = np.asarray(vectors, dtype=np.float32)
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]
        indices = []
        for doc_id, vec, meta in zip(ids, vectors, metadatas):
            indices.append(self.add(doc_id, vec, meta))
        return indices

    def delete(self, id: str) -> bool:
        """Mark node as deleted (soft tombstone to preserve graph connectivity)."""
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
        """
        active_mask = [not d for d in self.is_deleted]
        if all(active_mask):
            return

        active_ids = [self.ids[i] for i, act in enumerate(active_mask) if act]
        active_vecs = self.vectors[active_mask] if len(self.vectors) > 0 else np.empty((0, self.dim), dtype=np.float32)
        active_metas = [self.metadata[i] for i, act in enumerate(active_mask) if act]

        # Reset internal storage
        self.layers = []
        self.enter_node = None
        self.max_level = -1
        self.node_levels = {}
        self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.ids = []
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
        if self.enter_node is None or len(self.ids) == 0:
            return []

        query = np.asarray(query, dtype=np.float32).flatten()
        ef = max(ef_search or self.ef_search, k)

        curr_obj = self.enter_node
        curr_dist = self._dist(query, self.vectors[curr_obj])

        # 1. Greedy top-down traversal from max_level down to 1
        for lev in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[lev].get(curr_obj, set())
                for neighbor in neighbors:
                    d = self._dist(query, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # 2. Bottom layer 0: beam search with ef
        candidates = self._search_layer(query, [curr_obj], ef=ef, level=0)

        # 3. Filter deleted or non-matching metadata
        results = []
        for dist, node_idx in candidates:
            if self.is_deleted[node_idx]:
                continue
            meta = self.metadata[node_idx]
            if filter_fn is not None and not filter_fn(meta):
                continue

            score = float(distance_to_score(dist, metric=self.metric))
            results.append({
                "id": self.ids[node_idx],
                "score": score,
                "distance": dist,
                "metadata": meta,
                "layer_level": self.node_levels.get(node_idx, 0),
                "index_type": "hnsw",
            })
            if len(results) == k:
                break

        return results

    def get_graph_topology(self, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Export graph topology for 2D/3D visual inspection in the dashboard.
        """
        nodes = []
        edges = []
        node_limit = min(len(self.ids), max_nodes)

        for i in range(node_limit):
            if not self.is_deleted[i]:
                nodes.append({
                    "id": self.ids[i],
                    "idx": i,
                    "max_level": self.node_levels.get(i, 0),
                    "meta": self.metadata[i],
                })

        for lev_idx, layer_edges in enumerate(self.layers):
            for src, tgt_set in layer_edges.items():
                if src < node_limit and not self.is_deleted[src]:
                    for tgt in tgt_set:
                        if tgt < node_limit and not self.is_deleted[tgt] and src < tgt:
                            edges.append({
                                "source": self.ids[src],
                                "target": self.ids[tgt],
                                "level": lev_idx,
                            })

        return {
            "num_layers": len(self.layers),
            "max_level": self.max_level,
            "enter_node": self.ids[self.enter_node] if self.enter_node is not None else None,
            "nodes": nodes,
            "edges": edges,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
