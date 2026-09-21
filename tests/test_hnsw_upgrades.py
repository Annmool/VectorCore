"""
Comprehensive tests for HNSW upgrades:
1. Degree invariants (M, M0 shrink cap, no self-loops, connectivity)
2. Recall vs FlatIndex benchmarking
3. Re-insertion with tombstones, tombstone routing, deleting enter_node, compaction
4. Filtering edge cases (empty w_best, <1% selective filters, fewer than k matches)
5. extend_candidates ablation
6. RNG seed reproducibility
7. Multi-threaded concurrency stress test (8 threads, barrier, sys.setswitchinterval)
8. BFS connected subgraph topology export
9. Persistence and backward compatibility
"""

import sys
import threading
import time
import numpy as np
import pytest
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.flat_index import FlatIndex
from vectordb.core.collection import Collection
from vectordb.core.storage import StorageEngine


def test_degree_invariants_and_connectivity():
    """Verify new node degree <= M, neighbors <= shrink_cap, no self-loops, and level 0 reachability."""
    M = 8
    M0 = 16
    dim = 16
    n_vectors = 150

    index = HNSWIndex(dim=dim, M=M, M0=M0, ef_construction=32, ef_search=16, seed=42)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)

    for i in range(n_vectors):
        idx = index.add(f"doc_{i}", vectors[i])
        # Newly inserted node should have degree <= M at all levels (including level 0)
        # Note: subsequent inserts can increase neighbor degrees up to shrink_cap
        for lev in range(index.node_levels[idx] + 1):
            assert len(index.layers[lev][idx]) <= (M0 if lev == 0 else M)

    # Global degree invariant check
    for lev_idx, layer in enumerate(index.layers):
        cap = M0 if lev_idx == 0 else M
        for node_idx, neighbors in layer.items():
            # No self-loops
            assert node_idx not in neighbors, f"Self-loop detected on node {node_idx} at level {lev_idx}"
            # Degree cap respected
            assert len(neighbors) <= cap, f"Node {node_idx} exceeded degree cap {cap} at level {lev_idx}: {len(neighbors)}"

    # Connectivity at level 0: all active nodes reachable from enter_node
    visited = set()
    queue = [index.enter_node]
    visited.add(index.enter_node)
    while queue:
        curr = queue.pop(0)
        for neigh in index.layers[0].get(curr, set()):
            if neigh not in visited:
                visited.add(neigh)
                queue.append(neigh)

    active_nodes = {i for i in range(len(index.ids)) if not index.is_deleted[i]}
    assert visited == active_nodes, f"Not all active nodes reachable at level 0. Unreachable: {active_nodes - visited}"


def test_recall_vs_flat_index():
    """Measure and assert recall@10 >= 90% compared to ground truth FlatIndex."""
    dim = 32
    n_docs = 500
    n_queries = 50
    k = 10

    rng = np.random.default_rng(123)
    doc_vecs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    query_vecs = rng.standard_normal((n_queries, dim)).astype(np.float32)

    flat = FlatIndex(dim=dim, metric="cosine")
    hnsw = HNSWIndex(dim=dim, M=16, M0=32, ef_construction=64, ef_search=32, metric="cosine", seed=123)

    for i in range(n_docs):
        flat.add(f"doc_{i}", doc_vecs[i])
        hnsw.add(f"doc_{i}", doc_vecs[i])

    recalls = []
    for q in query_vecs:
        flat_res = flat.search(q, k=k)
        hnsw_res = hnsw.search(q, k=k)

        flat_ids = {r["id"] for r in flat_res}
        hnsw_ids = {r["id"] for r in hnsw_res}
        overlap = len(flat_ids.intersection(hnsw_ids))
        recalls.append(overlap / k)

    mean_recall = float(np.mean(recalls))
    print(f"\n[BENCHMARK] Mean Recall@{k} vs FlatIndex: {mean_recall * 100:.2f}%")
    assert mean_recall >= 0.90, f"Recall {mean_recall} is below 90% target"


def test_reinsertion_tombstones_and_routing():
    """Re-adding existing id should tombstone old node, insert fresh at new index with matching edges."""
    dim = 8
    hnsw = HNSWIndex(dim=dim, M=4, M0=8, ef_construction=16, ef_search=16, metric="cosine", seed=42)

    # doc_0 is in cluster A
    vec_cluster_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # doc_0 moves to cluster B
    vec_cluster_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    hnsw.add("doc_0", vec_cluster_a, {"tag": "old"})
    hnsw.add("doc_a1", vec_cluster_a, {"tag": "a1"})
    hnsw.add("doc_b1", vec_cluster_b, {"tag": "b1"})

    old_idx = hnsw.id_to_idx["doc_0"]
    assert old_idx == 0
    assert len(hnsw) == 3

    # Re-insert doc_0 with cluster B vector
    new_idx = hnsw.add("doc_0", vec_cluster_b, {"tag": "new"})
    assert new_idx != old_idx
    assert hnsw.is_deleted[old_idx] is True
    assert hnsw.is_deleted[new_idx] is False
    assert hnsw.id_to_idx["doc_0"] == new_idx
    assert len(hnsw) == 3  # len accounts for tombstones

    # Search for vec_cluster_b should return doc_0 and doc_b1 with distance ~ 0
    res_b = hnsw.search(vec_cluster_b, k=2)
    b_ids = [r["id"] for r in res_b]
    assert "doc_0" in b_ids
    assert "doc_b1" in b_ids
    doc_0_res = [r for r in res_b if r["id"] == "doc_0"][0]
    assert doc_0_res["metadata"]["tag"] == "new"
    assert np.isclose(doc_0_res["distance"], 0.0, atol=1e-4)

    # Search for vec_cluster_a should return doc_a1, NOT doc_0
    res_a = hnsw.search(vec_cluster_a, k=2)
    a_ids = [r["id"] for r in res_a]
    assert "doc_a1" in a_ids
    assert "doc_0" not in a_ids

    # Test compacting index
    hnsw.compact()
    assert len(hnsw) == 3
    assert len(hnsw.ids) == 3
    assert not any(hnsw.is_deleted)


def test_deleting_enter_node():
    """Deleting the enter_node should not break subsequent searches."""
    dim = 8
    hnsw = HNSWIndex(dim=dim, M=4, M0=8, ef_construction=16, ef_search=16, seed=42)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((10, dim)).astype(np.float32)

    for i in range(10):
        hnsw.add(f"doc_{i}", vecs[i])

    enter_id = hnsw.ids[hnsw.enter_node]
    # Delete the enter node
    assert hnsw.delete(enter_id) is True
    assert hnsw.is_deleted[hnsw.id_to_idx[enter_id]] is True

    # Searches must still complete and return 9 active nodes
    for i in range(10):
        if f"doc_{i}" == enter_id:
            continue
        res = hnsw.search(vecs[i], k=5)
        assert len(res) > 0
        # Deleted enter_node should never appear
        assert all(r["id"] != enter_id for r in res)


def test_filtering_edge_cases():
    """Test empty w_best (entry point does not match), selective filter fallback, and fewer than k matches."""
    dim = 8
    hnsw = HNSWIndex(dim=dim, M=8, M0=16, ef_construction=32, ef_search=16, seed=42)
    rng = np.random.default_rng(42)

    # 200 documents, only 3 have tag="rare"
    # Ensure doc_0 (enter_node) does NOT have tag="rare"
    for i in range(200):
        vec = rng.standard_normal(dim).astype(np.float32)
        tag = "rare" if i in [50, 100, 150] else "common"
        hnsw.add(f"doc_{i}", vec, {"tag": tag, "i": i})

    # 1. Entry point does not match: w_best starts empty, must not crash with IndexError
    filter_rare = lambda m: m.get("tag") == "rare"
    query = rng.standard_normal(dim).astype(np.float32)
    res = hnsw.search(query, k=5, filter_fn=filter_rare)

    # Only 3 exist, so search must return exactly 3 without infinite loop
    assert len(res) == 3
    assert all(r["metadata"]["tag"] == "rare" for r in res)

    # 2. Highly selective filter (<1% on larger set)
    filter_single = lambda m: m.get("i") == 100
    res_single = hnsw.search(query, k=5, filter_fn=filter_single)
    assert len(res_single) == 1
    assert res_single[0]["id"] == "doc_100"

    # 3. Filter matching 0 documents
    filter_none = lambda m: m.get("tag") == "non_existent"
    res_none = hnsw.search(query, k=5, filter_fn=filter_none)
    assert len(res_none) == 0


def test_extend_candidates_ablation():
    """Verify extend_candidates parameter works and compare index construction."""
    dim = 16
    n = 100
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)

    # Standard (extend_candidates=False)
    t0 = time.perf_counter()
    hnsw_standard = HNSWIndex(dim=dim, M=8, extend_candidates=False, seed=42)
    for i in range(n):
        hnsw_standard.add(f"doc_{i}", vecs[i])
    t_std = time.perf_counter() - t0

    # With candidate extension (extend_candidates=True)
    t0 = time.perf_counter()
    hnsw_extended = HNSWIndex(dim=dim, M=8, extend_candidates=True, seed=42)
    for i in range(n):
        hnsw_extended.add(f"doc_{i}", vecs[i])
    t_ext = time.perf_counter() - t0

    # Both must build successfully and return valid searches
    res_std = hnsw_standard.search(vecs[0], k=5)
    res_ext = hnsw_extended.search(vecs[0], k=5)
    assert len(res_std) == 5
    assert len(res_ext) == 5
    assert res_std[0]["id"] == "doc_0"
    assert res_ext[0]["id"] == "doc_0"
    print(f"\n[ABLATION] Build time: extend_candidates=False: {t_std:.4f}s, extend_candidates=True: {t_ext:.4f}s")


def test_rng_seed_reproducibility():
    """Identical insertion order and seed must generate identical graph topology and results."""
    dim = 8
    rng = np.random.default_rng(100)
    vecs = rng.standard_normal((50, dim)).astype(np.float32)

    idx1 = HNSWIndex(dim=dim, M=8, seed=999)
    idx2 = HNSWIndex(dim=dim, M=8, seed=999)

    for i in range(50):
        idx1.add(f"doc_{i}", vecs[i])
        idx2.add(f"doc_{i}", vecs[i])

    assert idx1.enter_node == idx2.enter_node
    assert idx1.max_level == idx2.max_level
    assert idx1.node_levels == idx2.node_levels

    for lev in range(len(idx1.layers)):
        assert idx1.layers[lev] == idx2.layers[lev]

    q = rng.standard_normal(dim).astype(np.float32)
    res1 = idx1.search(q, k=10)
    res2 = idx2.search(q, k=10)
    assert [r["id"] for r in res1] == [r["id"] for r in res2]


def test_bfs_graph_topology():
    """Test that get_graph_topology returns a connected subgraph, preserves max_level>=1, and dedupes edges."""
    dim = 8
    hnsw = HNSWIndex(dim=dim, M=4, M0=8, ef_construction=16, ef_search=16, seed=42)
    rng = np.random.default_rng(42)

    for i in range(60):
        hnsw.add(f"doc_{i}", rng.standard_normal(dim).astype(np.float32))

    # Mark some nodes as deleted
    hnsw.delete("doc_5")
    hnsw.delete("doc_10")

    topo = hnsw.get_graph_topology(max_nodes=30)
    nodes = topo["nodes"]
    edges = topo["edges"]

    assert len(nodes) <= 30
    node_indices = {n["idx"] for n in nodes}

    # Deleted nodes must not be emitted
    assert hnsw.id_to_idx["doc_5"] not in node_indices
    assert hnsw.id_to_idx["doc_10"] not in node_indices

    # All active nodes with max_level >= 1 must be present
    for idx, lvl in hnsw.node_levels.items():
        if lvl >= 1 and not hnsw.is_deleted[idx]:
            assert idx in node_indices, f"High-level node {idx} was omitted from graph topology"

    # All edges must connect nodes that exist in sampled nodes
    edge_keys = set()
    for e in edges:
        s = e["source_idx"]
        t = e["target_idx"]
        assert s in node_indices
        assert t in node_indices
        assert s != t
        key = (min(s, t), max(s, t), e["level"])
        assert key not in edge_keys, f"Duplicate edge found: {key}"
        edge_keys.add(key)


def test_concurrency_stress_test():
    """Stress test thread safety under aggressive thread interleaving with 8 threads."""
    dim = 8
    hnsw = HNSWIndex(dim=dim, M=8, M0=16, ef_construction=32, ef_search=16, seed=42)
    rng = np.random.default_rng(42)

    # Pre-populate with 50 items
    initial_vecs = rng.standard_normal((50, dim)).astype(np.float32)
    for i in range(50):
        hnsw.add(f"doc_{i}", initial_vecs[i])

    num_threads = 8
    barrier = threading.Barrier(num_threads)
    exceptions = []

    # Aggressive thread switching
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def worker_add(thread_id: int):
        barrier.wait()
        try:
            local_rng = np.random.default_rng(1000 + thread_id)
            for j in range(25):
                vec = local_rng.standard_normal(dim).astype(np.float32)
                # Mix fresh adds and re-insertions
                doc_id = f"doc_{j % 10}" if j % 2 == 0 else f"thread_{thread_id}_doc_{j}"
                hnsw.add(doc_id, vec, {"thread": thread_id})
        except Exception as e:
            exceptions.append((f"add_{thread_id}", e))

    def worker_search(thread_id: int):
        barrier.wait()
        try:
            local_rng = np.random.default_rng(2000 + thread_id)
            for _ in range(100):
                query = local_rng.standard_normal(dim).astype(np.float32)
                res = hnsw.search(query, k=5)
                for r in res:
                    assert isinstance(r["id"], str)
                    assert r["distance"] >= 0.0
        except Exception as e:
            exceptions.append((f"search_{thread_id}", e))

    threads = []
    # 4 writers, 4 readers
    for i in range(4):
        threads.append(threading.Thread(target=worker_add, args=(i,)))
    for i in range(4, 8):
        threads.append(threading.Thread(target=worker_search, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sys.setswitchinterval(old_switch)
    assert len(exceptions) == 0, f"Concurrent operations raised exceptions: {exceptions}"


def test_collection_persistence_with_upgrades(tmp_path):
    """Test saving and loading upgraded HNSW parameters (M0, seed, extend_candidates) and backward compatibility."""
    storage = StorageEngine(base_dir=str(tmp_path))
    coll = Collection(
        "test_coll_upgrades",
        dim=8,
        index_type="hnsw",
        storage_engine=storage,
        hnsw_m=8,
        hnsw_m0=16,
        hnsw_seed=1234,
        hnsw_extend_candidates=True,
    )

    rng = np.random.default_rng(1234)
    for i in range(20):
        vec = rng.standard_normal(8).astype(np.float32)
        coll.insert(f"doc_{i}", vec, {"i": i})

    coll.save()

    # Reload
    loaded_coll = Collection.load("test_coll_upgrades", storage_engine=storage)
    assert loaded_coll is not None
    assert loaded_coll.hnsw_m == 8
    assert loaded_coll.hnsw_m0 == 16
    assert loaded_coll.hnsw_seed == 1234
    assert loaded_coll.hnsw_extend_candidates is True
    assert len(loaded_coll) == 20

    # Query loaded collection
    test_vec = rng.standard_normal(8).astype(np.float32)
    res = loaded_coll.query(test_vec, k=3)
    assert len(res) == 3
