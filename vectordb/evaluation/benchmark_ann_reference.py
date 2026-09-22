"""
ANN Reference Benchmark Suite: Custom HNSW vs FAISS C++ HNSW / hnswlib.
Compares build throughput, query latency, and Recall@1/10 across standard ANN configurations.
Supports GloVe (Angular/Cosine), SIFT (Euclidean), and synthetic multi-cluster distributions.
"""

import os
import time
import argparse
from typing import Dict, Any, List, Optional
import numpy as np

from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.flat_index import FlatIndex
from vectordb.evaluation.metrics import recall_at_k


def load_or_generate_dataset(
    dataset_name: str = "synthetic",
    n_docs: int = 10000,
    n_queries: int = 100,
    dim: int = 32,
    seed: int = 42,
    data_dir: str = "./data/benchmarks",
) -> Dict[str, Any]:
    """Load real ANN benchmark dataset (GloVe/SIFT) or generate clustered Gaussian data."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if dataset_name == "glove-25":
        import urllib.request
        import h5py

        file_path = os.path.join(data_dir, "glove-25-angular.hdf5")
        if not os.path.exists(file_path):
            print(f"Downloading glove-25-angular.hdf5 to {file_path} (~120 MB)...")
            url = "http://ann-benchmarks.com/glove-25-angular.hdf5"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(file_path, "wb") as out:
                chunk_size = 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
            print("Download complete.")

        with h5py.File(file_path, "r") as f:
            train_data = np.array(f["train"][:n_docs], dtype=np.float32)
            test_data = np.array(f["test"][:n_queries], dtype=np.float32)

        return {
            "name": f"GloVe-25 (subset {n_docs:,})",
            "metric": "cosine",
            "dim": 25,
            "train": train_data,
            "queries": test_data,
        }

    # Default synthetic multi-cluster distribution
    print(f"Generating synthetic clustered dataset (n={n_docs:,}, queries={n_queries}, dim={dim})...")
    n_clusters = max(10, n_docs // 100)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    cluster_ids = rng.integers(0, n_clusters, size=n_docs)
    noise = rng.standard_normal((n_docs, dim)).astype(np.float32) * 0.15
    train = centroids[cluster_ids] + noise

    # Queries sampled near clusters
    query_cluster_ids = rng.integers(0, n_clusters, size=n_queries)
    query_noise = rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.15
    queries = centroids[query_cluster_ids] + query_noise

    return {
        "name": f"Synthetic Clustered ({n_docs:,} vectors, dim={dim})",
        "metric": "cosine",
        "dim": dim,
        "train": train,
        "queries": queries,
    }


def run_comparison(
    dataset: Dict[str, Any],
    M: int = 16,
    ef_construction: int = 64,
    ef_search: int = 32,
    k: int = 10,
) -> Dict[str, Any]:
    train = dataset["train"]
    queries = dataset["queries"]
    dim = dataset["dim"]
    metric = dataset["metric"]
    n_docs = len(train)
    n_queries = len(queries)

    print(f"\n{'='*75}")
    print(f"BENCHMARK: {dataset['name']} | M={M}, efC={ef_construction}, efS={ef_search}, k={k}")
    print(f"{'='*75}")

    # 1. Ground Truth (Exact Brute Force)
    print("Computing exact ground truth via FlatIndex...")
    flat = FlatIndex(dim=dim, metric=metric)
    flat.add_batch([f"doc_{i}" for i in range(n_docs)], train)
    ground_truth: List[List[str]] = []
    t0 = time.perf_counter()
    for q in queries:
        res = flat.search(q, k=k)
        ground_truth.append([r["id"] for r in res])
    t_flat = time.perf_counter() - t0
    flat_latency_ms = (t_flat / n_queries) * 1000

    # 2. Custom NumPy HNSW
    print("Benchmarking Custom NumPy HNSW...")
    t0 = time.perf_counter()
    custom_hnsw = HNSWIndex(
        dim=dim,
        M=M,
        M0=2 * M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        metric=metric,
        seed=42,
    )
    ef_search_list = [8, 16, 32, 64, 128]

    # 2. Custom NumPy HNSW Build
    print("Building Custom NumPy HNSW...")
    t0 = time.perf_counter()
    custom_hnsw = HNSWIndex(
        dim=dim,
        M=M,
        M0=2 * M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        metric=metric,
        seed=42,
    )
    for i in range(n_docs):
        custom_hnsw.add(f"doc_{i}", train[i])
    t_custom_build = time.perf_counter() - t0
    custom_throughput = n_docs / t_custom_build

    # Sweep ef_search for Custom HNSW
    custom_curve = []
    for ef in ef_search_list:
        t0 = time.perf_counter()
        custom_results = []
        for q in queries:
            res = custom_hnsw.search(q, k=k, ef_search=ef)
            custom_results.append([r["id"] for r in res])
        t_search = time.perf_counter() - t0
        latency_ms = (t_search / n_queries) * 1000
        qps = n_queries / t_search
        rec_k = float(np.mean([recall_at_k(ret, set(gt), k) for ret, gt in zip(custom_results, ground_truth)]))
        rec_1 = float(np.mean([recall_at_k(ret, set(gt[:1]), 1) for ret, gt in zip(custom_results, ground_truth)]))
        custom_curve.append({
            "ef": ef,
            "latency_ms": latency_ms,
            "qps": qps,
            "recall@1": rec_1,
            f"recall@{k}": rec_k,
        })

    # 3. FAISS C++ Reference (IndexHNSWFlat)
    faiss_curve = []
    t_faiss_build = 0.0
    faiss_throughput = 0.0
    try:
        import faiss

        print("Building FAISS C++ IndexHNSWFlat...")
        if metric == "cosine":
            norms = np.linalg.norm(train, axis=-1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            faiss_train = (train / norms).astype(np.float32)

            q_norms = np.linalg.norm(queries, axis=-1, keepdims=True)
            q_norms = np.where(q_norms < 1e-10, 1.0, q_norms)
            faiss_queries = (queries / q_norms).astype(np.float32)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_train = train.astype(np.float32)
            faiss_queries = queries.astype(np.float32)
            faiss_metric = faiss.METRIC_L2

        t0 = time.perf_counter()
        faiss_hnsw = faiss.IndexHNSWFlat(dim, M, faiss_metric)
        faiss_hnsw.hnsw.efConstruction = ef_construction
        faiss_hnsw.add(faiss_train)
        t_faiss_build = time.perf_counter() - t0
        faiss_throughput = n_docs / t_faiss_build

        # Sweep efSearch for FAISS
        for ef in ef_search_list:
            faiss_hnsw.hnsw.efSearch = ef
            t0 = time.perf_counter()
            _, I = faiss_hnsw.search(faiss_queries, k)
            t_search = time.perf_counter() - t0
            latency_ms = (t_search / n_queries) * 1000
            qps = n_queries / t_search

            faiss_results = [[f"doc_{idx}" for idx in row if idx >= 0] for row in I]
            rec_k = float(np.mean([recall_at_k(ret, set(gt), k) for ret, gt in zip(faiss_results, ground_truth)]))
            rec_1 = float(np.mean([recall_at_k(ret, set(gt[:1]), 1) for ret, gt in zip(faiss_results, ground_truth)]))
            faiss_curve.append({
                "ef": ef,
                "latency_ms": latency_ms,
                "qps": qps,
                "recall@1": rec_1,
                f"recall@{k}": rec_k,
            })
    except Exception as e:
        print(f"FAISS benchmark failed: {e}")

    # Print Build Comparison
    print("\n" + "="*85)
    print("INDEX CONSTRUCTION COMPARISON (M={}, efConstruction={})".format(M, ef_construction))
    print("="*85)
    print(f"{'Implementation':<32} | {'Build Time':<12} | {'Throughput':<16}")
    print("-" * 85)
    print(f"{'Custom NumPy HNSW':<32} | {t_custom_build:<12.2f}s | {custom_throughput:<16.0f} adds/s")
    if faiss_curve:
        print(f"{'FAISS IndexHNSWFlat (C++ AVX2)':<32} | {t_faiss_build:<12.2f}s | {faiss_throughput:<16.0f} adds/s")
    print("="*85)

    # Print Recall vs QPS Trade-off Table
    print("\n" + "="*85)
    print(f"RECALL vs. QPS TRADE-OFF FRONTIER SWEEP (k={k})")
    print("="*85)
    print(f"{'Implementation':<28} | {'efSearch':<9} | {'Recall@10':<10} | {'Latency':<10} | {'QPS':<10}")
    print("-" * 85)
    for pt in custom_curve:
        print(f"{'Custom NumPy HNSW':<28} | {pt['ef']:<9} | {pt[f'recall@{k}']*100:<9.2f}% | {pt['latency_ms']:<7.2f} ms | {pt['qps']:<10.0f}")
    if faiss_curve:
        print("-" * 85)
        for pt in faiss_curve:
            print(f"{'FAISS C++ IndexHNSWFlat':<28} | {pt['ef']:<9} | {pt[f'recall@{k}']*100:<9.2f}% | {pt['latency_ms']:<7.2f} ms | {pt['qps']:<10.0f}")
    print("="*85)

    return {
        "build": {
            "custom": {"time_s": t_custom_build, "throughput": custom_throughput},
            "faiss": {"time_s": t_faiss_build, "throughput": faiss_throughput},
        },
        "custom_curve": custom_curve,
        "faiss_curve": faiss_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="HNSW Benchmark against C++ Reference")
    parser.add_argument("--dataset", choices=["synthetic", "glove-25"], default="synthetic")
    parser.add_argument("--n-docs", type=int, default=10000)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dataset = load_or_generate_dataset(
        dataset_name=args.dataset,
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        dim=args.dim,
    )
    run_comparison(
        dataset,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        k=args.k,
    )


if __name__ == "__main__":
    main()
