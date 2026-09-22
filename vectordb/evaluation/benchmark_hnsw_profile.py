"""
Benchmark and Profiling script for HNSW Index optimizations (Tweak 12).
Evaluates insertion throughput and search latency using cProfile across multiple dataset sizes (1k, 5k, 10k).
Provides detailed profiling breakdown and before/after comparisons.
"""

import time
import cProfile
import pstats
import io
import argparse
from typing import Dict, Any
import numpy as np
from vectordb.core.hnsw_index import HNSWIndex


def benchmark_size(n_docs: int, n_queries: int = 100, dim: int = 32, seed: int = 42) -> Dict[str, Any]:
    print(f"\n{'='*25} BENCHMARK: {n_docs:,} VECTORS (dim={dim}) {'='*25}")
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    # 1. Profile Insertion
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    hnsw = HNSWIndex(dim=dim, M=16, M0=32, ef_construction=64, ef_search=32, metric="cosine", seed=seed)
    for i in range(n_docs):
        hnsw.add(f"doc_{i}", docs[i])
    t_insert = time.perf_counter() - t0
    pr.disable()

    # 2. Profile Search
    pr_search = cProfile.Profile()
    pr_search.enable()
    t0 = time.perf_counter()
    for q in queries:
        hnsw.search(q, k=10)
    t_search = time.perf_counter() - t0
    pr_search.disable()

    insert_throughput = n_docs / t_insert
    search_latency_ms = (t_search / n_queries) * 1000

    print(f"Insertion Time:       {t_insert:.4f}s ({insert_throughput:.1f} adds/sec)")
    print(f"Search Time:          {t_search:.4f}s ({search_latency_ms:.2f} ms/query across {n_queries} queries)")

    # Print top cProfile functions for insertion
    s_insert = io.StringIO()
    ps_insert = pstats.Stats(pr, stream=s_insert).sort_stats("tottime")
    ps_insert.print_stats(10)
    print("\n--- Top 10 Functions by Total Time (Insertion) ---")
    print(s_insert.getvalue())

    # Print top cProfile functions for search
    s_search = io.StringIO()
    ps_search = pstats.Stats(pr_search, stream=s_search).sort_stats("tottime")
    ps_search.print_stats(8)
    print("--- Top 8 Functions by Total Time (Search) ---")
    print(s_search.getvalue())

    return {
        "n_docs": n_docs,
        "n_queries": n_queries,
        "insert_time_s": t_insert,
        "insert_throughput": insert_throughput,
        "search_time_s": t_search,
        "search_latency_ms": search_latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="HNSW Profiling & Benchmarking")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000], help="Dataset sizes to benchmark")
    parser.add_argument("--queries", type=int, default=100, help="Number of search queries to run")
    parser.add_argument("--dim", type=int, default=32, help="Vector dimension")
    args = parser.parse_args()

    print("=== HNSW Index Optimization Performance Benchmark ===")
    print("Metrics: Pre-normalized unit vectors, batched NumPy distances, pairwise matrix, dynamic doubling buffer")

    results = []
    for sz in args.sizes:
        res = benchmark_size(n_docs=sz, n_queries=args.queries, dim=args.dim)
        results.append(res)

    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY & COMPARISON WITH UNOPTIMIZED BASELINE")
    print("="*70)
    print(f"{'Dataset Size':<15} | {'Insert Time':<12} | {'Throughput':<16} | {'Search Latency':<15}")
    print("-" * 70)
    # Baseline reference for 1000 docs
    print(f"{'1,000 (Baseline)':<15} | {'43.35s':<12} | {'23.1 adds/s':<16} | {'8.05 ms/query':<15}")
    for r in results:
        sz_str = f"{r['n_docs']:,} (Optimized)"
        ins_str = f"{r['insert_time_s']:.2f}s"
        th_str = f"{r['insert_throughput']:.1f} adds/s"
        lat_str = f"{r['search_latency_ms']:.2f} ms/query"
        print(f"{sz_str:<15} | {ins_str:<12} | {th_str:<16} | {lat_str:<15}")
    print("="*70)


if __name__ == "__main__":
    main()
