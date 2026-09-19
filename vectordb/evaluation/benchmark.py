"""
Benchmarking engine comparing Custom NumPy Indexes (Flat, IVF, HNSW) and FAISS on Speed vs. Recall.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from vectordb.core.flat_index import FlatIndex
from vectordb.core.ivf_index import IVFIndex
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.evaluation.metrics import recall_at_k


class VectorIndexBenchmark:
    """
    Runs systematic speed vs. recall benchmarks across indexing algorithms.
    """

    def __init__(self, dim: int = 384, num_vectors: int = 2000, num_queries: int = 100):
        self.dim = dim
        self.num_vectors = num_vectors
        self.num_queries = num_queries

        # Generate synthetic test vectors and normalize
        np.random.seed(42)
        raw_db = np.random.randn(num_vectors, dim).astype(np.float32)
        norms_db = np.linalg.norm(raw_db, axis=1, keepdims=True)
        self.db_vectors = raw_db / np.where(norms_db == 0, 1e-10, norms_db)

        raw_q = np.random.randn(num_queries, dim).astype(np.float32)
        norms_q = np.linalg.norm(raw_q, axis=1, keepdims=True)
        self.query_vectors = raw_q / np.where(norms_q == 0, 1e-10, norms_q)

        self.ids = [f"vec_{i}" for i in range(num_vectors)]

    def run_benchmark(self, k: int = 10) -> Dict[str, Any]:
        """
        Execute full benchmark suite across all index types.
        """
        results: Dict[str, Any] = {}

        # 1. Ground Truth Baseline: Custom Flat Index
        flat_idx = FlatIndex(dim=self.dim, metric="cosine")
        flat_idx.add_batch(self.ids, self.db_vectors)

        t0 = time.perf_counter()
        exact_results = []
        for q in self.query_vectors:
            res = flat_idx.search(q, k=k)
            exact_results.append([r["id"] for r in res])
        flat_time = time.perf_counter() - t0
        flat_latency_ms = (flat_time / self.num_queries) * 1000
        flat_qps = self.num_queries / max(flat_time, 1e-6)

        results["custom_flat"] = {
            "name": "Custom Flat (Exact)",
            "type": "exact",
            "latency_ms": round(flat_latency_ms, 3),
            "qps": round(flat_qps, 1),
            "recall@1": 1.0,
            f"recall@{k}": 1.0,
            "speedup": 1.0,
        }

        # 2. Custom IVF Index with varying nprobe
        ivf_nlist = max(int(np.sqrt(self.num_vectors)), 8)
        ivf_idx = IVFIndex(dim=self.dim, nlist=ivf_nlist, nprobe=4, metric="cosine")
        ivf_idx.add_batch(self.ids, self.db_vectors)

        for nprobe in [1, 2, 4, 8]:
            t0 = time.perf_counter()
            ivf_res_list = []
            for q in self.query_vectors:
                res = ivf_idx.search(q, k=k, nprobe=nprobe)
                ivf_res_list.append([r["id"] for r in res])
            ivf_time = time.perf_counter() - t0
            ivf_latency = (ivf_time / self.num_queries) * 1000
            ivf_qps = self.num_queries / max(ivf_time, 1e-6)

            recalls_k = [
                recall_at_k(ret, set(gt), k)
                for ret, gt in zip(ivf_res_list, exact_results)
            ]
            recalls_1 = [
                recall_at_k(ret, set(gt[:1]), 1)
                for ret, gt in zip(ivf_res_list, exact_results)
            ]

            results[f"custom_ivf_nprobe_{nprobe}"] = {
                "name": f"Custom IVF (nprobe={nprobe})",
                "type": "approximate",
                "latency_ms": round(ivf_latency, 3),
                "qps": round(ivf_qps, 1),
                "recall@1": round(float(np.mean(recalls_1)), 4),
                f"recall@{k}": round(float(np.mean(recalls_k)), 4),
                "speedup": round(flat_latency_ms / max(ivf_latency, 1e-6), 2),
            }

        # 3. Custom HNSW Index with varying efSearch
        hnsw_idx = HNSWIndex(
            dim=self.dim,
            M=16,
            ef_construction=64,
            ef_search=32,
            metric="cosine",
        )
        hnsw_idx.add_batch(self.ids, self.db_vectors)

        for ef in [16, 32, 64]:
            t0 = time.perf_counter()
            hnsw_res_list = []
            for q in self.query_vectors:
                res = hnsw_idx.search(q, k=k, ef_search=ef)
                hnsw_res_list.append([r["id"] for r in res])
            hnsw_time = time.perf_counter() - t0
            hnsw_latency = (hnsw_time / self.num_queries) * 1000
            hnsw_qps = self.num_queries / max(hnsw_time, 1e-6)

            recalls_k = [
                recall_at_k(ret, set(gt), k)
                for ret, gt in zip(hnsw_res_list, exact_results)
            ]
            recalls_1 = [
                recall_at_k(ret, set(gt[:1]), 1)
                for ret, gt in zip(hnsw_res_list, exact_results)
            ]

            results[f"custom_hnsw_ef_{ef}"] = {
                "name": f"Custom HNSW (ef={ef})",
                "type": "approximate",
                "latency_ms": round(hnsw_latency, 3),
                "qps": round(hnsw_qps, 1),
                "recall@1": round(float(np.mean(recalls_1)), 4),
                f"recall@{k}": round(float(np.mean(recalls_k)), 4),
                "speedup": round(flat_latency_ms / max(hnsw_latency, 1e-6), 2),
            }

        # 4. Optional: FAISS Benchmarking (if faiss is available)
        try:
            import faiss

            faiss_index = faiss.IndexFlatIP(self.dim)
            faiss_index.add(self.db_vectors)

            t0 = time.perf_counter()
            D, I = faiss_index.search(self.query_vectors, k)
            faiss_time = time.perf_counter() - t0
            faiss_latency = (faiss_time / self.num_queries) * 1000
            faiss_qps = self.num_queries / max(faiss_time, 1e-6)

            faiss_results = []
            for row in I:
                faiss_results.append([f"vec_{idx}" for idx in row if idx >= 0])

            faiss_recalls = [
                recall_at_k(ret, set(gt), k)
                for ret, gt in zip(faiss_results, exact_results)
            ]

            results["faiss_flat"] = {
                "name": "FAISS IndexFlatIP (C++ AVX2)",
                "type": "c_baseline",
                "latency_ms": round(faiss_latency, 3),
                "qps": round(faiss_qps, 1),
                "recall@1": 1.0,
                f"recall@{k}": round(float(np.mean(faiss_recalls)), 4),
                "speedup": round(flat_latency_ms / max(faiss_latency, 1e-6), 2),
            }

            # FAISS HNSW
            faiss_hnsw = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_INNER_PRODUCT)
            faiss_hnsw.hnsw.efSearch = 32
            faiss_hnsw.add(self.db_vectors)

            t0 = time.perf_counter()
            D_h, I_h = faiss_hnsw.search(self.query_vectors, k)
            faiss_h_time = time.perf_counter() - t0
            faiss_h_latency = (faiss_h_time / self.num_queries) * 1000
            faiss_h_qps = self.num_queries / max(faiss_h_time, 1e-6)

            faiss_h_results = [[f"vec_{idx}" for idx in row if idx >= 0] for row in I_h]
            faiss_h_recalls = [
                recall_at_k(ret, set(gt), k)
                for ret, gt in zip(faiss_h_results, exact_results)
            ]

            results["faiss_hnsw"] = {
                "name": "FAISS IndexHNSWFlat (C++)",
                "type": "c_approximate",
                "latency_ms": round(faiss_h_latency, 3),
                "qps": round(faiss_h_qps, 1),
                "recall@1": round(float(np.mean([recall_at_k(ret, set(gt[:1]), 1) for ret, gt in zip(faiss_h_results, exact_results)])), 4),
                f"recall@{k}": round(float(np.mean(faiss_h_recalls)), 4),
                "speedup": round(flat_latency_ms / max(faiss_h_latency, 1e-6), 2),
            }
        except Exception as e:
            print(f"[Benchmark] FAISS benchmark skipped: {e}")

        return {
            "dataset_info": {
                "num_vectors": self.num_vectors,
                "num_queries": self.num_queries,
                "dimension": self.dim,
                "k": k,
            },
            "results": results,
        }
