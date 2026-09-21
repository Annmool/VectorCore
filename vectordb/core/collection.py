"""
High-level Collection abstraction managing Vector Indexes, Metadata Filtering, and Storage.
"""

import threading
from typing import List, Dict, Any, Optional, Callable, Union, Literal
import numpy as np
import os
from vectordb.core.distance import MetricType
from vectordb.core.flat_index import FlatIndex
from vectordb.core.ivf_index import IVFIndex
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.quantization import ScalarQuantizer, ProductQuantizer
from vectordb.core.storage import StorageEngine

IndexType = Literal["flat", "ivf", "hnsw"]


def compile_metadata_filter(query_filter: Optional[Dict[str, Any]]) -> Optional[Callable[[Dict[str, Any]], bool]]:
    """
    Compile a Mongo-style metadata filter dictionary into a fast evaluation function.
    Supports:
    - Exact equality: `{"author": "Vaswani"}`
    - Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
    - Logical operators: `$and`, `$or`, `$not`
    """
    if not query_filter:
        return None

    def evaluate_predicate(val: Any, cond: Any) -> bool:
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$eq" and val != target:
                    return False
                elif op == "$ne" and val == target:
                    return False
                elif op == "$gt" and not (val > target):
                    return False
                elif op == "$gte" and not (val >= target):
                    return False
                elif op == "$lt" and not (val < target):
                    return False
                elif op == "$lte" and not (val <= target):
                    return False
                elif op == "$in" and val not in target:
                    return False
                elif op == "$nin" and val in target:
                    return False
            return True
        else:
            return val == cond

    def filter_func(meta: Dict[str, Any]) -> bool:
        if not meta:
            return False

        for k, v in query_filter.items():
            if k == "$and":
                if not all(compile_metadata_filter(sub_f)(meta) for sub_f in v):
                    return False
            elif k == "$or":
                if not any(compile_metadata_filter(sub_f)(meta) for sub_f in v):
                    return False
            elif k == "$not":
                if compile_metadata_filter(v)(meta):
                    return False
            else:
                field_val = meta.get(k)
                if field_val is None:
                    return False
                if not evaluate_predicate(field_val, v):
                    return False
        return True

    return filter_func


class Collection:
    """
    Manages vector storage, indexing (Flat, IVF, HNSW), quantization, and query routing.
    Thread-safe across operations via RLock.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        index_type: IndexType = "hnsw",
        metric: MetricType = "cosine",
        storage_engine: Optional[StorageEngine] = None,
        # Index-specific kwargs
        hnsw_m: int = 16,
        hnsw_m0: Optional[int] = None,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 32,
        hnsw_heuristic_neighbors: bool = True,
        hnsw_extend_candidates: bool = False,
        hnsw_seed: Optional[int] = None,
        ivf_nlist: int = 16,
        ivf_nprobe: int = 4,
    ):
        self.name = name
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.storage_engine = storage_engine or StorageEngine()

        self.hnsw_m = hnsw_m
        self.hnsw_m0 = hnsw_m0
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_heuristic_neighbors = hnsw_heuristic_neighbors
        self.hnsw_extend_candidates = hnsw_extend_candidates
        self.hnsw_seed = hnsw_seed
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe

        self._lock = threading.RLock()

        self.index: Union[FlatIndex, IVFIndex, HNSWIndex] = self._create_index(index_type)
        self.scalar_quantizer: Optional[ScalarQuantizer] = None
        self.product_quantizer: Optional[ProductQuantizer] = None

    def _create_index(self, itype: IndexType) -> Union[FlatIndex, IVFIndex, HNSWIndex]:
        if itype == "flat":
            return FlatIndex(dim=self.dim, metric=self.metric)
        elif itype == "ivf":
            return IVFIndex(dim=self.dim, nlist=self.ivf_nlist, nprobe=self.ivf_nprobe, metric=self.metric)
        elif itype == "hnsw":
            return HNSWIndex(
                dim=self.dim,
                M=self.hnsw_m,
                M0=self.hnsw_m0,
                ef_construction=self.hnsw_ef_construction,
                ef_search=self.hnsw_ef_search,
                metric=self.metric,
                heuristic_neighbors=self.hnsw_heuristic_neighbors,
                extend_candidates=self.hnsw_extend_candidates,
                seed=self.hnsw_seed,
            )
        else:
            raise ValueError(f"Unknown index type: {itype}")

    def __len__(self) -> int:
        with self._lock:
            return len(self.index)

    def insert(self, id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert or update a single vector with metadata."""
        with self._lock:
            self.index.add(id, vector, metadata)
            self.storage_engine.append_wal(self.name, "insert", {"id": id, "metadata": metadata})

    def insert_batch(
        self, ids: List[str], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Insert batch of vectors."""
        with self._lock:
            self.index.add_batch(ids, vectors, metadatas)
            self.storage_engine.append_wal(
                self.name, "insert_batch", {"count": len(ids), "ids": ids}
            )

    def delete(self, id: str) -> bool:
        """Delete vector by id."""
        with self._lock:
            success = self.index.delete(id)
            if success:
                self.storage_engine.append_wal(self.name, "delete", {"id": id})
            return success

    def compact(self) -> None:
        """Purge tombstones and rebuild active index."""
        with self._lock:
            if hasattr(self.index, "compact"):
                self.index.compact()

    def query(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query nearest neighbors with optional metadata filtering.
        """
        with self._lock:
            filter_fn = compile_metadata_filter(filter)

            if self.index_type == "flat":
                return self.index.search(vector, k=k, filter_fn=filter_fn)
            elif self.index_type == "ivf":
                return self.index.search(vector, k=k, nprobe=nprobe, filter_fn=filter_fn)
            elif self.index_type == "hnsw":
                return self.index.search(vector, k=k, ef_search=ef_search, filter_fn=filter_fn)
            return []

    def quantize_scalar(self) -> Dict[str, Any]:
        """Apply Int8 Scalar Quantization and calculate compression & reconstruction error."""
        with self._lock:
            if len(self.index.vectors) == 0:
                return {"status": "empty"}
            self.scalar_quantizer = ScalarQuantizer(symmetric=True)
            self.scalar_quantizer.fit(self.index.vectors)
            codes = self.scalar_quantizer.quantize(self.index.vectors)
            recon = self.scalar_quantizer.dequantize(codes)
            mse = float(np.mean((self.index.vectors - recon) ** 2))
            return {
                "status": "quantized",
                "type": "int8_scalar",
                "original_bytes_per_vec": self.dim * 4,
                "quantized_bytes_per_vec": self.dim * 1,
                "compression_ratio": 4.0,
                "reconstruction_mse": mse,
            }

    def quantize_product(self, num_subvectors: int = 8) -> Dict[str, Any]:
        """Apply Product Quantization and calculate compression & reconstruction error."""
        with self._lock:
            if len(self.index.vectors) == 0:
                return {"status": "empty"}
            self.product_quantizer = ProductQuantizer(num_subvectors=num_subvectors, num_clusters=256)
            self.product_quantizer.fit(self.index.vectors)
            codes = self.product_quantizer.quantize(self.index.vectors)
            recon = self.product_quantizer.dequantize(codes)
            mse = float(np.mean((self.index.vectors - recon) ** 2))
            return {
                "status": "quantized",
                "type": "product_quantization",
                "M": self.product_quantizer.M,
                "original_bytes_per_vec": self.dim * 4,
                "quantized_bytes_per_vec": self.product_quantizer.M,
                "compression_ratio": (self.dim * 4) / self.product_quantizer.M,
                "reconstruction_mse": mse,
            }

    def save(self) -> str:
        """Persist collection to disk."""
        with self._lock:
            config = {
                "name": self.name,
                "dim": self.dim,
                "index_type": self.index_type,
                "metric": self.metric,
                "hnsw_m": self.hnsw_m,
                "hnsw_m0": self.hnsw_m0,
                "hnsw_ef_construction": self.hnsw_ef_construction,
                "hnsw_ef_search": self.hnsw_ef_search,
                "hnsw_heuristic_neighbors": self.hnsw_heuristic_neighbors,
                "hnsw_extend_candidates": self.hnsw_extend_candidates,
                "hnsw_seed": self.hnsw_seed,
                "ivf_nlist": self.ivf_nlist,
                "ivf_nprobe": self.ivf_nprobe,
            }
            return self.storage_engine.save_collection(
                name=self.name,
                ids=self.index.ids,
                vectors=self.index.vectors,
                metadatas=self.index.metadata,
                deleted_mask=self.index.is_deleted,
                config=config,
            )

    @classmethod
    def load(cls, name: str, storage_engine: Optional[StorageEngine] = None) -> Optional["Collection"]:
        """Load collection from disk."""
        engine = storage_engine or StorageEngine()
        loaded = engine.load_collection(name)
        if loaded is None:
            return None

        config, ids, vectors, metadatas, deleted_mask = loaded
        coll = cls(
            name=config["name"],
            dim=config["dim"],
            index_type=config.get("index_type", "hnsw"),
            metric=config.get("metric", "cosine"),
            storage_engine=engine,
            hnsw_m=config.get("hnsw_m", 16),
            hnsw_m0=config.get("hnsw_m0", None),
            hnsw_ef_construction=config.get("hnsw_ef_construction", 64),
            hnsw_ef_search=config.get("hnsw_ef_search", 32),
            hnsw_heuristic_neighbors=config.get("hnsw_heuristic_neighbors", True),
            hnsw_extend_candidates=config.get("hnsw_extend_candidates", False),
            hnsw_seed=config.get("hnsw_seed", None),
            ivf_nlist=config.get("ivf_nlist", 16),
            ivf_nprobe=config.get("ivf_nprobe", 4),
        )
        coll.insert_batch(ids, vectors, metadatas)
        # Restore tombstone states
        coll.index.is_deleted = deleted_mask
        return coll

    def get_stats(self) -> Dict[str, Any]:
        """Return collection runtime and memory statistics."""
        with self._lock:
            num_vecs = len(self)
            raw_mem_bytes = num_vecs * self.dim * 4
            return {
                "name": self.name,
                "dim": self.dim,
                "index_type": self.index_type,
                "metric": self.metric,
                "count": num_vecs,
                "raw_memory_kb": round(raw_mem_bytes / 1024, 2),
                "is_quantized": self.scalar_quantizer is not None or self.product_quantizer is not None,
            }
