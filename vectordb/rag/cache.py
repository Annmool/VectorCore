"""
LRU Query and Vector Cache with latency measurement and hit-rate statistics.
"""

import time
import hashlib
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple


class LRUQueryCache:
    """
    Thread-safe-ready LRU Cache for queries and embeddings with hit/miss analytics.
    """

    def __init__(self, capacity: int = 500, ttl_seconds: float = 3600.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.total_latency_saved_ms = 0.0

    def _hash_key(self, key_data: Any) -> str:
        s = str(key_data).strip().lower()
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve cached result if valid and not expired."""
        k = self._hash_key(key)
        if k not in self._cache:
            self.misses += 1
            return None

        val, timestamp = self._cache[k]
        if time.time() - timestamp > self.ttl_seconds:
            # Expired
            del self._cache[k]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(k)
        self.hits += 1
        return val

    def put(self, key: Any, value: Any, estimated_computation_ms: float = 0.0) -> None:
        """Store item in cache with timestamp."""
        k = self._hash_key(key)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = (value, time.time())

        if len(self._cache) > self.capacity:
            # Pop oldest
            self._cache.popitem(last=False)

    def record_hit_savings(self, latency_ms: float) -> None:
        """Accumulate saved time from cache hit."""
        self.total_latency_saved_ms += latency_ms

    def clear(self) -> None:
        """Clear cache and reset stats."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_latency_saved_ms = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and performance statistics."""
        total_requests = self.hits + self.misses
        hit_ratio = (self.hits / total_requests) if total_requests > 0 else 0.0
        return {
            "size": len(self._cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_ratio_percent": round(hit_ratio * 100, 2),
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 2),
        }
