"""
Vector Quantization algorithms implemented from scratch in NumPy:
1. Scalar Quantization (Int8)
2. Product Quantization (PQ) with Asymmetric Distance Computation (ADC)
"""

from typing import Dict, Any, Tuple
import numpy as np


class ScalarQuantizer:
    """
    Int8 Scalar Quantization with symmetric or asymmetric scaling.
    Compresses 32-bit floating point vectors into 8-bit integers (4x compression).
    """

    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        self.scales: np.ndarray | None = None
        self.zero_points: np.ndarray | None = None
        self.dim: int | None = None
        self.is_trained = False

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """
        Calibrate scale and zero-point parameters across dimensions.
        vectors: shape (N, D)
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        self.dim = vectors.shape[1]

        if self.symmetric:
            # Symmetric: range [-max_abs, max_abs] mapped to [-127, 127]
            max_abs = np.max(np.abs(vectors), axis=0)
            max_abs = np.where(max_abs < 1e-8, 1e-8, max_abs)
            self.scales = max_abs / 127.0
            self.zero_points = np.zeros(self.dim, dtype=np.float32)
        else:
            # Asymmetric: range [min_val, max_val] mapped to [0, 255]
            min_val = np.min(vectors, axis=0)
            max_val = np.max(vectors, axis=0)
            diff = max_val - min_val
            diff = np.where(diff < 1e-8, 1e-8, diff)
            self.scales = diff / 255.0
            self.zero_points = min_val

        self.is_trained = True
        return self

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Convert float32 vectors to int8 / uint8 codes.
        """
        if not self.is_trained:
            self.fit(vectors)

        vectors = np.asarray(vectors, dtype=np.float32)
        if self.symmetric:
            quantized = np.round(vectors / self.scales)
            return np.clip(quantized, -127, 127).astype(np.int8)
        else:
            quantized = np.round((vectors - self.zero_points) / self.scales)
            return np.clip(quantized, 0, 255).astype(np.uint8)

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct float32 vectors from quantized codes.
        """
        if not self.is_trained:
            raise ValueError("Quantizer has not been calibrated.")

        codes = np.asarray(codes, dtype=np.float32)
        if self.symmetric:
            return codes * self.scales
        else:
            return (codes * self.scales) + self.zero_points

    def compute_distance_sq(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        Compute squared L2 distance between float32 query and int8 codes using dequantization.
        """
        recon = self.dequantize(codes)
        diff = recon - query
        return np.sum(diff * diff, axis=-1)


class ProductQuantizer:
    """
    Product Quantization (PQ) with Asymmetric Distance Computation (ADC).
    Decomposes D-dimensional space into M orthogonal sub-spaces of dimension d_sub = D / M.
    Each sub-space is quantized into K centroids (typically K=256 for 1 byte/sub-vector).
    A 1536-dim or 384-dim vector with M=48 is compressed down to 48 bytes! (8x-32x compression).
    """

    def __init__(self, num_subvectors: int = 8, num_clusters: int = 256, max_iter: int = 20):
        self.M = num_subvectors
        self.K = num_clusters  # 256 fits in uint8
        self.max_iter = max_iter
        self.codebooks: np.ndarray | None = None  # Shape: (M, K, d_sub)
        self.dim: int | None = None
        self.d_sub: int | None = None
        self.is_trained = False

    def _kmeans(self, data: np.ndarray, k: int, max_iter: int) -> np.ndarray:
        """Lightweight, fast K-Means in NumPy."""
        n_samples = data.shape[0]
        if n_samples <= k:
            # Pad if fewer samples than centroids
            if n_samples == 0:
                return np.zeros((k, data.shape[1]), dtype=np.float32)
            indices = np.random.choice(n_samples, size=k, replace=True)
            return data[indices].copy()

        # Initialize with k-means++-like selection
        centroids = [data[np.random.randint(n_samples)]]
        for _ in range(1, k):
            dist_sq = np.min(np.sum((data[:, np.newaxis, :] - np.array(centroids)) ** 2, axis=-1), axis=-1)
            probs = dist_sq / (np.sum(dist_sq) + 1e-10)
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(data[next_idx])
        centroids = np.array(centroids, dtype=np.float32)

        # Iterations
        for _ in range(max_iter):
            # Compute distance to each centroid: (N, K)
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            x_sq = np.sum(data ** 2, axis=1, keepdims=True)
            c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
            dists = x_sq + c_sq - 2.0 * np.dot(data, centroids.T)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    new_centroids[j] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return centroids

    def fit(self, vectors: np.ndarray) -> "ProductQuantizer":
        """
        Train M sub-space codebooks using K-Means.
        vectors: shape (N, D)
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n_samples, self.dim = vectors.shape

        if self.dim % self.M != 0:
            # Adjust M if not divisible
            candidates = [m for m in range(1, min(self.M + 1, self.dim + 1)) if self.dim % m == 0]
            self.M = max(candidates) if candidates else 1

        self.d_sub = self.dim // self.M
        self.codebooks = np.zeros((self.M, self.K, self.d_sub), dtype=np.float32)

        for m in range(self.M):
            sub_vectors = vectors[:, m * self.d_sub : (m + 1) * self.d_sub]
            self.codebooks[m] = self._kmeans(sub_vectors, self.K, self.max_iter)

        self.is_trained = True
        return self

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantize vectors into PQ codes.
        vectors: (N, D) -> codes: (N, M) of dtype uint8
        """
        if not self.is_trained:
            self.fit(vectors)

        vectors = np.asarray(vectors, dtype=np.float32)
        n_samples = vectors.shape[0]
        codes = np.zeros((n_samples, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub_vecs = vectors[:, m * self.d_sub : (m + 1) * self.d_sub]  # (N, d_sub)
            centroids = self.codebooks[m]  # (K, d_sub)

            # Compute distances to K centroids
            sub_sq = np.sum(sub_vecs ** 2, axis=1, keepdims=True)
            c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
            dists = sub_sq + c_sq - 2.0 * np.dot(sub_vecs, centroids.T)

            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate D-dimensional float32 vectors from PQ codes.
        codes: (N, M) -> reconstructed: (N, D)
        """
        if not self.is_trained:
            raise ValueError("ProductQuantizer not trained.")

        codes = np.asarray(codes, dtype=np.int64)
        n_samples = codes.shape[0]
        reconstructed = np.zeros((n_samples, self.dim), dtype=np.float32)

        for m in range(self.M):
            reconstructed[:, m * self.d_sub : (m + 1) * self.d_sub] = self.codebooks[m][codes[:, m]]

        return reconstructed

    def compute_adc_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distance lookup table for a given query vector.
        Lookup table shape: (M, K) containing squared L2 distances from
        each query sub-vector to all K sub-centroids.
        """
        query = np.asarray(query, dtype=np.float32).flatten()
        table = np.zeros((self.M, self.K), dtype=np.float32)

        for m in range(self.M):
            q_sub = query[m * self.d_sub : (m + 1) * self.d_sub]  # (d_sub,)
            centroids = self.codebooks[m]  # (K, d_sub)
            diff = centroids - q_sub  # (K, d_sub)
            table[m] = np.sum(diff * diff, axis=-1)

        return table

    def compute_distances_with_adc(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        Asymmetric Distance Computation (ADC):
        Compute squared distances between unquantized query and quantized dataset codes
        using table lookups without dequantizing vectors.
        """
        table = self.compute_adc_distance_table(query)  # (M, K)
        n_samples = codes.shape[0]
        distances = np.zeros(n_samples, dtype=np.float32)

        # Sum distances across all sub-spaces
        for m in range(self.M):
            distances += table[m, codes[:, m]]

        return distances

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quantizer parameters."""
        return {
            "M": self.M,
            "K": self.K,
            "dim": self.dim,
            "d_sub": self.d_sub,
            "is_trained": self.is_trained,
            "codebooks": self.codebooks.tolist() if self.codebooks is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductQuantizer":
        """Deserialize quantizer."""
        pq = cls(num_subvectors=data["M"], num_clusters=data["K"])
        pq.dim = data["dim"]
        pq.d_sub = data["d_sub"]
        pq.is_trained = data["is_trained"]
        if data["codebooks"] is not None:
            pq.codebooks = np.array(data["codebooks"], dtype=np.float32)
        return pq
