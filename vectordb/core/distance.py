"""
Vector distance and similarity metrics implemented from scratch with NumPy.
Supports both single-vector-to-matrix and matrix-to-matrix pairwise calculations.
"""

from typing import Literal
import numpy as np

MetricType = Literal["cosine", "l2", "dot", "manhattan"]

EPSILON = 1e-10


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector or 2D matrix of vectors along the last axis to unit L2 norm."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm < EPSILON, EPSILON, norm)
    return v / norm


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between u (1D or 2D) and v (1D or 2D).
    Returns values in range [-1, 1] (or [0, 1] if vectors are non-negative).
    """
    u_norm = normalize_vector(u)
    v_norm = normalize_vector(v)
    if u_norm.ndim == 1 and v_norm.ndim == 1:
        return float(np.dot(u_norm, v_norm))
    elif u_norm.ndim == 1 and v_norm.ndim == 2:
        return np.dot(v_norm, u_norm)
    elif u_norm.ndim == 2 and v_norm.ndim == 1:
        return np.dot(u_norm, v_norm)
    else:
        return np.dot(u_norm, v_norm.T)


def cosine_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance: 1 - cosine_similarity.
    Range [0, 2]. Smaller is closer.
    """
    return 1.0 - cosine_similarity(u, v)


def euclidean_distance(u: np.ndarray, v: np.ndarray, squared: bool = False) -> np.ndarray:
    """
    Compute Euclidean (L2) distance between u (1D or 2D) and v (1D or 2D).
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    if u.ndim == 1 and v.ndim == 1:
        diff = u - v
        sq_dist = np.dot(diff, diff)
        return float(sq_dist if squared else np.sqrt(sq_dist))
    elif u.ndim == 1 and v.ndim == 2:
        # u: (D,), v: (N, D) -> (N,)
        diff = v - u
        sq_dist = np.sum(diff * diff, axis=-1)
        return sq_dist if squared else np.sqrt(sq_dist)
    elif u.ndim == 2 and v.ndim == 1:
        # u: (N, D), v: (D,) -> (N,)
        diff = u - v
        sq_dist = np.sum(diff * diff, axis=-1)
        return sq_dist if squared else np.sqrt(sq_dist)
    else:
        # u: (M, D), v: (N, D) -> (M, N)
        # Using ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u, v>
        u_sq = np.sum(u * u, axis=-1, keepdims=True)  # (M, 1)
        v_sq = np.sum(v * v, axis=-1, keepdims=True).T  # (1, N)
        cross = np.dot(u, v.T)  # (M, N)
        sq_dist = np.maximum(u_sq + v_sq - 2.0 * cross, 0.0)
        return sq_dist if squared else np.sqrt(sq_dist)


def dot_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute inner dot product between u and v.
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim == 1 and v.ndim == 1:
        return float(np.dot(u, v))
    elif u.ndim == 1 and v.ndim == 2:
        return np.dot(v, u)
    elif u.ndim == 2 and v.ndim == 1:
        return np.dot(u, v)
    else:
        return np.dot(u, v.T)


def manhattan_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute Manhattan (L1) distance between u and v.
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim == 1 and v.ndim == 1:
        return float(np.sum(np.abs(u - v)))
    elif (u.ndim == 1 and v.ndim == 2) or (u.ndim == 2 and v.ndim == 1):
        return np.sum(np.abs(u - v), axis=-1)
    else:
        # Broadcasting u (M, 1, D) - v (1, N, D) -> (M, N)
        return np.sum(np.abs(u[:, np.newaxis, :] - v[np.newaxis, :, :]), axis=-1)


def compute_distance(u: np.ndarray, v: np.ndarray, metric: MetricType = "cosine") -> np.ndarray:
    """
    Unified distance function. Lower value always means closer/more similar.
    For dot product, returns negative dot product so smaller is better.
    """
    if metric == "cosine":
        return cosine_distance(u, v)
    elif metric == "l2":
        return euclidean_distance(u, v)
    elif metric == "dot":
        # Return negative dot product so smaller distance = higher similarity
        return -dot_product(u, v)
    elif metric == "manhattan":
        return manhattan_distance(u, v)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def distance_to_score(dist: float | np.ndarray, metric: MetricType = "cosine") -> float | np.ndarray:
    """
    Convert distance to a normalized similarity score in range [0, 1] where 1 is identical.
    """
    if metric == "cosine":
        # dist in [0, 2] -> sim = 1 - dist in [-1, 1] -> normalized to [0, 1]
        sim = 1.0 - dist
        return np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    elif metric == "l2" or metric == "manhattan":
        # Use exponential decay: score = 1 / (1 + dist)
        return 1.0 / (1.0 + dist)
    elif metric == "dot":
        # dist is -dot_product
        dot_val = -dist
        # Sigmoid or linear scaled
        return 1.0 / (1.0 + np.exp(-np.clip(dot_val, -10, 10)))
    return 1.0 / (1.0 + dist)
