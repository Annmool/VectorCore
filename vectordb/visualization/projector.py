"""
Dimensionality Reduction and 2D/3D Embedding Space Projector.
Implements Principal Component Analysis (PCA) via Singular Value Decomposition (SVD)
from scratch in pure NumPy, with continuous eigenspace coordinate projection and cluster mapping.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


# Curated palette tailored for the Obsidian Matrix dark theme
NEON_PALETTE = [
    "#00f59b",  # Bioluminescent Mint
    "#06b6d4",  # Cyan Blue
    "#818cf8",  # Electric Indigo
    "#a855f7",  # Vivid Purple
    "#fbbf24",  # Champagne Gold
    "#f43f5e",  # Crimson Rose
    "#38bdf8",  # Sky Blue
    "#fb923c",  # Solar Orange
    "#34d399",  # Emerald Green
    "#ec4899",  # Hot Pink
    "#a3e635",  # Lime Accent
    "#e879f9",  # Fuchsia
]


class EmbeddingProjector:
    """
    Projects high-dimensional vector embeddings (e.g. 384d) into a calibrated 2D plane
    for interactive exploration, semantic clustering, and query vector localization.
    """

    def __init__(self):
        self.mean_vector: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None  # shape: (dim, 2)
        self.singular_values: Optional[np.ndarray] = None
        self.explained_variance: List[float] = [0.0, 0.0]
        self.scale: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.fitted: bool = False

    def fit_transform(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: List[Dict[str, Any]],
        target_span: float = 800.0,
    ) -> Dict[str, Any]:
        """
        Fits PCA model on vector matrix and returns scaled 2D point positions,
        cluster color assignments, variance ratios, and data bounds.
        """
        n_vectors = len(vectors)
        if n_vectors == 0:
            return {
                "points": [],
                "clusters": [],
                "explained_variance": [0.0, 0.0],
                "total_points": 0,
            }

        arr = np.asarray(vectors, dtype=np.float32)
        dim = arr.shape[1]

        # For very small datasets, produce clean synthetic spread
        if n_vectors < 2:
            self.mean_vector = arr[0].copy()
            self.components = np.zeros((dim, 2), dtype=np.float32)
            self.components[0, 0] = 1.0
            self.components[1, 1] = 1.0
            self.fitted = True
            return {
                "points": [{
                    "id": ids[0],
                    "x": 0.0,
                    "y": 0.0,
                    "cluster": metadata[0].get("source", "Document"),
                    "color": NEON_PALETTE[0],
                    "title": metadata[0].get("title", ids[0]),
                    "snippet": (metadata[0].get("text", "")[:140] + "...") if metadata[0].get("text") else "",
                    "meta": metadata[0],
                }],
                "clusters": [{"name": metadata[0].get("source", "Document"), "color": NEON_PALETTE[0], "count": 1}],
                "explained_variance": [100.0, 0.0],
                "total_points": 1,
            }

        # 1. Center the vector matrix
        self.mean_vector = np.mean(arr, axis=0)
        centered = arr - self.mean_vector

        # 2. Compute SVD
        try:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
            self.singular_values = s
            # Top 2 principal directions (shape: dim x 2)
            self.components = vt[:2, :].T

            # Variance explained
            total_variance = float(np.sum(s ** 2))
            if total_variance > 1e-9:
                ev1 = float((s[0] ** 2) / total_variance) * 100.0
                ev2 = float((s[1] ** 2) / total_variance) * 100.0 if len(s) > 1 else 0.0
            else:
                ev1, ev2 = 50.0, 50.0
            self.explained_variance = [round(ev1, 1), round(ev2, 1)]
        except Exception:
            # Fallback to random orthogonal projection if SVD encounters non-convergence
            rnd = np.random.default_rng(42)
            rand_proj = rnd.standard_normal((dim, 2)).astype(np.float32)
            q, _ = np.linalg.qr(rand_proj)
            self.components = q
            self.explained_variance = [50.0, 50.0]

        # 3. Project centered points to 2D
        coords_2d = np.dot(centered, self.components)  # (N, 2)

        # 4. Normalize and scale to target canvas space [-target_span/2, target_span/2]
        min_vals = np.min(coords_2d, axis=0)
        max_vals = np.max(coords_2d, axis=0)
        ranges = np.maximum(max_vals - min_vals, 1e-6)

        # Center point
        mid_x = (min_vals[0] + max_vals[0]) / 2.0
        mid_y = (min_vals[1] + max_vals[1]) / 2.0

        max_range = float(np.max(ranges))
        self.scale = (target_span * 0.85) / max(max_range, 1e-6)
        self.offset_x = mid_x
        self.offset_y = mid_y
        self.fitted = True

        scaled_x = (coords_2d[:, 0] - self.offset_x) * self.scale
        scaled_y = (coords_2d[:, 1] - self.offset_y) * self.scale

        # 5. Group by cluster / document source for color tagging
        cluster_map: Dict[str, str] = {}
        cluster_counts: Dict[str, int] = {}
        color_idx = 0

        points = []
        for i in range(n_vectors):
            meta = metadata[i] if i < len(metadata) else {}
            # Grouping label: source name or paper title
            cluster_name = (
                meta.get("source")
                or meta.get("paper_title")
                or meta.get("title")
                or "General Knowledge"
            )
            # Shorten if too long
            if len(cluster_name) > 28:
                cluster_name = cluster_name[:25] + "..."

            if cluster_name not in cluster_map:
                cluster_map[cluster_name] = NEON_PALETTE[color_idx % len(NEON_PALETTE)]
                cluster_counts[cluster_name] = 0
                color_idx += 1

            cluster_counts[cluster_name] += 1
            color = cluster_map[cluster_name]

            raw_text = meta.get("text", "") or ""
            snippet = raw_text[:140] + ("..." if len(raw_text) > 140 else "")

            points.append({
                "id": ids[i],
                "x": round(float(scaled_x[i]), 2),
                "y": round(float(scaled_y[i]), 2),
                "cluster": cluster_name,
                "color": color,
                "title": meta.get("title") or meta.get("source") or ids[i],
                "snippet": snippet,
                "meta": meta,
            })

        cluster_summary = [
            {"name": name, "color": cluster_map[name], "count": cluster_counts[name]}
            for name in cluster_map
        ]

        return {
            "points": points,
            "clusters": cluster_summary,
            "explained_variance": self.explained_variance,
            "total_points": n_vectors,
            "scale": self.scale,
        }

    def project_query(self, query_vec: np.ndarray) -> Tuple[float, float]:
        """
        Projects an arbitrary query vector into the established 2D coordinate system.
        """
        if not self.fitted or self.mean_vector is None or self.components is None:
            return 0.0, 0.0

        q = np.asarray(query_vec, dtype=np.float32).flatten()
        q_centered = q - self.mean_vector
        coords = np.dot(q_centered, self.components)

        x_scaled = (coords[0] - self.offset_x) * self.scale
        y_scaled = (coords[1] - self.offset_y) * self.scale
        return round(float(x_scaled), 2), round(float(y_scaled), 2)
