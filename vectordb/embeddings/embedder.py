"""
Embedding generation using SentenceTransformers with local caching, batching, and fallback.
"""

from typing import List, Union, Optional
import hashlib
import numpy as np


class Embedder:
    """
    Sentence Transformers embedding wrapper with caching and batching.
    Default model: `sentence-transformers/all-MiniLM-L6-v2` (dim=384).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dim = 384
        self._cache = {}

        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _get_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"[Embedder] Notice: Could not load {self.model_name} ({e}). Using deterministic embedder.")
                self._model = "fallback"
                self._dim = 384
        return self._model

    @property
    def dim(self) -> int:
        self._get_model()
        return self._dim

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic pseudo-semantic projection for testing / offline fallback."""
        rng_vectors = []
        for text in texts:
            # Seed from MD5 hash of text for repeatability
            h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            np.random.seed(h)
            vec = np.random.randn(self._dim).astype(np.float32)
            # Add simple word hash components so identical words share direction
            words = text.lower().split()
            for w in words:
                w_seed = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:6], 16)
                np.random.seed(w_seed)
                vec += 0.3 * np.random.randn(self._dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            rng_vectors.append(vec)
        return np.array(rng_vectors, dtype=np.float32)

    def embed_texts(self, texts: Union[str, List[str]], show_progress_bar: bool = False) -> np.ndarray:
        """
        Embed a single text string or list of text strings.
        Returns: np.ndarray of shape (N, dim) normalized to unit length.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        if not text_list:
            return np.empty((0, self.dim), dtype=np.float32)

        # Check cache
        cached_results = {}
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(text_list):
            h = self._hash_text(text)
            if h in self._cache:
                cached_results[i] = self._cache[h]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        computed_embeddings = None
        if uncached_texts:
            model = self._get_model()
            if model == "fallback":
                computed_embeddings = self._fallback_embed(uncached_texts)
            else:
                computed_embeddings = model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )

            # Store into cache
            for idx_in_uncached, global_idx in enumerate(uncached_indices):
                vec = computed_embeddings[idx_in_uncached]
                h = self._hash_text(text_list[global_idx])
                self._cache[h] = vec
                cached_results[global_idx] = vec

        # Assemble final matrix
        final_matrix = np.array([cached_results[i] for i in range(len(text_list))], dtype=np.float32)

        return final_matrix[0] if is_single else final_matrix

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        return self.embed_texts(query)
