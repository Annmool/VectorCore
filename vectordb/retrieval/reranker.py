"""
Cross-Encoder Neural Reranking module.
Re-scores and re-ranks top retrieval candidates using full query-document cross-attention.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class CrossEncoderReranker:
    """
    Neural Cross-Encoder reranker.
    Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

        if device is None:
            try:
                import torch
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _get_model(self):
        if self._model is None:
            try:
                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception:
                    pass
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                print(f"[Reranker] Notice: Neural cross-encoder unavailable ({e}). Using lexical-density fallback.")
                self._model = "fallback"
        return self._model

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _fallback_rerank(self, query: str, candidate_texts: List[str]) -> np.ndarray:
        """Heuristic fallback reranking based on query term frequency & exact span match."""
        q_terms = set(query.lower().split())
        scores = []
        for text in candidate_texts:
            t_lower = text.lower()
            overlap = sum(1 for term in q_terms if term in t_lower)
            density = overlap / max(len(q_terms), 1)
            # Bonus for exact query substring
            if query.lower() in t_lower:
                density += 0.5
            scores.append(density)
        return np.array(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate list from hybrid search down to top_k results.
        Each candidate should contain 'text' or 'metadata.text'.
        If min_score is provided, filters out candidates below this relevance threshold.
        """
        if not candidates:
            return []

        model = self._get_model()
        doc_texts = []
        for c in candidates:
            text = c.get("text") or c.get("metadata", {}).get("text", "") or ""
            doc_texts.append(text)

        if model == "fallback":
            raw_scores = self._fallback_rerank(query, doc_texts)
            norm_scores = self._sigmoid(raw_scores * 2.0 - 1.0)
        else:
            pairs = [[query, text] for text in doc_texts]
            raw_scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            norm_scores = self._sigmoid(np.asarray(raw_scores, dtype=np.float32))

        # Attach rerank score and original rank
        reranked = []
        for orig_rank, (candidate, score) in enumerate(zip(candidates, norm_scores)):
            item = dict(candidate)
            item["original_rank"] = orig_rank + 1
            item["rerank_score"] = float(score)
            reranked.append(item)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Apply relevance thresholding if min_score is specified
        if min_score is not None and reranked:
            top_score = reranked[0]["rerank_score"]
            effective_min = max(min_score, top_score * 0.15)
            reranked = [r for r in reranked if r["rerank_score"] >= effective_min]

        return reranked[:top_k]
