"""
Custom BM25Okapi Inverted Index search engine built from scratch.
Provides exact term-frequency BM25 retrieval without external dependencies.
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np


class BM25Index:
    """
    BM25Okapi Inverted Index Engine.
    Formula:
    IDF(q_i) = ln( (N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1 )
    Score(D, Q) = sum_{q_i in Q} [ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))) ]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []
        self.doc_metas: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
        self.is_deleted: List[bool] = []

        # Term frequency per doc: list of Dict[term, freq]
        self.doc_term_freqs: List[Dict[str, int]] = []
        # Inverted index: term -> list of internal doc indices containing term
        self.inverted_index: Dict[str, List[int]] = {}
        # Document frequency: term -> number of docs containing term
        self.doc_freqs: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases, strips punctuation, and tokens are extracted."""
        text = text.lower()
        # Find all alphanumeric tokens
        tokens = re.findall(r"\b[a-z0-9_]+\b", text)
        return tokens

    def add_document(self, id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """Add a single document to the BM25 index."""
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        doc_idx = len(self.doc_ids)

        term_freqs: Dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        self.doc_ids.append(id)
        self.doc_texts.append(text)
        self.doc_metas.append(meta or {})
        self.doc_lengths.append(doc_len)
        self.doc_term_freqs.append(term_freqs)
        self.is_deleted.append(False)

        # Update inverted index and document frequencies
        for term in term_freqs.keys():
            if term not in self.inverted_index:
                self.inverted_index[term] = []
                self.doc_freqs[term] = 0
            self.inverted_index[term].append(doc_idx)
            self.doc_freqs[term] += 1

        self.corpus_size += 1
        self.avgdl = sum(self.doc_lengths) / max(self.corpus_size, 1)
        self.idf_cache.clear()  # Invalidate cached IDFs
        return doc_idx

    def add_documents(
        self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """Add batch of documents."""
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]
        indices = []
        for doc_id, text, meta in zip(ids, texts, metadatas):
            indices.append(self.add_document(doc_id, text, meta))
        return indices

    def _get_idf(self, term: str) -> float:
        """Compute IDF for a term with caching."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        n_q = self.doc_freqs.get(term, 0)
        # BM25Okapi IDF formula
        idf = math.log(1.0 + (self.corpus_size - n_q + 0.5) / (n_q + 0.5))
        self.idf_cache[term] = max(idf, 0.0)  # Avoid negative IDF for very frequent words
        return self.idf_cache[term]

    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute BM25 scores for matching documents and return top-k results.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        # Find candidate documents containing at least one query term
        candidate_doc_indices: Set[int] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                for idx in self.inverted_index[token]:
                    if not self.is_deleted[idx]:
                        candidate_doc_indices.add(idx)

        if not candidate_doc_indices:
            return []

        # Score candidates
        scores: Dict[int, float] = {}
        for doc_idx in candidate_doc_indices:
            if filter_fn is not None and not filter_fn(self.doc_metas[doc_idx]):
                continue

            doc_len = self.doc_lengths[doc_idx]
            term_freqs = self.doc_term_freqs[doc_idx]
            doc_score = 0.0

            for token in query_tokens:
                if token in term_freqs:
                    freq = term_freqs[token]
                    idf = self._get_idf(token)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1e-5)))
                    doc_score += idf * (numerator / denominator)

            if doc_score > 0:
                scores[doc_idx] = doc_score

        if not scores:
            return []

        # Sort by BM25 score descending
        sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)[:k]
        max_score = max(scores.values()) if scores else 1.0

        results = []
        for idx in sorted_indices:
            raw_score = scores[idx]
            # Normalized BM25 score to [0, 1] relative to top hit
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append({
                "id": self.doc_ids[idx],
                "score": norm_score,
                "raw_bm25_score": raw_score,
                "text": self.doc_texts[idx],
                "metadata": self.doc_metas[idx],
                "retriever": "bm25",
            })

        return results
