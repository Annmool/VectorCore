"""
Document Chunking implementations from scratch:
1. Fixed-Size Chunker (with character/token stride overlap)
2. Sentence-Boundary Chunker (respects natural grammatical boundaries)
3. Semantic Chunker (identifies embedding-distance shifts between consecutive sentences)
"""

import re
from typing import List, Dict, Any, Optional, Callable, Literal
import numpy as np
from vectordb.ingestion.loader import Document

ChunkingStrategy = Literal["fixed", "sentence", "semantic"]


class Chunk:
    """Represents a chunk of text with traceable metadata and chunk index."""

    def __init__(
        self,
        text: str,
        chunk_id: str,
        source: str,
        strategy: str,
        metadata: Optional[Dict[str, Any]] = None,
        char_start: int = 0,
        char_end: int = 0,
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.source = source
        self.strategy = strategy
        self.metadata = metadata or {}
        self.char_start = char_start
        self.char_end = char_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "strategy": self.strategy,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"<Chunk id='{self.chunk_id}' chars={len(self.text)} strategy='{self.strategy}'>"


class BaseChunker:
    """Base class for all chunkers."""

    def split_text(self, text: str, source: str = "doc", base_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        raise NotImplementedError

    def split_document(self, doc: Document) -> List[Chunk]:
        source = doc.metadata.get("id") or doc.metadata.get("source") or doc.metadata.get("title", "doc")
        return self.split_text(doc.content, source=source, base_meta=doc.metadata)


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size window chunker with configurable chunk size and overlap.
    Breaks on word boundaries near chunk_size rather than arbitrary characters.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source: str = "doc", base_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If not at the end of text, find closest space to avoid splitting words
            if end < text_len:
                space_pos = text.rfind(" ", start, end)
                if space_pos != -1 and space_pos > start + (self.chunk_size // 2):
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = dict(base_meta or {})
                meta["chunk_index"] = chunk_idx
                meta["strategy"] = "fixed"
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"{source}_fixed_{chunk_idx}",
                        source=source,
                        strategy="fixed",
                        metadata=meta,
                        char_start=start,
                        char_end=end,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break
            # Step forward by chunk_size - overlap
            step = max(end - start - self.chunk_overlap, 1)
            start = start + step

        return chunks


class SentenceBoundaryChunker(BaseChunker):
    """
    Sentence-boundary chunker:
    Extracts individual grammatical sentences, then packs them into chunks up to max_chunk_size
    with sentence-level overlap.
    """

    def __init__(self, max_chunk_size: int = 600, sentence_overlap: int = 1):
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text on sentence endings while protecting common abbreviations."""
        # Clean lines and split on ., ?, ! followed by space or newline
        sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\"'(\[])", text)
        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned

    def split_text(self, text: str, source: str = "doc", base_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        i = 0
        chunk_idx = 0

        while i < len(sentences):
            current_sentences = []
            curr_length = 0
            j = i

            while j < len(sentences):
                sent = sentences[j]
                if curr_length + len(sent) + 1 > self.max_chunk_size and current_sentences:
                    break
                current_sentences.append(sent)
                curr_length += len(sent) + 1
                j += 1

            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                meta = dict(base_meta or {})
                meta["chunk_index"] = chunk_idx
                meta["sentence_count"] = len(current_sentences)
                meta["strategy"] = "sentence"
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"{source}_sent_{chunk_idx}",
                        source=source,
                        strategy="sentence",
                        metadata=meta,
                    )
                )
                chunk_idx += 1

            if j >= len(sentences):
                break
            # Advance with overlap
            i = max(j - self.sentence_overlap, i + 1)

        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic Chunker:
    1. Splits document into sentences.
    2. Embeds each sentence using embed_fn.
    3. Computes cosine distance between consecutive sentences.
    4. Calculates breakpoint threshold (e.g. 80th percentile of distance gradients).
    5. Splits at semantic shifts to keep conceptually homogeneous blocks together.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        distance_threshold_percentile: float = 80.0,
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
    ):
        self.embed_fn = embed_fn
        self.percentile = distance_threshold_percentile
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_text(self, text: str, source: str = "doc", base_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        sentences = SentenceBoundaryChunker._split_into_sentences(text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    text=text.strip(),
                    chunk_id=f"{source}_sem_0",
                    source=source,
                    strategy="semantic",
                    metadata=base_meta or {},
                )
            ]

        # If no embedder provided, fall back to sentence chunking
        if self.embed_fn is None:
            fallback = SentenceBoundaryChunker(max_chunk_size=self.max_chunk_size)
            return fallback.split_text(text, source=source, base_meta=base_meta)

        embeddings = self.embed_fn(sentences)
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        embeddings = embeddings / norms

        # Consecutive cosine distances
        distances = []
        for idx in range(len(embeddings) - 1):
            cos_sim = float(np.dot(embeddings[idx], embeddings[idx + 1]))
            cos_dist = 1.0 - cos_sim
            distances.append(cos_dist)

        threshold = float(np.percentile(distances, self.percentile)) if distances else 0.5

        # Group sentences based on threshold and size limits
        chunks: List[Chunk] = []
        current_group: List[str] = [sentences[0]]
        current_len = len(sentences[0])
        chunk_idx = 0

        for idx in range(len(distances)):
            next_sentence = sentences[idx + 1]
            dist = distances[idx]
            is_semantic_shift = dist > threshold

            # Check if adding exceeds max size or shift detected and min size reached
            if (is_semantic_shift and current_len >= self.min_chunk_size) or (
                current_len + len(next_sentence) > self.max_chunk_size
            ):
                chunk_text = " ".join(current_group).strip()
                meta = dict(base_meta or {})
                meta["chunk_index"] = chunk_idx
                meta["semantic_shift_score"] = float(dist)
                meta["strategy"] = "semantic"
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"{source}_sem_{chunk_idx}",
                        source=source,
                        strategy="semantic",
                        metadata=meta,
                    )
                )
                chunk_idx += 1
                current_group = [next_sentence]
                current_len = len(next_sentence)
            else:
                current_group.append(next_sentence)
                current_len += len(next_sentence) + 1

        if current_group:
            chunk_text = " ".join(current_group).strip()
            meta = dict(base_meta or {})
            meta["chunk_index"] = chunk_idx
            meta["strategy"] = "semantic"
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=f"{source}_sem_{chunk_idx}",
                    source=source,
                    strategy="semantic",
                    metadata=meta,
                )
            )

        return chunks


def compare_chunking_strategies(
    text: str, embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compare Fixed-size, Sentence-based, and Semantic chunking on the same input text.
    Returns metrics (chunk counts, avg lengths, length variance, sample chunks).
    """
    fixed_chunker = FixedSizeChunker(chunk_size=400, chunk_overlap=80)
    sent_chunker = SentenceBoundaryChunker(max_chunk_size=450, sentence_overlap=1)
    sem_chunker = SemanticChunker(embed_fn=embed_fn, max_chunk_size=500)

    fixed_chunks = fixed_chunker.split_text(text, source="comparison")
    sent_chunks = sent_chunker.split_text(text, source="comparison")
    sem_chunks = sem_chunker.split_text(text, source="comparison")

    def get_stats(chunks: List[Chunk]) -> Dict[str, Any]:
        lengths = [len(c.text) for c in chunks]
        return {
            "count": len(chunks),
            "avg_length": float(np.mean(lengths)) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "std_length": float(np.std(lengths)) if lengths else 0,
            "chunks": [c.to_dict() for c in chunks],
        }

    return {
        "fixed": get_stats(fixed_chunks),
        "sentence": get_stats(sent_chunks),
        "semantic": get_stats(sem_chunks),
    }
