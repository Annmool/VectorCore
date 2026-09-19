"""
Unit tests for Document Chunkers (Fixed, Sentence-boundary, and Semantic).
"""

from vectordb.ingestion.chunker import (
    FixedSizeChunker,
    SentenceBoundaryChunker,
    SemanticChunker,
    compare_chunking_strategies,
)


SAMPLE_TEXT = """Hierarchical Navigable Small World graphs are multi-layer proximity graphs.
They provide logarithmic complexity for approximate nearest neighbor search.
Product Quantization on the other hand is a lossy compression technique.
PQ decomposes vector space into orthogonal sub-spaces.
Each sub-space is clustered with K-Means."""


def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(SAMPLE_TEXT, source="test_doc")
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 120
        assert c.source == "test_doc"


def test_sentence_chunker():
    chunker = SentenceBoundaryChunker(max_chunk_size=150, sentence_overlap=1)
    chunks = chunker.split_text(SAMPLE_TEXT, source="test_doc")
    assert len(chunks) >= 2
    # Ensure no sentence cuts in the middle of a word
    for c in chunks:
        assert c.text.endswith(".")


def test_chunking_comparison():
    comp = compare_chunking_strategies(SAMPLE_TEXT)
    assert "fixed" in comp
    assert "sentence" in comp
    assert "semantic" in comp
    assert comp["fixed"]["count"] > 0
    assert comp["sentence"]["count"] > 0
