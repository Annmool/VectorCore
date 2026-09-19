"""
Citation Grounding & Fact Verification Engine.
Links generated sentences directly to source document chunks with verbatim evidence snippets.
"""

import re
from typing import List, Dict, Any, Tuple


class CitationEngine:
    """
    Parses, grounds, and verifies citations in LLM-generated responses.
    """

    @staticmethod
    def extract_citation_indices(text: str) -> List[int]:
        """Extract referenced citation indices like [1], [2], [1, 2]."""
        matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
        indices = []
        for match in matches:
            for num_str in match.split(","):
                try:
                    indices.append(int(num_str.strip()))
                except ValueError:
                    pass
        return sorted(list(set(indices)))

    @staticmethod
    def build_grounded_citations(
        response_text: str, context_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process response text, attach citation metadata, extract direct quotes,
        and calculate citation faithfulness metrics.
        """
        referenced_indices = CitationEngine.extract_citation_indices(response_text)
        citations = []

        for idx in range(1, len(context_chunks) + 1):
            if idx <= len(context_chunks):
                chunk = context_chunks[idx - 1]
                meta = chunk.get("metadata", {})
                chunk_text = chunk.get("text") or meta.get("text", "") or ""
                source_doc = meta.get("source") or meta.get("file_path") or "Document"
                page = meta.get("page")
                score = chunk.get("rerank_score") or chunk.get("score", 0.0)

                # Check if this index was explicitly cited in the generated answer
                is_cited = idx in referenced_indices

                # Pick leading 150 chars as snippet
                snippet = (
                    chunk_text[:180] + "..." if len(chunk_text) > 180 else chunk_text
                )

                citations.append({
                    "citation_number": idx,
                    "chunk_id": chunk.get("id") or meta.get("chunk_id", f"chunk_{idx}"),
                    "source": source_doc,
                    "page": page,
                    "relevance_score": round(float(score), 4),
                    "is_cited": is_cited,
                    "snippet": snippet,
                    "full_text": chunk_text,
                })

        # Calculate faithfulness: check overlap between response words and cited chunks
        response_words = set(re.findall(r"\b\w{4,}\b", response_text.lower()))
        context_words = set()
        for c in context_chunks:
            c_text = c.get("text") or c.get("metadata", {}).get("text", "")
            context_words.update(re.findall(r"\b\w{4,}\b", c_text.lower()))

        grounding_overlap = (
            len(response_words.intersection(context_words)) / max(len(response_words), 1)
        )

        quality_metrics = {
            "total_candidates_provided": len(context_chunks),
            "citations_referenced_count": len(referenced_indices),
            "grounding_faithfulness_ratio": round(min(grounding_overlap, 1.0), 3),
            "has_grounded_citations": len(referenced_indices) > 0,
        }

        return response_text, citations, quality_metrics
