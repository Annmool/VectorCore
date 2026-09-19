"""
Multi-backend RAG Generator supporting Gemini, OpenAI, Ollama, and Built-in Extractive Synthesis.
Always guarantees grounded, cited responses.
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from vectordb.rag.citation import CitationEngine


SYSTEM_PROMPT = """You are an accurate, research-grade AI Assistant powered by a custom Vector Database and RAG system.
Your job is to answer the user's question strictly based on the provided Context Chunks.

Guidelines:
1. Ground every substantive statement in the provided context using bracketed numerical citations like [1], [2], or [1, 2].
2. If the answer cannot be determined from the context, state clearly that the provided corpus does not contain the required details.
3. Be structured, concise, and professional. Use markdown formatting where helpful.
"""


def build_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Construct prompt with numbered context snippets."""
    context_blocks = []
    for idx, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source") or meta.get("file_path") or "Document"
        text = chunk.get("text") or meta.get("text", "")
        context_blocks.append(f"[{idx}] Source: {source}\n{text}")

    context_str = "\n\n".join(context_blocks)
    prompt = f"""Context Chunks:
{context_str}

User Question: {query}

Please provide a detailed, accurate response with inline citations [1], [2] pointing to the relevant sources:"""
    return prompt


class RAGGenerator:
    """
    RAG Orchestrator handling LLM generation, fallback synthesis, and citation grounding.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        preferred_provider: str = "auto",  # 'gemini', 'openai', 'ollama', 'smart_extractive'
    ):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.ollama_url = ollama_url
        self.preferred_provider = preferred_provider

    def _generate_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini REST API."""
        if not self.gemini_api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[RAGGenerator] Gemini call failed: {e}")
        return None

    def _generate_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI REST API."""
        if not self.openai_api_key:
            return None
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[RAGGenerator] OpenAI call failed: {e}")
        return None

    def _generate_ollama(self, prompt: str, model: str = "llama3.2") -> Optional[str]:
        """Call local Ollama instance."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": model,
            "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json().get("response")
        except Exception:
            pass
        return None

    def _smart_extractive_synthesizer(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Built-in Smart Extractive Synthesis Engine.
        Synthesizes a coherent, cited answer by extracting most salient factual sentences
        and organizing them with sectioning and direct citations.
        """
        if not context_chunks:
            return "No relevant context chunks were found to answer this question."

        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))

        extracted_findings = []
        for idx, chunk in enumerate(context_chunks[:5], 1):
            text = chunk.get("text") or chunk.get("metadata", {}).get("text", "")
            sentences = re.split(r"(?<=[.?!])\s+", text)

            scored_sentences = []
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 20:
                    continue
                s_terms = set(re.findall(r"\b\w{3,}\b", sent_clean.lower()))
                overlap = len(q_terms.intersection(s_terms))
                scored_sentences.append((overlap, sent_clean))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s for score, s in scored_sentences[:2] if score > 0]

            if not top_sentences and sentences:
                top_sentences = [sentences[0].strip()]

            for s in top_sentences:
                extracted_findings.append((idx, s))

        if not extracted_findings:
            first_chunk_text = context_chunks[0].get("text", "")[:300]
            return f"Based on the available corpus [1]: {first_chunk_text}..."

        # Assemble synthesized answer
        answer_parts = [
            f"Based on the retrieved context regarding **'{query}'**, the key findings are:\n"
        ]
        seen_sentences = set()
        for idx, s in extracted_findings:
            if s not in seen_sentences:
                seen_sentences.add(s)
                answer_parts.append(f"- {s} [{idx}]")

        answer_parts.append(
            f"\n*Summary:* The retrieved evidence from {len(context_chunks)} relevant chunks answers the inquiry by highlighting the core mechanisms described above."
        )
        return "\n".join(answer_parts)

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer with provenance, citations, and timing breakdown.
        """
        start_time = time.time()
        prompt = build_rag_prompt(query, context_chunks)
        active_provider = provider or self.preferred_provider
        response_text = None
        used_engine = "smart_extractive"

        if active_provider in ["auto", "gemini"] and self.gemini_api_key:
            res = self._generate_gemini(prompt)
            if res:
                response_text = res
                used_engine = "gemini-1.5-flash"

        if response_text is None and (active_provider in ["auto", "openai"] and self.openai_api_key):
            res = self._generate_openai(prompt)
            if res:
                response_text = res
                used_engine = "gpt-4o-mini"

        if response_text is None and (active_provider in ["auto", "ollama"]):
            res = self._generate_ollama(prompt)
            if res:
                response_text = res
                used_engine = "ollama"

        if response_text is None:
            response_text = self._smart_extractive_synthesizer(query, context_chunks)
            used_engine = "smart_extractive_synthesizer"

        gen_latency_ms = round((time.time() - start_time) * 1000, 2)

        # Ground citations
        final_answer, citations, metrics = CitationEngine.build_grounded_citations(
            response_text, context_chunks
        )

        return {
            "answer": final_answer,
            "citations": citations,
            "engine": used_engine,
            "latency_ms": gen_latency_ms,
            "metrics": metrics,
        }
