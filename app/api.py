"""
FastAPI Backend Server for Mini Vector Database and RAG Layer.
Exposes REST endpoints for indexing, chunking, hybrid search, reranking, RAG, quantization, and benchmarks.
"""

import os
import time
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from vectordb.core.collection import Collection
from vectordb.core.storage import StorageEngine
from vectordb.embeddings.embedder import Embedder
from vectordb.ingestion.loader import DocumentLoader, Document
from vectordb.ingestion.chunker import (
    FixedSizeChunker,
    SentenceBoundaryChunker,
    SemanticChunker,
    compare_chunking_strategies,
)
from vectordb.retrieval.bm25 import BM25Index
from vectordb.retrieval.hybrid import HybridRetriever
from vectordb.retrieval.reranker import CrossEncoderReranker
from vectordb.rag.cache import LRUQueryCache
from vectordb.rag.generator import RAGGenerator
from vectordb.evaluation.metrics import evaluate_retrieval_system
from vectordb.evaluation.dataset import PAPERS_CORPUS, EVALUATION_QA_PAIRS
from vectordb.evaluation.benchmark import VectorIndexBenchmark


app = FastAPI(
    title="Mini Vector Database & RAG Core API",
    description="Full-stack vector database built from scratch with HNSW, IVF, BM25, RRF, Cross-Encoder Reranking, and Grounded Citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class SystemState:
    def __init__(self):
        self.storage = StorageEngine(base_dir="./data/collections")
        self.embedder = Embedder()
        self.collection = Collection(
            name="ai_research_kb",
            dim=self.embedder.dim,
            index_type="hnsw",
            metric="cosine",
            storage_engine=self.storage,
            hnsw_m=16,
            hnsw_ef_construction=64,
            hnsw_ef_search=32,
        )
        self.bm25 = BM25Index()
        self.hybrid_retriever = HybridRetriever(self.collection, self.bm25, self.embedder)
        self.reranker = CrossEncoderReranker()
        self.cache = LRUQueryCache(capacity=500)
        self.rag_generator = RAGGenerator()
        self.documents_metadata: List[Dict[str, Any]] = []

state = SystemState()


# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    index_type: Optional[Literal["flat", "ivf", "hnsw"]] = None
    filter: Optional[Dict[str, Any]] = None
    ef_search: Optional[int] = 32
    nprobe: Optional[int] = 4


class HybridSearchRequest(BaseModel):
    query: str
    k: int = 5
    fusion_method: Literal["rrf", "convex"] = "rrf"
    alpha: float = 0.65
    rrf_k: int = 60
    filter: Optional[Dict[str, Any]] = None


class RerankRequest(BaseModel):
    query: str
    k: int = 5
    initial_k: int = 15
    fusion_method: Literal["rrf", "convex"] = "rrf"
    alpha: float = 0.65
    filter: Optional[Dict[str, Any]] = None


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    use_hybrid: bool = True
    provider: Optional[str] = "auto"
    filter: Optional[Dict[str, Any]] = None


class TextIngestRequest(BaseModel):
    text: str
    source_name: str = "custom_document"
    chunking_strategy: Literal["fixed", "sentence", "semantic"] = "semantic"
    chunk_size: int = 500
    chunk_overlap: int = 80
    metadata: Optional[Dict[str, Any]] = None


class ChunkCompareRequest(BaseModel):
    text: str


class QuantizationRequest(BaseModel):
    type: Literal["int8_scalar", "product_quantization"] = "int8_scalar"
    num_subvectors: int = 8


class SwitchIndexRequest(BaseModel):
    index_type: Literal["flat", "ivf", "hnsw"]


def seed_knowledge_base(force: bool = False):
    if len(state.collection) > 0 and not force:
        return {"status": "already_seeded", "count": len(state.collection)}

    if force:
        # Re-instantiate fresh collection & BM25
        state.collection = Collection(
            name="ai_research_kb",
            dim=state.embedder.dim,
            index_type="hnsw",
            metric="cosine",
            storage_engine=state.storage,
            hnsw_m=8,
            hnsw_ef_construction=64,
            hnsw_ef_search=32,
        )
        state.bm25 = BM25Index()
        state.hybrid_retriever.collection = state.collection
        state.hybrid_retriever.bm25 = state.bm25

    chunker = SentenceBoundaryChunker(max_chunk_size=300, sentence_overlap=1)

    all_chunks = []
    for paper in PAPERS_CORPUS:
        doc = Document(content=paper["content"], metadata=paper)
        chunks = chunker.split_document(doc)
        all_chunks.extend(chunks)

    chunk_texts = [c.text for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]
    chunk_metas = [c.to_dict() for c in all_chunks]

    embeddings = state.embedder.embed_texts(chunk_texts)
    state.collection.insert_batch(chunk_ids, embeddings, chunk_metas)
    state.bm25.add_documents(chunk_ids, chunk_texts, chunk_metas)

    return {"status": "seeded", "chunks_indexed": len(all_chunks), "documents": len(PAPERS_CORPUS)}


@app.on_event("startup")
def startup_event():
    try:
        seed_knowledge_base(force=False)
        print(f"[VectorDB] Startup complete. Indexed {len(state.collection)} chunks.")
    except Exception as e:
        print(f"[VectorDB] Startup notice: {e}")


@app.get("/api/status")
def get_system_status():
    """Return runtime system status, index statistics, and cache metrics."""
    return {
        "status": "online",
        "collection": state.collection.get_stats(),
        "cache": state.cache.get_stats(),
        "embedder_dim": state.embedder.dim,
        "active_index": state.collection.index_type,
    }


@app.post("/api/collection/seed")
def seed_corpus_endpoint():
    """Seed or re-seed the built-in AI research knowledge base."""
    return seed_knowledge_base()


@app.post("/api/collection/switch-index")
def switch_index(req: SwitchIndexRequest):
    """Dynamically switch index between Flat, IVF, and HNSW."""
    # Rebuild new index with current vectors
    old_ids = state.collection.index.ids
    old_vecs = state.collection.index.vectors
    old_metas = state.collection.index.metadata
    old_del = state.collection.index.is_deleted

    new_coll = Collection(
        name=state.collection.name,
        dim=state.collection.dim,
        index_type=req.index_type,
        metric=state.collection.metric,
        storage_engine=state.storage,
    )
    if len(old_ids) > 0:
        new_coll.insert_batch(old_ids, old_vecs, old_metas)
        new_coll.index.is_deleted = old_del

    state.collection = new_coll
    state.hybrid_retriever.collection = new_coll

    return {
        "status": "success",
        "switched_to": req.index_type,
        "count": len(state.collection),
    }


@app.post("/api/chunking/compare")
def compare_chunking(req: ChunkCompareRequest):
    """Compare Fixed, Sentence, and Semantic chunking on user text."""
    return compare_chunking_strategies(req.text, embed_fn=state.embedder.embed_texts)


@app.post("/api/documents/ingest")
def ingest_text(req: TextIngestRequest):
    """Ingest, chunk, and index arbitrary raw text."""
    t0 = time.perf_counter()
    doc = Document(content=req.text, metadata=req.metadata or {"source": req.source_name})

    if req.chunking_strategy == "fixed":
        chunker = FixedSizeChunker(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    elif req.chunking_strategy == "sentence":
        chunker = SentenceBoundaryChunker(max_chunk_size=req.chunk_size)
    else:
        chunker = SemanticChunker(embed_fn=state.embedder.embed_texts, max_chunk_size=req.chunk_size)

    chunks = chunker.split_document(doc)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced 0 chunks.")

    chunk_texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_metas = [c.to_dict() for c in chunks]

    embeddings = state.embedder.embed_texts(chunk_texts)
    state.collection.insert_batch(chunk_ids, embeddings, chunk_metas)
    state.bm25.add_documents(chunk_ids, chunk_texts, chunk_metas)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "status": "success",
        "chunks_indexed": len(chunks),
        "strategy": req.chunking_strategy,
        "latency_ms": elapsed_ms,
        "chunks": [c.to_dict() for c in chunks],
    }


@app.post("/api/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("semantic"),
    chunk_size: int = Form(500),
):
    """Upload and index PDF, Markdown, or Text files."""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        docs = DocumentLoader.load_file(tmp_path)
        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract text from file.")

        # Update source metadata
        for d in docs:
            d.metadata["source"] = file.filename

        if chunking_strategy == "fixed":
            chunker = FixedSizeChunker(chunk_size=chunk_size)
        elif chunking_strategy == "sentence":
            chunker = SentenceBoundaryChunker(max_chunk_size=chunk_size)
        else:
            chunker = SemanticChunker(embed_fn=state.embedder.embed_texts, max_chunk_size=chunk_size)

        all_chunks = []
        for d in docs:
            all_chunks.extend(chunker.split_document(d))

        chunk_texts = [c.text for c in all_chunks]
        chunk_ids = [f"{file.filename}_{c.chunk_id}" for c in all_chunks]
        chunk_metas = [c.to_dict() for c in all_chunks]

        embeddings = state.embedder.embed_texts(chunk_texts)
        state.collection.insert_batch(chunk_ids, embeddings, chunk_metas)
        state.bm25.add_documents(chunk_ids, chunk_texts, chunk_metas)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_or_sections": len(docs),
            "chunks_created": len(all_chunks),
            "strategy": chunking_strategy,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/search/dense")
def search_dense(req: QueryRequest):
    """Dense vector search using currently active index (Flat / IVF / HNSW)."""
    t0 = time.perf_counter()
    embed_start = time.perf_counter()
    q_vec = state.embedder.embed_query(req.query)
    embed_latency = (time.perf_counter() - embed_start) * 1000

    search_start = time.perf_counter()
    results = state.collection.query(
        q_vec,
        k=req.k,
        filter=req.filter,
        ef_search=req.ef_search,
        nprobe=req.nprobe,
    )
    search_latency = (time.perf_counter() - search_start) * 1000
    total_latency = (time.perf_counter() - t0) * 1000

    return {
        "query": req.query,
        "results": results,
        "index_type": state.collection.index_type,
        "timing": {
            "embedding_ms": round(embed_latency, 2),
            "search_ms": round(search_latency, 2),
            "total_ms": round(total_latency, 2),
        },
    }


@app.post("/api/search/bm25")
def search_bm25(req: QueryRequest):
    """Pure BM25 keyword search."""
    t0 = time.perf_counter()
    results = state.bm25.search(req.query, k=req.k)
    search_latency = (time.perf_counter() - t0) * 1000

    return {
        "query": req.query,
        "results": results,
        "timing": {"search_ms": round(search_latency, 2)},
    }


@app.post("/api/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    """Hybrid search combining Dense (HNSW/IVF) and Sparse (BM25) with RRF / Convex fusion."""
    t0 = time.perf_counter()
    results = state.hybrid_retriever.search(
        query=req.query,
        k=req.k,
        fusion_method=req.fusion_method,
        alpha=req.alpha,
        rrf_k=req.rrf_k,
        filter=req.filter,
    )
    total_latency = (time.perf_counter() - t0) * 1000

    return {
        "query": req.query,
        "results": results,
        "fusion_method": req.fusion_method,
        "timing": {"total_ms": round(total_latency, 2)},
    }


@app.post("/api/search/rerank")
def search_rerank(req: RerankRequest):
    """Hybrid search followed by Cross-Encoder neural reranking."""
    t0 = time.perf_counter()
    # 1. First stage: hybrid search
    t_hybrid_0 = time.perf_counter()
    candidates = state.hybrid_retriever.search(
        query=req.query,
        k=req.initial_k,
        fusion_method=req.fusion_method,
        alpha=req.alpha,
        filter=req.filter,
    )
    hybrid_time = (time.perf_counter() - t_hybrid_0) * 1000

    # 2. Second stage: cross-encoder rerank
    t_rerank_0 = time.perf_counter()
    reranked = state.reranker.rerank(req.query, candidates, top_k=req.k)
    rerank_time = (time.perf_counter() - t_rerank_0) * 1000
    total_latency = (time.perf_counter() - t0) * 1000

    return {
        "query": req.query,
        "results": reranked,
        "timing": {
            "hybrid_retrieval_ms": round(hybrid_time, 2),
            "cross_encoder_ms": round(rerank_time, 2),
            "total_ms": round(total_latency, 2),
        },
    }


@app.post("/api/rag/ask")
def rag_ask(req: RAGRequest):
    """
    End-to-End RAG Query:
    Checks Query Cache -> Hybrid Search -> Neural Rerank -> LLM Generation -> Inline Citations.
    """
    t_start = time.perf_counter()
    cache_key = f"rag:{req.query}:{req.top_k}:{req.use_reranker}:{req.use_hybrid}"
    cached = state.cache.get(cache_key)

    if cached:
        cached_resp = dict(cached)
        cached_resp["cache_hit"] = True
        cached_resp["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        state.cache.record_hit_savings(cached.get("total_latency_ms", 50.0))
        return cached_resp

    # 1. Retrieval
    t_ret_0 = time.perf_counter()
    if req.use_hybrid:
        candidates = state.hybrid_retriever.search(
            query=req.query,
            k=max(req.top_k * 2, 10),
            filter=req.filter,
        )
    else:
        q_vec = state.embedder.embed_query(req.query)
        candidates = state.collection.query(q_vec, k=req.top_k, filter=req.filter)
    retrieval_ms = (time.perf_counter() - t_ret_0) * 1000

    # 2. Rerank
    t_rerank_0 = time.perf_counter()
    if req.use_reranker and candidates:
        final_context = state.reranker.rerank(req.query, candidates, top_k=req.top_k)
    else:
        final_context = candidates[:req.top_k]
    rerank_ms = (time.perf_counter() - t_rerank_0) * 1000

    # 3. Generation & Grounding
    t_gen_0 = time.perf_counter()
    gen_result = state.rag_generator.generate_answer(
        query=req.query, context_chunks=final_context, provider=req.provider
    )
    gen_ms = (time.perf_counter() - t_gen_0) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000

    response_payload = {
        "query": req.query,
        "answer": gen_result["answer"],
        "citations": gen_result["citations"],
        "engine": gen_result["engine"],
        "quality_metrics": gen_result["metrics"],
        "context_chunks": final_context,
        "cache_hit": False,
        "latency_breakdown": {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "total_latency_ms": round(total_ms, 2),
    }

    state.cache.put(cache_key, response_payload, estimated_computation_ms=total_ms)
    return response_payload


@app.post("/api/quantization/run")
def run_quantization(req: QuantizationRequest):
    """Run live Scalar or Product Quantization and return compression & reconstruction stats."""
    if req.type == "int8_scalar":
        return state.collection.quantize_scalar()
    else:
        return state.collection.quantize_product(num_subvectors=req.num_subvectors)


@app.post("/api/benchmark/run")
def run_benchmarks():
    """Run full Speed vs. Recall benchmark comparing Flat, IVF, HNSW, and FAISS."""
    benchmark = VectorIndexBenchmark(dim=state.embedder.dim, num_vectors=1000, num_queries=50)
    return benchmark.run_benchmark(k=10)


@app.post("/api/evaluate/run")
def run_evaluation():
    """Evaluate Retrieval System (Recall@K, MRR, NDCG) on benchmark dataset."""
    query_eval_results = []

    for item in EVALUATION_QA_PAIRS:
        question = item["question"]
        gt_ids = set(item["ground_truth_doc_ids"])

        # Hybrid search
        results = state.hybrid_retriever.search(question, k=10)
        retrieved_source_ids = []
        for r in results:
            meta = r.get("metadata", {})
            src = meta.get("id") or meta.get("source") or meta.get("file_path", "")
            # Clean chunk id to base doc id
            for gt in gt_ids:
                if gt in src or gt in r.get("id", ""):
                    retrieved_source_ids.append(gt)
                    break
            else:
                retrieved_source_ids.append(src)

        query_eval_results.append({
            "retrieved_ids": retrieved_source_ids,
            "ground_truth_ids": list(gt_ids),
        })

    summary = evaluate_retrieval_system(query_eval_results, k_values=[1, 3, 5, 10])
    return {
        "summary_metrics": summary,
        "num_test_queries": len(EVALUATION_QA_PAIRS),
        "corpus_tested": "AI & Vector Search Seminal Papers",
    }


@app.get("/api/hnsw/graph")
def get_hnsw_topology():
    """Return HNSW graph topology for 2D/3D visualization."""
    if isinstance(state.collection.index, Collection) or hasattr(state.collection.index, "get_graph_topology"):
        return state.collection.index.get_graph_topology(max_nodes=40)
    return {"status": "not_hnsw", "active_index": state.collection.index_type}


@app.get("/api/cache/stats")
def get_cache_stats():
    """Return LRU cache analytics."""
    return state.cache.get_stats()


@app.post("/api/cache/clear")
def clear_cache():
    """Clear query cache."""
    state.cache.clear()
    return {"status": "cleared"}


# Mount static frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
