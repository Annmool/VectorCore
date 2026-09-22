<div align="center">

# ⚡ VectorCore

**A High-Performance Mini Vector Database & Grounded RAG Engine Built Completely from Mathematical First Principles in Python & NumPy.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-VectorCore%20Console-00f59b?style=for-the-badge&logo=render&logoColor=08090b)](https://vectorcore-tz2v.onrender.com/)

[![Python](https://img.shields.io/badge/Python-3.10%2B-00f59b?style=flat-square&logo=python&logoColor=08090b)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-00f59b?style=flat-square&logo=fastapi&logoColor=08090b)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-00f59b?style=flat-square&logo=docker&logoColor=08090b)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-fbbf24?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-00f59b?style=flat-square)](tests/)

🌐 **Live Console:** [https://vectorcore-tz2v.onrender.com](https://vectorcore-tz2v.onrender.com) &nbsp;|&nbsp; 📖 **API Docs:** [https://vectorcore-tz2v.onrender.com/docs](https://vectorcore-tz2v.onrender.com/docs)

*Zero black-box wrappers (no Chroma, no Pinecone, no FAISS dependencies for indexing algorithms).*

</div>

---

## 📌 Table of Contents
1. [Overview & Architecture](#-overview--architecture)
2. [Core Innovations](#-core-innovations)
   - [1. Indexing Algorithms (Flat, IVF, HNSW)](#1-indexing-algorithms)
   - [2. Vector Quantization (Int8 & PQ-ADC)](#2-vector-quantization--memory-compression)
   - [3. Multi-Strategy Chunking](#3-multi-strategy-chunking)
   - [4. Hybrid Search & Neural Reranker](#4-hybrid-search--neural-reranking)
   - [5. Grounded RAG with Inline Citations](#5-grounded-rag--inline-citations)
3. [Obsidian Matrix Console](#-obsidian-matrix-console)
4. [Quickstart & Local Setup](#-quickstart--local-setup)
5. [Docker & Cloud Deployment](#-docker--cloud-deployment)
6. [Evaluation & Benchmarks](#-evaluation--benchmarks)
7. [API Reference](#-api-reference)
8. [Project Structure](#-project-structure)

---

## 🏗️ Overview & Architecture

VectorCore is an end-to-end vector search and retrieval-augmented generation engine implementing every layer of modern vector database engineering from scratch:

```
                                  +---------------------------------------+
                                  | Ingestion: PDF / Markdown / Code / TXT|
                                  +-------------------+-------------------+
                                                      |
                                    +-----------------+-----------------+
                                    | 3-Way Chunking (Fixed / Sentence  |
                                    | / Semantic Distance Gradient)     |
                                    +-----------------+-----------------+
                                                      |
                                                      v
                                        +-------------+-------------+
                                        | Sentence Transformers     |
                                        | (all-MiniLM-L6-v2 / 384d) |
                                        +-------------+-------------+
                                                      |
                   +----------------------------------+----------------------------------+
                   |                                                                     |
                   v                                                                     v
+------------------------------------+                                +------------------------------------+
| Custom Vector DB Core (NumPy)      |                                | Sparse Lexical Index (from scratch)|
| - Flat (Exact Brute-Force Cosine)  |                                | - BM25Okapi Inverted Index         |
| - IVF Index (K-Means Voronoi Cells)|                                | - Term frequency saturation (k1)   |
| - HNSW Skip-Graph (M, efSearch)    |                                | - Document length penalty (b)      |
| - Scalar (Int8) & PQ Quantization  |                                +------------------+-----------------+
+------------------+-----------------+                                                   |
                   |                                                                     |
                   +----------------------------------+----------------------------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      | Hybrid Rank Fusion (RRF)      |
                                      | RRF(d) = sum[ w / (k + rank) ]|
                                      +---------------+---------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      | Neural Cross-Encoder Reranker |
                                      | (ms-marco-MiniLM-L-6-v2)      |
                                      +---------------+---------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      | Grounded RAG & Inline Cites   |
                                      | - 2-Tier Semantic LRU Cache   |
                                      | - Verified [1], [2] Citations |
                                      | - Latency Waterfall Telemetry |
                                      +-------------------------------+
```

---

## 🔬 Core Innovations

### 1. Indexing Algorithms
- **Flat Index (`vectordb/core/flat_index.py`)**: Exact matrix-vector cosine/Euclidean distance computation in NumPy, serving as the ground-truth baseline with MongoDB-style metadata filtering (`$eq`, `$gt`, `$in`, `$and`, `$or`).
- **Inverted File (IVF) Index (`vectordb/core/ivf_index.py`)**: Custom K-Means clustering partitions high-dimensional vector space into Voronoi cells. At query time, only the $n_{probe}$ closest centroids are scanned ($\mathcal{O}(n_{probe} \cdot \frac{N}{K} \cdot d)$).
- **Hierarchical Navigable Small World (HNSW) (`vectordb/core/hnsw_index.py`)**:
  - Probabilistic multi-layer skip-graph hierarchy ($\mathcal{O}(\log N)$ complexity).
  - Greedy nearest-neighbor search routing across upper skip layers.
  - `SELECT-NEIGHBORS-HEURISTIC` ensuring angular diversity and preventing clustering bottlenecks.
  - Dynamic graph topology export and real-time 2.5D visualizer.

### 2. Vector Quantization & Memory Compression
- **Int8 Scalar Quantization (`vectordb/core/quantization.py`)**: Compresses continuous 32-bit floats into signed 8-bit integers with calibrated min/max scaling factors ($\mathbf{4\times\text{ memory reduction}}$).
- **Product Quantization (PQ)**: Decomposes $D$-dimensional space into $M=8$ orthogonal sub-vectors and trains 256 sub-centroids per sub-space.
- **Asymmetric Distance Computation (ADC)**: Computes query-to-database distances directly in quantized code space using centroid distance lookup tables—**zero vector decompression required at query time**.

### 3. Multi-Strategy Chunking
- **Fixed-Size Chunking (`vectordb/ingestion/chunker.py`)**: Exact character/token windows with sliding overlap.
- **Sentence-Boundary Chunking**: Natural grammatical sentence preservation using lookaheads.
- **Semantic Distance Chunking**: Computes cosine distances between consecutive sentence embeddings and detects semantic shifts at distance gradient thresholds.

### 4. Hybrid Search & Neural Reranking
- **BM25Okapi Engine (`vectordb/retrieval/bm25.py`)**: Built from scratch with inverted posting lists, term frequencies, document lengths, and smoothed IDF.
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k_{rrf} + \text{rank}_m(d)}$$
- **Cross-Encoder Neural Reranker (`vectordb/retrieval/reranker.py`)**: Full query-document cross-attention using `cross-encoder/ms-marco-MiniLM-L-6-v2` with Sigmoid score calibration.

### 5. Grounded RAG & Inline Citations
- Strict factual answer synthesis with verified inline bracketed citations (`[1]`, `[2]`).
- Every citation is bidirectionally linked to the paper source, section, relevance score, and underlying text snippet.
- **Two-Tier Semantic Cache (`vectordb/rag/cache.py`)**: LRU in-memory cache for sub-millisecond query reuse.
- **Latency Waterfall**: Fine-grained telemetry measuring Retrieval $\to$ Rerank $\to$ Generation execution times.

---

## 🎨 Obsidian Matrix Console

The project includes an interactive web console served directly by FastAPI without complex node builds:
- **Obsidian Matrix Palette**: Volcanic titanium black (`#08090b`), bioluminescent mint (`#00f59b`), and champagne gold (`#fbbf24`).
- **Interactive 2.5D HNSW Visualizer**: Real-time HTML5 Canvas rendering of the skip-graph layers, entry point routing, and cluster connectivity.
- **Multi-Strategy Search Studio**: Live 4-column comparison of Dense Vector vs. Sparse BM25 vs. Hybrid RRF vs. Cross-Encoder Reranked outputs.
- **Chunking Lab & Quantization Comparator**: Live side-by-side experimentation.

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Annmool/VectorCore.git
cd VectorCore

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run_server.py
```
Open **`http://localhost:8000`** in your browser. API docs available at **`http://localhost:8000/docs`**.

### 3. Run Automated Tests
```bash
python -m pytest -v
```

---

## 🐳 Docker & Cloud Deployment

### Run with Docker Compose
```bash
docker compose up -d --build
```

### Build & Run Container Directly
```bash
docker build -t vectorcore .
docker run -p 8000:8000 -v ./data:/app/data vectorcore
```

---

## 📊 Evaluation & Benchmarks

### 1. Research Corpus Benchmark (15 Documents & Grounded QA)
VectorCore includes an evaluation suite against a 15-question research benchmark dataset using domain papers:

| Index Type | Build Time (15 docs + Embeddings) | Query Latency (P95) | Recall@5 |
| :--- | :--- | :--- | :--- |
| **Flat (Exact)** | Instant ($\mathcal{O}(1)$) | ~1.8 ms | **1.000** |
| **IVF (K-Means)** | ~180 ms | ~0.6 ms | **0.942** |
| **HNSW (Skip-Graph)** | ~210 ms | **~0.25 ms** | **0.986** |
| **Product Quantization (PQ)** | ~450 ms | ~0.4 ms | **0.912** |

### 2. Standalone HNSW Scaling Benchmark (32-dim Cosine)
Empirical benchmarks comparing the optimized NumPy HNSW engine against the unoptimized baseline on a standard Python 3.12 64-bit runtime:

| Dataset Size | Insert Time | Insert Throughput | Query Latency (100 queries) | Speedup vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **1,000 Vectors (Baseline)** | 43.35s | 23.1 adds/sec | 8.05 ms/query | 1.0x (Reference) |
| **1,000 Vectors (Optimized)** | **2.06s** | **485.3 adds/sec** | **1.21 ms/query** | **21.0x build speedup** (6.6x query speedup) |
| **5,000 Vectors (Optimized)** | **14.92s** | **335.1 adds/sec** | **1.73 ms/query** | Follows near-ideal $\mathcal{O}(N \log N)$ |
| **10,000 Vectors (Optimized)** | **32.71s** | **305.7 adds/sec** | **1.99 ms/query** | **305+ adds/sec**, sub-2ms latency |

> [!NOTE]
> **Empirical Measurement vs. Historical Placeholders**:
> Earlier documentation referenced an unverified ~620 ms placeholder for 10k vectors. The real, empirically profiled build time for 10,000 vectors in pure Python/NumPy is **32.71s** (305 adds/sec) with sub-2ms query latency.
> 
> **Scaling Dynamics ($\mathcal{O}(N \log N)$)**:
> The mild throughput taper from 1k (485 adds/s) to 10k (306 adds/s) is mathematically expected: as the graph grows, the number of hierarchical skip layers and beam-search traversal hops increases logarithmically ($\frac{10000 \ln 10000}{5000 \ln 5000} \approx 2.16\times$ theoretical vs $2.19\times$ observed). Profiling confirms the candidate heuristic pairwise matrix consumes $< 2.7\%$ of runtime.
>
> **Thread-Safety Verified**:
> Concurrent readers and writers run thread-safe under an `RLock`, verified with `test_concurrency_stress_test` under aggressive thread interleaving (8 threads, barrier synchronization, `sys.setswitchinterval(1e-6)`).

*Run the benchmark locally:*
```bash
python -m vectordb.evaluation.benchmark_hnsw_profile --sizes 1000 5000 10000
```

### 3. Reference Comparison: Custom NumPy HNSW vs. FAISS C++ HNSW
Direct side-by-side evaluation against Meta's `faiss.IndexHNSWFlat` (compiled C++ with AVX2 instruction set) on 10,000 vectors from the standard GloVe-25 word embedding benchmark (`M=16`, `efConstruction=64`, Cosine Metric, `k=10`).

> [!NOTE]
> **Performance Trade-Off Framing**:
> The custom pure Python/NumPy implementation trades ~150–200x query throughput for complete architectural transparency and zero compiled native C++ build dependencies, while maintaining a modest recall advantage (e.g. **98.30% vs 97.10%** at `efSearch=32`) at comparable parameters due to $M_0=2M=32$ Layer 0 connectivity. For lightweight, self-contained microservices or educational architectures, sub-millisecond query latency (~0.9ms) offers an attractive sweet spot without requiring C++ compilation toolchains.

#### Recall@10 vs. QPS Trade-Off Frontier (GloVe-25, 10,000 Subset)
Sweeping `efSearch` over $[8, 16, 32, 64, 128]$ against exact brute-force ground truth:

| `efSearch` | Custom HNSW Recall@10 | FAISS C++ Recall@10 | Custom Latency | Custom QPS | FAISS Latency | FAISS QPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | **88.60%** | 80.70% | 0.61 ms | 1,639 | 0.003 ms | 336,587 |
| **16** | **93.70%** | 90.50% | 0.70 ms | 1,432 | 0.003 ms | 320,821 |
| **32** | **98.30%** | 97.10% | 0.90 ms | 1,107 | 0.005 ms | 207,555 |
| **64** | **99.60%** | 99.40% | 1.31 ms | 765 | 0.008 ms | 130,890 |
| **128** | **100.00%** | 99.80% | 2.03 ms | 493 | 0.017 ms | 59,221 |

#### Index Construction Comparison (10,000 Vectors)
| Implementation | Language / Kernel | Build Time | Build Throughput |
| :--- | :--- | :--- | :--- |
| **Custom HNSW** | Pure Python / NumPy | 17.12s | 584 adds/sec |
| **FAISS IndexHNSWFlat** | C++ AVX2 Multi-threaded | 0.09s | 111,887 adds/sec |

*Run the parameter sweep locally:*
```bash
python -m vectordb.evaluation.benchmark_ann_reference --dataset glove-25 --n-docs 10000 --n-queries 100
```

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | System health, vector count, active index, and cache statistics |
| `POST` | `/api/rag/ask` | Grounded RAG query with citations and latency waterfall |
| `POST` | `/api/search/multi` | 4-way search comparison (Dense, BM25, Hybrid, Rerank) |
| `GET` | `/api/hnsw/graph` | HNSW graph nodes, edges, levels, and entry point |
| `POST` | `/api/chunking/compare` | Compare Fixed, Sentence, and Semantic chunkers |
| `POST` | `/api/quantization/int8` | Execute Int8 scalar quantization benchmark |
| `POST` | `/api/quantization/pq` | Train Product Quantizer and compute reconstruction MSE |
| `POST` | `/api/benchmark/run` | Benchmark Flat, IVF, HNSW, and FAISS |
| `POST` | `/api/benchmark/eval` | Evaluate Recall@1/5, MRR, MAP, and NDCG |
| `POST` | `/api/upload` | Ingest PDF, Markdown, Python code, or Plaintext |

---

## 📂 Project Structure

```
VectorCore/
├── app/
│   ├── api.py                    # FastAPI application & REST endpoints
│   └── frontend/
│       ├── index.html            # Obsidian Matrix Console UI
│       ├── favicon.svg           # VectorCore brand favicon
│       ├── css/styles.css        # Matrix design system & micro-animations
│       └── js/app.js             # Canvas visualizer & interactive client logic
├── data/
│   └── collections/              # Persistent vector storage & WAL logs
├── tests/
│   ├── test_chunker.py           # Unit tests for 3 chunking strategies
│   ├── test_distance.py          # Vectorized distance metric tests
│   ├── test_indexes.py           # Flat, IVF, and HNSW validation tests
│   └── test_quantization.py      # Int8 & Product Quantization tests
├── vectordb/
│   ├── core/                     # Vector database engine
│   │   ├── distance.py           # Cosine, L2, Dot, Manhattan distances
│   │   ├── flat_index.py         # Exact brute-force index + filtering
│   │   ├── ivf_index.py          # IVF K-Means Voronoi partitioning
│   │   ├── hnsw_index.py         # Multi-layer HNSW skip-graph
│   │   ├── quantization.py       # Scalar & Product Quantization (ADC)
│   │   ├── collection.py         # Document collection & CRUD
│   │   └── storage.py            # Disk serialization & WAL persistence
│   ├── embeddings/               # Sentence-Transformers embedding wrapper
│   ├── evaluation/               # Benchmark dataset, IR metrics & FAISS suite
│   ├── ingestion/                # Document loaders & chunkers
│   ├── rag/                      # RAG synthesizer, citations & LRU cache
│   └── retrieval/                # BM25Okapi, RRF Hybrid fusion & Cross-Encoder
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml            # Container orchestration with volume mounts
├── requirements.txt              # Production dependencies
├── run_server.py                 # Server entrypoint
└── README.md                     # Documentation
```

---

## 📜 License
MIT License. Built for high-performance retrieval and vector database systems education.
