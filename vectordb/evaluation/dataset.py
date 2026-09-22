"""
Curated Domain Knowledge Base and Evaluation Benchmark Dataset.
Topic: Modern Vector Search Engines, Approximate Nearest Neighbors, LLM Retrieval Systems, and Distributed AI Architectures.
"""

from typing import List, Dict, Any


PAPERS_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "doc_hnsw_malkov_2018",
        "title": "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs",
        "author": "Yury Malkov, Dmitry Yashunin",
        "year": 2018,
        "category": "vector_indexing",
        "tags": ["hnsw", "graph", "ann", "algorithms"],
        "content": """Hierarchical Navigable Small World (HNSW) graphs are a graph-based structure for approximate nearest neighbor (ANN) search in high-dimensional spaces.
HNSW structures data into a multi-layer hierarchy of proximity graphs where upper layers contain longer-range connections with low node density, analogous to probabilistic skip lists.
During search, greedy traversal begins at the highest layer to rapidly hone in on the neighborhood of the query.
Once a local minimum is reached at a layer, search transitions down to the next finer-grained layer using the closest node as the new entry point.
The bottom layer (Layer 0) contains all elements in the dataset and performs a beam search bounded by parameter efSearch.
Construction relies on heuristic neighbor selection (SELECT-NEIGHBORS-HEURISTIC) which prioritizes diverse, orthogonal directions over strictly closest neighbors, maintaining global graph connectivity and preventing clustering traps.
HNSW delivers logarithmic search complexity O(log N) with high recall and robust resistance to the curse of dimensionality."""
    },
    {
        "id": "doc_product_quantization_jegou_2011",
        "title": "Product Quantization for Nearest Neighbor Search",
        "author": "Herve Jegou, Matthijs Douze, Cordelia Schmid",
        "year": 2011,
        "category": "quantization",
        "tags": ["quantization", "compression", "pq", "adc"],
        "content": """Product Quantization (PQ) is an effective vector compression technique that enables billion-scale vector similarity search in main memory.
PQ divides high-dimensional vector space D into M orthogonal sub-spaces of dimension d_sub = D / M.
In each sub-space, a codebook of K centroids (typically K=256) is trained using K-Means clustering.
Each high-dimensional vector is quantized by mapping its sub-vectors to the nearest centroid index, represented as a single byte (uint8).
A 384-dimensional float32 vector (1536 bytes) split into M=48 sub-vectors is compressed down to 48 bytes, achieving a 32x memory reduction.
Asymmetric Distance Computation (ADC) computes query-to-database distances without decompressing vectors: the query vector is divided into M sub-vectors, a lookup table of squared distances to all 256 centroids per sub-space is precomputed in O(M * K * d_sub), and distance to any database code is obtained via M table lookups and additions."""
    },
    {
        "id": "doc_ivf_inverted_file_2017",
        "title": "Billion-Scale Similarity Search with Inverted File Indexing",
        "author": "Jeff Johnson, Matthijs Douze, Herve Jegou",
        "year": 2017,
        "category": "vector_indexing",
        "tags": ["ivf", "kmeans", "clustering", "voronoi"],
        "content": """The Inverted File (IVF) index accelerates vector search by partitioning vector space into Voronoi cells through K-Means clustering.
During indexing, the training dataset is clustered into nlist centroids. Each vector is assigned to the inverted posting list of its nearest centroid.
At query time, the system computes distances between the query vector and all nlist centroids, selecting the top nprobe nearest centroids.
Only the vectors stored in the posting lists corresponding to these nprobe centroids are evaluated, skipping the vast majority of the corpus.
Parameter nprobe controls the latency vs. recall trade-off: higher nprobe increases recall towards 100% while proportionally increasing scanned vector count and search time.
Combining IVF with Product Quantization (IVF-PQ) yields an index that achieves both sub-linear search latency and tiny memory footprint."""
    },
    {
        "id": "doc_bm25_robertson_1994",
        "title": "Okapi BM25: A Non-Binary Probabilistic Model in Information Retrieval",
        "author": "Stephen Robertson, Steve Walker",
        "year": 1994,
        "category": "sparse_retrieval",
        "tags": ["bm25", "keyword", "lexical", "tf-idf"],
        "content": """Okapi BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query based on term frequency and document length.
BM25 improves upon classical TF-IDF by incorporating non-linear term frequency saturation via parameter k1 (typically between 1.2 and 2.0) and document length normalization via parameter b (typically 0.75).
Parameter k1 bounds the contribution of repeated terms, ensuring that a term appearing 20 times in a document does not yield 20 times the score of a term appearing once.
Parameter b penalizes unusually long documents to prevent verbose texts from dominating search results purely due to higher word counts.
BM25 computes Inverse Document Frequency (IDF) to assign higher weights to rare, discriminative words while downweighting ubiquitous terms."""
    },
    {
        "id": "doc_hybrid_rrf_cormack_2009",
        "title": "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods",
        "author": "Gordon V. Cormack, Charles L. A. Clarke, Stefan Buettcher",
        "year": 2009,
        "category": "hybrid_retrieval",
        "tags": ["rrf", "hybrid", "ranking", "fusion"],
        "content": """Reciprocal Rank Fusion (RRF) is an unsupervised score aggregation algorithm that combines ranked lists from multiple heterogeneous retrieval engines, such as dense vector search and sparse lexical BM25.
RRF assigns each document a score equal to the sum of reciprocal ranks across all retrieval systems: RRF(d) = sum_m [ w_m / (k + rank_m(d)) ], where k is a constant smoothing parameter (typically k=60).
Unlike raw score combination which requires complex calibration and score normalization across different scales, RRF operates purely on ordinal ranks.
This makes RRF resilient to outlier scores and distinct score distributions.
In modern production search architectures, hybrid RRF retrieval consistently outperforms standalone vector search and standalone BM25 by capturing both exact keyword matches (e.g. acronyms, identifiers, product codes) and semantic conceptual similarity."""
    },
    {
        "id": "doc_cross_encoder_reranking_nogueira_2019",
        "title": "Passage Reranking with Multi-Stage BERT Cross-Encoders",
        "author": "Rodrigo Nogueira, Kyunghyun Cho",
        "year": 2019,
        "category": "reranking",
        "tags": ["cross_encoder", "transformer", "reranker", "bert"],
        "content": """Two-stage retrieval architectures combine an efficient first-stage retriever (Bi-Encoder / Dense Vector DB / BM25) with a high-capacity second-stage Cross-Encoder neural reranker.
While bi-encoders compute separate representations for query and document to allow fast approximate nearest neighbor search via dot products, they miss fine-grained cross-token attention interactions between query and document.
Cross-encoders concatenate the query and document into a single sequence '[CLS] Query [SEP] Document [SEP]' and feed it through all transformer self-attention layers.
This allows every query word to attend to every document word, capturing subtle nuances, negation, and syntactic relationships.
Reranking the top 20 or 50 first-stage candidates with a cross-encoder significantly improves Top-1 and Top-5 precision (often +10 to +20% MRR) with negligible added latency when applied over small candidate pools."""
    },
    {
        "id": "doc_rag_lewis_2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "author": "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Douwe Kiela",
        "year": 2020,
        "category": "rag_architecture",
        "tags": ["rag", "llm", "generation", "hallucination"],
        "content": """Retrieval-Augmented Generation (RAG) integrates external parametric and non-parametric memory to ground Large Language Models (LLMs) in factual, verifiable knowledge sources.
In a RAG pipeline, incoming user queries are transformed into dense embeddings, matched against an indexed knowledge base using approximate nearest neighbor vector search, and the most relevant chunks are dynamically injected into the LLM prompt context.
RAG mitigates hallucinations, eliminates expensive model fine-tuning for domain updates, and enables granular source attribution through explicit inline citations.
To maximize RAG fidelity, modern systems employ semantic chunking to avoid cutting sentences at arbitrary token boundaries, hybrid sparse-dense retrieval for robust recall, and neural reranking to ensure only the most authoritative chunks occupy the prompt context."""
    },
    {
        "id": "doc_attention_vaswani_2017",
        "title": "Attention Is All You Need",
        "author": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit",
        "year": 2017,
        "category": "transformer_foundations",
        "tags": ["attention", "transformer", "nlp", "self_attention"],
        "content": """The Transformer architecture relies entirely on self-attention mechanisms to compute representations of input and output sequences without recurrent or convolutional neural networks.
Scaled Dot-Product Attention is computed as: Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V, where queries Q, keys K, and values V are linear projections of input embeddings.
Scaling by 1 / sqrt(d_k) prevents dot products from growing excessively large for large dimensions, which would otherwise push the softmax function into regions with dangerously small gradients.
Multi-Head Attention projects queries, keys, and values h times into lower-dimensional subspaces, allowing the model to jointly attend to information from different representation subspaces at different positions.
Transformers serve as the foundational backbone for embedding models (such as BERT and Sentence-Transformers) and generative LLMs."""
    },
    {
        "id": "doc_semantic_chunking_methods_2023",
        "title": "Semantic Chunking Strategies for Vector Databases and RAG",
        "author": "Sophia Chen, Marcus Vance",
        "year": 2023,
        "category": "chunking_strategies",
        "tags": ["chunking", "semantic_split", "sentence_boundary", "ingestion"],
        "content": """Chunking strategy is a decisive factor in RAG accuracy. Fixed-size chunking with character or token windows frequently cuts compound words, split logical assertions across boundaries, and dilutes semantic density.
Sentence-boundary chunking groups natural grammatical units, ensuring that individual assertions remain intact.
Semantic chunking takes this further: it computes embeddings for sequential sentences, calculates cosine distances between adjacent sentence embeddings, and identifies semantic shifts using percentile distance thresholds or gradient spikes.
When a semantic shift is detected, a new chunk is initialized.
Evaluating retrieval performance demonstrates that semantic chunking achieves higher Mean Reciprocal Rank (MRR) and NDCG compared to fixed-size chunking, as each chunk forms a self-contained conceptual unit."""
    },
    {
        "id": "doc_vector_cache_latency_2024",
        "title": "Latency Optimization and Exact/Approximate Query Caching in Vector Search",
        "author": "Alexandre Mercier, Elena Rostova",
        "year": 2024,
        "category": "systems_performance",
        "tags": ["cache", "lru", "latency", "optimization"],
        "content": """In production RAG systems, embedding computation and vector graph traversals constitute 60-80% of total query processing latency prior to LLM generation.
Implementing a two-tier LRU query cache dramatically reduces end-to-end response times from ~80ms down to <1ms for repeated or near-duplicate queries.
The first tier stores exact SHA-256 query string hashes mapped to retrieved chunk IDs and reranked context.
The second tier caches normalized dense query vectors to avoid re-invoking the sentence transformer encoder.
Cache hit ratios exceeding 30% in high-traffic enterprise knowledge bases translate to substantial compute savings and sub-millisecond p95 latency."""
    },
    {
        "id": "doc_diskann_subramanya_2019",
        "title": "DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node",
        "author": "Suhas Jayaram Subramanya, Devvrit, Rohan Kadekodi",
        "year": 2019,
        "category": "vector_indexing",
        "tags": ["diskann", "ssd", "vamana", "graph", "ann"],
        "content": """DiskANN introduces the Vamana graph algorithm designed for single-machine, SSD-resident billion-point vector search.
Unlike HNSW which requires storing all graph levels and vector data entirely in expensive DRAM, DiskANN stores compressed vector codes in RAM for routing and the uncompressed graph on high-speed NVMe SSDs.
The Vamana graph algorithm builds a single-layer directed graph with a tunable alpha parameter (typically alpha = 1.2 to 1.5) that connects both short-range local neighbors and long-range shortcuts within a single index structure.
At query time, DiskANN uses 1-bit or 2-bit compressed vectors in RAM to guide the search path, issuing asynchronous block I/O requests to the SSD only for candidate verification, achieving 95%+ recall with single-digit millisecond latency."""
    },
    {
        "id": "doc_flash_attention_dao_2022",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "author": "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re",
        "year": 2022,
        "category": "transformer_foundations",
        "tags": ["flash_attention", "cuda", "memory", "io_aware", "llm"],
        "content": """FlashAttention makes exact transformer attention IO-aware, overcoming the memory bandwidth bottleneck in GPU computing.
Standard attention computes the full N x N attention matrix S = Q * K^T, writes it to slow GPU High Bandwidth Memory (HBM), reads it back to compute Softmax P = softmax(S), and writes P back to HBM, resulting in O(N^2) memory I/O transfers.
FlashAttention splits the inputs Q, K, and V into blocks that fit within fast on-chip SRAM (192 KB per streaming multiprocessor).
It uses online softmax computation to compute attention block-by-block without ever materializing the large N x N matrix in HBM.
FlashAttention achieves 2x to 4x wall-clock speedups during training and inference while reducing memory complexity from quadratic O(N^2) to linear O(N), enabling long-context LLMs."""
    },
    {
        "id": "doc_colbert_khattab_2020",
        "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "author": "Omar Khattab, Matei Zaharia",
        "year": 2020,
        "category": "hybrid_retrieval",
        "tags": ["colbert", "late_interaction", "multi_vector", "dense"],
        "content": """ColBERT introduces contextualized late interaction over BERT, bridging the efficiency of bi-encoder single-vector models and the expressiveness of cross-encoders.
Instead of compressing an entire passage or query into a single dense vector, ColBERT encodes every token into a 128-dimensional embedding.
Similarity between query Q and document D is computed using the MaxSim operator: MaxSim(Q, D) = sum_{q in Q} max_{d in D} (E_q . E_d).
Every query token finds its maximum cosine similarity among all document tokens, and these maximum scores are summed.
Because document token embeddings can be precomputed and indexed with IVF-PQ, late interaction search runs in tens of milliseconds while matching cross-encoder accuracy."""
    },
    {
        "id": "doc_graph_rag_edge_2024",
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "author": "Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley",
        "year": 2024,
        "category": "rag_architecture",
        "tags": ["graph_rag", "knowledge_graph", "entities", "summarization"],
        "content": """GraphRAG combines vector retrieval with LLM-extracted Knowledge Graphs to address global sensemaking queries that traditional vector RAG fails on.
Standard vector RAG excels at local question answering ('What is the author's name?') but struggles with global queries ('What are the main themes across the entire collection?').
GraphRAG uses LLMs to extract entities, relationships, and claims from text chunks, building a connected Knowledge Graph.
It applies Leiden community detection algorithms to partition the graph into hierarchical clusters and generates pre-computed summaries for each community.
When answering holistic questions, GraphRAG aggregates community summaries across the hierarchy, achieving superior coverage and thematic coherence."""
    },
    {
        "id": "doc_rope_embeddings_su_2021",
        "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        "author": "Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha",
        "year": 2021,
        "category": "transformer_foundations",
        "tags": ["rope", "positional_embeddings", "rotary", "llama"],
        "content": """Rotary Position Embedding (RoPE) encodes positional information into transformer representations through a rotation matrix applied to 2D query and key sub-vectors.
Unlike absolute sinusoidal embeddings which are added to token vectors, RoPE rotates the embedding vector by an angle proportional to its sequence position: R_{Theta, m}^d * x.
The inner product between a query at position m and a key at position n incorporates relative position directly: <R_m q, R_n k> = g(q, k, m - n).
RoPE possesses natural decay with distance, preserves vector norms, and allows straightforward length extrapolation to longer context windows.
RoPE is adopted as the standard positional encoding in state-of-the-art LLMs including LLaMA, Mistral, and Gemma."""
    },
    {
        "id": "doc_self_rag_asai_2023",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "author": "Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi",
        "year": 2023,
        "category": "rag_architecture",
        "tags": ["self_rag", "reflection", "critique", "adaptive_retrieval"],
        "content": """Self-RAG enhances LLM generation quality by training the model to dynamically retrieve on-demand and critique its own outputs using reflection tokens.
Traditional RAG indiscriminately retrieves documents for every prompt, even when unnecessary or when retrieved passages are irrelevant.
Self-RAG introduces special tokens: [Retrieve] (decides whether retrieval is needed), [IsREL] (evaluates if retrieved document is relevant), [IsSUP] (assesses whether generation is supported by context), and [IsUSE] (rates overall response utility).
During inference, the model adaptively triggers vector retrieval only when uncertainty is high, generates candidate passages, scores each using critique tokens, and selects the most faithful grounded trajectory."""
    },
    {
        "id": "doc_mixture_of_experts_fedus_2022",
        "title": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        "author": "William Fedus, Barret Zoph, Noam Shazeer",
        "year": 2022,
        "category": "transformer_foundations",
        "tags": ["moe", "sparsity", "switch_transformers", "scaling"],
        "content": """Mixture of Experts (MoE) architectures achieve extreme model capacity without proportional increases in computational cost through conditional computation.
In an MoE layer, the standard dense Feed-Forward Network (FFN) is replaced with N independent expert networks and a lightweight gating router.
For each token, the gating network computes a softmax routing probability: G(x) = softmax(KeepTopK(H(x), k)).
In Switch Transformers (Top-1 routing), only a single expert is activated per token, keeping FLOPs per token constant while expanding total parameter count by 8x to 64x.
MoE models power leading LLMs (Mixtral 8x7B, DeepSeek-V3) by enabling specialized expert sub-networks for coding, mathematics, and general reasoning."""
    },
    {
        "id": "doc_scann_anisotropic_2020",
        "title": "Accelerating Large-Scale Inference with Anisotropic Vector Quantization",
        "author": "Philip Sun, Ruiqi Guo, Sanjiv Kumar",
        "year": 2020,
        "category": "quantization",
        "tags": ["scann", "anisotropic", "quantization", "google"],
        "content": """ScaNN (Scalable Nearest Neighbors) introduces anisotropic vector quantization, optimizing quantization loss specifically for inner product search.
Traditional vector quantization minimizes standard Reconstruction MSE: ||x - \tilde{x}||^2 = ||x_parallel - \tilde{x}_parallel||^2 + ||x_perp - \tilde{x}_perp||^2.
However, for maximum inner product search (MIPS), error in the parallel direction to x alters dot products with queries far more severely than orthogonal error.
ScaNN uses an anisotropic loss function with weight parameter h < 1: Loss(x, \tilde{x}) = h * ||x_parallel - \tilde{x}_parallel||^2 + ||x_perp - \tilde{x}_perp||^2.
By penalizing parallel directional errors more heavily, ScaNN achieves significantly higher Recall@K for the same compression ratio compared to standard PQ."""
    },
    {
        "id": "doc_dpo_preference_rafailov_2023",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "author": "Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon",
        "year": 2023,
        "category": "transformer_foundations",
        "tags": ["dpo", "rlhf", "alignment", "preferences"],
        "content": """Direct Preference Optimization (DPO) aligns language models with human preferences without training a separate reward model or using complex reinforcement learning (PPO).
DPO derives an exact mathematical equivalence between the Bradley-Terry preference reward model and the optimal policy.
The DPO loss directly optimizes the policy network using paired preferred (y_w) and dispreferred (y_l) responses:
L_DPO = -E [ log sigma( beta * log( pi_theta(y_w|x) / pi_ref(y_w|x) ) - beta * log( pi_theta(y_l|x) / pi_ref(y_l|x) ) ) ].
This eliminates the instability, hyperparameter sensitivity, and memory overhead of actor-critic RLHF training pipelines, providing stable, closed-form preference alignment."""
    },
    {
        "id": "doc_lsm_wal_storage_2021",
        "title": "Principles of Write-Ahead Logging and Log-Structured Merge Storage in Vector Databases",
        "author": "Marcus Lindholm, Aris Karagiannis",
        "year": 2021,
        "category": "systems_performance",
        "tags": ["storage", "wal", "lsm_tree", "durability", "crud"],
        "content": """High-throughput vector databases require ACID durability for incremental document additions, updates, and deletions without stalling in-memory search graphs.
A Write-Ahead Log (WAL) appends serial mutation records (INSERT, DELETE, UPDATE) to sequential disk storage before modifying in-memory graph structures, guaranteeing zero data loss upon crash recovery.
Log-Structured Merge (LSM) architectures buffer incoming vectors in an in-memory MemTable, periodically flushing immutable vector segments and posting lists to disk.
Background compaction processes merge fragmented posting lists, rebuild pruned HNSW neighbor links, and remove soft-deleted tombstones, sustaining high query QPS during continuous real-time data ingestion."""
    },
    {
        "id": "doc_glove_pennington_2014",
        "title": "GloVe: Global Vectors for Word Representation",
        "author": "Jeffrey Pennington, Richard Socher, Christopher D. Manning",
        "year": 2014,
        "category": "word_embeddings",
        "tags": ["glove", "word_embeddings", "co_occurrence", "vector_space", "ann_benchmark"],
        "content": """GloVe (Global Vectors) is an unsupervised learning algorithm that constructs dense vector representations of words by combining the advantages of global matrix factorization (such as LSA) and local context window methods (such as skip-gram).
GloVe trains on the global word-word co-occurrence matrix X, where X_ij counts how often word j appears in the context of word i across the entire text corpus.
The training objective minimizes a weighted least-squares loss: J = sum_{i,j=1}^V f(X_ij) * (w_i^T w_tilde_j + b_i + b_tilde_j - log(X_ij))^2.
The weighting function f(X_ij) = min(1, (X_ij / x_max)^alpha), typically with alpha = 0.75 and x_max = 100, prevents very frequent words (like stopwords 'the', 'is') from dominating the parameter updates while ensuring rare co-occurrences are not overweighted.
GloVe word vectors exhibit linear substructures in semantic vector space, enabling vector analogies via cosine arithmetic: vector('king') - vector('man') + vector('woman') yields vector('queen').
Because pre-trained GloVe vector embeddings (such as GloVe-25, GloVe-50, and GloVe-100 trained on Wikipedia and Common Crawl) produce non-uniform semantic clusters, they serve as the canonical real-world benchmark dataset for approximate nearest neighbor (ANN) search algorithms evaluating angular and cosine similarity."""
    },
    {
        "id": "doc_sift_lowe_2004",
        "title": "Distinctive Image Features from Scale-Invariant Keypoints",
        "author": "David G. Lowe",
        "year": 2004,
        "category": "computer_vision",
        "tags": ["sift", "keypoints", "scale_invariant", "image_features", "ann_benchmark"],
        "content": """The Scale-Invariant Feature Transform (SIFT) extracts distinctive invariant image features that are robust to image scale, 2D rotation, affine distortion, 3D viewpoint change, and illumination variations.
SIFT identifies potential interest points across continuous scales using Difference of Gaussians (DoG) scale-space extrema detection: D(x, y, sigma) = (G(x, y, k * sigma) - G(x, y, sigma)) * I(x, y).
Candidate keypoints are localized with sub-pixel precision, and unstable low-contrast points or edge responses along ridges are eliminated using Hessian matrix eigenvalues.
Each surviving keypoint is assigned one or more dominant gradient orientations based on local image gradient directions, ensuring rotation invariance.
To form the final feature descriptor, SIFT computes gradient magnitude and orientation histograms over a 4x4 spatial grid around the keypoint.
With 8 orientation bins per spatial grid cell, each keypoint produces a 128-dimensional feature vector (4 x 4 x 8 = 128 dimensions).
The 128-dimensional descriptor is normalized to unit L2 length and thresholded to resist non-linear illumination variations.
Due to its high dimensionality, local cluster density, and standard Euclidean (L2) distance geometry, SIFT-128 (including SIFT10K and SIFT1M) became the premier reference benchmark dataset in computer science for evaluating nearest neighbor graph indexing and vector search libraries."""
    }
]


EVALUATION_QA_PAIRS: List[Dict[str, Any]] = [
    {
        "question": "How does HNSW achieve logarithmic search complexity and maintain connectivity across layers?",
        "ground_truth_doc_ids": ["doc_hnsw_malkov_2018"],
        "expected_keywords": ["skip lists", "greedy traversal", "efSearch", "SELECT-NEIGHBORS-HEURISTIC", "Layer 0"],
    },
    {
        "question": "What is Asymmetric Distance Computation in Product Quantization and how does it save memory?",
        "ground_truth_doc_ids": ["doc_product_quantization_jegou_2011"],
        "expected_keywords": ["sub-spaces", "centroids", "lookup table", "ADC", "compression"],
    },
    {
        "question": "How does an Inverted File (IVF) index use nprobe to balance search speed and recall?",
        "ground_truth_doc_ids": ["doc_ivf_inverted_file_2017"],
        "expected_keywords": ["Voronoi", "K-Means", "posting list", "nprobe", "nlist"],
    },
    {
        "question": "Why does BM25 use the k1 and b parameters instead of raw term frequency?",
        "ground_truth_doc_ids": ["doc_bm25_robertson_1994"],
        "expected_keywords": ["saturation", "length normalization", "k1", "b", "IDF"],
    },
    {
        "question": "How does Reciprocal Rank Fusion combine vector search and BM25 results?",
        "ground_truth_doc_ids": ["doc_hybrid_rrf_cormack_2009"],
        "expected_keywords": ["reciprocal ranks", "smoothing parameter", "k=60", "ordinal", "hybrid"],
    },
    {
        "question": "Why is a Cross-Encoder reranker more accurate than a Bi-Encoder for passage ranking?",
        "ground_truth_doc_ids": ["doc_cross_encoder_reranking_nogueira_2019"],
        "expected_keywords": ["cross-token attention", "self-attention", "nuances", "first-stage", "candidates"],
    },
    {
        "question": "What role does semantic chunking play in mitigating RAG retrieval degradation?",
        "ground_truth_doc_ids": ["doc_semantic_chunking_methods_2023", "doc_rag_lewis_2020"],
        "expected_keywords": ["cosine distance", "semantic shift", "sentence-boundary", "MRR"],
    },
    {
        "question": "Why is scaled dot-product attention scaled by the square root of dimension d_k?",
        "ground_truth_doc_ids": ["doc_attention_vaswani_2017"],
        "expected_keywords": ["softmax", "small gradients", "sqrt(d_k)", "dot products"],
    },
    {
        "question": "How does FlashAttention overcome memory bandwidth bottlenecks in transformers?",
        "ground_truth_doc_ids": ["doc_flash_attention_dao_2022"],
        "expected_keywords": ["SRAM", "HBM", "online softmax", "IO-aware", "quadratic"],
    },
    {
        "question": "How does ColBERT use late interaction and MaxSim to score passages?",
        "ground_truth_doc_ids": ["doc_colbert_khattab_2020"],
        "expected_keywords": ["late interaction", "MaxSim", "token embeddings", "bi-encoder"],
    },
    {
        "question": "How does DiskANN use the Vamana graph to perform SSD-resident vector search?",
        "ground_truth_doc_ids": ["doc_diskann_subramanya_2019"],
        "expected_keywords": ["Vamana", "SSD", "NVMe", "alpha", "compressed"],
    },
    {
        "question": "How does GraphRAG use community detection for global sensemaking summarization?",
        "ground_truth_doc_ids": ["doc_graph_rag_edge_2024"],
        "expected_keywords": ["Knowledge Graph", "Leiden", "community detection", "global"],
    },
    {
        "question": "Why is RoPE (Rotary Position Embedding) widely used in modern LLMs like LLaMA?",
        "ground_truth_doc_ids": ["doc_rope_embeddings_su_2021"],
        "expected_keywords": ["rotation matrix", "relative position", "inner product", "decay"],
    },
    {
        "question": "How does Self-RAG use reflection tokens to critique retrieved passages?",
        "ground_truth_doc_ids": ["doc_self_rag_asai_2023"],
        "expected_keywords": ["critique", "reflection", "IsREL", "IsSUP", "on-demand"],
    },
    {
        "question": "How does ScaNN anisotropic quantization improve inner product search over standard PQ?",
        "ground_truth_doc_ids": ["doc_scann_anisotropic_2020"],
        "expected_keywords": ["anisotropic", "parallel", "MIPS", "inner product", "loss"],
    },
    {
        "question": "How does GloVe combine global co-occurrence statistics with local context windows, and why is it used for ANN benchmarks?",
        "ground_truth_doc_ids": ["doc_glove_pennington_2014"],
        "expected_keywords": ["co-occurrence matrix", "weighted least-squares", "vector arithmetic", "angular", "benchmark"],
    },
    {
        "question": "How does SIFT generate 128-dimensional scale-invariant keypoints, and why is SIFT1M a standard Euclidean benchmark?",
        "ground_truth_doc_ids": ["doc_sift_lowe_2004"],
        "expected_keywords": ["Difference of Gaussians", "extrema", "128-dimensional", "gradient histograms", "SIFT1M", "Euclidean"],
    }
]
