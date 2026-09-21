/**
 * VectorCore Left-Sidebar Console Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Tab Navigation
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const tabTitle = document.getElementById("current-tab-title");
  const tabDesc = document.getElementById("current-tab-desc");

  const tabMetadata = {
    "rag-playground": {
      title: "RAG & Grounded Citations",
      desc: "Ask research questions with verified inline citations directly linked to underlying vector chunks.",
    },
    "search-studio": {
      title: "Search Studio & Multi-Strategy Lab",
      desc: "Compare Dense Vector, BM25 Keyword, Hybrid RRF, and Cross-Encoder Reranked outputs side-by-side.",
    },
    "hnsw-visualizer": {
      title: "HNSW Skip-Graph Topology Inspector",
      desc: "Explore multi-layer hierarchical graph structure, entry points, and small-world connections.",
    },
    "chunking-lab": {
      title: "Chunking Strategy Comparator Lab",
      desc: "Evaluate Fixed-Size, Sentence-Boundary, and Semantic Distance Gradient chunking in real time.",
    },
    "quantization-lab": {
      title: "Quantization & Memory Compression",
      desc: "Evaluate Int8 Scalar and Product Quantization (PQ) compression ratios and reconstruction error.",
    },
    "benchmark-hub": {
      title: "Benchmarks & IR Evaluation Suite",
      desc: "Benchmark search latency vs. recall against FAISS and evaluate Recall@K, MRR, and NDCG.",
    },
    "ingest-hub": {
      title: "Document Ingestion & File Uploader",
      desc: "Upload PDFs, Markdown files, or Python code into the custom Vector Database.",
    },
  };

  navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navItems.forEach((n) => n.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const activePane = document.getElementById(`pane-${targetTab}`);
      if (activePane) activePane.classList.add("active");

      if (tabMetadata[targetTab]) {
        tabTitle.textContent = tabMetadata[targetTab].title;
        tabDesc.textContent = tabMetadata[targetTab].desc;
      }

      if (targetTab === "hnsw-visualizer") {
        fetchAndRenderHNSWGraph();
      }
    });
  });

  // Global Telemetry Polling
  async function refreshSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.status === "online") {
        document.getElementById("status-index").textContent = data.collection.index_type.toUpperCase();
        document.getElementById("status-vectors").textContent = data.collection.count;
        document.getElementById("status-cache").textContent = `${data.cache.hit_ratio_percent}%`;
        document.getElementById("index-type-select").value = data.collection.index_type;
      }
    } catch (e) {
      console.warn("Status poll error:", e);
    }
  }
  refreshSystemStatus();

  // Seed Knowledge Base Button
  document.getElementById("btn-reseed").addEventListener("click", async () => {
    const btn = document.getElementById("btn-reseed");
    btn.disabled = true;
    btn.innerHTML = "Seeding...";
    try {
      const res = await fetch("/api/collection/seed", { method: "POST" });
      const data = await res.json();
      alert(`Knowledge base ready: ${data.chunks_indexed || data.count} chunks indexed across 20 papers!`);
      refreshSystemStatus();
      fetchAndRenderHNSWGraph();
    } catch (e) {
      alert(`Seeding failed: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg> Seed Papers`;
    }
  });

  // Index Switcher
  document.getElementById("index-type-select").addEventListener("change", async (e) => {
    const newIndex = e.target.value;
    try {
      const res = await fetch("/api/collection/switch-index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_type: newIndex }),
      });
      const data = await res.json();
      if (data.status === "success") {
        refreshSystemStatus();
        fetchAndRenderHNSWGraph();
      }
    } catch (err) {
      console.error("Failed to switch index:", err);
    }
  });

  // Sample Query Chips
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      document.getElementById("rag-query-input").value = q;
      document.getElementById("rag-form").dispatchEvent(new Event("submit"));
    });
  });

  // =========================================================================
  // Tab 1: RAG Q&A Execution
  // =========================================================================
  const ragForm = document.getElementById("rag-form");
  const ragSubmitBtn = document.getElementById("btn-submit-rag");
  const ragAnswerText = document.getElementById("rag-answer-text");
  const ragLatencyBreakdown = document.getElementById("rag-latency-breakdown");
  const ragEngineBadge = document.getElementById("rag-engine-badge");
  const ragCacheBadge = document.getElementById("rag-cache-badge");
  const citationsListBody = document.getElementById("citations-list-body");
  const citationsCount = document.getElementById("citations-count");

  ragForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = document.getElementById("rag-query-input").value.trim();
    if (!query) return;

    const useHybrid = document.getElementById("rag-toggle-hybrid").checked;
    const useRerank = document.getElementById("rag-toggle-rerank").checked;
    const topK = parseInt(document.getElementById("rag-topk-select").value, 10);

    ragSubmitBtn.disabled = true;
    ragSubmitBtn.innerHTML = `<span>Thinking...</span>`;
    ragAnswerText.innerHTML = `<div class="placeholder-state"><p>Retrieving vectors, reranking, and generating grounded answer...</p></div>`;

    try {
      const res = await fetch("/api/rag/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: topK,
          use_hybrid: useHybrid,
          use_reranker: useRerank,
        }),
      });

      const data = await res.json();
      renderRAGResponse(data);
      refreshSystemStatus();
    } catch (err) {
      ragAnswerText.innerHTML = `<div class="placeholder-state" style="color: var(--brand-rose);"><p>Error: ${err.message}</p></div>`;
    } finally {
      ragSubmitBtn.disabled = false;
      ragSubmitBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg><span>Generate RAG Answer</span>`;
    }
  });

  function renderRAGResponse(data) {
    ragEngineBadge.textContent = `Engine: ${data.engine}`;
    ragCacheBadge.style.display = data.cache_hit ? "inline-block" : "none";

    let formattedHtml = data.answer
      .replace(/\n\n/g, "</p><p>")
      .replace(/^- (.*)$/gm, "<li>$1</li>")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>");

    if (formattedHtml.includes("<li>")) {
      formattedHtml = formattedHtml.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
    }
    formattedHtml = `<p>${formattedHtml}</p>`;

    formattedHtml = formattedHtml.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (match, nums) => {
      const numList = nums.split(",").map((n) => n.trim());
      return numList
        .map(
          (n) => `<span class="citation-badge" data-cite-num="${n}">[${n}]</span>`
        )
        .join("");
    });

    ragAnswerText.innerHTML = formattedHtml;

    citationsCount.textContent = `${data.citations.length} Sources`;
    citationsListBody.innerHTML = "";

    data.citations.forEach((c) => {
      const citeCard = document.createElement("div");
      citeCard.className = `citation-card ${c.is_cited ? "is-cited" : ""}`;
      citeCard.id = `cite-card-${c.citation_number}`;
      citeCard.innerHTML = `
        <div class="cite-card-header">
          <span class="cite-number">[${c.citation_number}]</span>
          <span class="cite-score">Score: ${(c.relevance_score * 100).toFixed(1)}%</span>
        </div>
        <div class="cite-source">${c.source}</div>
        <div class="cite-snippet">"${c.snippet}"</div>
      `;
      citationsListBody.appendChild(citeCard);
    });

    document.querySelectorAll(".citation-badge").forEach((badge) => {
      const num = badge.getAttribute("data-cite-num");
      const targetCard = document.getElementById(`cite-card-${num}`);
      
      badge.addEventListener("mouseenter", () => {
        if (targetCard) {
          targetCard.classList.add("highlighted");
        }
      });
      badge.addEventListener("mouseleave", () => {
        if (targetCard) {
          targetCard.classList.remove("highlighted");
        }
      });
      badge.addEventListener("click", () => {
        if (targetCard) {
          targetCard.scrollIntoView({ behavior: "smooth", block: "center" });
          targetCard.classList.add("highlighted");
          setTimeout(() => targetCard.classList.remove("highlighted"), 2000);
        }
      });
    });

    if (data.latency_breakdown) {
      ragLatencyBreakdown.style.display = "block";
      const { retrieval_ms, rerank_ms, generation_ms, total_ms } = data.latency_breakdown;
      document.getElementById("time-retrieval").textContent = `${retrieval_ms}ms`;
      document.getElementById("time-rerank").textContent = `${rerank_ms}ms`;
      document.getElementById("time-gen").textContent = `${generation_ms}ms`;
      document.getElementById("time-total").textContent = `${total_ms}ms`;

      const maxTime = Math.max(total_ms, 1);
      document.getElementById("fill-retrieval").style.width = `${(retrieval_ms / maxTime) * 100}%`;
      document.getElementById("fill-rerank").style.width = `${(rerank_ms / maxTime) * 100}%`;
      document.getElementById("fill-gen").style.width = `${(generation_ms / maxTime) * 100}%`;
    }
  }

  // =========================================================================
  // Tab 2: Search Studio
  // =========================================================================
  const studioAlphaSlider = document.getElementById("studio-alpha-slider");
  const alphaDisplay = document.getElementById("alpha-val-display");
  studioAlphaSlider.addEventListener("input", (e) => {
    alphaDisplay.textContent = e.target.value;
  });

  document.getElementById("btn-studio-search").addEventListener("click", async () => {
    const query = document.getElementById("studio-query-input").value.trim();
    if (!query) return;

    const fusionMode = document.getElementById("studio-fusion-mode").value;
    const alpha = parseFloat(studioAlphaSlider.value);
    const yearFilter = document.getElementById("studio-year-filter").value;
    const filter = yearFilter !== "all" ? { year: { $gte: parseInt(yearFilter, 10) } } : null;

    // Dense
    fetch("/api/search/dense", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5, filter }),
    }).then((r) => r.json()).then((d) => renderStrategyCol("dense", d.results, d.timing.total_ms));

    // BM25
    fetch("/api/search/bm25", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5, filter }),
    }).then((r) => r.json()).then((d) => renderStrategyCol("bm25", d.results, d.timing.search_ms));

    // Hybrid
    fetch("/api/search/hybrid", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5, fusion_method: fusionMode, alpha, filter }),
    }).then((r) => r.json()).then((d) => renderStrategyCol("hybrid", d.results, d.timing.total_ms));

    // Rerank
    fetch("/api/search/rerank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5, initial_k: 15, fusion_method: fusionMode, alpha, filter }),
    }).then((r) => r.json()).then((d) => renderStrategyCol("rerank", d.results, d.timing.total_ms));
  });

  function renderStrategyCol(colType, items, latencyMs) {
    const colContainer = document.getElementById(`${colType}-results-col`);
    const timeBadge = document.getElementById(`${colType}-time-badge`);
    timeBadge.textContent = `${latencyMs || 0}ms`;

    if (!items || items.length === 0) {
      colContainer.innerHTML = `<p class="empty-hint">No matches found.</p>`;
      return;
    }

    colContainer.innerHTML = items
      .map((item, idx) => {
        const title = item.metadata?.title || item.metadata?.source || item.id || `Result #${idx + 1}`;
        const score = item.rerank_score || item.score || 0;
        const text = (item.text || item.metadata?.text || "").substring(0, 100);
        return `
          <div class="result-mini-card">
            <div class="result-mini-head">
              <span>#${idx + 1}</span>
              <span>${(score * 100).toFixed(1)}%</span>
            </div>
            <div class="result-mini-title">${title}</div>
            <div class="result-mini-snippet">${text}...</div>
          </div>
        `;
      })
      .join("");
  }

  // =========================================================================
  // Tab 3: HNSW Graph Visualizer Canvas
  // =========================================================================
  async function fetchAndRenderHNSWGraph() {
    try {
      const res = await fetch(`/api/hnsw/graph?t=${Date.now()}`);
      const data = await res.json();
      renderHNSWCanvas(data);
    } catch (e) {
      console.warn("HNSW Graph fetch error:", e);
    }
  }

  document.getElementById("btn-refresh-hnsw").addEventListener("click", fetchAndRenderHNSWGraph);

  function renderHNSWCanvas(data) {
    const canvas = document.getElementById("hnsw-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const statsBanner = document.getElementById("hnsw-stats-row");
    statsBanner.innerHTML = `
      <span class="badge">Num Layers: ${data.num_layers || 1}</span>
      <span class="badge">Max Level: ${data.max_level || 0}</span>
      <span class="badge">Visualized Nodes: ${data.nodes?.length || 0}</span>
      <span class="badge">Total Edges: ${data.edges?.length || 0}</span>
      <span class="badge badge-engine">Entry Node: ${data.enter_node || "N/A"}</span>
    `;

    if (!data.nodes || data.nodes.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "14px 'Plus Jakarta Sans'";
      ctx.textAlign = "center";
      ctx.fillText("No nodes in HNSW index. Click 'Seed Papers' in the top bar.", width / 2, height / 2);
      return;
    }

    const layerCount = Math.max(data.num_layers, 2);
    const topMargin = 60;
    const bottomMargin = 70;
    const availableHeight = height - topMargin - bottomMargin;
    const layerSpacing = availableHeight / Math.max(layerCount - 1, 1);

    const layerNodes = {};
    for (let l = 0; l < layerCount; l++) {
      layerNodes[l] = [];
    }

    data.nodes.forEach((node) => {
      const lev = Math.min(node.max_level || 0, layerCount - 1);
      layerNodes[lev].push(node);
    });

    const nodePositions = {};
    for (let l = 0; l < layerCount; l++) {
      const y = height - bottomMargin - l * layerSpacing;
      const nodesAtL = layerNodes[l];
      const count = nodesAtL.length;
      
      nodesAtL.forEach((node, idx) => {
        const xSpan = width - 200;
        const x = count === 1 ? width / 2 : 100 + (idx * xSpan) / (count - 1);
        const yOffset = count > 6 ? (idx % 2 === 0 ? -10 : 10) : 0;
        nodePositions[node.id] = {
          x: x,
          y: y + yOffset,
          level: l,
          id: node.id,
          meta: node.meta,
          title: node.meta?.title || node.meta?.source || node.id,
        };
      });
    }

    data.nodes.forEach((node, idx) => {
      if (!nodePositions[node.id]) {
        const xSpan = width - 200;
        const x = 100 + (idx * xSpan) / Math.max(data.nodes.length - 1, 1);
        nodePositions[node.id] = {
          x: x,
          y: height - bottomMargin,
          level: 0,
          id: node.id,
          meta: node.meta,
          title: node.meta?.title || node.meta?.source || node.id,
        };
      }
    });

    // Draw Layer Guide Planes
    for (let l = 0; l < layerCount; l++) {
      const y = height - bottomMargin - l * layerSpacing;
      
      const grad = ctx.createLinearGradient(40, y, width - 40, y);
      grad.addColorStop(0, "rgba(0, 245, 155, 0.01)");
      grad.addColorStop(0.5, l > 0 ? "rgba(251, 191, 36, 0.08)" : "rgba(0, 245, 155, 0.05)");
      grad.addColorStop(1, "rgba(0, 245, 155, 0.01)");

      ctx.fillStyle = grad;
      ctx.fillRect(40, y - 22, width - 80, 44);

      ctx.strokeStyle = l > 0 ? "rgba(251, 191, 36, 0.3)" : "rgba(0, 245, 155, 0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 40, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = l > 0 ? "#fbbf24" : "#00f59b";
      ctx.font = "bold 10px 'JetBrains Mono'";
      ctx.textAlign = "left";
      const layerLabel = l === 0
        ? "Layer 0 (Base Graph)"
        : (l === data.max_level ? `Layer ${l} (Top Express Skip)` : `Layer ${l} (Express Skip)`);
      ctx.fillText(layerLabel, 44, y - 8);
    }

    // Draw Edges
    if (data.edges) {
      data.edges.forEach((edge) => {
        const p1 = nodePositions[edge.source];
        const p2 = nodePositions[edge.target];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);

          if (p1.level === p2.level) {
            const midX = (p1.x + p2.x) / 2;
            const midY = (p1.y + p2.y) / 2 - (Math.abs(p1.x - p2.x) > 150 ? 15 : 0);
            ctx.quadraticCurveTo(midX, midY, p2.x, p2.y);
            ctx.strokeStyle = edge.level > 0 ? "rgba(251, 191, 36, 0.55)" : "rgba(0, 245, 155, 0.28)";
            ctx.lineWidth = edge.level > 0 ? 1.6 : 1.1;
          } else {
            const cx = (p1.x + p2.x) / 2 + 15;
            const cy = (p1.y + p2.y) / 2;
            ctx.quadraticCurveTo(cx, cy, p2.x, p2.y);
            ctx.strokeStyle = "rgba(244, 63, 94, 0.45)";
            ctx.lineWidth = 1.4;
          }
          ctx.stroke();
        }
      });
    }

    // Draw Nodes
    data.nodes.forEach((node) => {
      const pos = nodePositions[node.id];
      if (!pos) return;

      const isEnterNode = node.id === data.enter_node;
      const radius = isEnterNode ? 9 : (pos.level > 0 ? 7 : 5.5);

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

      if (isEnterNode) {
        ctx.fillStyle = "#f43f5e";
        ctx.shadowColor = "#f43f5e";
        ctx.shadowBlur = 12;
      } else if (pos.level > 0) {
        ctx.fillStyle = "#fbbf24";
        ctx.shadowColor = "#fbbf24";
        ctx.shadowBlur = 8;
      } else {
        ctx.fillStyle = "#00f59b";
        ctx.shadowColor = "#00f59b";
        ctx.shadowBlur = 5;
      }

      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px 'Plus Jakarta Sans'";
      ctx.textAlign = "center";
      const label = pos.title.length > 16 ? pos.title.substring(0, 14) + ".." : pos.title;
      ctx.fillText(label, pos.x, pos.y + radius + 12);
    });

    // Hover tooltip listener
    if (!canvas._hasHoverListener) {
      canvas._hasHoverListener = true;
      let hoveredNodeId = null;

      canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        let found = null;
        Object.values(nodePositions).forEach((p) => {
          const dx = p.x - mouseX;
          const dy = p.y - mouseY;
          if (Math.sqrt(dx * dx + dy * dy) < 16) {
            found = p;
          }
        });

        if (found && hoveredNodeId !== found.id) {
          hoveredNodeId = found.id;
          canvas.style.cursor = "pointer";
          canvas.title = `${found.title}\nID: ${found.id}\nLevel: ${found.level}`;
        } else if (!found && hoveredNodeId) {
          hoveredNodeId = null;
          canvas.style.cursor = "default";
          canvas.title = "";
        }
      });
    }
  }

  // =========================================================================
  // Tab 4: Chunking Comparator
  // =========================================================================
  const sampleChunkingText = `Hierarchical Navigable Small World (HNSW) graphs are multi-layer proximity structures for approximate nearest neighbor search. Upper layers contain long-range connections for fast greedy routing, while lower layers provide dense connections for local accuracy.
Product Quantization (PQ) is an orthogonal vector compression technique. It divides high-dimensional vectors into M sub-vectors and clusters each sub-space with K-Means. At query time, Asymmetric Distance Computation (ADC) uses precomputed distance lookup tables.
BM25 is a probabilistic ranking function based on term frequencies and document lengths. It incorporates parameter k1 for frequency saturation and parameter b for length normalization. Combining dense vectors with BM25 via Reciprocal Rank Fusion ensures high recall for both keywords and semantics.`;

  document.getElementById("btn-load-sample-text").addEventListener("click", () => {
    document.getElementById("chunking-text-input").value = sampleChunkingText;
  });

  document.getElementById("btn-run-chunking-compare").addEventListener("click", async () => {
    const text = document.getElementById("chunking-text-input").value.trim();
    if (!text) return;

    const btn = document.getElementById("btn-run-chunking-compare");
    btn.disabled = true;

    try {
      const res = await fetch("/api/chunking/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();

      document.getElementById("chunk-comparison-container").style.display = "grid";

      ["fixed", "sentence", "semantic"].forEach((strat) => {
        const stats = data[strat];
        document.getElementById(`stats-${strat}`).innerHTML = `
          <span>Count: <strong>${stats.count}</strong></span> &bull; 
          <span>Avg: <strong>${stats.avg_length.toFixed(0)} chars</strong></span>
        `;

        const listEl = document.getElementById(`list-${strat}-chunks`);
        listEl.innerHTML = stats.chunks
          .map(
            (c, i) => `
          <div class="chunk-block">
            <div class="chunk-block-header">
              <span>Chunk #${i + 1}</span>
              <span>${c.text.length} chars</span>
            </div>
            <div class="chunk-block-text">${c.text}</div>
          </div>
        `
          )
          .join("");
      });
    } catch (err) {
      alert(`Chunking comparison error: ${err.message}`);
    } finally {
      btn.disabled = false;
    }
  });

  // =========================================================================
  // Tab 5: Quantization Lab
  // =========================================================================
  document.getElementById("btn-run-int8").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-int8");
    btn.disabled = true;
    try {
      const res = await fetch("/api/quantization/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "int8_scalar" }),
      });
      const d = await res.json();
      document.getElementById("int8-metrics-box").innerHTML = `
        <div class="metric-row"><span>Compression:</span> <strong>${d.compression_ratio}x</strong></div>
        <div class="metric-row"><span>Raw Size:</span> <strong>${d.original_bytes_per_vec}B &rarr; ${d.quantized_bytes_per_vec}B</strong></div>
        <div class="metric-row"><span>Recon MSE:</span> <strong>${d.reconstruction_mse.toFixed(6)}</strong></div>
      `;
    } finally {
      btn.disabled = false;
    }
  });

  document.getElementById("btn-run-pq").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-pq");
    const mVal = parseInt(document.getElementById("pq-m-select").value, 10);
    btn.disabled = true;
    try {
      const res = await fetch("/api/quantization/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "product_quantization", num_subvectors: mVal }),
      });
      const d = await res.json();
      document.getElementById("pq-metrics-box").innerHTML = `
        <div class="metric-row"><span>Sub-spaces (M):</span> <strong>${d.M}</strong></div>
        <div class="metric-row"><span>Compression:</span> <strong>${d.compression_ratio.toFixed(1)}x</strong></div>
        <div class="metric-row"><span>Vector Size:</span> <strong>${d.original_bytes_per_vec}B &rarr; ${d.quantized_bytes_per_vec}B</strong></div>
        <div class="metric-row"><span>Recon MSE:</span> <strong>${d.reconstruction_mse.toFixed(6)}</strong></div>
      `;
    } finally {
      btn.disabled = false;
    }
  });

  // =========================================================================
  // Tab 6: Benchmarks & IR Evaluation
  // =========================================================================
  document.getElementById("btn-run-benchmark").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-benchmark");
    const container = document.getElementById("benchmark-results-container");
    btn.disabled = true;
    container.innerHTML = `<p class="empty-hint">Running 50 queries across Flat, IVF, HNSW, and FAISS...</p>`;

    try {
      const res = await fetch("/api/benchmark/run", { method: "POST" });
      const data = await res.json();

      let tableHtml = `
        <table class="benchmark-table">
          <thead>
            <tr>
              <th>Index Algorithm</th>
              <th>Type</th>
              <th>Latency (ms)</th>
              <th>QPS</th>
              <th>Recall@1</th>
              <th>Recall@10</th>
              <th>Speedup</th>
            </tr>
          </thead>
          <tbody>
      `;

      Object.values(data.results).forEach((r) => {
        tableHtml += `
          <tr>
            <td><strong>${r.name}</strong></td>
            <td><span class="badge">${r.type}</span></td>
            <td>${r.latency_ms}ms</td>
            <td>${r.qps}</td>
            <td>${(r["recall@1"] * 100).toFixed(1)}%</td>
            <td>${(r["recall@10"] * 100).toFixed(1)}%</td>
            <td><strong style="color: var(--brand-emerald);">${r.speedup}x</strong></td>
          </tr>
        `;
      });

      tableHtml += `</tbody></table>`;
      container.innerHTML = tableHtml;
    } catch (err) {
      container.innerHTML = `<p style="color: var(--brand-rose);">Benchmark error: ${err.message}</p>`;
    } finally {
      btn.disabled = false;
    }
  });

  document.getElementById("btn-run-eval").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-eval");
    const container = document.getElementById("eval-metrics-container");
    btn.disabled = true;
    container.innerHTML = `<p class="empty-hint">Evaluating QA ground-truth pairs...</p>`;

    try {
      const res = await fetch("/api/evaluate/run", { method: "POST" });
      const data = await res.json();
      const m = data.summary_metrics;

      container.innerHTML = `
        <div class="eval-metric-card">
          <div class="eval-metric-title">Recall@1</div>
          <div class="eval-metric-num">${(m["recall@1"] * 100).toFixed(1)}%</div>
        </div>
        <div class="eval-metric-card">
          <div class="eval-metric-title">Recall@5</div>
          <div class="eval-metric-num">${(m["recall@5"] * 100).toFixed(1)}%</div>
        </div>
        <div class="eval-metric-card">
          <div class="eval-metric-title">MRR</div>
          <div class="eval-metric-num">${m.mrr.toFixed(3)}</div>
        </div>
        <div class="eval-metric-card">
          <div class="eval-metric-title">MAP</div>
          <div class="eval-metric-num">${m.map.toFixed(3)}</div>
        </div>
        <div class="eval-metric-card">
          <div class="eval-metric-title">NDCG@5</div>
          <div class="eval-metric-num">${(m["ndcg@5"] * 100).toFixed(1)}%</div>
        </div>
      `;
    } catch (err) {
      container.innerHTML = `<p style="color: var(--brand-rose);">Evaluation error: ${err.message}</p>`;
    } finally {
      btn.disabled = false;
    }
  });

  // =========================================================================
  // Tab 7: Document Ingestion / File Upload
  // =========================================================================
  const dropZone = document.getElementById("file-drop-zone");
  const fileInput = document.getElementById("file-input");
  let selectedFile = null;

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      selectedFile = e.target.files[0];
      dropZone.querySelector(".drop-prompt").innerHTML = `Selected: <strong>${selectedFile.name}</strong> (${(selectedFile.size / 1024).toFixed(1)} KB)`;
    }
  });

  document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert("Please select a file to upload.");
      return;
    }

    const strat = document.getElementById("upload-strategy").value;
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("chunking_strategy", strat);

    const btn = document.getElementById("btn-submit-upload");
    const statusBox = document.getElementById("upload-status-box");
    btn.disabled = true;
    statusBox.innerHTML = `<p>Extracting, chunking (${strat}), and embedding...</p>`;

    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      statusBox.innerHTML = `
        <div class="badge badge-cache" style="padding: 8px 12px;">
          Successfully indexed "${data.filename}": Created ${data.chunks_created} vector chunks!
        </div>
      `;
      refreshSystemStatus();
    } catch (err) {
      statusBox.innerHTML = `<p style="color: var(--brand-rose);">Upload failed: ${err.message}</p>`;
    } finally {
      btn.disabled = false;
    }
  });
});
