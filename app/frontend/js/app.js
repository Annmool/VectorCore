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
    "embedding-projector": {
      title: "2D Semantic Embedding Space Projector",
      desc: "Interactive Principal Component Analysis (PCA) projection of all vector embeddings with cluster inspection and query localization.",
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
      } else if (targetTab === "embedding-projector") {
        fetchAndRenderProjector();
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
      alert(`Knowledge base ready: ${data.chunks_indexed || data.count} chunks indexed across ${data.documents || 22} papers!`);
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

    citationsCount.textContent = `${data.citations.length} ${data.citations.length === 1 ? "Source" : "Sources"}`;
    citationsListBody.innerHTML = "";

    if (data.citations.length === 0) {
      citationsListBody.innerHTML = `<div class="placeholder-state" style="padding: 1.5rem 1rem; text-align: center; color: var(--text-muted);"><p>No sources met the relevance threshold.</p></div>`;
    }

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
  // Tab 3: HNSW Graph Visualizer Canvas & Traversal Debugger
  // =========================================================================
  let hnswGraphData = null;
  let hnswTraceData = null;
  let currentTraceStep = -1;
  let traceTimer = null;
  let traceSpeed = 500;

  async function fetchAndRenderHNSWGraph() {
    try {
      const res = await fetch(`/api/hnsw/graph?t=${Date.now()}`);
      const data = await res.json();
      hnswGraphData = data;
      renderHNSWCanvas(data);
    } catch (e) {
      console.warn("HNSW Graph fetch error:", e);
    }
  }

  document.getElementById("btn-refresh-hnsw").addEventListener("click", () => {
    stopTracePlayback();
    hnswTraceData = null;
    currentTraceStep = -1;
    document.getElementById("hnsw-playback-controls").style.display = "none";
    document.getElementById("hnsw-telemetry-hud").style.display = "none";
    fetchAndRenderHNSWGraph();
  });

  // Trace Query Traversal Form
  document.getElementById("btn-trace-hnsw").addEventListener("click", executeHNSWTrace);
  document.getElementById("hnsw-query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      executeHNSWTrace();
    }
  });

  async function executeHNSWTrace() {
    const q = document.getElementById("hnsw-query-input").value.trim();
    if (!q) return;

    const btn = document.getElementById("btn-trace-hnsw");
    btn.disabled = true;
    btn.innerHTML = `<span>Tracing...</span>`;

    try {
      const res = await fetch("/api/hnsw/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k: 5 }),
      });
      const data = await res.json();
      if (!data.steps || data.steps.length === 0) {
        alert("No traversal steps recorded. Please seed the graph first.");
        return;
      }

      hnswTraceData = data;
      currentTraceStep = 0;
      document.getElementById("hnsw-playback-controls").style.display = "flex";
      document.getElementById("hnsw-telemetry-hud").style.display = "flex";

      renderHNSWCanvas(hnswGraphData);
      startTracePlayback();
    } catch (err) {
      console.error("Trace error:", err);
      alert("Failed to trace query route: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>Trace Search Route</span>`;
    }
  }

  function startTracePlayback() {
    stopTracePlayback();
    document.getElementById("btn-trace-playpause").textContent = "⏸ Pause";
    traceTimer = setInterval(() => {
      if (!hnswTraceData) return;
      if (currentTraceStep < hnswTraceData.steps.length - 1) {
        currentTraceStep++;
        renderHNSWCanvas(hnswGraphData);
      } else {
        stopTracePlayback();
      }
    }, traceSpeed);
  }

  function stopTracePlayback() {
    if (traceTimer) {
      clearInterval(traceTimer);
      traceTimer = null;
    }
    const playBtn = document.getElementById("btn-trace-playpause");
    if (playBtn) playBtn.textContent = "▶ Play";
  }

  document.getElementById("btn-trace-playpause").addEventListener("click", () => {
    if (traceTimer) {
      stopTracePlayback();
    } else {
      if (hnswTraceData && currentTraceStep >= hnswTraceData.steps.length - 1) {
        currentTraceStep = 0;
      }
      startTracePlayback();
    }
  });

  document.getElementById("btn-trace-next").addEventListener("click", () => {
    stopTracePlayback();
    if (hnswTraceData && currentTraceStep < hnswTraceData.steps.length - 1) {
      currentTraceStep++;
      renderHNSWCanvas(hnswGraphData);
    }
  });

  document.getElementById("btn-trace-prev").addEventListener("click", () => {
    stopTracePlayback();
    if (hnswTraceData && currentTraceStep > 0) {
      currentTraceStep--;
      renderHNSWCanvas(hnswGraphData);
    }
  });

  document.getElementById("hnsw-trace-speed").addEventListener("change", (e) => {
    traceSpeed = parseInt(e.target.value, 10);
    if (traceTimer) {
      startTracePlayback();
    }
  });

  document.getElementById("btn-trace-clear").addEventListener("click", () => {
    stopTracePlayback();
    hnswTraceData = null;
    currentTraceStep = -1;
    document.getElementById("hnsw-playback-controls").style.display = "none";
    document.getElementById("hnsw-telemetry-hud").style.display = "none";
    renderHNSWCanvas(hnswGraphData);
  });

  function renderHNSWCanvas(data) {
    if (!data) return;
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

    // Draw Base Edges
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
            ctx.strokeStyle = edge.level > 0 ? "rgba(251, 191, 36, 0.4)" : "rgba(0, 245, 155, 0.2)";
            ctx.lineWidth = edge.level > 0 ? 1.4 : 1.0;
          } else {
            const cx = (p1.x + p2.x) / 2 + 15;
            const cy = (p1.y + p2.y) / 2;
            ctx.quadraticCurveTo(cx, cy, p2.x, p2.y);
            ctx.strokeStyle = "rgba(244, 63, 94, 0.35)";
            ctx.lineWidth = 1.2;
          }
          ctx.stroke();
        }
      });
    }

    // Draw Traversed Laser Hops if Trace is Active
    if (hnswTraceData && currentTraceStep >= 0) {
      const steps = hnswTraceData.steps;
      for (let s = 0; s <= currentTraceStep && s < steps.length; s++) {
        const step = steps[s];
        if (step.from_node && step.to_node) {
          const fromPos = nodePositions[step.from_node];
          const toPos = nodePositions[step.to_node];
          if (fromPos && toPos) {
            ctx.beginPath();
            ctx.moveTo(fromPos.x, fromPos.y);
            ctx.lineTo(toPos.x, toPos.y);

            // Glowing neon laser line
            ctx.strokeStyle = "#00f59b";
            ctx.shadowColor = "#00f59b";
            ctx.shadowBlur = 12;
            ctx.lineWidth = s === currentTraceStep ? 3.5 : 2.2;
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        }
      }

      // Update Telemetry HUD
      const activeStep = steps[currentTraceStep];
      if (activeStep) {
        document.getElementById("hud-layer").textContent = `Layer: ${activeStep.layer}`;
        document.getElementById("hud-node").textContent = `Current Node: ${activeStep.to_node}`;
        document.getElementById("hud-dist").textContent = `Dist: ${activeStep.distance.toFixed(3)}`;
        document.getElementById("hud-desc").textContent = activeStep.description;
        document.getElementById("hnsw-step-label").textContent = `Step ${currentTraceStep + 1} / ${steps.length}`;
      }
    }

    // Draw Nodes
    data.nodes.forEach((node) => {
      const pos = nodePositions[node.id];
      if (!pos) return;

      const isEnterNode = node.id === data.enter_node;
      const isCurrentTraceTarget = (
        hnswTraceData &&
        currentTraceStep >= 0 &&
        hnswTraceData.steps[currentTraceStep]?.to_node === node.id
      );

      const radius = isCurrentTraceTarget ? 11 : (isEnterNode ? 9 : (pos.level > 0 ? 7 : 5.5));

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

      if (isCurrentTraceTarget) {
        ctx.fillStyle = "#00f59b";
        ctx.shadowColor = "#00f59b";
        ctx.shadowBlur = 20;
      } else if (isEnterNode) {
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

      ctx.strokeStyle = isCurrentTraceTarget ? "#08090b" : "#ffffff";
      ctx.lineWidth = isCurrentTraceTarget ? 3 : 1.5;
      ctx.stroke();

      // Outer animated ring for current target
      if (isCurrentTraceTarget) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 245, 155, 0.7)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.fillStyle = isCurrentTraceTarget ? "#00f59b" : "#94a3b8";
      ctx.font = isCurrentTraceTarget ? "bold 11px 'Plus Jakarta Sans'" : "10px 'Plus Jakarta Sans'";
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
  // Tab 4: 2D Semantic Embedding Space Projector Logic
  // =========================================================================
  let projData = null;
  let projQueryData = null;
  let projPan = { x: 0, y: 0 };
  let projZoom = 1.0;
  let isDraggingProj = false;
  let dragStartProj = { x: 0, y: 0 };
  let hoveredPointProj = null;
  let activeClusterFilterProj = null;
  let projAnimFrame = null;
  let queryPulsePhase = 0;

  async function fetchAndRenderProjector() {
    try {
      const res = await fetch(`/api/visualization/embedding-space?t=${Date.now()}`);
      const data = await res.json();
      projData = data;

      // Update Telemetry Pills
      if (data.explained_variance) {
        document.getElementById("proj-badge-variance").textContent =
          `PCA Variance: PC1 ${data.explained_variance[0]}% | PC2 ${data.explained_variance[1]}%`;
      }
      document.getElementById("proj-badge-count").textContent = `Vectors: ${data.total_points || 0}`;

      // Populate Legend Bar
      const legendContainer = document.getElementById("proj-legend-list");
      legendContainer.innerHTML = "";

      const allPill = document.createElement("div");
      allPill.className = `legend-pill ${activeClusterFilterProj === null ? "active" : ""}`;
      allPill.innerHTML = `<span>All Clusters</span> <span class="legend-pill-count">${data.total_points || 0}</span>`;
      allPill.addEventListener("click", () => {
        activeClusterFilterProj = null;
        document.querySelectorAll(".legend-pill").forEach((p) => p.classList.remove("active"));
        allPill.classList.add("active");
        renderProjectorCanvas();
      });
      legendContainer.appendChild(allPill);

      (data.clusters || []).forEach((c) => {
        const pill = document.createElement("div");
        pill.className = `legend-pill ${activeClusterFilterProj === c.name ? "active" : ""}`;
        pill.innerHTML = `
          <span class="legend-dot-proj" style="background:${c.color}; color:${c.color};"></span>
          <span>${c.name}</span>
          <span class="legend-pill-count">${c.count}</span>
        `;
        pill.addEventListener("click", () => {
          if (activeClusterFilterProj === c.name) {
            activeClusterFilterProj = null;
            document.querySelectorAll(".legend-pill").forEach((p) => p.classList.remove("active"));
            allPill.classList.add("active");
          } else {
            activeClusterFilterProj = c.name;
            document.querySelectorAll(".legend-pill").forEach((p) => p.classList.remove("active"));
            pill.classList.add("active");
          }
          renderProjectorCanvas();
        });
        legendContainer.appendChild(pill);
      });

      renderProjectorCanvas();
      startProjectorAnimation();
    } catch (e) {
      console.warn("Projector fetch error:", e);
    }
  }

  function startProjectorAnimation() {
    if (projAnimFrame) cancelAnimationFrame(projAnimFrame);
    function loop() {
      queryPulsePhase = (queryPulsePhase + 0.05) % (Math.PI * 2);
      if (projQueryData) {
        renderProjectorCanvas();
      }
      projAnimFrame = requestAnimationFrame(loop);
    }
    projAnimFrame = requestAnimationFrame(loop);
  }

  function renderProjectorCanvas() {
    const canvas = document.getElementById("projector-canvas");
    if (!canvas || !projData) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    ctx.save();
    // Center point with pan and zoom transforms
    ctx.translate(width / 2 + projPan.x, height / 2 + projPan.y);
    ctx.scale(projZoom, projZoom);

    // 1. Draw Subtle Coordinate Grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1 / projZoom;
    const gridSize = 60;
    const ext = 800;
    for (let x = -ext; x <= ext; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, -ext);
      ctx.lineTo(x, ext);
      ctx.stroke();
    }
    for (let y = -ext; y <= ext; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(-ext, y);
      ctx.lineTo(ext, y);
      ctx.stroke();
    }

    // 2. Draw Axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1.2 / projZoom;
    ctx.beginPath();
    ctx.moveTo(-ext, 0);
    ctx.lineTo(ext, 0);
    ctx.moveTo(0, -ext);
    ctx.lineTo(0, ext);
    ctx.stroke();

    // 3. Draw Laser Rays from Query Point to Top-K Nearest Neighbors
    if (projQueryData && projQueryData.top_k) {
      const qx = projQueryData.query_2d.x;
      const qy = projQueryData.query_2d.y;

      const idToPoint = {};
      projData.points.forEach((p) => { idToPoint[p.id] = p; });

      projQueryData.top_k.forEach((res, rank) => {
        const targetPt = idToPoint[res.id];
        if (targetPt) {
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(targetPt.x, targetPt.y);

          ctx.strokeStyle = "rgba(0, 245, 155, 0.6)";
          ctx.shadowColor = "#00f59b";
          ctx.shadowBlur = 8;
          ctx.lineWidth = Math.max(1.2, (2.4 - rank * 0.3)) / projZoom;
          ctx.setLineDash([4 / projZoom, 4 / projZoom]);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.shadowBlur = 0;

          // Rank label at midpoint
          const midX = (qx + targetPt.x) / 2;
          const midY = (qy + targetPt.y) / 2;
          ctx.fillStyle = "rgba(10, 14, 20, 0.85)";
          ctx.fillRect(midX - 14 / projZoom, midY - 8 / projZoom, 28 / projZoom, 16 / projZoom);
          ctx.fillStyle = "#00f59b";
          ctx.font = `bold ${10 / projZoom}px 'JetBrains Mono'`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(`#${rank + 1}`, midX, midY);
        }
      });
    }

    // 4. Draw Vector Points
    projData.points.forEach((pt) => {
      const isFilteredOut = activeClusterFilterProj !== null && pt.cluster !== activeClusterFilterProj;
      const isHovered = hoveredPointProj && hoveredPointProj.id === pt.id;
      const isTopKMatch = projQueryData && projQueryData.top_k?.some((k) => k.id === pt.id);

      const baseRadius = isHovered ? 8 : (isTopKMatch ? 7 : 5);
      const radius = baseRadius / projZoom;

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);

      if (isFilteredOut) {
        ctx.fillStyle = "rgba(75, 85, 99, 0.25)";
        ctx.fill();
        return;
      }

      ctx.fillStyle = pt.color;
      if (isHovered || isTopKMatch) {
        ctx.shadowColor = pt.color;
        ctx.shadowBlur = 14;
      } else {
        ctx.shadowColor = pt.color;
        ctx.shadowBlur = 5;
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = isTopKMatch ? "#ffffff" : "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = (isTopKMatch ? 2 : 1) / projZoom;
      ctx.stroke();

      // Top-K Match Halo
      if (isTopKMatch) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, (radius + 4 / projZoom), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 245, 155, 0.8)";
        ctx.lineWidth = 1.5 / projZoom;
        ctx.stroke();
      }
    });

    // 5. Draw Animated Query Beacon Star
    if (projQueryData) {
      const qx = projQueryData.query_2d.x;
      const qy = projQueryData.query_2d.y;

      // Pulsing outer ripple rings
      const pulseSize = (14 + Math.sin(queryPulsePhase) * 6) / projZoom;
      ctx.beginPath();
      ctx.arc(qx, qy, pulseSize, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0, 245, 155, ${0.4 + Math.cos(queryPulsePhase) * 0.3})`;
      ctx.lineWidth = 2 / projZoom;
      ctx.stroke();

      // Inner pulsating core
      ctx.beginPath();
      ctx.arc(qx, qy, 8 / projZoom, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#00f59b";
      ctx.shadowBlur = 18;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Query Beacon Tag
      ctx.fillStyle = "#00f59b";
      ctx.font = `bold ${11 / projZoom}px 'Plus Jakarta Sans'`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(`Query: "${projQueryData.query.substring(0, 22)}..."`, qx, qy + (pulseSize + 4 / projZoom));
    }

    ctx.restore();
  }

  // Project Query Handler
  document.getElementById("btn-project-query").addEventListener("click", executeProjectQuery);
  document.getElementById("proj-query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      executeProjectQuery();
    }
  });

  async function executeProjectQuery() {
    const q = document.getElementById("proj-query-input").value.trim();
    if (!q) return;

    const btn = document.getElementById("btn-project-query");
    btn.disabled = true;
    btn.innerHTML = `<span>Projecting...</span>`;

    try {
      const res = await fetch("/api/visualization/project-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k: 5 }),
      });
      const data = await res.json();
      projQueryData = data;
      document.getElementById("btn-clear-proj-query").style.display = "inline-flex";

      // Smoothly pan towards query coordinates
      projPan.x = -data.query_2d.x * projZoom;
      projPan.y = -data.query_2d.y * projZoom;
      renderProjectorCanvas();
    } catch (err) {
      console.error("Query projection error:", err);
      alert("Failed to project query: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line></svg> <span>Project & Find Top-K</span>`;
    }
  }

  // Clear query button
  document.getElementById("btn-clear-proj-query").addEventListener("click", () => {
    projQueryData = null;
    document.getElementById("btn-clear-proj-query").style.display = "none";
    document.getElementById("proj-query-input").value = "";
    renderProjectorCanvas();
  });

  // Preset sample chips
  document.querySelectorAll(".sample-chip-sm").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.getElementById("proj-query-input").value = chip.getAttribute("data-q");
      executeProjectQuery();
    });
  });

  // Reset zoom button
  document.getElementById("btn-reset-proj-zoom").addEventListener("click", () => {
    projPan = { x: 0, y: 0 };
    projZoom = 1.0;
    renderProjectorCanvas();
  });

  // Recalculate PCA button
  document.getElementById("btn-refresh-projector").addEventListener("click", fetchAndRenderProjector);

  // Close details panel button
  document.getElementById("btn-close-proj-detail").addEventListener("click", () => {
    document.getElementById("proj-details-panel").style.display = "none";
  });

  // Canvas Mouse Pan, Zoom, and Hover Listeners
  const projCanvas = document.getElementById("projector-canvas");
  const projTooltip = document.getElementById("proj-hover-tooltip");

  if (projCanvas) {
    // Wheel to Zoom
    projCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.min(Math.max(projZoom * zoomFactor, 0.4), 6.0);

      const rect = projCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - projCanvas.width / 2;
      const mouseY = e.clientY - rect.top - projCanvas.height / 2;

      projPan.x -= mouseX * (newZoom / projZoom - 1);
      projPan.y -= mouseY * (newZoom / projZoom - 1);
      projZoom = newZoom;

      renderProjectorCanvas();
    }, { passive: false });

    // Drag to Pan
    projCanvas.addEventListener("mousedown", (e) => {
      if (e.button === 0) {
        isDraggingProj = true;
        dragStartProj = { x: e.clientX - projPan.x, y: e.clientY - projPan.y };
      }
    });

    window.addEventListener("mouseup", () => {
      isDraggingProj = false;
    });

    projCanvas.addEventListener("mousemove", (e) => {
      const rect = projCanvas.getBoundingClientRect();
      if (isDraggingProj) {
        projPan.x = e.clientX - dragStartProj.x;
        projPan.y = e.clientY - dragStartProj.y;
        renderProjectorCanvas();
        projTooltip.style.display = "none";
        return;
      }

      if (!projData || !projData.points) return;

      // Transform mouse to canvas coordinates
      const scaleX = projCanvas.width / rect.width;
      const scaleY = projCanvas.height / rect.height;
      const canvasMouseX = (e.clientX - rect.left) * scaleX;
      const canvasMouseY = (e.clientY - rect.top) * scaleY;

      // Transform into data eigenspace
      const dataX = (canvasMouseX - projCanvas.width / 2 - projPan.x) / projZoom;
      const dataY = (canvasMouseY - projCanvas.height / 2 - projPan.y) / projZoom;

      let closest = null;
      let minDistance = 14 / projZoom;

      projData.points.forEach((pt) => {
        if (activeClusterFilterProj !== null && pt.cluster !== activeClusterFilterProj) return;
        const dx = pt.x - dataX;
        const dy = pt.y - dataY;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < minDistance) {
          minDistance = d;
          closest = pt;
        }
      });

      if (closest) {
        hoveredPointProj = closest;
        projCanvas.style.cursor = "pointer";

        // Position floating tooltip
        projTooltip.style.display = "block";
        projTooltip.style.left = `${e.clientX - rect.left}px`;
        projTooltip.style.top = `${e.clientY - rect.top}px`;

        document.getElementById("tooltip-cluster").textContent = closest.cluster;
        document.getElementById("tooltip-cluster").style.color = closest.color;
        document.getElementById("tooltip-id").textContent = closest.id;
        document.getElementById("tooltip-title").textContent = closest.title;
        document.getElementById("tooltip-snippet").textContent = closest.snippet;

        renderProjectorCanvas();
      } else {
        if (hoveredPointProj) {
          hoveredPointProj = null;
          projCanvas.style.cursor = "grab";
          projTooltip.style.display = "none";
          renderProjectorCanvas();
        }
      }
    });

    // Click to Open Details Panel
    projCanvas.addEventListener("click", () => {
      if (hoveredPointProj) {
        const pt = hoveredPointProj;
        const panel = document.getElementById("proj-details-panel");
        panel.style.display = "block";

        document.getElementById("proj-detail-title").textContent = pt.title;
        document.getElementById("proj-detail-badge").textContent = pt.cluster;
        document.getElementById("proj-detail-badge").style.borderColor = pt.color;
        document.getElementById("proj-detail-badge").style.color = pt.color;

        const rawText = pt.meta?.text || pt.snippet || "No text available.";
        document.getElementById("proj-detail-text").textContent = rawText;

        const metaEl = document.getElementById("proj-detail-meta");
        metaEl.innerHTML = `
          <span><strong>ID:</strong> ${pt.id}</span>
          <span><strong>Source:</strong> ${pt.meta?.source || pt.cluster}</span>
          <span><strong>2D Coords:</strong> (${pt.x}, ${pt.y})</span>
          ${pt.meta?.year ? `<span><strong>Year:</strong> ${pt.meta.year}</span>` : ""}
          ${pt.meta?.section ? `<span><strong>Section:</strong> ${pt.meta.section}</span>` : ""}
        `;

        panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
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
