<div align="center">
  
  # 🐴 H A Y A G R I V A
  
  ### *The Editorial, SOTA Hybrid Conversational RAG Oracle*
  
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![LangChain](https://img.shields.io/badge/LangChain-1C3C3A?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
  [![Qdrant](https://img.shields.io/badge/Qdrant-FF4B4B?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
  [![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
  [![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)

  *A local-first, cloud-serverless hybrid conversational knowledge engine engineered to resolve RAG constraints using reciprocal rank fusion, parent-document injection, knowledge GraphRAG, and high-availability model fallbacks.*

</div>

---

## ── Architectural Pipeline & Optimization Stages

Naive RAG pipelines suffer from word-matching limitations (semantic search missing exact keywords), context dilution (passing small chunks that lack surrounding context), and the "lost-in-the-middle" LLM attention degradation.

**Hayagriva** implements a SOTA **4-Stage Retrieval & Re-ranking Pipeline** alongside GraphRAG to solve these bottlenecks:

```
                  ┌────────────────────────┐
                  │       User Query       │
                  └───────────┬────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   [ DENSE SEMANTIC SEARCH ]      [ SPARSE KEYWORD SEARCH ]
    ChromaDB or Qdrant Cloud        Okapi BM25 Formulation
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │ Reciprocal Rank Fusion (RRF)│
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │    Parent-Document Swap     │
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │ Cross-Encoder Re-ranking   │
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  High-Availability LLM Chain│
               └─────────────────────────────┘
```

### The 4 Core Ingestion & Retrieval Stages
1. **Stage 1: Hybrid Search:** Runs a parallel search using **Okapi BM25** (for exact keyword/phrase recall) and **Dense Embeddings** (via `all-MiniLM-L6-v2` locally or `gemini-embedding-2` in cloud mode for conceptual semantic matching).
2. **Stage 2: Reciprocal Rank Fusion (RRF):** Merges Dense and Sparse ranking lists using RRF scoring to optimize results:
   \[
   RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{60 + r_m(d)}
   \]
3. **Stage 3: Parent-Document In-Metadata Injection:** Embeds small child chunks (200 chars) for precise retrieval, but stores the full parent paragraph (1000 chars) in the child chunk's metadata payload. Upon retrieval, the child chunk is swapped for its parent text, giving the LLM rich context without requiring a local filesystem document store.
4. **Stage 4: Cross-Encoder Re-ranking:** Re-scores parent documents using `ms-marco-MiniLM-L-6-v2` (Local) or Cohere Rerank v3 (Cloud) to evaluate token-level joint self-attention, mitigating LLM positioning bias ("lost-in-the-middle").

---

## ── Advanced RAG Orchestration Features

### 1. Multi-Query Expansion
To prevent vocabulary mismatches, user queries are intercepted and silently rewritten into **3 distinct variations** using synonyms and alternate phrasing (leveraging `gemini-3.5-flash` or fallback LLMs).
* *Example:* If you ask *"Why is he bad?"*, it expands to:
  1. *"What makes Thalric a villain?"*
  2. *"What are the antagonist's motives?"*
  3. *"Why is Thalric's behavior considered malevolent?"*
The engine searches Qdrant/Chroma and BM25 for all 3 variations simultaneously, capturing target paragraphs even if the phrasing differs from the user's initial question.

### 2. Self-Reflective RAG (The Grader)
To prevent hallucinations, retrieved paragraphs are sent to a strict **"Grader AI"** immediately after re-ranking.
* The Grader reads the context and votes **`yes`** or **`no`** on whether it contains the actual answer to the query.
* If **`yes`**, the system streams the generated response.
* If **`no`**, the system bypasses final LLM execution entirely and outputs a clean, safe message: *"I cannot find the answer in the provided documents."* This mathematically prevents the LLM from guessing based on irrelevant text.

### 3. Knowledge GraphRAG Integration
Hayagriva supports multi-hop reasoning by querying a knowledge graph alongside the vector database:
* **Decoupled Ingestion:** A local script (`build_graph.py`) uses `LLMGraphTransformer` to extract `(Node)-[Relation]->(Node)` facts from your books and syncs them up to a **Neo4j AuraDB** cloud instance.
* **Production Retrieval:** At runtime on Vercel, the RAG engine uses the raw `neo4j` Python driver to query 1-hop facts for entities extracted from the query. This bypasses heavy dependency imports, preventing function bundle bloat.

---

## ── Design Language: "Editorial Neo-Minimalism"

Hayagriva replaces default chat bubbles with an editorial, scholarly manuscript layout:
* **Typographic Hierarchy:** Combines the luxury serif **Cormorant Garamond** (for queries and narrative paragraphs) with **JetBrains Mono** (for metadata, telemetry, and footnote numbers) to create a clean, academic look.
* **Marked.js Integration:** The client-side utilizes `marked.js` to render headings (`h1`-`h6`), blockquotes, lists, and standard formatting dynamically as the stream flows.
* **Exegesis Panel:** Footnote citations (e.g. `[P.2 [89%]]`) slide open a side drawer displaying the verbatim text snippet used for the answer, alongside its source file name and relevance score.
* **Stopwatch & Telemetry Metrics:** Telemetry status messages (*"Scanning Qdrant..."*, *"Traversing Neo4j..."*) stream in real-time alongside a millisecond stopwatch. Upon completion, a metrics bar displays pipeline latency, vector chunks retrieved, and GraphRAG facts utilized.

---

## ── Directory Structure

```
Hayagriva/
├── data/
│   ├── documents/              # Directory for source files (.pdf, .txt, .md)
│   └── store/                  # Local ingestion log cache
├── static/                     # Glassmorphic Front-End
│   ├── index.html              # Classical editorial interface
│   ├── style.css               # Obsidian & Bronze styling
│   └── app.js                  # marked.js stream parser & stopwatch
├── src/                        # FastAPI Application
│   ├── config.py               # Settings manager (loads environment)
│   ├── vector_store.py         # Embedding wrappers & database configurations
│   ├── reranker.py             # Local/Cloud re-ranker wrappers
│   ├── llm.py                  # High-Availability Gemini fallback chain
│   ├── graph_store.py          # Raw Neo4j driver query pipeline
│   ├── ingest.py               # Incremental parent-child ingester
│   └── api.py                  # Synchronous FastAPI SSE thread endpoints
├── main.py                     # Entry point
├── build_graph.py              # Local Graph extraction script
├── vercel.json                 # Vercel deployment spec
├── requirements.txt            # Lightweight cloud requirements (Vercel)
└── requirements-local.txt      # Local requirements (torch/sentence-transformers)
```

---

## ── Complete Deployment Guide

### 1. Local Deployment (Local Mode)

Local mode runs entirely on your local CPU utilizing Ollama and Sentence-Transformers:

#### A. Install Requirements
```bash
git clone https://github.com/your-username/Hayagriva.git
cd Hayagriva

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-local.txt
```

#### B. Configure Ollama
Ensure [Ollama](https://ollama.com/) is installed and running, then pull the local LLM:
```bash
ollama pull qwen3.5:2b
```

#### C. Index Local Documents
Place your `.pdf`, `.txt`, or `.md` files in `data/documents/`, then run the ingester:
```bash
python -m src.ingest
```

#### D. Start the App
```bash
python main.py
```
Open **`http://localhost:8000`** in your browser.

---

### 2. Serverless Cloud Deployment (Vercel)

When a `GEMINI_API_KEY` is present in the environment variables, Hayagriva automatically boots in **Cloud Mode**, bypassing PyTorch/Chroma binary dependencies to remain within Vercel serverless size limits.

#### A. Provision Cloud Services
1. **Google AI Studio:** Get a free API Key for [Gemini](https://aistudio.google.com/).
2. **Qdrant Cloud:** Set up a free cluster on [Qdrant Cloud](https://qdrant.tech/) and obtain your Cluster URL and API Key.
3. **Neo4j AuraDB (Optional):** Deploy a free database instance on [Neo4j AuraDB](https://neo4j.com/cloud/platform/auradb/) for Knowledge Graph features.
4. **Cohere Rerank (Optional):** Obtain a free API key from [Cohere](https://cohere.com/) to enable advanced cloud re-ranking.

#### B. Setup Environment Variables on Vercel
Import your GitHub repository to Vercel and add the following **Environment Variables**:
* `GEMINI_API_KEY`: Your Gemini API Key.
* `QDRANT_URL`: Your Qdrant Cluster URL (e.g. `https://xxx.aws.qdrant.io:6333`).
* `QDRANT_API_KEY`: Your Qdrant API Key.
* `COHERE_API_KEY`: *(Optional)* Your Cohere API Key for re-ranking.
* `NEO4J_URI`: *(Optional)* Your Neo4j instance URI.
* `NEO4J_USERNAME`: *(Optional)* `neo4j`.
* `NEO4J_PASSWORD`: *(Optional)* Your Neo4j instance password.

#### C. Extract and Push the Knowledge Graph (Local Script)
Because extracting graph relationships requires dozens of consecutive LLM prompts, this process must be run locally:
```bash
# Set environment variables in a local .env file matching AuraDB credentials
python build_graph.py data/documents/your_book.pdf
```
This script indexes the vector chunks to Qdrant, extracts relationship triplets, and pushes them directly to Neo4j AuraDB.

---

## ── FAQ, Edge Cases & Troubleshooting

### Q1: Vercel deploy fails with "Bundle Size Exceeded (500 MB)"
* **The Cause:** Importing heavy deep-learning dependencies (such as `torch`, `sentence-transformers`, or binary SQLite bindings for `chromadb`) exceeds the serverless deployment storage limits.
* **The Fix:** Ensure you are using the separate requirements files. Vercel utilizes `requirements.txt` (which has been pruned of `langchain-neo4j`, `langchain-experimental`, `torch`, and `chromadb`). Make sure you do not add these packages back to `requirements.txt`. Local executions should utilize `requirements-local.txt` instead.

### Q2: Gemini API returns `429 RESOURCE_EXHAUSTED` (Rate Limiting)
* **The Cause:** Google's Free Tier Gemini API restricts usage to 15 Requests Per Minute (RPM) and 1,000 Requests Per Day (RPD).
* **The Fix:** Hayagriva incorporates high-availability strategies to mitigate this:
  1. **Failover Chain:** LLM queries automatically attempt `gemini-3.5-flash` -> `gemini-3.1-flash-lite` -> `gemini-2.5-flash-lite` -> `gemini-2.5-flash`.
  2. **Instant Failover:** Setting `max_retries=0` prevents the system from hanging on backoff loops, falling back to the next model in milliseconds.
  3. **Embedding Fallback:** If `gemini-embedding-2` returns a 429 during ingestion, it instantly falls back to `gemini-embedding-001`.

### Q3: Why does the typing indicator stay stuck on "Initializing telemetry..."?
* **The Cause:** Standard proxy services (Vercel, Nginx, Cloudflare) buffer server responses, which holds back Server-Sent Events (SSE) from flushing to the client in real-time.
* **The Fix:** Hayagriva passes explicit headers to bypass buffering. Ensure your proxy configuration respects:
  `X-Accel-Buffering: no` and `Cache-Control: no-cache, no-transform`.

### Q4: Document upload fails with errors on Vercel
* **The Cause:** Ephemeral filesystems on Vercel are read-only.
* **The Fix:** Hayagriva handles uploads by writing files temporarily to `/tmp` (which is writable on Vercel serverless containers) and deletes them immediately after ingestion.

### Q5: Upload restrictions on Vercel
* **The Cause:** Vercel functions have a maximum execution duration limit (10s on hobby tier, up to 300s on pro tier) and free-tier Gemini API limitations.
* **The Fix:** Ingesting large files (>40 pages or >90 chunks) on serverless will throw a descriptive error instructing the user to perform ingestion locally using `python -m src.ingest`.

### Q6: Scanned PDFs don't return answers
* **The Cause:** Scanned PDFs are images and do not contain extractable text.
* **The Fix:** The quality scan checker rejects documents with average characters per page < 80. You must OCR your PDF documents before indexing them.
