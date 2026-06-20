# Hayagriva: Hybrid Conversational RAG Oracle

Hayagriva is a premium, near-SOTA Retrieval-Augmented Generation (RAG) platform. It features a local-first architecture built with **FastAPI**, **LangChain**, and **ChromaDB**, designed to run completely locally on CPU/Ollama or deploy serverlessly to **Vercel** using Cloud APIs (Gemini + Qdrant Cloud).

The system features an **"Editorial Neo-Minimalism"** glassmorphic dashboard interface, utilizing elegant serif typography (*Cormorant Garamond*) and a sliding annotation drawer to display verbatim citations from indexed documents.

---

## ── Key Architectural Features

*   **Stage 1: Hybrid Retrieval:** Integrates **BM25 Sparse Retrieval** (for exact keyword/terminology matching) and **Dense Vector Embeddings** (Chroma/Qdrant using `all-MiniLM-L6-v2` for semantic concepts).
*   **Stage 2: Reciprocal Rank Fusion (RRF):** Blends dense and sparse retrieval ranks dynamically using the RRF algorithm, securing matches that naive semantic vector search misses.
*   **Stage 3: Parent-Document Retrieval:** Chunks are stored hierarchically: child nodes (200 characters) are embedded for high-precision search, but are swapped with their larger parent nodes (1000 characters) during retrieval, giving the LLM rich narrative context.
*   **Stage 4: Cross-Encoder Re-ranking:** Employs `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU to re-score candidate parent documents, mitigating the "lost-in-the-middle" context window dilution problem.
*   **Conversational Query Condensation:** Utilizes multi-turn chat history to rephrase pronoun-dependent follow-up questions into standalone queries before hitting retrieval.
*   **Real-time Streaming:** Leverages FastAPI and the Javascript Streams API to deliver token-by-token streaming via Server-Sent Events (SSE).

---

## ── System Diagram

```
[User Query]
     │
     ▼
[Stage 1: Hybrid Retrieval] ──► Dense Vector (Chroma) & Sparse Keyword (BM25)
     │
     ▼
[Stage 2: RRF Merging]       ──► Reciprocal Rank Fusion of child chunks (top 15)
     │
     ▼
[Stage 3: Parent Lookup]    ──► Swapping retrieved 200-char children with 1000-char parents
     │
     ▼
[Stage 4: Re-ranking]       ──► Re-scoring parent documents via local CPU Cross-Encoder
     │
     ▼
[Generation Stream]         ──► Chat memory rephrasing + FastAPI Server-Sent Events (SSE)
```

---

## ── File Structure

```
Hayagriva/
├── data/
│   ├── documents/              # Place PDF, TXT, or MD files here
│   └── store/                  # Local store for Parent Documents (JSON-based)
├── static/                     # Premium Front-End UI
│   ├── index.html              # Classical editorial dashboard
│   ├── style.css               # Obsidian & Bronze styling
│   └── app.js                  # Asynchronous SSE stream receiver
├── src/                        # Backend Source
│   ├── config.py               # Settings manager (reads environment)
│   ├── vector_store.py         # Embedding & database configurations
│   ├── reranker.py             # Local Cross-Encoder re-ranker
│   ├── ingest.py               # Incremental file scanning pipeline
│   └── api.py                  # FastAPI routing & SSE streams
├── main.py                     # Entry launcher
├── vercel.json                 # Vercel deployment routing configuration
└── requirements.txt            # Package dependencies
```

---

## ── Local Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Hayagriva.git
    cd Hayagriva
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ollama Configuration:**
    Make sure [Ollama](https://ollama.com/) is installed and running, then pull the model:
    ```bash
    ollama pull qwen3.5:2b
    ```

5.  **Run Ingestion:**
    Place your PDFs inside `data/documents/` and index them:
    ```bash
    python -m src.ingest
    ```

6.  **Start the Server:**
    ```bash
    python main.py
    ```
    Open **`http://localhost:8000`** in your browser.

---

## ── Vercel Serverless Deployment

To deploy this project to Vercel in **Cloud Mode** (which swaps local models for Cloud API calls to stay within Vercel's size limits):

1.  Create a project on [Vercel](https://vercel.com).
2.  Set the following **Environment Variables** in your Vercel Dashboard:
    *   `GEMINI_API_KEY`: Your Google Gemini API Key.
    *   *(Optional)* `COHERE_API_KEY`: To enable cloud-based re-ranking.
    *   *(Optional)* `QDRANT_URL` & `QDRANT_API_KEY`: To connect to Qdrant Cloud.
3.  Deploy the repository. Vercel will build the Python API functions automatically.
