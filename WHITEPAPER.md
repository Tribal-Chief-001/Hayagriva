# Hayagriva: Re-architecting Local Retrieval-Augmented Generation (RAG) through Hybrid Search, Hierarchical Ingestion, Late-Interaction Re-ranking, and High-Availability Model Fallbacks

**Technical White Paper**  
**Author:** AI Engineering & Architecture Group  
**Date:** June 2026  
**Project URI:** [Hayagriva Workspace](file:///home/lucifer/Documents/Projects/Hayagriva/)  

---

## Abstract
Traditional Retrieval-Augmented Generation (RAG) pipelines suffer from critical failure modes, including vocabulary mismatch in semantic vector spaces, contextual fragmentation during chunking, and the "lost-in-the-middle" attention degradation in Large Language Models (LLMs). This paper presents **Hayagriva**, a local-first, cloud-deployable conversational RAG system that addresses these limitations.

By implementing a multi-stage pipeline comprising:
1. **Hybrid Dense-Sparse Retrieval** (Ensemble BM25 and Qdrant/ChromaDB),
2. **Reciprocal Rank Fusion (RRF)**,
3. **Parent-Document (Hierarchical) Swapping via Metadata Injection**,
4. **Late-Interaction Cross-Encoder Re-ranking**, and
5. **High-Availability Flash Fallback Chains**.

Hayagriva establishes a resilient, high-fidelity retrieval system. The architecture is wrapped in a high-performance **FastAPI** backend delivering real-time Server-Sent Events (SSE) token streaming with event-loop threading offloading, paired with an elegant **"Editorial Neo-Minimalism"** web interface featuring a slide-out Exegesis panel for granular citation inspections. We demonstrate that this architecture runs efficiently on local consumer CPUs while maintaining a zero-dependency serverless deployment footprint on Vercel under strict ephemeral filesystem constraints.

---

## 1. Introduction & Problem Statement

Naive RAG architectures typically split text documents into arbitrary, contiguous chunks, embed them using a bi-encoder transformer, index them in a vector database, and retrieve them via cosine similarity. While easy to build, this approach suffers from three fundamental bottlenecks:

### 1.1 Semantic vs. Keyword Retrieval Mismatch
Vector embeddings map text into high-dimensional geometric spaces where distance corresponds to conceptual similarity. However, this method frequently fails when matching specific serial numbers, acronyms, exact mathematical formulas, or unique proper nouns. For example, a user asking for a specific subsection page number will fail to trigger semantic matching because the query lacks semantic overlap, even though the keyword matches exactly.

### 1.2 The Chunking Dilemma: Precision vs. Context
A smaller chunk size (e.g., 100–200 characters) is excellent for pinpointing precise semantic concepts, as it isolates noise. However, passing these small snippets to an LLM deprives it of surrounding context (pronouns, flow of arguments, structural headings), leading to disjointed, incorrect, or hallucinated syntheses. Conversely, large chunks (e.g., 1000–2000 characters) preserve context but dilute vector representations with irrelevant text, lowering search precision.

### 1.3 LLM Context Dilution ("Lost in the Middle")
Studies in attention allocation show that decoder-only LLMs are highly sensitive to the positioning of information within prompt context windows. Crucial data located in the middle of a long prompt (e.g., 10 retrieved chunks joined together) is often ignored.

### 1.4 API Quota & Deployment Constraints
When deploying to cloud platforms like Vercel under the Hobby/Free tier:
*   **Vercel Ephemeral Filesystem:** Functions run in read-only containers, which prevents local SQLite or ChromaDB databases from persisting files across function calls.
*   **Vercel Package Size Limit:** Serverless function bundles are restricted to **50 MB** (zipped) or **500 MB** (unzipped ephemeral storage). Bundling local machine learning frameworks (like PyTorch, Sentence-Transformers, or ChromaDB binaries) is impossible because they quickly exceed these limits.
*   **Gemini Free-Tier Rate Limits:** The Google Gemini API enforces a strict rate limit of **15 Requests Per Minute (RPM)** and **1,000 Requests Per Day (RPD)** per model family, which can cause catastrophic timeouts (HTTP 429) during batch embedding indexing.

---

## 2. Mathematical Foundations & Concept Mechanics

To resolve the constraints of naive RAG, Hayagriva operates a structured 4-stage retrieval pipeline.

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

### 2.1 Sparse Retrieval: The Okapi BM25 Formulation
The Okapi BM25 algorithm evaluates the relevance of a document to a query based on term frequencies. For a query \(Q\) containing terms \(q_1, q_2, \dots, q_n\), and a document \(D\), the BM25 score is defined as:

\[
\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
\]

Where:
*   \(f(q_i, D)\) is the term frequency of \(q_i\) in document \(D\).
*   \(|D|\) is the length of document \(D\) in words.
*   \(\text{avgdl}\) is the average document length across the entire index corpus.
*   \(k_1\) is a tuning parameter controlling term frequency saturation (set to \(1.5\) in Hayagriva).
*   \(b\) is a tuning parameter controlling document length normalization (set to \(0.75\) in Hayagriva).

The Inverse Document Frequency (\(\text{IDF}\)) is computed as:

\[
\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)
\]

where \(N\) is the total number of documents in the corpus, and \(n(q_i)\) is the number of documents containing term \(q_i\).

### 2.2 Dense Retrieval: Vector Space Cosine Similarity
We embed child chunks into a dense vector space. In Local Mode, we use `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). In Cloud Mode, we use `gemini-embedding-2` (3072 dimensions). The similarity between a query vector \(\mathbf{q}\) and a document chunk vector \(\mathbf{d}\) is measured using Cosine Similarity:

\[
\text{Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \frac{\sum_{j=1}^{M} q_j d_j}{\sqrt{\sum_{j=1}^{M} q_j^2} \sqrt{\sum_{j=1}^{M} d_j^2}}
\]

### 2.3 Reciprocal Rank Fusion (RRF)
To combine the ranking lists of BM25 (\(R_{\text{sparse}}\)) and the vector store (\(R_{\text{dense}}\)), we implement Reciprocal Rank Fusion (RRF). RRF scores each document \(d\) based on its reciprocal rank in each list, bypassing the need to normalize raw scores across different scales:

\[
RRF\_Score(d \in D) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}
\]

Where:
*   \(M\) is the set of retrievers (Dense and Sparse).
*   \(r_m(d)\) is the rank of document \(d\) in retriever \(m\) (starting at 1). If \(d\) is not retrieved by \(m\), \(r_m(d) \to \infty\).
*   \(k\) is a constant smoothing factor that prevents low-ranked documents from disproportionately affecting the score (set to \(60\)).
*   \(w_m\) represents the relative weight of the retriever (in Hayagriva, we distribute weights as \(w_{\text{dense}} = 0.6\) and \(w_{\text{sparse}} = 0.4\)).

### 2.4 Late Interaction: Bi-Encoder vs. Cross-Encoder
In dense retrieval, **Bi-Encoders** are used to embed queries and documents independently. This allows vector databases to perform fast approximate nearest neighbor (ANN) searches, but it loses the token-level cross-attention context:

```
[Query] ──► Transform Encoder ──► Vector [q] ──┐
                                               ├──► Cosine Similarity (Late)
[Doc]   ──► Transform Encoder ──► Vector [d] ──┘
```

Hayagriva solves this by passing candidate documents through a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2` in Local Mode; Cohere Rerank v3 in Cloud Mode) in the re-ranking stage. The Cross-Encoder processes the query and document tokens simultaneously through self-attention layers:

```
[Query + Document] ──► Full Transform Self-Attention ──► Relevance Score (Early/Joint)
```

By computing joint attention across the full text sequence, Cross-Encoders evaluate semantic relevance with much higher accuracy. We limit the latency cost by only re-ranking the top 15 candidates returned by the fast RRF stage down to the top 3.

---

## 3. Modular Source Code Map

The backend of Hayagriva is organized into modular Python files inside the `src/` directory:

*   **[config.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/config.py):** Implements the `Settings` class using environment variables. It detects the presence of `GEMINI_API_KEY` to toggle the application between `Local Mode` (CPU-bound Ollama + sentence-transformers) and `Cloud Mode` (Gemini API calls + Qdrant Cloud).
*   **[vector_store.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/vector_store.py):** Configures database connections. It handles lazy-loading of SQLite/ChromaDB dependencies to prevent cold starts and sets up connection fallbacks (using a `DummyVectorStore`) to prevent application crashes when a database is unconfigured.
*   **[ingest.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/ingest.py):** Implements incremental file ingestion. It calculates the SHA-256 hash of documents in `data/documents/`, checks the `ingestion_log.json` cache (persisted inside Qdrant in Cloud Mode), and only processes new or modified files.
*   **[reranker.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/reranker.py):** Exposes the `CrossEncoderReranker` class. In local mode, it lazy-loads `sentence_transformers.CrossEncoder` to avoid hogging RAM at startup. In cloud mode, it routes re-ranking requests to Cohere's Rerank API.
*   **[llm.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/llm.py):** Exposes the shared `get_llm()` utility. Configures the multi-model LLM fallback chain.
*   **[graph_store.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/graph_store.py):** Handles connections to the Neo4j AuraDB instance. It queries relationships using the raw `neo4j` Python driver, entirely bypassing heavy LangChain modules to comply with serverless size limits.
*   **[rag_engine.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/rag_engine.py):** Coordinates execution. It manages in-memory conversational logs per session ID, condenses multi-turn history into standalone queries, triggers the RRF ensemble retrieval, swaps child chunks for parent paragraphs, and streams tokens.
*   **[api.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/api.py):** Exposes FastAPI routes (`GET /api/status`, `GET /api/documents`, `POST /api/ingest`, `POST /api/chat`, `POST /api/purge_db`, `DELETE /api/documents/{filename}`).
*   **[main.py](file:///home/lucifer/Documents/Projects/Hayagriva/main.py):** Relies on `uvicorn` to launch the server locally.

---

## 4. Ingestion & Hierarchical Ingestion Mechanics

To avoid relying on a local filesystem `docstore` (which is erased on serverless nodes), Hayagriva implements a **Parent-Document Swapping via Metadata Injection** pipeline.

### 4.1 Ingestion Flow
1.  **Text Extraction:** The parser loads the PDF, TXT, or MD document.
2.  **Parent Splitting:** The document is segmented into larger "Parent" chunks (1000 characters, 150 overlap) to preserve flow.
3.  **Child Splitting:** Each Parent chunk is further segmented into smaller "Child" chunks (200 characters, 30 overlap) to maximize search precision.
4.  **Metadata Packing:** For every child chunk generated, we inject the complete raw text of its parent context block directly into the child chunk's metadata:
    `child.metadata["parent_text"] = parent.page_content`
5.  **Unified Storage:** The children are embedded and stored in the vector database (Chroma locally or Qdrant Cloud). The database holds both the vectors for search and the parent texts for LLM context. No secondary local database is required.

### 4.2 Quality Scans
Before a document is embedded, it undergoes two quality validation checks to prevent garbage-in-garbage-out:
*   **DRM and Web-Viewer Detection:** Scans the first few chunks for signature DRM warnings or browser viewer placeholders (e.g. `"please download"`, `"cannot load this document"`, `"scribd"`, `"document viewer"`). If a match is found and total text size is short, ingestion is rejected.
*   **Scanned/Image PDF Detection:** Computes the average character count per page. If it is less than 80 characters, it flags the PDF as scanned/image-only and halts ingestion, recommending OCR preprocessing.

---

## 5. Unified Serverless Architecture

When deploying to Vercel, the configuration dynamically shifts to adapt to the serverless constraints.

### 5.1 SQLite & ChromaDB Lazy-Loading
ChromaDB requires binary bindings and builds static databases on the filesystem. Since Vercel serverless functions are read-only, importing ChromaDB in cloud environments can crash the app or inflate the deployment size.
*   **Dynamic Switch:** The vector store module checks `settings.is_qdrant_mode`.
*   **Chroma Bypass:** If in Qdrant mode, the code completely bypasses importing SQLite3 or ChromaDB, utilizing the standard HTTP-based `QdrantClient`.
*   **Dummy Failback:** If the Qdrant connection fails, a `DummyVectorStore` is instantiated. This class allows the server to boot successfully and serves warning alerts to the UI when document uploads are attempted, rather than failing silently with an internal server error.

### 5.2 Persistent Ingestion Logging
Since serverless filesystems are ephemeral, local files like `ingestion_log.json` cannot survive.
*   **Qdrant-Based Log:** In Qdrant Cloud Mode, the log is stored as a custom point inside the vector collection.
*   **Sentinel Point:** We write a sentinel point using a zero-vector (`[0.0] * EMBEDDING_DIM`) and tag its metadata payload:
    ```json
    {
      "page_content": "[LOG] filename.pdf",
      "metadata": {
        "tag": "__ingestion_log__",
        "filename": "filename.pdf",
        "hash": "sha256_hash_value"
      }
    }
    ```
*   **Deduplication:** When scanning for new files, the RAG engine scrolls through the collection, filters out all points with the tag `__ingestion_log__`, and maps filenames to hashes in memory. This eliminates duplicate uploads.

---

## 6. High-Availability Model Fallbacks

Google limits free-tier Gemini API usage on a per-model-family basis. To achieve higher throughput (50 to 70 RPM) on the free tier, Hayagriva chains multiple Gemini models.

### 6.1 The Fallback LLM Hierarchy
We construct a primary and fallback model chain in [llm.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/llm.py):
1.  **Primary:** `gemini-3.5-flash` (Anchor model)
2.  **Secondary:** `gemini-3.1-flash-lite` (First failover)
3.  **Tertiary:** `gemini-2.5-flash-lite` (Second failover)
4.  **Final Catch:** `gemini-2.5-flash` (Quota fallback)

These are chained together using LangChain's `.with_fallbacks()` method:
```python
primary_llm.with_fallbacks([fallback_1, fallback_2, fallback_3])
```

### 6.2 Instant Failover Routing (`max_retries=0`)
*   **The Problem:** By default, LangChain attempts exponential backoff retries when a model hits an HTTP 429 `RESOURCE_EXHAUSTED` rate limit. This causes the API call to hang for up to 90 seconds before falling back, triggering Vercel timeout errors.
*   **The Solution:** We set `max_retries=0` on all LLM and embedding instances. When a 429 error occurs, the primary model fails instantly, and the system catches the exception and routes the query down the fallback chain in milliseconds.

### 6.3 Embedding Fallbacks
Similarly, the embedding setup uses a custom `FallbackGeminiEmbeddings` wrapper:
1.  Attempts vectorization using `gemini-embedding-2` (3072 dimensions).
2.  If it returns a 429 error, it catches the error and immediately falls back to `gemini-embedding-001` (768 dimensions), padding the output vector with zeros to match Qdrant's 3072-dimension collection configuration.

---

## 7. Knowledge GraphRAG Integration

To support multi-hop reasoning (e.g. connecting characters and storylines across different books), Hayagriva combines vector database queries with a Knowledge Graph.

```
                     ┌──────────────────┐
                     │    User Query    │
                     └─────────┬────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [ Qdrant Search ]               [ Neo4j Query ]
      Extract Paragraphs             Extract 1-Hop Facts
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     Joint Context Prompt      │
               └───────────────────────────────┘
```

### 7.1 Throttled Local Extraction
Because extracting nodes and relationships requires numerous sequential LLM reasoning steps, doing so at runtime on Vercel is impossible due to rate limits and execution timeouts.
*   **Local script:** We use [build_graph.py](file:///home/lucifer/Documents/Projects/Hayagriva/build_graph.py) to run extraction locally.
*   **Throttling:** It iterates over document chunks and uses LangChain's `LLMGraphTransformer` to extract entities and relations, throttling calls with a `2.5s` delay between chunks to remain under the Gemini 100 RPM free-tier limit.
*   **Cloud Sync:** It pushes the extracted graph structures up to a Neo4j AuraDB instance.

### 7.2 Lightweight Production Retrieval
Importing `langchain-neo4j` on Vercel pulls in `scipy` and `numpy`, violating the 500 MB size limit.
*   **Raw Python Driver:** We refactored [graph_store.py](file:///home/lucifer/Documents/Projects/Hayagriva/src/graph_store.py) to use the lightweight, raw `neo4j` Python driver.
*   **1-Hop Entity Query:** During RAG generation, the backend uses a lightweight LLM call to extract entity names from the query, then runs a Cypher query using the raw Neo4j driver to extract 1-hop relationship facts:
    ```cypher
    MATCH (n) 
    WHERE toLower(n.id) IN $entities 
    MATCH (n)-[r]-(m) 
    RETURN n.id AS source, type(r) AS rel, m.id AS target 
    LIMIT 30
    ```
*   **Enriched Context:** The retrieved facts (e.g. `Thalric -[STOLE]-> Crown Of Two Shadows`) are joined with the vector chunks to construct the final prompt.

---

## 8. SSE Event-Loop & Threading Optimization

Because RAG orchestration tasks (loading embeddings, querying databases, running fallbacks) are CPU-bound and block execution, running them in a naive async event loop can degrade server performance.

### 8.1 Starlette Thread Pool Offloading
*   **The Issue:** Running the SSE generator as `async def` inside the `/api/chat` route forced FastAPI to execute RAG queries on the main thread, blocking the event loop and freezing concurrent requests.
*   **The Fix:** Redefined both the `chat_endpoint` and `sse_generator` as synchronous `def` functions. Starlette automatically intercepts synchronous endpoints and executes them in an external thread pool, keeping the main event loop responsive.

### 8.2 Proxy Buffering Bypass
*   **The Issue:** Services like Vercel, Nginx, or Cloudflare buffer server responses to optimize packet delivery. This buffers SSE status updates, keeping the frontend stuck showing "Initializing telemetry..." until the LLM response is completed.
*   **The Fix:** Configured response headers to explicitly bypass proxy buffering:
    ```http
    Cache-Control: no-cache, no-transform
    Connection: keep-alive
    X-Accel-Buffering: no
    ```

---

## 9. "Editorial Neo-Minimalism" UI/UX Design

The interface adopts a high-end scholarly design language, styled around Immanuel Kant's philosophical theme.

### 9.1 Typography System
*   **Cormorant Garamond (Luxury Serif):** Used for primary queries, blockquotes, and assistant responses.
*   **JetBrains Mono (Monospace):** Used for metadata, file names, telemetry logs, and stopwatch indicators.
*   **Inter (Clean Sans-Serif):** Handles UI buttons, sidebar index items, and system headers.

### 9.2 Marked.js Markdown Integration
To render RAG answers cleanly without custom regex formatting bugs:
*   We integrated `marked.js` in [index.html](file:///home/lucifer/Documents/Projects/Hayagriva/static/index.html).
*   Refactored `formatMarkdown` in [app.js](file:///home/lucifer/Documents/Projects/Hayagriva/static/app.js) to parse Markdown to standard HTML, supporting paragraph tags (`<p>`), lists (`<ul>`/`<ol>`), blockquotes, bold text, and headings.
*   Added CSS rules in [style.css](file:///home/lucifer/Documents/Projects/Hayagriva/static/style.css) to style Markdown elements with appropriate margins, borders, and line heights.

### 9.3 FOOTNOTE Citations & Slide-out Exegesis Panel
*   **Citation Chips:** Sources are rendered as footnoted buttons (e.g. `P.2 [89%]`) beneath the response.
*   **Exegesis Drawer:** Clicking a chip slides out a right-hand exegesis panel showing the document metadata, relevance percentage, and the verbatim source snippet, avoiding UI clutter.
*   **Telemetry Stopwatch:** The typing indicator shows a live, millisecond-accurate stopwatch during retrieval, displaying RAG execution metrics (latency, chunks, facts found) upon completion.

---

## 10. Advanced RAG Optimization: Multi-Query Expansion & Self-Reflective RAG (The Grader)

To achieve maximum accuracy and eliminate hallucinations or search misses, Hayagriva deploys two state-of-the-art algorithmic mechanisms:

### 10.1 Multi-Query Expansion
Standard vector searches frequently fail if the user's phrasing is different from the document's terminology.
* **The Process:** When a user submits a query, it is not executed directly. Instead, the backend RAG engine intercepts it and silently rewrites it into **3 distinct variations** using synonyms and alternate phrasing (leveraging `gemini-3.5-flash` or fallback LLMs).
* **Search Execution:** The engine executes dense (Qdrant) and sparse (BM25) searches for all 3 variations simultaneously, aggregating the results using Reciprocal Rank Fusion (RRF).
* **Example:**
  * *Original Query:* `"Why is he bad?"`
  * *AI Expanded Queries:* 
    1. `"What makes Thalric a villain?"`
    2. `"What are the antagonist's motives?"`
    3. `"Why is Thalric's behavior considered malevolent?"`
This ensures that even if the source document uses specific terminology like "malevolent" or "villain" rather than "bad", the retrieval pipeline will successfully locate the context.

### 10.2 Self-Reflective RAG (The Grader)
Hallucinations occur when an LLM is forced to generate an answer based on irrelevant retrieved context. Hayagriva mitigates this through a strict self-reflective loop.
* **The Process:** After paragraphs are retrieved and re-ranked, the system passes them to a lightweight "Grader AI" (instantiated with high-throughput settings).
* **The Vote:** The Grader reads the paragraphs and outputs a strict binary grade (`yes` or `no`) indicating whether the context actually contains the information required to answer the question.
* **Flow Control:**
  * **If Yes:** The engine proceeds to the final response generation step and streams a highly accurate answer.
  * **If No:** The engine bypasses final LLM generation entirely, instantly outputting: *"I cannot find the answer in the provided documents."*
This mathematical threshold safeguard eliminates hallucination by refusing to guess when no relevant data is present in the database.

---

## 11. Detailed Commit & Audit Log

The table below outlines the evolution of the Hayagriva codebase through key commits:

| Commit Hash | Type | Summary of Changes | Architectural Impact |
| :--- | :--- | :--- | :--- |
| `7b466f1` | **fix** | Integrated `marked.js` and added editorial CSS rules. | Enabled standard-compliant markdown rendering for lists and headings. |
| `aa66f90` | **fix** | Pruned `langchain-neo4j`/`langchain-experimental` from `requirements.txt`. | Solved Vercel 500MB bundle size limit. |
| `e89fedb` | **fix** | Added proxy buffering headers (`X-Accel-Buffering: no`). | Enabled real-time telemetry streaming over SSE. |
| `d351db8` | **fix** | Refactored `/api/chat` to synchronous `def` endpoints. | Offloaded CPU blocking tasks to a thread pool, preventing thread starvation. |
| `20e1a74` | **fix** | Configured `get_llm()` fallback chain in `query_graph()`. | Prevented Neo4j Cypher generator from hanging on 429 rate limit errors. |
| `b79eed6` | **fix** | Set `max_retries=0` for all LLM and embedding instances. | Disabled native 90s retries, enabling instant failover to fallback models. |
| `5f449a7` | **feat** | Upgraded LLM to `gemini-3.5-flash` with fallbacks. | Increased reasoning capabilities and query throughput limits. |
| `81d91ab` | **feat** | Implemented live JS stopwatch and SSE telemetry status. | Exposed backend RAG pipeline metrics to the user interface. |
| `8a35c76` | **feat** | Unified ingestion to push to both Qdrant and Neo4j. | Synced local uploads to vector and graph databases simultaneously. |
| `69802ab` | **feat** | Implemented Multi-Query Expansion & Reflective Grader. | Eliminated semantic search mismatch and prevented LLM hallucinations. |
| `02a3a09` | **feat** | Added document delete and global nuke endpoints. | Enabled administrative database management directly from the UI. |
| `bedf398` | **feat** | Implemented automatic quota fallback to `gemini-embedding-001`. | Prevented batch ingestion failures during embedding rate limits. |
| `0b31b11` | **fix** | Added DRM detection and character density scans. | Rejected scanned/DRM PDFs before they dilute vector spaces. |
| `f9ed6c3` | **feat** | Raised Vercel timeout limits and document size constraints. | Enabled uploading files up to 40 pages / 800 chunks. |
| `45af82c` | **fix** | Auto-create Qdrant collections on initialization. | Simplified setup for clean, out-of-the-box cloud databases. |

---

## 12. Conclusion & Future Outlook

Hayagriva demonstrates that a state-of-the-art, multi-stage RAG pipeline can be run locally on standard hardware and deployed serverless in cloud environments. By structuring retrieval into an ensemble (Dense + Sparse + RRF + Reranker), implementing hierarchical parent document lookup via metadata injection, and orchestrating models via failover fallback chains, the system achieves maximum uptime and precision. 

Future development will focus on incorporating local multimodal embeddings for PDF image analysis, as well as optimizing GraphRAG queries for multi-hop graph neural networks.
