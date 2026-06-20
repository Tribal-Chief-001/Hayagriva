# Roadmap: Unified Serverless RAG Deployment Guide

This document outlines the step-by-step roadmap to push **Hayagriva** cleanly to GitHub and deploy it serverlessly on Vercel.

By using our upgraded **In-Metadata Parent-Text Injection** architecture, the application is completely serverless. It stores parent context paragraphs directly inside the child chunk vector metadata. No files are written to Vercel's read-only disk at runtime, allowing users to upload and query *any* document dynamically.

---

## ── Phase 1: Local Pre-Indexing (Optional)

You can index your default set of documents locally before pushing to Git:

1.  Place your default PDFs/TXTs/MDs inside `data/documents/`.
2.  Run the ingestion pipeline locally:
    ```bash
    python -m src.ingest
    ```
3.  This writes vectors and parent text metadata directly into `chroma_db/`. You can commit `chroma_db/` to Git. Vercel will bundle it, and it will run read-only out-of-the-box.

---

## ── Phase 2: The Perfect GitHub Upload

To ensure your portfolio is clean, secure, and ready for recruiters:

1.  **Initialize Git:**
    ```bash
    git init
    ```
2.  **Verify Gitignore:**
    Ensure `.gitignore` is present. Run `git status` to verify `venv/` and `.env` are excluded.
3.  **Commit & Push:**
    ```bash
    git add .
    git commit -m "feat: implement unified serverless parent-child RAG with drag-and-drop uploads"
    git branch -M main
    git remote add origin https://github.com/your-username/Hayagriva.git
    git push -u origin main
    ```

---

## ── Phase 3: Setting Up Cloud Services

To support dynamic uploads at runtime on your live website, you must connect the app to a cloud vector store so it can save newly uploaded vectors.

1.  **Create a Qdrant Cloud Cluster (Free Tier):**
    *   Sign up at [Qdrant Cloud Console](https://cloud.qdrant.io/) (no credit card required).
    *   Click **Create Cluster**. Qdrant will spin up a hosted cluster in seconds.
    *   Copy the **API Key** generated for you.
    *   Copy the **Cluster URL** endpoint (e.g. `https://xxx-xxxx.aws.qdrant.io:6333`).
2.  **Get a Google Gemini API Key:**
    *   Sign up at [Google AI Studio](https://aistudio.google.com/).
    *   Generate a free API key to run LLM (`gemini-1.5-flash`) and embeddings (`models/text-embedding-004`).

---

## ── Phase 4: Vercel Deployment

1.  Log in to [Vercel](https://vercel.com) and import the `Hayagriva` repository.
2.  In the project settings under **Environment Variables**, add the following keys:
    *   `GEMINI_API_KEY`: Your Google Gemini API Key (Required for Cloud mode).
    *   `QDRANT_URL`: Your Qdrant Cloud endpoint URL (Required for dynamic uploads).
    *   `QDRANT_API_KEY`: Your Qdrant Cloud API key.
    *   *(Optional)* `COHERE_API_KEY`: If you want to use Cohere Rerank in the cloud.
3.  Click **Deploy**. Vercel will build the FastAPI backend using `vercel.json` and serve the static glassmorphic frontend.

---

## ── Phase 5: Verification & UI Health Check

1.  Open your deployed Vercel URL.
2.  Verify the **System State** widget displays **CLOUD MODE (GEMINI)** and connects to your cloud vector store.
3.  Drag and drop *any* PDF from your local machine into the **DRAG & DROP SCROLL** box.
4.  Once indexed, verify it appears in the **INDEXED TEXTS** catalog.
5.  Run a query about your uploaded document, verify the token-by-token stream, and click the generated footnote citation pill to slide open the annotation drawer showing the verbatim parent text segment retrieved directly from Qdrant Cloud!
