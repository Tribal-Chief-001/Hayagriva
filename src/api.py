import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import settings
from src.ingest import run_ingestion, get_ingestion_log, ingest_document_bytes
from src.rag_engine import RAGEngine

app = FastAPI(title="Hayagriva API", description="Hybrid RAG Knowledge Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared RAG engine instance
rag_engine = RAGEngine()


class ChatRequest(BaseModel):
    message: str
    session_id: str


# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    """Returns the operational status and configuration of the RAG engine."""
    return {
        "status": "online",
        "mode": "Cloud Mode (Gemini)" if settings.is_cloud_mode else "Local Mode (Ollama)",
        "config": {
            "is_cloud_mode": settings.is_cloud_mode,
            "is_qdrant_mode": settings.is_qdrant_mode,
            "llm_model": settings.GEMINI_LLM_MODEL if settings.is_cloud_mode else settings.OLLAMA_MODEL_NAME,
            "embeddings": f"{settings.GEMINI_EMBEDDING_MODEL} (Gemini)" if settings.is_cloud_mode else settings.EMBEDDING_MODEL_NAME,
            "vector_store": "Qdrant Cloud" if settings.is_qdrant_mode else "Chroma DB (Local)"
        }
    }


# ---------------------------------------------------------------------------
# Chat (streaming SSE)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Streams RAG tokens and citation metadata using Server-Sent Events."""
    async def sse_generator():
        try:
            for event in rag_engine.query(request.message, request.session_id):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except Exception as e:
            yield f"event: token\ndata: {json.dumps(f'[Server error: {e}]')}\n\n"
            yield "event: done\ndata: \"\"\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Document Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF, TXT, or MD file upload, chunks it, and indexes it into
    the vector store. Returns JSON with chunk count on success.
    """
    # Validate extension before reading bytes
    allowed = {".pdf", ".txt", ".md"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unsupported file type '{suffix}'. Upload a .pdf, .txt, or .md file."}
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "The uploaded file is empty."}
            )

        result = ingest_document_bytes(file_bytes, file.filename)

        if result.get("status") == "success":
            print(f"[API] '{file.filename}' ingested. Refreshing BM25 index...")
            rag_engine.refresh_bm25_retriever()

        return result

    except ValueError as e:
        # Known user-facing errors (bad file, empty PDF, wrong credentials…)
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        # Unexpected server errors
        print(f"[API] Upload error for '{file.filename}': {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error during ingestion: {e}"}
        )


# ---------------------------------------------------------------------------
# Directory Ingest (local / CLI trigger)
# ---------------------------------------------------------------------------

@app.post("/api/ingest")
def trigger_ingestion():
    """Scans the documents directory and ingests any new/modified files."""
    try:
        result = run_ingestion()
        if result.get("ingested"):
            print("[API] Directory ingest done. Refreshing BM25...")
            rag_engine.refresh_bm25_retriever()
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ---------------------------------------------------------------------------
# Indexed Documents List
# ---------------------------------------------------------------------------

@app.get("/api/documents")
def list_documents():
    """Returns a list of all ingested documents."""
    try:
        log = get_ingestion_log()
        return {
            "documents": [
                {"filename": fname, "hash": fhash[:10] + "..."}
                for fname, fhash in log.items()
            ]
        }
    except Exception as e:
        return {"documents": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Static Frontend
# ---------------------------------------------------------------------------

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    """Serves the main frontend page."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Hayagriva. Place static/index.html to enable the web client."}
