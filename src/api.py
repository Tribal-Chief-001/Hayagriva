import json
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import settings
from src.ingest import run_ingestion, get_ingestion_log, ingest_document_bytes
from src.rag_engine import RAGEngine

app = FastAPI(title="Hayagriva API", description="Hybrid RAG Knowledge Engine API")

# Add CORS Middleware to support development workflows
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the shared RAG engine
rag_engine = RAGEngine()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.get("/api/status")
def get_status():
    """Returns the operational status and mode of the RAG engine."""
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

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Streams RAG tokens and citation metadata using Server-Sent Events (SSE)."""
    async def sse_generator():
        for event in rag_engine.query(request.message, request.session_id):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles dynamic user file uploads, parsing and storing them in the vector database."""
    try:
        file_bytes = await file.read()
        result = ingest_document_bytes(file_bytes, file.filename)
        
        if result.get("status") == "success":
            # Refresh BM25 index with the newly added chunks
            print(f"[API] Ingestion complete for '{file.filename}'. Refreshing BM25 Retriever index...")
            rag_engine.refresh_bm25_retriever()
            
        return result
    except Exception as e:
        return {"status": "error", "message": f"Failed to upload document: {e}"}

@app.post("/api/ingest")
def trigger_ingestion():
    """Triggers the document scanning and ingestion process."""
    result = run_ingestion()
    if result.get("status") == "success" and result.get("ingested"):
        # Refresh BM25 index with the newly added chunks
        print("[API] Ingestion complete. Refreshing BM25 Retriever index...")
        rag_engine.refresh_bm25_retriever()
    return result

@app.get("/api/documents")
def list_documents():
    """Returns a list of all currently ingested documents and their hashes."""
    log = get_ingestion_log()
    return {
        "documents": [
            {"filename": fname, "hash": fhash[:10] + "..."} 
            for fname, fhash in log.items()
        ]
    }

# Mount static folder for frontend asset delivery
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    """Serves the main frontend page."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Hayagriva. Create static/index.html to view the web client."}
