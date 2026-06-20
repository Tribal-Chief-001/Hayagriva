import os
import tempfile
import hashlib
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.vector_store import get_embeddings, get_vector_store

INGESTION_LOG_PATH = Path(settings.PARENT_STORE_DIR) / "ingestion_log.json"

def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA-256 hash of a file to detect changes."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_ingestion_log() -> dict:
    """Reads the ingestion log containing filename-to-hash mappings."""
    if INGESTION_LOG_PATH.exists():
        try:
            with open(INGESTION_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Ingest] Error reading ingestion log: {e}")
            return {}
    return {}

def save_ingestion_log(log: dict):
    """Saves the ingestion log containing filename-to-hash mappings."""
    try:
        # Ensure parent directories exist
        os.makedirs(INGESTION_LOG_PATH.parent, exist_ok=True)
        with open(INGESTION_LOG_PATH, "w") as f:
            json.dump(log, f, indent=4)
    except Exception as e:
        print(f"[Ingest] Error writing ingestion log: {e}")

def split_and_inject_parent(pages, parent_splitter, child_splitter) -> list:
    """Splits pages into parents, then parents into children, injecting parent text into children metadata."""
    # 1. Split pages into parent documents
    parent_docs = parent_splitter.split_documents(pages)
    
    child_docs = []
    for parent in parent_docs:
        # Split this parent into child chunks
        children = child_splitter.split_documents([parent])
        for child in children:
            # Inject parent text into the child chunk's metadata
            child.metadata["parent_text"] = parent.page_content
            # Keep original source metadata
            child.metadata["source"] = parent.metadata.get("source", "Unknown")
            child.metadata["page"] = parent.metadata.get("page", 1)
            child_docs.append(child)
            
    return child_docs

def ingest_document_bytes(file_bytes: bytes, filename: str) -> dict:
    """Processes uploaded document bytes in-memory, splits and uploads to vector store."""
    suffix = Path(filename).suffix.lower()
    
    # Save the bytes to a temp file in /tmp (safe for Vercel/Serverless)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
    try:
        # Load documents based on file type
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")
            
        pages = loader.load()
        
        # Enrich metadata
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            if "page" not in page.metadata:
                page.metadata["page"] = i + 1
                
        # Splitters
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        
        child_docs = split_and_inject_parent(pages, parent_splitter, child_splitter)
        
        # Upload child chunks to vector store
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)
        vector_store.add_documents(child_docs)
        
        # Track hash log locally if file-writing is allowed (e.g. locally)
        try:
            log = get_ingestion_log()
            # Calculate hash from bytes
            hasher = hashlib.sha256()
            hasher.update(file_bytes)
            log[filename] = hasher.hexdigest()
            save_ingestion_log(log)
        except Exception as e:
            print(f"[Ingest] Skipping log update (read-only environment): {e}")
            
        print(f"[Ingest] In-memory upload of '{filename}' complete: {len(child_docs)} chunks added.")
        return {
            "status": "success",
            "filename": filename,
            "chunks": len(child_docs)
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_ingestion() -> dict:
    """Scans settings.DOCUMENTS_DIR and ingests new/modified documents incrementally."""
    doc_dir = Path(settings.DOCUMENTS_DIR)
    if not doc_dir.exists():
        os.makedirs(doc_dir, exist_ok=True)
        return {"status": "success", "message": "Documents directory created.", "ingested": [], "skipped": []}
    
    # Initialize RAG vector store
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    
    # Define parent and child splitters for Hierarchical / Parent-Child RAG
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    
    log = get_ingestion_log()
    ingested_files = []
    skipped_files = []
    
    # Scan files in documents folder
    supported_extensions = {".pdf", ".txt", ".md"}
    files_to_process = [p for p in doc_dir.iterdir() if p.suffix.lower() in supported_extensions]
    
    if not files_to_process:
        return {"status": "success", "message": "No documents found to process.", "ingested": [], "skipped": []}
    
    print(f"[Ingest] Found {len(files_to_process)} document(s) in {doc_dir}")
    
    for file_path in files_to_process:
        filename = file_path.name
        current_hash = calculate_sha256(file_path)
        
        # Check if file has already been ingested with the same hash
        if filename in log and log[filename] == current_hash:
            skipped_files.append(filename)
            print(f"[Ingest] Skipped '{filename}' (already ingested and unchanged)")
            continue
            
        print(f"[Ingest] Ingesting '{filename}'...")
        
        try:
            # Read bytes to feed into ingest_document_bytes
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                
            ingest_document_bytes(file_bytes, filename)
            ingested_files.append(filename)
            
        except Exception as e:
            print(f"[Ingest] Failed to ingest '{filename}': {e}")
            
    return {
        "status": "success",
        "ingested": ingested_files,
        "skipped": skipped_files
    }

if __name__ == "__main__":
    result = run_ingestion()
    print("\nIngestion Summary:", json.dumps(result, indent=2))
