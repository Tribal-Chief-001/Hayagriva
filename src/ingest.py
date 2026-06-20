import os
import tempfile
import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings
from src.vector_store import get_embeddings, get_vector_store

# ---------------------------------------------------------------------------
# Ingestion Log — stored inside Qdrant as a tagged "__meta__" document so it
# survives Vercel's read-only / ephemeral filesystem.
# Locally we also mirror it to disk for fast reads.
# ---------------------------------------------------------------------------

_LOG_COLLECTION_TAG = "__ingestion_log__"
_LOCAL_LOG_PATH = Path(settings.PARENT_STORE_DIR) / "ingestion_log.json"


def _get_qdrant_client():
    """Returns a raw qdrant_client if in Qdrant mode, else None."""
    if not settings.is_qdrant_mode:
        return None
    try:
        import qdrant_client
        return qdrant_client.QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=8
        )
    except Exception as e:
        print(f"[Ingest] Warning: could not connect to Qdrant for log ops: {e}")
        return None


def get_ingestion_log() -> dict:
    """
    Returns the filename→sha256 map of all ingested documents.
    Reads from Qdrant payload first (cloud-safe), falls back to disk.
    """
    # Try Qdrant-based log — scroll ALL points and filter in Python
    # (avoids requiring a payload index on metadata.tag)
    if settings.is_qdrant_mode:
        try:
            client = _get_qdrant_client()
            if client:
                log = {}
                offset = None
                while True:
                    batch, next_offset = client.scroll(
                        collection_name=settings.QDRANT_COLLECTION,
                        limit=500,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    for point in batch:
                        payload = point.payload or {}
                        meta = payload.get("metadata", {})
                        if meta.get("tag") == _LOG_COLLECTION_TAG:
                            fname = meta.get("filename")
                            fhash = meta.get("hash")
                            if fname and fhash:
                                log[fname] = fhash
                    if next_offset is None:
                        break
                    offset = next_offset
                return log
        except Exception as e:
            print(f"[Ingest] Could not read Qdrant ingestion log: {e}")

    # Fallback: local disk (works locally, silently fails on Vercel)
    try:
        import json
        if _LOCAL_LOG_PATH.exists():
            with open(_LOCAL_LOG_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Ingest] Could not read local ingestion log: {e}")

    return {}


def save_ingestion_log(log: dict):
    """Persist log to disk (local only — Vercel uses Qdrant directly in upsert_log_entry)."""
    try:
        import json
        os.makedirs(_LOCAL_LOG_PATH.parent, exist_ok=True)
        with open(_LOCAL_LOG_PATH, "w") as f:
            json.dump(log, f, indent=4)
    except Exception as e:
        print(f"[Ingest] Skipping local log write (read-only env): {e}")


def upsert_log_entry(filename: str, file_hash: str):
    """
    Records a successfully ingested document in the Qdrant store as a tagged
    metadata-only record (zero vector, safe sentinel approach using a dummy payload).
    Also mirrors to disk when possible.
    """
    # Mirror to disk
    try:
        import json
        log = {}
        if _LOCAL_LOG_PATH.exists():
            with open(_LOCAL_LOG_PATH, "r") as f:
                log = json.load(f)
        log[filename] = file_hash
        save_ingestion_log(log)
    except Exception:
        pass

    # Upsert into Qdrant as a tagged dummy point
    if settings.is_qdrant_mode:
        try:
            import qdrant_client
            from qdrant_client.http import models as qmodels
            client = _get_qdrant_client()
            if client:
                # Use a deterministic ID based on filename so upsert is idempotent
                point_id = int(hashlib.md5(f"log:{filename}".encode()).hexdigest(), 16) % (2**63)
                client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=[
                        qmodels.PointStruct(
                            id=point_id,
                            vector=[0.0] * settings.EMBEDDING_DIM,
                            payload={
                                "page_content": f"[LOG] {filename}",
                                "metadata": {
                                    "tag": _LOG_COLLECTION_TAG,
                                    "filename": filename,
                                    "hash": file_hash
                                }
                            }
                        )
                    ]
                )
                print(f"[Ingest] Log entry upserted into Qdrant for '{filename}'.")
        except Exception as e:
            print(f"[Ingest] Warning: could not upsert log entry into Qdrant: {e}")


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def split_and_inject_parent(pages: list, parent_splitter, child_splitter) -> list:
    """
    Splits pages → parents → children, injecting each parent's raw text into
    every child chunk's metadata for later parent-doc retrieval.
    """
    parent_docs = parent_splitter.split_documents(pages)
    child_docs = []

    for parent in parent_docs:
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_text"] = parent.page_content
            child.metadata["source"] = parent.metadata.get("source", "Unknown")
            child.metadata["page"] = parent.metadata.get("page", 1)
            child_docs.append(child)

    return child_docs


# ---------------------------------------------------------------------------
# Core ingestion entry-points
# ---------------------------------------------------------------------------

def ingest_document_bytes(file_bytes: bytes, filename: str) -> dict:
    """
    Processes a document from raw bytes (for API uploads).
    Batches vector store writes in groups of 30 to stay within the
    Gemini free-tier 100 RPM limit without any blocking sleep.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload a .pdf, .txt, or .md file.")

    # Reject empty bytes before writing to /tmp
    if not file_bytes:
        raise ValueError(f"The uploaded file '{filename}' is empty (0 bytes).")

    # Write to /tmp (writable on Vercel)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Load — catch pypdf/IO errors and surface as clean ValueError
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        try:
            pages = loader.load()
        except Exception as load_err:
            raise ValueError(
                f"Could not parse '{filename}': {load_err}. "
                "If this is a PDF, ensure it is not password-protected or image-only (scanned)."
            ) from load_err

        if not pages:
            raise ValueError(f"Document '{filename}' appears to be empty or could not be parsed.")

        # Enrich metadata
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            if "page" not in page.metadata:
                page.metadata["page"] = i + 1

        # Chunk
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.PARENT_CHUNK_OVERLAP
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHILD_CHUNK_OVERLAP
        )
        child_docs = split_and_inject_parent(pages, parent_splitter, child_splitter)

        if not child_docs:
            raise ValueError(f"No text could be extracted from '{filename}'. Is the PDF scanned/image-based?")

        # --- Pre-flight quality checks ---

        # 1. Detect DRM-protected or browser-viewer PDFs (e.g. Scribd, Z-Library web viewer)
        #    These contain error messages instead of real book content.
        all_text = " ".join(d.page_content for d in child_docs).lower()
        drm_signals = ["please download", "cannot load this document", "document viewer",
                       "scribd", "buy the full", "preview only", "sample chapter"]
        matched_signal = next((s for s in drm_signals if s in all_text), None)
        if matched_signal and len(all_text) < 2000:
            raise ValueError(
                f"'{filename}' appears to be a DRM-protected or browser-preview PDF "
                f"(detected: '{matched_signal}'). Please download the actual PDF file "
                f"rather than saving it from a web viewer like Scribd or Z-Library."
            )

        # 2. Detect scanned / image-only PDFs (very little extractable text per page)
        total_text_len = sum(len(p.page_content.strip()) for p in pages)
        avg_chars_per_page = total_text_len / len(pages)
        if avg_chars_per_page < 80:
            raise ValueError(
                f"'{filename}' appears to be a scanned or image-based PDF "
                f"(avg {avg_chars_per_page:.0f} chars/page, expected >80). "
                f"Please use a text-based PDF or convert it using OCR first."
            )

        # Filter out log-tagged documents from Qdrant scroll so BM25 ignores them
        content_docs = [d for d in child_docs if d.metadata.get("tag") != _LOG_COLLECTION_TAG]

        on_vercel = os.getenv("VERCEL") == "1"
        VERCEL_MAX_PAGES  = 40   # 40 pages ≈ 800 chunks ≈ 40 embed calls ≈ 120s → fits in 300s
        VERCEL_MAX_CHUNKS = 800  # Hard chunk ceiling for any file type on Vercel

        # Enforce limits on Vercel to prevent hitting the 300s maxDuration
        if on_vercel:
            if len(pages) > VERCEL_MAX_PAGES:
                raise ValueError(
                    f"Document has {len(pages)} pages — Vercel can only process up to "
                    f"{VERCEL_MAX_PAGES} pages per upload. Please split the document "
                    f"or run ingestion locally: python -m src.ingest"
                )
            if len(content_docs) > VERCEL_MAX_CHUNKS:
                raise ValueError(
                    f"Document produces {len(content_docs)} chunks — Vercel limit is "
                    f"{VERCEL_MAX_CHUNKS}. Please split the document "
                    f"or run ingestion locally: python -m src.ingest"
                )

        print(f"[Ingest] '{filename}': {len(pages)} pages → {len(content_docs)} child chunks. Uploading in batches...")

        # --- Direct Qdrant upsert (bypasses LangChain's internal double-embedding) ---
        # We embed each batch ourselves, build PointStructs, and upsert directly.
        # This gives full control over rate limiting and avoids double-embedding.
        if settings.is_qdrant_mode:
            import qdrant_client as qc
            from qdrant_client.http import models as qmodels
            import uuid, time

            raw_client = qc.QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=8
            )
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            raw_emb = GoogleGenerativeAIEmbeddings(
                model=settings.GEMINI_EMBEDDING_MODEL,
                google_api_key=settings.GEMINI_API_KEY
            )

            BATCH_SIZE = 20
            total_batches = -(-len(content_docs) // BATCH_SIZE)

            for i in range(0, len(content_docs), BATCH_SIZE):
                batch = content_docs[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                print(f"[Ingest] Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

                texts = [d.page_content for d in batch]

                # Retry with exponential backoff on transient 429s (local mode only)
                retries, backoff = 5, 10.0
                while True:
                    try:
                        vectors = raw_emb.embed_documents(texts)
                        break
                    except Exception as emb_err:
                        err_str = str(emb_err)
                        if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and retries > 0 and not on_vercel:
                            print(f"[Ingest] Rate limit hit — retrying batch {batch_num} in {backoff:.0f}s ({retries} retries left)...")
                            time.sleep(backoff)
                            retries -= 1
                            backoff *= 2.0
                        else:
                            raise

                points = [
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                    )
                    for doc, vector in zip(batch, vectors)
                ]

                raw_client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
                print(f"[Ingest] Batch {batch_num}/{total_batches} upserted.")

                # Sleep between batches to stay under 100 RPM (Gemini free tier):
                #   Vercel: 1s sleep → 3s/batch (embed ~2s + sleep 1s) → 40 batches = 120s ≪ 300s
                #   Local:  20s sleep → handles large books without hitting rate limit
                if i + BATCH_SIZE < len(content_docs):
                    time.sleep(1 if on_vercel else 20)

        else:
            # Local Chroma path — LangChain handles embedding internally
            embeddings = get_embeddings()
            vector_store = get_vector_store(embeddings)
            for i in range(0, len(content_docs), 50):
                vector_store.add_documents(content_docs[i:i + 50])


        # Record in ingestion log
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        upsert_log_entry(filename, hasher.hexdigest())

        print(f"[Ingest] '{filename}' complete: {len(content_docs)} chunks indexed.")
        return {"status": "success", "filename": filename, "chunks": len(content_docs)}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_ingestion() -> dict:
    """Scans DOCUMENTS_DIR and ingests new/modified documents incrementally."""
    doc_dir = Path(settings.DOCUMENTS_DIR)
    if not doc_dir.exists():
        os.makedirs(doc_dir, exist_ok=True)
        return {"status": "success", "message": "Documents directory created.", "ingested": [], "skipped": []}

    log = get_ingestion_log()
    supported = {".pdf", ".txt", ".md"}
    files = [p for p in doc_dir.iterdir() if p.suffix.lower() in supported]

    if not files:
        return {"status": "success", "message": "No documents found.", "ingested": [], "skipped": []}

    print(f"[Ingest] Found {len(files)} document(s) in {doc_dir}")
    ingested, skipped = [], []

    for file_path in files:
        fname = file_path.name

        # SHA-256 deduplication
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha.update(block)
        current_hash = sha.hexdigest()

        if fname in log and log[fname] == current_hash:
            skipped.append(fname)
            print(f"[Ingest] Skipped '{fname}' (unchanged).")
            continue

        print(f"[Ingest] Ingesting '{fname}'...")
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            ingest_document_bytes(file_bytes, fname)
            ingested.append(fname)
        except Exception as e:
            print(f"[Ingest] Failed '{fname}': {e}")

    return {"status": "success", "ingested": ingested, "skipped": skipped}


if __name__ == "__main__":
    import json
    result = run_ingestion()
    print("\nIngestion Summary:", json.dumps(result, indent=2))
