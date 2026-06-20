import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # Server configuration
    PORT: int = int(os.getenv("PORT", "8000"))

    # API Keys & Cloud settings
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
    QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None

    # Directory paths
    CHROMA_DB_DIR: str = str(PROJECT_ROOT / "chroma_db")
    PARENT_STORE_DIR: str = str(PROJECT_ROOT / "data" / "store")
    DOCUMENTS_DIR: str = str(PROJECT_ROOT / "data" / "documents")

    # Model configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_MODEL_NAME: str = "qwen3.5:2b"
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"

    # Vector store settings
    QDRANT_COLLECTION: str = "hayagriva_child_chunks"
    # Dimension of gemini-embedding-2 output vectors
    EMBEDDING_DIM: int = 3072

    # Ingestion chunking — kept small to minimise API calls on the free tier.
    # Parent: 800 chars  →  Child: 250 chars  →  ~3 children per parent.
    # A 15-page PDF ≈ 45 parents ≈ 135 children  →  5 batches of 30 → well under 100 RPM.
    PARENT_CHUNK_SIZE: int = 800
    PARENT_CHUNK_OVERLAP: int = 100
    CHILD_CHUNK_SIZE: int = 250
    CHILD_CHUNK_OVERLAP: int = 25

    @property
    def is_cloud_mode(self) -> bool:
        """Determines if we are running in cloud mode using Gemini."""
        return self.GEMINI_API_KEY is not None

    @property
    def is_qdrant_mode(self) -> bool:
        """Determines if we are running in cloud vector store mode using Qdrant."""
        return self.QDRANT_URL is not None and self.QDRANT_API_KEY is not None


# Instantiate settings
settings = Settings()

# Ensure local directories exist (no-op on Vercel's read-only filesystem)
try:
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.PARENT_STORE_DIR, exist_ok=True)
except OSError:
    pass
