import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env if present
load_dotenv()

class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # Server configuration
    PORT: int = int(os.getenv("PORT", "8000"))

    # Cloud Config
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
    NEO4J_USERNAME: Optional[str] = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Directory paths
    CHROMA_DB_DIR: str = str(PROJECT_ROOT / "chroma_db")
    PARENT_STORE_DIR: str = str(PROJECT_ROOT / "data" / "store")
    DOCUMENTS_DIR: str = str(PROJECT_ROOT / "data" / "documents")

    # Model configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    OLLAMA_MODEL_NAME: str = "qwen3.5:2b"
    GEMINI_LLM_MODEL: str = "gemini-3.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"

    # Vector store settings
    QDRANT_COLLECTION: str = "hayagriva_child_chunks"
    # Dimension of gemini-embedding-2 output vectors
    EMBEDDING_DIM: int = 3072

    # Ingestion chunking
    PARENT_CHUNK_SIZE: int = 3000
    PARENT_CHUNK_OVERLAP: int = 300
    CHILD_CHUNK_SIZE: int = 1000
    CHILD_CHUNK_OVERLAP: int = 100

    @property
    def is_cloud_mode(self) -> bool:
        """Determines if we are running in cloud mode using Gemini."""
        return self.GEMINI_API_KEY is not None

    @property
    def is_qdrant_mode(self) -> bool:
        """Determines if we are running in cloud vector store mode using Qdrant."""
        return bool(self.QDRANT_URL and self.QDRANT_API_KEY)

    @property
    def is_graph_enabled(self) -> bool:
        """Determines if Neo4j graph features are enabled."""
        return bool(self.NEO4J_URI and self.NEO4J_USERNAME and self.NEO4J_PASSWORD)


# Instantiate settings
settings = Settings()

# Ensure local directories exist (no-op on Vercel's read-only filesystem)
try:
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.PARENT_STORE_DIR, exist_ok=True)
except OSError:
    pass
