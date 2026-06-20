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
    GEMINI_LLM_MODEL: str = "gemini-1.5-flash"  # or gemini-2.0-flash
    
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

# Ensure directories exist
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.PARENT_STORE_DIR, exist_ok=True)
