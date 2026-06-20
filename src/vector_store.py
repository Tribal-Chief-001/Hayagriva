import os
from src.config import settings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class FallbackGeminiEmbeddings(Embeddings):
    """
    Wrapper around GoogleGenerativeAIEmbeddings that automatically falls back
    from gemini-embedding-2 to gemini-embedding-001 if the daily quota is exhausted.
    """
    def __init__(self, google_api_key: str, model_primary: str = "gemini-embedding-2", model_fallback: str = "gemini-embedding-001"):
        self.google_api_key = google_api_key
        self.model_primary = model_primary
        self.model_fallback = model_fallback
        
        # Initialize primary and fallback embedding models
        self.primary_emb = GoogleGenerativeAIEmbeddings(model=self.model_primary, google_api_key=self.google_api_key, max_retries=0)
        self.fallback_emb = GoogleGenerativeAIEmbeddings(model=self.model_fallback, google_api_key=self.google_api_key, max_retries=0)
        self._use_fallback = False

    def _embed_with_fallback(self, func_name: str, *args, **kwargs):
        if self._use_fallback:
            return getattr(self.fallback_emb, func_name)(*args, **kwargs)

        try:
            return getattr(self.primary_emb, func_name)(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                print(f"[Embeddings] Primary embedding model '{self.model_primary}' failed due to quota/rate limit: {e}. Falling back to '{self.model_fallback}'...")
                self._use_fallback = True
                try:
                    return getattr(self.fallback_emb, func_name)(*args, **kwargs)
                except Exception as fallback_err:
                    raise ValueError(
                        "Gemini API rate limit or daily quota exceeded on both primary and fallback models. "
                        "Free tier is limited to 15 requests/minute and 1,000 requests/day. "
                        "Please wait a minute before uploading again, or check your Gemini quota/billing settings."
                    ) from fallback_err
            else:
                raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_fallback("embed_documents", texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_fallback("embed_query", text)


def get_embeddings():
    """Initializes embeddings depending on active mode (Local vs. Cloud)."""
    if settings.is_cloud_mode:
        print("[VectorStore] Initializing Gemini Fallback Embeddings (Cloud Mode)")
        return FallbackGeminiEmbeddings(
            google_api_key=settings.GEMINI_API_KEY
        )
    else:
        print("[VectorStore] Initializing Hugging Face Sentence-Transformers Embeddings (Local Mode)")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )


def get_vector_store(embeddings):
    """Initializes and returns the child-chunk vector store (Chroma or Qdrant)."""
    if settings.is_qdrant_mode:
        print("[VectorStore] Initializing Qdrant Cloud Vector Store")
        from langchain_qdrant import QdrantVectorStore
        import qdrant_client
        from qdrant_client.http import models as qdrant_models
        try:
            client = qdrant_client.QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=8  # Stay within Vercel's 10s limit
            )
            collection_name = settings.QDRANT_COLLECTION

            # Auto-create the collection if it does not exist
            try:
                exists = client.collection_exists(collection_name)
            except AttributeError:
                try:
                    client.get_collection(collection_name)
                    exists = True
                except Exception:
                    exists = False

            if not exists:
                print(f"[VectorStore] Collection '{collection_name}' not found. Creating with dim={settings.EMBEDDING_DIM}...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                print(f"[VectorStore] Collection '{collection_name}' created successfully.")

            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
        except Exception as e:
            print(f"[VectorStore] WARNING: Qdrant init failed ({e}). Using DummyVectorStore.")
            return _make_dummy("Qdrant connection failed. Check QDRANT_URL and QDRANT_API_KEY in Vercel Settings.")
    else:
        if os.getenv("VERCEL") == "1":
            print("[VectorStore] WARNING: Running on Vercel without Qdrant credentials. Uploads disabled.")
            return _make_dummy("Qdrant credentials missing. Set QDRANT_URL and QDRANT_API_KEY in Vercel Settings.")

        print(f"[VectorStore] Initializing Local Chroma Vector Store at: {settings.CHROMA_DB_DIR}")
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=settings.QDRANT_COLLECTION
        )


def _make_dummy(error_message: str):
    """Returns a DummyVectorStore that safely no-ops retrieval and raises on writes."""
    class DummyVectorStore:
        client = None
        collection_name = settings.QDRANT_COLLECTION

        def as_retriever(self, **kwargs):
            class DummyRetriever:
                def invoke(self, query):
                    print("[VectorStore] Retrieval skipped: vector store is inactive.")
                    return []
            return DummyRetriever()

        def get(self):
            return {"documents": [], "metadatas": []}

        def add_texts(self, texts, metadatas=None, **kwargs):
            print("[VectorStore] add_texts: vector store is inactive.")
            return []

        def add_documents(self, documents, **kwargs):
            raise ValueError(error_message)

    return DummyVectorStore()
