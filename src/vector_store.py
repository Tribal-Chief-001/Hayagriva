import os
from src.config import settings


def get_embeddings():
    """Initializes embeddings depending on active mode (Local vs. Cloud)."""
    if settings.is_cloud_mode:
        print("[VectorStore] Initializing Gemini Embeddings (Cloud Mode)")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
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
