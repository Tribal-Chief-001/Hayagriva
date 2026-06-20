import os
from src.config import settings

def get_embeddings():
    """Initializes embeddings depending on active mode (Local vs. Cloud)."""
    if settings.is_cloud_mode:
        print("[VectorStore] Initializing Gemini Embeddings (Cloud Mode)")
        from langchain_google_genai import GoogleGenAIEmbeddings
        return GoogleGenAIEmbeddings(
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
        from langchain_community.vectorstores import Qdrant
        import qdrant_client
        client = qdrant_client.QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        return Qdrant(
            client=client,
            collection_name="hayagriva_child_chunks",
            embeddings=embeddings
        )
    else:
        if os.getenv("VERCEL") == "1":
            print("[VectorStore] WARNING: Running on Vercel without Qdrant Cloud credentials. Chroma is unsupported on Vercel.")
            class DummyVectorStore:
                client = None
                collection_name = "hayagriva_child_chunks"
                def as_retriever(self, **kwargs):
                    class DummyRetriever:
                        def invoke(self, query):
                            print("[VectorStore] Retrieval skipped: Qdrant credentials missing.")
                            return []
                    return DummyRetriever()
                def get(self):
                    return {"documents": [], "metadatas": []}
                def add_texts(self, texts, metadatas=None, **kwargs):
                    print("[VectorStore] Cannot add texts: Vector store is inactive on Vercel (missing Qdrant credentials).")
                    return []
            return DummyVectorStore()

        print(f"[VectorStore] Initializing Local Chroma Vector Store at: {settings.CHROMA_DB_DIR}")
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name="hayagriva_child_chunks"
        )
