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
                api_key=settings.QDRANT_API_KEY
            )
            collection_name = "hayagriva_child_chunks"
            
            # Check if collection exists; if not, create it automatically
            try:
                exists = client.collection_exists(collection_name)
            except AttributeError:
                try:
                    client.get_collection(collection_name)
                    exists = True
                except Exception:
                    exists = False
            
            if not exists:
                print(f"[VectorStore] Qdrant collection '{collection_name}' does not exist. Creating it...")
                try:
                    dummy_emb = embeddings.embed_query("test")
                    vector_size = len(dummy_emb)
                except Exception:
                    vector_size = 3072  # Default fallback for gemini-embedding-2
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                print(f"[VectorStore] Collection '{collection_name}' created successfully with dimension {vector_size}.")

            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
        except Exception as e:
            print(f"[VectorStore] WARNING: Failed to initialize Qdrant Cloud Vector Store ({e}). Falling back to dummy storage to prevent startup crash.")
            class DummyVectorStore:
                client = None
                collection_name = "hayagriva_child_chunks"
                def as_retriever(self, **kwargs):
                    class DummyRetriever:
                        def invoke(self, query):
                            print("[VectorStore] Retrieval skipped: Qdrant connection failed.")
                            return []
                    return DummyRetriever()
                def get(self):
                    return {"documents": [], "metadatas": []}
                def add_texts(self, texts, metadatas=None, **kwargs):
                    print("[VectorStore] Cannot add texts: Qdrant connection failed.")
                    return []
                def add_documents(self, documents, **kwargs):
                    raise ValueError("Qdrant connection failed. Check your QDRANT_URL and QDRANT_API_KEY environment variables in your Vercel project Settings.")
            return DummyVectorStore()
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
                def add_documents(self, documents, **kwargs):
                    raise ValueError("Qdrant credentials missing. Set QDRANT_URL and QDRANT_API_KEY environment variables in your Vercel project Settings.")
            return DummyVectorStore()

        print(f"[VectorStore] Initializing Local Chroma Vector Store at: {settings.CHROMA_DB_DIR}")
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name="hayagriva_child_chunks"
        )
