import os
from src.config import settings

class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        self.mode = "local" if not settings.is_cloud_mode else "cloud"
        
        # If in local mode, we prepare the lazy loader
        if self.mode == "local":
            print("[Reranker] Initialized in LOCAL mode. Model will load on first rerank request.")
        else:
            print("[Reranker] Initialized in CLOUD mode.")

    def _load_local_model(self):
        """Lazy load the local CrossEncoder model to save memory/startup time."""
        if self.model is None:
            print("[Reranker] Lazy loading local CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
            from sentence_transformers import CrossEncoder as STCrossEncoder
            self.model = STCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            print("[Reranker] Local CrossEncoder model loaded successfully!")

    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        if not documents:
            return []

        # ----------------------------------------------------
        # CLOUD MODE: Use Cohere Rerank API if key exists, otherwise bypass
        # ----------------------------------------------------
        if self.mode == "cloud":
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                try:
                    print("[Reranker] Running Cloud Cohere Rerank...")
                    from langchain_community.llms import Cohere
                    # We can use cohere client directly or langchain community wrapper
                    import cohere
                    co = cohere.Client(cohere_api_key)
                    doc_texts = [doc.page_content for doc in documents]
                    response = co.rerank(
                        model="rerank-english-v2.0",
                        query=query,
                        documents=doc_texts,
                        top_n=top_k
                    )
                    
                    reranked_docs = []
                    for result in response.results:
                        idx = result.index
                        doc = documents[idx]
                        doc.metadata["relevance_score"] = float(result.relevance_score)
                        reranked_docs.append(doc)
                    return reranked_docs
                except Exception as e:
                    print(f"[Reranker] Cohere Rerank failed (falling back to raw docs): {e}")
            else:
                print("[Reranker] Cloud mode active but COHERE_API_KEY not found. Bypassing re-ranking.")
            
            # Fallback: return top_k documents directly, setting dummy scores
            for i, doc in enumerate(documents[:top_k]):
                doc.metadata["relevance_score"] = 1.0 - (i * 0.1)
            return documents[:top_k]

        # ----------------------------------------------------
        # LOCAL MODE: Run CPU Sentence-Transformers Cross-Encoder
        # ----------------------------------------------------
        try:
            self._load_local_model()
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)
            
            # Pair and sort
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = []
            for doc, score in doc_scores[:top_k]:
                doc.metadata["relevance_score"] = float(score)
                reranked_docs.append(doc)
                
            print(f"[Reranker] Local CrossEncoder reranked {len(documents)} docs down to {len(reranked_docs)}")
            return reranked_docs
        except Exception as e:
            print(f"[Reranker] Local rerank failed (falling back to raw docs): {e}")
            for i, doc in enumerate(documents[:top_k]):
                doc.metadata["relevance_score"] = 1.0 - (i * 0.1)
            return documents[:top_k]
