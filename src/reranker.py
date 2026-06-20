import os
from src.config import settings


class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        self.mode = "local" if not settings.is_cloud_mode else "cloud"

        if self.mode == "local":
            print("[Reranker] Initialized in LOCAL mode. CrossEncoder loads on first use.")
        else:
            print("[Reranker] Initialized in CLOUD mode.")

    def _load_local_model(self):
        """Lazy-loads the local CrossEncoder to avoid startup time cost."""
        if self.model is None:
            print("[Reranker] Loading local CrossEncoder model...")
            from sentence_transformers import CrossEncoder as STCrossEncoder
            self.model = STCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            print("[Reranker] Local CrossEncoder ready.")

    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        if not documents:
            return []

        # ------------------------------------------------------------------
        # CLOUD MODE: Cohere Rerank API (if key present), else bypass
        # ------------------------------------------------------------------
        if self.mode == "cloud":
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                try:
                    print("[Reranker] Running Cohere Rerank (cloud)...")
                    import cohere
                    co = cohere.Client(cohere_api_key)
                    doc_texts = [doc.page_content for doc in documents]
                    response = co.rerank(
                        model="rerank-english-v3.0",   # latest stable model
                        query=query,
                        documents=doc_texts,
                        top_n=top_k
                    )
                    reranked = []
                    for result in response.results:
                        doc = documents[result.index]
                        doc.metadata["relevance_score"] = float(result.relevance_score)
                        reranked.append(doc)
                    return reranked
                except Exception as e:
                    print(f"[Reranker] Cohere Rerank failed (bypassing): {e}")
            else:
                print("[Reranker] COHERE_API_KEY not set. Bypassing rerank.")

            # Fallback: return top_k with descending dummy scores
            for i, doc in enumerate(documents[:top_k]):
                doc.metadata["relevance_score"] = round(1.0 - i * 0.1, 2)
            return documents[:top_k]

        # ------------------------------------------------------------------
        # LOCAL MODE: Sentence-Transformers CrossEncoder
        # ------------------------------------------------------------------
        try:
            self._load_local_model()
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)

            doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            reranked = []
            for doc, score in doc_scores[:top_k]:
                doc.metadata["relevance_score"] = float(score)
                reranked.append(doc)

            print(f"[Reranker] Local rerank: {len(documents)} → {len(reranked)} docs.")
            return reranked
        except Exception as e:
            print(f"[Reranker] Local rerank failed (bypassing): {e}")
            for i, doc in enumerate(documents[:top_k]):
                doc.metadata["relevance_score"] = round(1.0 - i * 0.1, 2)
            return documents[:top_k]
