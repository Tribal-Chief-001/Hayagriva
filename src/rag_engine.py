import os
import hashlib
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.config import settings
from src.vector_store import get_embeddings, get_vector_store
from src.reranker import CrossEncoderReranker

_LOG_TAG = "__ingestion_log__"


class RAGEngine:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store = get_vector_store(self.embeddings)
        self.reranker = CrossEncoderReranker()

        # Local session-based memory
        # Format: { session_id: [ {"role": "user"/"assistant", "content": str} ] }
        self.session_histories = {}

        # BM25 retriever — built lazily from vector store contents
        self.bm25_retriever = None
        self.refresh_bm25_retriever()

    def refresh_bm25_retriever(self):
        """Loads all content child chunks from the vector store and builds the BM25 index."""
        try:
            documents = []

            # Qdrant path — use scroll API (no .get())
            if (hasattr(self.vector_store, "client")
                    and hasattr(self.vector_store, "collection_name")
                    and self.vector_store.client is not None):

                print(f"[RAGEngine] Scrolling Qdrant '{self.vector_store.collection_name}' for BM25...")
                offset = None
                while True:
                    batch, next_offset = self.vector_store.client.scroll(
                        collection_name=self.vector_store.collection_name,
                        limit=500,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    for point in batch:
                        payload = point.payload or {}
                        meta = payload.get("metadata", {})
                        # Skip ingestion-log sentinel records
                        if meta.get("tag") == _LOG_TAG:
                            continue
                        text = payload.get("page_content", "")
                        if text:
                            documents.append(Document(page_content=text, metadata=meta))
                    if next_offset is None:
                        break
                    offset = next_offset

            # Chroma path
            elif hasattr(self.vector_store, "get"):
                data = self.vector_store.get()
                if data and data.get("documents"):
                    for i, text in enumerate(data["documents"]):
                        meta = (data["metadatas"] or [{}])[i]
                        if meta.get("tag") == _LOG_TAG:
                            continue
                        documents.append(Document(page_content=text, metadata=meta))

            if not documents:
                print("[RAGEngine] No content documents found. BM25 inactive.")
                self.bm25_retriever = None
                return

            print(f"[RAGEngine] Building BM25Retriever with {len(documents)} chunks.")
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 10

        except Exception as e:
            print(f"[RAGEngine] BM25 build failed: {e}")
            self.bm25_retriever = None

    def get_llm(self):
        """Returns the appropriate chat model (Gemini for cloud, Ollama for local)."""
        if settings.is_cloud_mode:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_LLM_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.2
            )
        else:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=settings.OLLAMA_MODEL_NAME, temperature=0.2)

    def _resolve_parents(self, child_docs: list) -> list:
        """Extracts parent text from child chunk metadata, deduplicates by content hash."""
        parent_docs = []
        seen = set()

        for child in child_docs:
            parent_text = child.metadata.get("parent_text") or child.page_content
            text_hash = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                parent_docs.append(Document(
                    page_content=parent_text,
                    metadata={
                        "source": child.metadata.get("source", "Unknown"),
                        "page": child.metadata.get("page", 1)
                    }
                ))

        return parent_docs

    def retrieve_context(self, query: str) -> list:
        """Hybrid retrieval: Dense + BM25 → RRF fusion → parent resolve → rerank."""
        # 1. Dense retrieval
        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        try:
            dense_results = dense_retriever.invoke(query)
            # Filter out log-tag sentinels
            dense_results = [d for d in dense_results if d.metadata.get("tag") != _LOG_TAG]
        except Exception as e:
            print(f"[RAGEngine] Dense retrieval error: {e}")
            dense_results = []

        # 2. BM25 retrieval
        sparse_results = []
        if self.bm25_retriever:
            try:
                sparse_results = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"[RAGEngine] BM25 error: {e}")

        # 3. Reciprocal Rank Fusion
        all_chunks: dict = {}

        def apply_rrf(results, weight=1.0):
            for rank, doc in enumerate(results):
                key = doc.page_content
                if key not in all_chunks:
                    all_chunks[key] = {"doc": doc, "score": 0.0}
                all_chunks[key]["score"] += weight / (60.0 + rank)

        apply_rrf(dense_results, weight=0.6)
        apply_rrf(sparse_results, weight=0.4)

        top_chunks = [
            item["doc"]
            for item in sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)[:15]
        ]

        # 4. Resolve child → parent
        parent_docs = self._resolve_parents(top_chunks)

        # 5. Cross-Encoder rerank
        return self.reranker.rerank(query, parent_docs, top_k=3)

    def get_history_text(self, session_id: str) -> str:
        history = self.session_histories.get(session_id, [])
        if not history:
            return ""
        formatted = []
        for msg in history[-6:]:  # Last 3 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def condense_query(self, query: str, history_text: str) -> str:
        """Converts a follow-up question into a standalone query using conversation history."""
        if not history_text:
            return query
        llm = self.get_llm()
        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone question in its original language. "
            "Be brief.\n\n"
            f"History:\n{history_text}\n\n"
            f"Follow-up: {query}\n"
            "Standalone:"
        )
        try:
            response = llm.invoke(prompt)
            condensed = response.content.strip()
            print(f"[RAGEngine] Condensed: '{query}' → '{condensed}'")
            return condensed
        except Exception as e:
            print(f"[RAGEngine] Condensation failed: {e}")
            return query

    def query(self, user_query: str, session_id: str):
        """Executes a query and yields SSE events for FastAPI streaming."""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []

        history_text = self.get_history_text(session_id)
        standalone_query = self.condense_query(user_query, history_text)

        # Retrieve + rerank
        retrieved_docs = self.retrieve_context(standalone_query)

        # Emit sources first
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "score": doc.metadata.get("relevance_score", 0.0),
                "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            })
        yield {"event": "sources", "data": sources}

        # Build prompt
        if retrieved_docs:
            context_text = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page', 0)}]\n{doc.page_content}"
                for doc in retrieved_docs
            ])
        else:
            context_text = "[No relevant documents found in the corpus.]"

        final_prompt = (
            "You are Hayagriva, a horse-headed avatar of knowledge and wisdom, "
            "serving as an expert RAG AI assistant.\n"
            "Answer the user's question using ONLY the retrieved context below. "
            "If the context does not contain the answer, say exactly: "
            "\"I cannot find the answer in the provided documents.\" "
            "Keep the tone wise, intellectual, and direct.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Question: {standalone_query}\n"
            "Answer:"
        )

        # Stream response
        llm = self.get_llm()
        full_response = ""

        try:
            print(f"[RAGEngine] Streaming LLM response for: {standalone_query}")
            for chunk in llm.stream(final_prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                yield {"event": "token", "data": token}
        except Exception as e:
            error_msg = f"\n[Error generating response: {e}]"
            yield {"event": "token", "data": error_msg}
            full_response += error_msg

        # Persist session history
        self.session_histories[session_id].append({"role": "user", "content": user_query})
        self.session_histories[session_id].append({"role": "assistant", "content": full_response})

        yield {"event": "done", "data": ""}
