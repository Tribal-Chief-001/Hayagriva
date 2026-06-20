import os
import hashlib
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.config import settings
from src.vector_store import get_embeddings, get_vector_store
from src.reranker import CrossEncoderReranker

class RAGEngine:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store = get_vector_store(self.embeddings)
        self.reranker = CrossEncoderReranker()
        
        # Local session-based memory
        # Format: { session_id: [ {"role": "user"/"assistant", "content": str} ] }
        self.session_histories = {}
        
        # BM25 retriever will be initialized dynamically from Chroma contents
        self.bm25_retriever = None
        self.refresh_bm25_retriever()

    def refresh_bm25_retriever(self):
        """Loads all child chunks from the vector store and builds the BM25 index."""
        try:
            documents = []
            # Check if vector store is Qdrant
            if hasattr(self.vector_store, "client") and hasattr(self.vector_store, "collection_name") and self.vector_store.client is not None:
                print(f"[RAGEngine] Scrolling Qdrant collection '{self.vector_store.collection_name}' to build BM25 index...")
                points, _ = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                for point in points:
                    payload = point.payload or {}
                    # LangChain Qdrant stores page_content and metadata in payload keys
                    text = payload.get("page_content", "")
                    metadata = payload.get("metadata", {})
                    if text:
                        documents.append(Document(page_content=text, metadata=metadata))
            elif hasattr(self.vector_store, "get"):
                # Check if vector store is initialized and has documents (Chroma DB)
                data = self.vector_store.get()
                if data and "documents" in data and data["documents"]:
                    for i in range(len(data["documents"])):
                        text = data["documents"][i]
                        metadata = data["metadatas"][i] if data["metadatas"] else {}
                        documents.append(Document(page_content=text, metadata=metadata))
            
            if not documents:
                print("[RAGEngine] No documents found in vector store. BM25 is inactive.")
                self.bm25_retriever = None
                return
                
            print(f"[RAGEngine] Initializing BM25Retriever with {len(documents)} child chunks.")
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 10  # Retrieve top 10 for ensemble
        except Exception as e:
            print(f"[RAGEngine] Failed to build BM25Retriever: {e}")
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
            return ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                temperature=0.2
            )

    def _resolve_parents(self, child_docs: list) -> list:
        """Extracts parent text from child chunk metadata, falling back to child content if missing."""
        parent_docs = []
        seen_parent_texts = set()
        
        for child in child_docs:
            parent_text = child.metadata.get("parent_text")
            
            # If no parent_text is present, use child content itself
            if not parent_text:
                parent_text = child.page_content
                
            text_hash = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
            if text_hash not in seen_parent_texts:
                seen_parent_texts.add(text_hash)
                
                # Construct parent Document object
                parent_doc = Document(
                    page_content=parent_text,
                    metadata={
                        "source": child.metadata.get("source", "Unknown"),
                        "page": child.metadata.get("page", 1)
                    }
                )
                parent_docs.append(parent_doc)
                
        return parent_docs

    def retrieve_context(self, query: str) -> list:
        """Retrieves and re-ranks parent documents using hybrid retrieval."""
        # 1. Gather Dense Chunks
        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        dense_results = dense_retriever.invoke(query)
        
        # 2. Gather Sparse Chunks
        sparse_results = []
        if self.bm25_retriever:
            try:
                sparse_results = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"[RAGEngine] BM25 query failed: {e}")
                
        # 3. Reciprocal Rank Fusion (RRF) on child chunks
        all_chunks = {}
        
        def apply_rrf(results, weight=1.0):
            for rank, doc in enumerate(results):
                doc_id = doc.page_content # Use content as key to deduplicate chunks
                if doc_id not in all_chunks:
                    all_chunks[doc_id] = {"doc": doc, "score": 0.0}
                all_chunks[doc_id]["score"] += weight * (1.0 / (60.0 + rank))
                
        apply_rrf(dense_results, weight=0.6)
        apply_rrf(sparse_results, weight=0.4)
        
        # Sort chunks by RRF score
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        top_chunks = [item["doc"] for item in sorted_chunks[:15]]
        
        # 4. Resolve child chunks to parent documents (from child metadata)
        parent_docs = self._resolve_parents(top_chunks)
        
        # 5. Re-rank using Cross-Encoder
        final_docs = self.reranker.rerank(query, parent_docs, top_k=3)
        return final_docs

    def get_history_text(self, session_id: str) -> str:
        """Formates the chat history for input into prompts."""
        history = self.session_histories.get(session_id, [])
        if not history:
            return ""
        formatted = []
        for msg in history[-6:]: # Keep last 3 turns (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def condense_query(self, query: str, history_text: str) -> str:
        """Converts follow-up questions to standalone queries based on history."""
        if not history_text:
            return query
            
        llm = self.get_llm()
        prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question (which captures the context of the history) in its original language. Keep it brief.

Conversation History:
{history_text}

Follow-up Question: {query}
Standalone Question:"""
        
        try:
            response = llm.invoke(prompt)
            condensed = response.content.strip()
            print(f"[RAGEngine] Condensed '{query}' -> '{condensed}'")
            return condensed
        except Exception as e:
            print(f"[RAGEngine] Query condensation failed: {e}")
            return query

    def query(self, user_query: str, session_id: str):
        """Executes a query and yields events for FastAPI streaming."""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
            
        history_text = self.get_history_text(session_id)
        
        # 1. Condense follow-up query
        standalone_query = self.condense_query(user_query, history_text)
        
        # 2. Retrieve contexts
        retrieved_docs = self.retrieve_context(standalone_query)
        
        # Format sources to send to frontend first
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "score": doc.metadata.get("relevance_score", 0.0),
                "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
            
        yield {"event": "sources", "data": sources}
        
        # 3. Build Final Contextual Prompt
        context_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page', 0)}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        final_prompt = f"""You are Hayagriva, a horse-headed avatar of knowledge and wisdom, serving as an expert RAG AI assistant.
Answer the user's question using ONLY the retrieved context below. If the context does not contain the answer, say "I cannot find the answer in the provided documents." Keep the tone wise, intellectual, and direct.

Context:
{context_text}

Chat History:
{history_text}

Question: {standalone_query}
Answer:"""

        # 4. Stream response
        llm = self.get_llm()
        full_response = ""
        
        try:
            print(f"[RAGEngine] Invoking LLM for query: {standalone_query}")
            for chunk in llm.stream(final_prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                yield {"event": "token", "data": token}
        except Exception as e:
            error_msg = f"\n[Error generating response: {e}]"
            yield {"event": "token", "data": error_msg}
            full_response += error_msg
            
        # Save session history
        self.session_histories[session_id].append({"role": "user", "content": user_query})
        self.session_histories[session_id].append({"role": "assistant", "content": full_response})
        
        yield {"event": "done", "data": ""}
