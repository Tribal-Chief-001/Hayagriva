import os
import json
import time
import hashlib
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.config import settings
from src.vector_store import get_embeddings, get_vector_store
from src.reranker import CrossEncoderReranker
from src.graph_store import get_graph, query_graph

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
        """Returns the appropriate chat model with automatic high-throughput fallbacks."""
        from src.llm import get_llm as shared_get_llm
        return shared_get_llm(temperature=0.2)

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

    def retrieve_context(self, queries: list) -> list:
        """Hybrid retrieval: Dense + BM25 over multiple query variations → RRF fusion → parent resolve → rerank."""
        all_chunks: dict = {}

        def apply_rrf(results, weight=1.0):
            for rank, doc in enumerate(results):
                key = doc.page_content
                if key not in all_chunks:
                    all_chunks[key] = {"doc": doc, "score": 0.0}
                all_chunks[key]["score"] += weight / (60.0 + rank)

        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        for q in queries:
            # 1. Dense retrieval
            try:
                dense_results = dense_retriever.invoke(q)
                dense_results = [d for d in dense_results if d.metadata.get("tag") != _LOG_TAG]
                apply_rrf(dense_results, weight=0.6)
            except Exception as e:
                print(f"[RAGEngine] Dense retrieval error for '{q}': {e}")

            # 2. BM25 retrieval
            if self.bm25_retriever:
                try:
                    sparse_results = self.bm25_retriever.invoke(q)
                    apply_rrf(sparse_results, weight=0.4)
                except Exception as e:
                    print(f"[RAGEngine] BM25 error for '{q}': {e}")

        top_chunks = [
            item["doc"]
            for item in sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)[:15]
        ]

        # 4. Resolve child → parent
        parent_docs = self._resolve_parents(top_chunks)

        # 5. Cross-Encoder rerank against the primary query
        return self.reranker.rerank(queries[0], parent_docs, top_k=3)

    def get_history_text(self, session_id: str) -> str:
        history = self.session_histories.get(session_id, [])
        if not history:
            return ""
        formatted = []
        for msg in history[-6:]:  # Last 3 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def condense_and_expand_query(self, query: str, history_text: str) -> list:
        """Converts a follow-up question into standalone variations (Multi-Query Expansion)."""
        llm = self.get_llm()
        prompt = (
            "Given the conversation history and a user question, generate 3 different standalone versions of the question.\n"
            "Use synonyms and different phrasing to maximize semantic search retrieval. Return exactly 3 lines, one for each variation.\n"
            "Do not include numbers or bullet points at the start of the lines.\n\n"
            f"History:\n{history_text if history_text else 'None'}\n\n"
            f"Question: {query}\n"
            "Variations:"
        )
        try:
            response = llm.invoke(prompt)
            
            content = response.content
            if isinstance(content, list):
                content = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
            elif not isinstance(content, str):
                content = str(content)
                
            lines = [line.strip().lstrip("-*1234567890. ") for line in content.strip().split("\n") if line.strip()]
            if len(lines) >= 1:
                print(f"[RAGEngine] Expanded queries: {lines[:3]}")
                return lines[:3]
        except Exception as e:
            print(f"[RAGEngine] Expansion failed: {e}")
        return [query]

    def grade_documents(self, query: str, docs: list) -> bool:
        """Self-Reflective RAG Grader: Checks if retrieved docs are relevant to the query."""
        if not docs:
            return False
        llm = self.get_llm()
        context_text = "\n\n".join([d.page_content for d in docs])
        prompt = (
            "You are a strict grading assistant evaluating if a retrieved document is relevant to a user's question.\n"
            "If the document contains ANY keywords, concepts, or semantic meaning that could help answer the question, grade it as 'yes'.\n"
            "If the document is completely unrelated and useless, grade it as 'no'.\n"
            "Return ONLY 'yes' or 'no' without any punctuation.\n\n"
            f"Question: {query}\n\n"
            f"Context: {context_text[:4000]}\n\n"
            "Is relevant? (yes/no):"
        )
        try:
            res = llm.invoke(prompt)
            
            content = res.content
            if isinstance(content, list):
                content = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
            elif not isinstance(content, str):
                content = str(content)
                
            decision = content.strip().lower()
            print(f"[RAGEngine] Grader decision for '{query}': {decision}")
            return "yes" in decision
        except Exception as e:
            print(f"[RAGEngine] Grader failed ({e}), defaulting to True")
            return True

    def query(self, user_query: str, session_id: str):
        """Executes a query and yields SSE events for FastAPI streaming."""
        start_time = time.time()
        
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []

        yield {"event": "status", "data": "Refining user query..."}
        history_text = self.get_history_text(session_id)
        queries = self.condense_and_expand_query(user_query, history_text)
        main_query = queries[0]

        yield {"event": "status", "data": "Scanning Qdrant vector database..."}
        # Retrieve + rerank using Multi-Query Expansion
        retrieved_docs = self.retrieve_context(queries)

        yield {"event": "status", "data": "Cross-Encoder Grading..."}

        # Self-Reflective Grader
        is_relevant = self.grade_documents(main_query, retrieved_docs)
        if not is_relevant:
            print("[RAGEngine] Grader deemed documents irrelevant. Bypassing final generation.")
            msg = "I cannot find the answer in the provided documents. The retrieved context does not appear to be relevant to your question."
            yield {"event": "token", "data": msg}
            self.session_histories[session_id].append({"role": "user", "content": user_query})
            self.session_histories[session_id].append({"role": "assistant", "content": msg})
            
            latency = round(time.time() - start_time, 2)
            metrics = {"latency": latency, "chunks": 0, "graph": False}
            yield {"event": "metrics", "data": metrics}
            
            yield {"event": "done", "data": ""}
            return

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

        # Build Vector Prompt Context
        if retrieved_docs:
            context_text = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page', 0)}]\n{doc.page_content}"
                for doc in retrieved_docs
            ])
        else:
            context_text = "[No relevant documents found in the corpus.]"
            
        # -------------------------------------------------------------
        # GRAPH RAG INTEGRATION (Simultaneous Graph Traversal)
        # -------------------------------------------------------------
        graph_facts = ""
        if settings.is_graph_enabled:
            yield {"event": "status", "data": "Traversing Neo4j Knowledge Graph..."}
            print(f"[RAGEngine] Querying Neo4j Knowledge Graph...")
            graph = get_graph()
            if graph:
                # We use the main_query to traverse the graph relationships
                graph_result = query_graph(main_query, graph)
                if graph_result:
                    print(f"[RAGEngine] Graph found relevant facts: {graph_result[:100]}...")
                    graph_facts = f"\n\n[KNOWLEDGE GRAPH FACTS]:\n{graph_result}\n"
                else:
                    print(f"[RAGEngine] Graph returned no facts.")
        # -------------------------------------------------------------

        final_prompt = (
            "You are Hayagriva, a horse-headed avatar of knowledge and wisdom, "
            "serving as an expert RAG AI assistant.\n"
            "Answer the user's question using the retrieved context below (which may include paragraphs and structural graph facts). "
            "The user may ask analytical questions about the text (e.g. 'Why did the author...', 'What is the theme...'). "
            "You MUST use your reasoning to answer these based on the provided text.\n"
            "If the question is completely unrelated to the context, say exactly: "
            "\"I cannot find the answer in the provided documents.\"\n"
            "Keep the tone wise, intellectual, and direct.\n\n"
            f"Context:\n{context_text}{graph_facts}\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Question: {main_query}\n"
            "Answer:"
        )

        # Stream response
        llm = self.get_llm()
        full_response = ""

        try:
            yield {"event": "status", "data": "Generating response..."}
            print(f"[RAGEngine] Streaming LLM response for: {main_query}")
            for chunk in llm.stream(final_prompt):
                token = chunk.content if hasattr(chunk, "content") else chunk
                
                if isinstance(token, list):
                    token = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in token])
                elif not isinstance(token, str):
                    token = str(token)
                    
                full_response += token
                yield {"event": "token", "data": token}
        except Exception as e:
            error_msg = f"\n[Error generating response: {e}]"
            yield {"event": "token", "data": error_msg}
            full_response += error_msg

        # Persist session history
        self.session_histories[session_id].append({"role": "user", "content": user_query})
        self.session_histories[session_id].append({"role": "assistant", "content": full_response})

        # Calculate final metrics
        latency = round(time.time() - start_time, 2)
        metrics = {
            "latency": latency,
            "chunks": len(retrieved_docs),
            "graph": bool(graph_facts)
        }
        yield {"event": "metrics", "data": metrics}

        yield {"event": "done", "data": ""}
