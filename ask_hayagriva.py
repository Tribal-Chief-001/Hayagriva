from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOllama(
    model="qwen3.5:2b"
)

question = input("Ask Hayagriva: ")

docs = retriever.invoke(question)

context = "\n\n".join(
    doc.page_content for doc in docs
)

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nHayagriva:\n")
print(response.content)