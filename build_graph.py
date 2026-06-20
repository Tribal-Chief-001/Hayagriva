import os
import time
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from langchain_neo4j import Neo4jGraph
from src.ingest import ingest_document_bytes, delete_document_from_db

def extract_and_upload_graph(file_path: str):
    print(f"============================================================")
    print(f"Neo4j Knowledge Graph Extraction: {Path(file_path).name}")
    print(f"============================================================")
    
    if not settings.is_graph_enabled:
        print("[ERROR] Neo4j is not configured. Please add NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD to your .env file.")
        return

    try:
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to Neo4j database: {e}")
        return

    filename = Path(file_path).name

    # 0. Clean any existing entries (Idempotence check to prevent duplicates)
    print(f"[*] Phase 1: Cleaning existing entries for {filename} to prevent duplicates...")
    delete_document_from_db(filename)

    # 1. Upload to Qdrant (Vector Database) First
    print(f"\n[*] Phase 2: Uploading to Qdrant Vector Database...")
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        # We temporarily set VERCEL=0 to ensure the ingest script uses local rate-limits (20s sleep) 
        # instead of Vercel's fast-fail rate limits.
        os.environ["VERCEL"] = "0"
        result = ingest_document_bytes(file_bytes, filename)
        print(f"[*] Qdrant Upload Success: {result['chunks']} chunks indexed.")
    except Exception as e:
        print(f"[ERROR] Failed to upload to Qdrant: {e}")
        print(f"[*] Proceeding to Graph Extraction anyway...")

    # 2. Load document for Graph Extraction
    print(f"\n[*] Phase 3: Loading document for Neo4j Graph Extraction: {file_path}")
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        print(f"[ERROR] Unsupported file type: {ext}")
        return
        
    pages = loader.load()
    if not pages:
        print("[ERROR] Document is empty.")
        return

    # 2. Chunking (We use large chunks for the graph to give the LLM more context per call)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=300
    )
    chunks = text_splitter.split_documents(pages)
    print(f"[*] Split into {len(chunks)} chunks.")

    # 3. Setup LLM Graph Transformer
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0
    )
    
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 4. Process chunks with strict rate limiting
    print(f"[*] Beginning graph extraction (Using Gemini 1.5 Flash).")
    print(f"[*] Note: Throttling to 2 seconds per chunk to respect 100 RPM free-tier limit.")
    
    for i, chunk in enumerate(chunks):
        print(f"    -> Processing chunk {i+1}/{len(chunks)}...")
        try:
            # We must pass it as a list
            graph_documents = llm_transformer.convert_to_graph_documents([chunk])
            if graph_documents:
                # Upsert into Neo4j
                graph.add_graph_documents(
                    graph_documents, 
                    baseEntityLabel=True, 
                    include_source=True
                )
                print(f"       + Added {len(graph_documents[0].nodes)} nodes and {len(graph_documents[0].relationships)} edges.")
            else:
                print(f"       ~ No relationships found in this chunk.")
        except Exception as e:
            print(f"       ! Error processing chunk {i+1}: {e}")
            
        # STRICT RATE LIMITING (Wait 2.5 seconds to ensure we stay under 100 RPM)
        if i < len(chunks) - 1:
            time.sleep(2.5)

    print(f"============================================================")
    print(f"[*] Graph Extraction Complete! Entities pushed to Neo4j.")
    print(f"============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a Knowledge Graph from a document and push to Neo4j.")
    parser.add_argument("file_path", help="Path to the PDF, TXT, or MD file.")
    args = parser.parse_args()
    
    extract_and_upload_graph(args.file_path)
