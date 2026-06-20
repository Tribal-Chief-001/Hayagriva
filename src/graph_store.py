import os
from langchain_neo4j import Neo4jGraph
from src.config import settings

def get_graph() -> Neo4jGraph | None:
    """Returns an initialized Neo4jGraph connection if enabled, otherwise None."""
    if not settings.is_graph_enabled:
        return None
    try:
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE
        )
        return graph
    except Exception as e:
        print(f"[GraphStore] Could not connect to Neo4j: {e}")
        return None

def query_graph(query: str, graph: Neo4jGraph) -> str:
    """
    Translates a natural language query to Cypher, runs it against the graph,
    and returns a summary of the facts found.
    """
    if not graph or not settings.is_cloud_mode:
        return ""
        
    try:
        from langchain_neo4j import GraphCypherQAChain
        from src.llm import get_llm
        
        llm = get_llm(temperature=0)
        
        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            cypher_llm=llm,
            qa_llm=llm,
            verbose=False,
            allow_dangerous_requests=True # Required by newer Langchain versions for Cypher QA
        )
        
        # We wrap in a broad try/except because if the graph is empty or the Cypher query fails,
        # we don't want it to crash the entire RAG pipeline.
        result = chain.invoke({"query": query})
        return result.get("result", "")
    except Exception as e:
        print(f"[GraphStore] Cypher Query failed: {e}")
        return ""
