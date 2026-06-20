import os
import json
from neo4j import GraphDatabase
from src.config import settings
from src.llm import get_llm

def get_graph() -> GraphDatabase.driver or None:
    """Returns an initialized Neo4j driver if enabled, otherwise None."""
    if not settings.is_graph_enabled:
        return None
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        # Verify connection by running a quick ping
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run("RETURN 1")
        return driver
    except Exception as e:
        print(f"[GraphStore] Could not connect to Neo4j: {e}")
        return None

def query_graph(query: str, driver) -> str:
    """
    Extracts key entity names from the query using the LLM, retrieves their
    1-hop relationship facts using a raw Neo4j Cypher query, and formats them.
    """
    if not driver or not settings.is_cloud_mode:
        return ""
        
    try:
        # 1. Use the lightweight LLM chain to extract character/object/location entities
        llm = get_llm(temperature=0)
        prompt = (
            "Extract the key character names, locations, objects, or main entities from the user's query.\n"
            "Return them as a simple comma-separated list of words/phrases, and nothing else. No bullet points or numbering.\n\n"
            f"Query: {query}\n"
            "Entities:"
        )
        res = llm.invoke(prompt)
        
        content = res.content
        if isinstance(content, list):
            content = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
        elif not isinstance(content, str):
            content = str(content)
            
        entities = [e.strip().lower() for e in content.split(",") if e.strip()]
        if not entities:
            return ""
            
        # 2. Query Neo4j for 1-hop facts related to these entities
        facts = []
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            cypher = (
                "MATCH (n) "
                "WHERE toLower(n.id) IN $entities "
                "MATCH (n)-[r]-(m) "
                "RETURN n.id AS source, type(r) AS rel, m.id AS target "
                "LIMIT 30"
            )
            result = session.run(cypher, entities=entities)
            for record in result:
                src = record['source']
                tgt = record['target']
                rel = record['rel']
                # Skip document chunk nodes (usually 32 character hex hashes)
                if len(src) == 32 or len(tgt) == 32:
                    continue
                fact = f"{src} -[{rel}]-> {tgt}"
                if fact not in facts:
                    facts.append(fact)
                    
        if not facts:
            return ""
            
        return "\n".join(facts)
    except Exception as e:
        print(f"[GraphStore] Cypher Query failed: {e}")
        return ""
    finally:
        try:
            driver.close()
        except Exception:
            pass
