import os

import pytest


REQUIRED_ENV_VARS = ("GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


@pytest.mark.skipif(
    any(not os.getenv(name) for name in REQUIRED_ENV_VARS),
    reason="RAG smoke test requires Google and Qdrant credentials.",
)
def test_pipeline():
    from agents.rag_builder import ingest_into_qdrant, retrieve_from_qdrant

    fake_extracted_map = {
        "science_paper.pdf": (
            "The mitochondria is the powerhouse of the cell. It generates most of the chemical energy "
            "needed to power the cell's biochemical reactions. Chemical energy produced by the mitochondria "
            "is stored in a small molecule called adenosine triphosphate (ATP). " 
            * 20 
        ),
        "https://qdrant.tech/blog": (
            "Qdrant is a high-performance, massive-scale Vector Database for the next generation of AI. "
            "It is written in Rust to ensure memory safety and speed. It uses HNSW algorithm for fast approximate nearest neighbor search."
        )
    }

    print("Starting RAG Ingestion...")
    
    chunks_created = ingest_into_qdrant(fake_extracted_map)
    print(f"✅ Successfully chunked and inserted {chunks_created} vectors into Qdrant.\n")

    query = "What programming language is Qdrant built with?"
    print(f"Searching Qdrant for: '{query}'\n")
    
    results = retrieve_from_qdrant(query, top_k=2)
    
    print("--- Retrieved Context ---")
    print(results)
    print("----------------------------")

if __name__ == "__main__":
    test_pipeline()
