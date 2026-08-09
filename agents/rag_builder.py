import os
import math
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langsmith import traceable

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",output_dimensionality=768)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 6
CACHE_DIR = Path(".cache")
INGESTION_CACHE_FILE = CACHE_DIR / "ingestion_cache.json"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "agent_documents"

if QDRANT_URL and QDRANT_API_KEY:
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print("Connecting to Local Qdrant...")
    client = QdrantClient(path="./local_qdrant_db")

if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


def _load_ingestion_cache() -> set[str]:
    try:
        payload = json.loads(INGESTION_CACHE_FILE.read_text(encoding="utf-8"))
        return set(payload.get("signatures", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _save_ingestion_cache(signatures: set[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INGESTION_CACHE_FILE.write_text(
        json.dumps({"signatures": sorted(signatures)}, indent=2),
        encoding="utf-8",
    )


def _source_signature(source_name: str, content: str) -> str:
    normalized = f"{source_name}\n{content}".encode("utf-8", errors="ignore")
    return hashlib.sha256(normalized).hexdigest()

@traceable(name="qdrant_ingestion", run_type="chain")
def ingest_into_qdrant(extracted_map: Dict[str, str]) -> Dict[str, int]:
    """
    Takes the raw extracted text dictionary, chunks it, and upserts it into Qdrant.
    Returns the number of chunks successfully ingested and skipped sources.
    """
    if not extracted_map:
        return {"added_chunks": 0, "skipped_sources": 0}

    documents = []
    cache = _load_ingestion_cache()
    skipped_sources = 0
    for source_name, content in extracted_map.items():
        signature = _source_signature(source_name, content)
        if signature in cache:
            skipped_sources += 1
            continue
        doc = Document(page_content=content, metadata={"source": source_name})
        documents.append(doc)
        cache.add(signature)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    
    if chunked_docs:
        vector_store.add_documents(chunked_docs)
        _save_ingestion_cache(cache)
        
    return {"added_chunks": len(chunked_docs), "skipped_sources": skipped_sources}

def select_top_k_for_context(extracted_map: Dict[str, str]) -> int:
    """
    Choose a retrieval depth based on how much source text was extracted in the
    current turn.

    Smaller inputs get a wider retrieval window so we cover most or all of the
    available chunks. Larger inputs get a tighter cap so the prompt stays
    manageable.
    """
    if not extracted_map:
        return DEFAULT_TOP_K

    total_chars = sum(len(content) for content in extracted_map.values())
    estimated_chunks = max(1, math.ceil(total_chars / CHUNK_SIZE))

    if estimated_chunks <= 3:
        return min(8, estimated_chunks + 2)
    if estimated_chunks <= 8:
        return min(8, estimated_chunks + 1)
    return DEFAULT_TOP_K

@traceable(name="qdrant_retrieval", run_type="retriever")
def retrieve_from_qdrant(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Embeds the user query, searches Qdrant for the most relevant chunks,
    and returns them as a single formatted string for the LLM.
    """
    retrieved_docs = vector_store.similarity_search(query, k=top_k)
    
    if not retrieved_docs:
        return ""

    formatted_chunks = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown Source")
        formatted_chunks.append(f"--- Chunk {i+1} from {source} ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_chunks)
