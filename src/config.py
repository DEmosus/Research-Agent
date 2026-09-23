"""
Central configuration for the research agent.

Hardcoded values (model name, the Pinecone index name, a chunk size, a
batch size). Pulling them into one module with env-var overrides means tuning
the system never means hunting through five files for a magic number.
"""

import os

# --- LLM / embeddings ---
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_CHAT_TEMPERATURE = float(os.getenv("OPENAI_CHAT_TEMPERATURE", "0"))
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# --- Pinecone ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langgraph-research-agent")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# --- SerpAPI ---
SERPAPI_ENGINE = os.getenv("SERPAPI_ENGINE", "google")
SERPAPI_NUM_RESULTS = int(os.getenv("SERPAPI_NUM_RESULTS", "5"))

# --- RAG retrieval ---
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_FILTERED_TOP_K = int(os.getenv("RAG_FILTERED_TOP_K", "6"))

# --- Agent loop ---
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))

# --- Ingestion pipeline ---
ARXIV_SEARCH_QUERY = os.getenv("ARXIV_SEARCH_QUERY", "cat:cs.AI")
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# --- Filesystem ---
DATA_DIR = os.getenv("DATA_DIR", "data")
ARXIV_JSON_PATH = os.path.join(DATA_DIR, "arxiv_dataset.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
