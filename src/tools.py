"""
Research tools available to the oracle: fetching an ArXiv abstract by ID,
a general web search via SerpAPI, and two Pinecone-backed RAG searches over
the ingested paper chunks (one filtered to a specific paper, one broad).

As in the rest of this project, every external client (the embedding
encoder, the Pinecone index, the SerpAPI params) is built lazily via a
cached ``get_*`` factory rather than at import time, so this module can be
imported — and its pure helpers unit-tested — without any API keys set.
"""

import os
import re
from functools import lru_cache
from typing import List

import requests
from langchain_core.tools import tool

from . import config

# --- ArXiv abstract fetch -------------------------------------------------

# Matches the <blockquote class="abstract mathjax"> block on an arxiv.org/abs
# page and captures everything after the "Abstract:" label inside it.
_ABSTRACT_PATTERN = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL,
)


@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str) -> str:
    """Fetches the abstract from an ArXiv paper given its ArXiv ID.

    Args:
        arxiv_id: The ArXiv paper ID, e.g. "1706.03762".

    Returns:
        The extracted abstract text from the ArXiv paper's abstract page.
    """
    res = requests.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=15)
    match = _ABSTRACT_PATTERN.search(res.text)
    return match.group(1) if match else "Abstract not found."


# --- Web search (SerpAPI) --------------------------------------------------

@lru_cache(maxsize=1)
def get_serpapi_params() -> dict:
    """Return the base SerpAPI request params, built on first use."""
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "SERPAPI_KEY is not set. Copy .env.example to .env and add your "
            "SerpAPI key (https://serpapi.com) before running the agent."
        )
    return {"engine": config.SERPAPI_ENGINE, "api_key": api_key}


def _format_search_results(results: list) -> str:
    return "\n---\n".join(
        "\n".join([r["title"], r.get("snippet", ""), r["link"]]) for r in results
    )


@tool("web_search")
def web_search(query: str) -> str:
    """Finds general knowledge information using a Google search.

    Args:
        query: The search query string.

    Returns:
        A formatted string of the top search results (title, snippet,
        link), or "No results found." if the search returned nothing.
    """
    from serpapi import GoogleSearch  # imported lazily: avoids a hard

    # import-time dependency for code paths that never call this tool.
    search = GoogleSearch(
        {**get_serpapi_params(), "q": query, "num": config.SERPAPI_NUM_RESULTS}
    )
    results = search.get_dict().get("organic_results", [])
    return _format_search_results(results) if results else "No results found."


# --- RAG search (Pinecone) --------------------------------------------------

@lru_cache(maxsize=1)
def get_encoder():
    """Return a cached OpenAIEncoder, constructed on first use."""
    from semantic_router.encoders import OpenAIEncoder

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
            "your OpenAI key before running the agent."
        )
    return OpenAIEncoder(name=config.OPENAI_EMBED_MODEL, openai_api_key=api_key)


@lru_cache(maxsize=1)
def get_pinecone_index():
    """Return a cached handle to the (already-populated) Pinecone index.

    This only *connects*; it never creates the index. Creating and
    populating it is the ingestion pipeline's job (see src/ingest.py) — a
    query-time tool has no business provisioning infrastructure, and a
    clear error here is much more useful than silently querying an empty
    index that was just auto-created.
    """
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Copy .env.example to .env and add "
            "your Pinecone key (https://app.pinecone.io) before running the agent."
        )
    pc = Pinecone(api_key=api_key)
    if not pc.indexes.exists(config.PINECONE_INDEX_NAME):
        raise RuntimeError(
            f"Pinecone index '{config.PINECONE_INDEX_NAME}' doesn't exist yet. "
            "Run `python -m src.ingest` first to build and populate it."
        )
    return pc.Index(config.PINECONE_INDEX_NAME)


def format_rag_contexts(matches: list) -> str:
    """Formats Pinecone match results into a readable string for the LLM.

    Args:
        matches: A list of Pinecone match dicts, each with a 'metadata'
            dict containing 'title', 'chunk', and 'arxiv_id'.

    Returns:
        A formatted string of document titles, chunks, and ArXiv IDs.
    """
    blocks = []
    for match in matches:
        meta = match["metadata"]
        blocks.append(
            f"Title: {meta['title']}\nChunk: {meta['chunk']}\nArXiv ID: {meta['arxiv_id']}\n"
        )
    return "\n---\n".join(blocks)


def _embed_query(query: str) -> List[float]:
    """Encode a single query string into a single flat embedding vector.

    encoder([query]) returns a *list of embeddings* (shape (1, dims)) since
    the encoder is designed to batch-embed many strings at once. Pinecone's
    index.query(vector=...) wants one flat vector, not a list containing
    one vector — so this takes [0] to unwrap it. (The original notebook
    passed the un-unwrapped list straight through; see docs/decisions.md.)
    """
    return get_encoder()([query])[0]


@tool
def rag_search_filter(query: str, arxiv_id: str) -> str:
    """Finds information from the ArXiv database using a natural language query and a specific ArXiv ID.

    Args:
        query: The search query in natural language.
        arxiv_id: The ArXiv ID of the specific paper to filter by.

    Returns:
        A formatted string of relevant document contexts.
    """
    index = get_pinecone_index()
    matches = index.query(
        vector=_embed_query(query),
        top_k=config.RAG_FILTERED_TOP_K,
        include_metadata=True,
        filter={"arxiv_id": arxiv_id},
    )["matches"]
    return format_rag_contexts(matches)


@tool("rag_search")
def rag_search(query: str) -> str:
    """Finds specialist information on AI using a natural language query.

    Args:
        query: The search query in natural language.

    Returns:
        A formatted string of relevant document contexts.
    """
    index = get_pinecone_index()
    matches = index.query(
        vector=_embed_query(query),
        top_k=config.RAG_TOP_K,
        include_metadata=True,
    )["matches"]
    return format_rag_contexts(matches)
