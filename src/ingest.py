"""
Offline ingestion pipeline: builds the knowledge base the agent's RAG tools
query at runtime.

    extract_from_arxiv -> download_pdfs -> expand_df (chunk) -> ensure_index -> populate_index

Run as a script (`python -m src.ingest`) to do all five steps end to end.
Each step is also a plain function so it can be re-run individually — e.g.
in a notebook, or to re-chunk without re-hitting the ArXiv API.

This is a batch job, not a request-time path, so unlike tools.py/oracle.py
it doesn't need lazy singletons for its own sake — but it reuses
tools.get_encoder() rather than constructing a second OpenAIEncoder, so
ingestion and querying always embed with the exact same model.
"""

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests

from . import config
from .tools import get_encoder

ARXIV_NAMESPACE = "{http://www.w3.org/2005/Atom}"


# --- Step 1: ArXiv metadata -------------------------------------------------


def extract_from_arxiv(
    search_query: str = config.ARXIV_SEARCH_QUERY,
    max_results: int = config.ARXIV_MAX_RESULTS,
    json_file_path: str = config.ARXIV_JSON_PATH,
) -> pd.DataFrame:
    """Fetch paper metadata from the ArXiv API and save it as JSON.

    Args:
        search_query: ArXiv API search query, e.g. "cat:cs.AI".
        max_results: Maximum number of papers to retrieve.
        json_file_path: Where to save the raw paper metadata as JSON.

    Returns:
        A DataFrame with one row per paper: title, summary, authors,
        arxiv_id, url, pdf_link.
    """
    os.makedirs(os.path.dirname(json_file_path) or ".", exist_ok=True)
    url = f"http://export.arxiv.org/api/query?search_query={search_query}&max_results={max_results}"
    response = requests.get(url, timeout=30)
    root = ET.fromstring(response.content)

    papers = []
    for entry in root.findall(f"{ARXIV_NAMESPACE}entry"):
        title = entry.find(f"{ARXIV_NAMESPACE}title").text.strip()
        summary = entry.find(f"{ARXIV_NAMESPACE}summary").text.strip()
        authors = [
            a.find(f"{ARXIV_NAMESPACE}name").text
            for a in entry.findall(f"{ARXIV_NAMESPACE}author")
        ]
        paper_url = entry.find(f"{ARXIV_NAMESPACE}id").text
        arxiv_id = paper_url.split("/")[-1]
        pdf_link = next(
            (
                link.attrib["href"]
                for link in entry.findall(f"{ARXIV_NAMESPACE}link")
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )
        papers.append(
            {
                "title": title,
                "summary": summary,
                "authors": authors,
                "arxiv_id": arxiv_id,
                "url": paper_url,
                "pdf_link": pdf_link,
            }
        )

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(papers)} paper records to {json_file_path}")

    return pd.DataFrame(papers)


# --- Step 2: PDF download ---------------------------------------------------


def download_pdfs(
    df: pd.DataFrame, download_folder: str = config.DATA_DIR
) -> pd.DataFrame:
    """Download each paper's PDF, skipping any already present on disk.

    Args:
        df: DataFrame with a 'pdf_link' column.
        download_folder: Directory to save PDFs into.

    Returns:
        df with an added 'pdf_file_name' column (None for failed downloads).
    """
    os.makedirs(download_folder, exist_ok=True)
    pdf_file_names = []

    for _, row in df.iterrows():
        pdf_link = row["pdf_link"]
        file_name = os.path.join(download_folder, pdf_link.split("/")[-1] + ".pdf")

        if os.path.exists(file_name):
            # PDF that's already on disk.
            print(f"Already downloaded, skipping: {file_name}")
            pdf_file_names.append(file_name)
            continue

        try:
            response = requests.get(pdf_link, timeout=30)
            response.raise_for_status()
            with open(file_name, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {file_name}")
            pdf_file_names.append(file_name)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {pdf_link}: {e}")
            pdf_file_names.append(None)

    df["pdf_file_name"] = pdf_file_names
    return df


# --- Step 3: chunking --------------------------------------------------------


def load_and_chunk_pdf(pdf_file_name: str, chunk_size: int = config.CHUNK_SIZE):
    """Load a PDF and split it into overlapping text chunks.

    Args:
        pdf_file_name: Path to the PDF file.
        chunk_size: Max characters per chunk.

    Returns:
        A list of langchain Document chunks.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"Loading and chunking: {pdf_file_name}")
    pages = PyPDFLoader(pdf_file_name).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(pages)


def expand_df(df: pd.DataFrame, chunk_size: int = config.CHUNK_SIZE) -> pd.DataFrame:
    """Expand one row per paper into one row per chunk, with sibling links.

    Args:
        df: DataFrame with 'pdf_file_name', 'arxiv_id', 'title', 'summary',
            'authors', and 'url' columns.
        chunk_size: Max characters per chunk, forwarded to load_and_chunk_pdf.

    Returns:
        A new DataFrame, one row per chunk, with 'id' (f"{arxiv_id}#{i}"),
        'chunk' (the chunk text), and 'prechunk_id'/'postchunk_id' linking
        each chunk to its neighbors within the same paper (empty string at
        either end of a paper's chunk sequence).
    """
    expanded_rows = []

    for _, row in df.iterrows():
        if not row.get("pdf_file_name"):
            continue
        try:
            chunks = load_and_chunk_pdf(row["pdf_file_name"], chunk_size=chunk_size)
        except Exception as e:
            print(f"Error processing {row['pdf_file_name']}: {e}")
            continue

        n = len(chunks)
        for i, chunk in enumerate(chunks):
            expanded_rows.append(
                {
                    "id": f"{row['arxiv_id']}#{i}",
                    "title": row["title"],
                    "summary": row["summary"],
                    "authors": row["authors"],
                    "arxiv_id": row["arxiv_id"],
                    "url": row["url"],
                    "chunk": chunk.page_content,
                    "prechunk_id": "" if i == 0 else f"{row['arxiv_id']}#{i - 1}",
                    "postchunk_id": "" if i == n - 1 else f"{row['arxiv_id']}#{i + 1}",
                }
            )

    return pd.DataFrame(expanded_rows)


# --- Step 4: Pinecone index --------------------------------------------------


def ensure_pinecone_index(dimension: int):
    """Create the Pinecone index if it doesn't exist yet, and return a handle to it.

    Uses the modern pinecone-client API (`pc.indexes.exists` /
    `pc.indexes.create`), whose `create()` blocks until the index is ready
    by default — no manual polling loop needed (the original notebook
    polled `describe_index(...).status['ready']` in a `while` loop against
    an older client version).
    """
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Copy .env.example to .env and add "
            "your Pinecone key before running ingestion."
        )
    pc = Pinecone(api_key=api_key)

    if not pc.indexes.exists(config.PINECONE_INDEX_NAME):
        print(
            f"Creating Pinecone index '{config.PINECONE_INDEX_NAME}' (dim={dimension})..."
        )
        pc.indexes.create(
            name=config.PINECONE_INDEX_NAME,
            dimension=dimension,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION
            ),
        )
    else:
        print(f"Pinecone index '{config.PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(config.PINECONE_INDEX_NAME)


# --- Step 5: embed + upsert --------------------------------------------------


def populate_index(
    expanded_df: pd.DataFrame, index, batch_size: int = config.EMBED_BATCH_SIZE
) -> None:
    """Embed each chunk and upsert it into Pinecone, in batches.

    Args:
        expanded_df: One row per chunk (see expand_df).
        index: A connected Pinecone index handle (see ensure_pinecone_index).
        batch_size: How many chunks to embed/upsert per API call.
    """
    from tqdm.auto import tqdm

    encoder = get_encoder()

    for start in tqdm(
        range(0, len(expanded_df), batch_size), desc="Embedding + upserting"
    ):
        batch = expanded_df.iloc[start : start + batch_size].to_dict(orient="records")
        ids = [r["id"] for r in batch]
        chunks = [r["chunk"] for r in batch]
        metadata = [
            {"arxiv_id": r["arxiv_id"], "title": r["title"], "chunk": r["chunk"]}
            for r in batch
        ]
        embeddings = encoder(chunks)
        index.upsert(vectors=zip(ids, embeddings, metadata))

    print(
        f"Upserted {len(expanded_df)} chunks. Index stats: {index.describe_index_stats()}"
    )


# --- End-to-end pipeline + CLI -----------------------------------------------


def run_pipeline(
    search_query: str = config.ARXIV_SEARCH_QUERY,
    max_results: int = config.ARXIV_MAX_RESULTS,
    chunk_size: int = config.CHUNK_SIZE,
    batch_size: int = config.EMBED_BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
) -> None:
    """Run the full ingestion pipeline: extract -> download -> chunk -> index -> populate."""
    json_path = os.path.join(data_dir, "arxiv_dataset.json")

    df = extract_from_arxiv(search_query, max_results, json_path)
    df = download_pdfs(df, data_dir)
    expanded = expand_df(df, chunk_size=chunk_size)
    if expanded.empty:
        print(
            "No chunks produced (all PDF downloads/parses failed) — nothing to index."
        )
        return

    dims = len(get_encoder()([expanded["chunk"].iloc[0]])[0])
    index = ensure_pinecone_index(dimension=dims)
    populate_index(expanded, index, batch_size=batch_size)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Build the ArXiv knowledge base: fetch papers, chunk them, and index them in Pinecone.",
    )
    parser.add_argument(
        "--query",
        default=config.ARXIV_SEARCH_QUERY,
        help="ArXiv search query (default: cat:cs.AI).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=config.ARXIV_MAX_RESULTS,
        help="Max papers to fetch.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_SIZE,
        help="Max characters per chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBED_BATCH_SIZE,
        help="Embedding/upsert batch size.",
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help="Where to save JSON metadata and PDFs.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing required environment variable: OPENAI_API_KEY")
        return 1
    try:
        run_pipeline(
            search_query=args.query,
            max_results=args.max_results,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - surface any failure clearly to the CLI user
        print(f"Ingestion failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
