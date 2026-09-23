"""
Tests for the ingestion pipeline.

Only expand_df's chunk-linking logic is pure enough to test without live
network/API access (ArXiv, PDF downloads, OpenAI embeddings, and Pinecone
are all real I/O — see docs/development-log.md for how those paths were
validated instead: by reading the installed client libraries' current
signatures directly, since this sandbox has no route to those services).

Run with: python -m pytest tests/ -v
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.ingest import expand_df


def _fake_chunks(texts):
    """Build fake langchain Document-like objects (only .page_content is used)."""
    docs = []
    for t in texts:
        d = MagicMock()
        d.page_content = t
        docs.append(d)
    return docs


def test_expand_df_links_chunks_within_a_paper():
    df = pd.DataFrame([
        {
            "pdf_file_name": "paper1.pdf", "arxiv_id": "1111.1111",
            "title": "Paper One", "summary": "Summary one",
            "authors": ["A. Author"], "url": "https://arxiv.org/abs/1111.1111",
        },
    ])

    with patch("src.ingest.load_and_chunk_pdf", return_value=_fake_chunks(["a", "b", "c"])):
        result = expand_df(df)

    assert list(result["id"]) == ["1111.1111#0", "1111.1111#1", "1111.1111#2"]
    assert list(result["chunk"]) == ["a", "b", "c"]
    # First chunk has no predecessor, last has no successor.
    assert result.iloc[0]["prechunk_id"] == ""
    assert result.iloc[0]["postchunk_id"] == "1111.1111#1"
    assert result.iloc[1]["prechunk_id"] == "1111.1111#0"
    assert result.iloc[1]["postchunk_id"] == "1111.1111#2"
    assert result.iloc[2]["prechunk_id"] == "1111.1111#1"
    assert result.iloc[2]["postchunk_id"] == ""


def test_expand_df_skips_rows_with_no_pdf():
    df = pd.DataFrame([
        {"pdf_file_name": None, "arxiv_id": "2222.2222", "title": "T", "summary": "S",
         "authors": [], "url": "u"},
    ])
    result = expand_df(df)
    assert result.empty


def test_expand_df_continues_after_a_failed_paper():
    df = pd.DataFrame([
        {"pdf_file_name": "bad.pdf", "arxiv_id": "3333.3333", "title": "Bad", "summary": "S",
         "authors": [], "url": "u"},
        {"pdf_file_name": "good.pdf", "arxiv_id": "4444.4444", "title": "Good", "summary": "S",
         "authors": [], "url": "u"},
    ])

    def side_effect(pdf_file_name, chunk_size=512):
        if pdf_file_name == "bad.pdf":
            raise ValueError("corrupt PDF")
        return _fake_chunks(["only chunk"])

    with patch("src.ingest.load_and_chunk_pdf", side_effect=side_effect):
        result = expand_df(df)

    assert list(result["arxiv_id"]) == ["4444.4444"]
