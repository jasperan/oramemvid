import os
import pytest
from unittest.mock import patch
from oramemvid.ingest import extract_text, chunk_text, ingest_file, ingest_text
from oramemvid.embeddings import OllamaEmbedding

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_chunk_text_basic():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        words = chunk.split()
        assert len(words) <= 110


def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_chunk_text_overlap():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    overlap = set(first_words[-10:]) & set(second_words[:10])
    assert len(overlap) > 0


def test_extract_text_txt():
    path = os.path.join(FIXTURES, "sample.txt")
    text = extract_text(path)
    assert "Oracle Database" in text
    assert "VECTOR_DISTANCE" in text


def test_ingest_text_creates_frames(db_conn):
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        result = ingest_text(
            conn=db_conn, text="Test content for ingest unique zzz111. " * 50,
            uri="test://ingest-text-unique1", provider=provider,
        )
    assert result["total_frames"] >= 1
    assert result["uri"] == "test://ingest-text-unique1"


def test_ingest_file_txt(db_conn):
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    path = os.path.join(FIXTURES, "sample.txt")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        result = ingest_file(conn=db_conn, file_path=path, provider=provider)
    assert result["doc_id"] is not None
    assert result["total_frames"] >= 1
    assert result["filename"] == "sample.txt"


def test_ingest_duplicate_file_skips(db_conn):
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    path = os.path.join(FIXTURES, "sample.txt")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        r1 = ingest_file(conn=db_conn, file_path=path, provider=provider)
        r2 = ingest_file(conn=db_conn, file_path=path, provider=provider)
    assert r2["skipped"] is True
    assert r2["doc_id"] == r1["doc_id"]
