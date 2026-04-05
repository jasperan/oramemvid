import pytest
from unittest.mock import patch
from oramemvid.search import search_text, search_vector, search_hybrid
from oramemvid.frames import create_frame
from oramemvid.embeddings import OllamaEmbedding


@pytest.fixture
def seeded_frames(db_conn):
    """Insert some frames for search testing."""
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    contents = [
        "Oracle Database supports vector embeddings natively for search zzzsearch1.",
        "Python is a popular programming language for data science zzzsearch2.",
        "FastAPI is a modern web framework for building APIs with Python zzzsearch3.",
    ]
    ids = []
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        for i, c in enumerate(contents):
            fid = create_frame(conn=db_conn, uri=f"test://search-unique/{i}", content=c, provider=provider)
            ids.append(fid)
    return ids


def test_search_text(db_conn, seeded_frames):
    results = search_text(db_conn, "Oracle Database", top_k=5)
    assert len(results) >= 1
    assert any("Oracle" in r["content"] for r in results)


def test_search_text_no_results(db_conn, seeded_frames):
    results = search_text(db_conn, "xyznonexistent999qqq", top_k=5)
    assert len(results) == 0


def test_search_vector(db_conn, seeded_frames):
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        results = search_vector(db_conn, "database embeddings", provider=provider, top_k=5)
    assert len(results) >= 1


def test_search_hybrid(db_conn, seeded_frames):
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        results = search_hybrid(db_conn, "Python web framework", provider=provider, top_k=5)
    assert len(results) >= 1


def test_search_text_with_time_filter(db_conn, seeded_frames):
    results = search_text(db_conn, "Python", top_k=5, time_from="2020-01-01", time_to="2099-12-31")
    assert len(results) >= 1
