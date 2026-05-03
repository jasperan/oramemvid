import os
from unittest.mock import patch
from oramemvid.embeddings import OllamaEmbedding
from oramemvid.llm import OllamaLLM
from oramemvid.ingest import ingest_file
from oramemvid.search import search_text, search_vector, search_hybrid
from oramemvid.frames import list_frames
from oramemvid.memory_cards import list_memory_cards

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _mock_embed(text):
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    values = [int(h[i:i+2], 16) / 255.0 for i in range(0, len(h), 2)]
    return (values + [0.0] * 384)[:384]


def _mock_extract(content):
    return [
        {"entity": "Oracle", "slot": "feature", "value": "vector search", "kind": "Fact", "confidence": 0.9}
    ]


def test_full_pipeline(db_conn, tmp_path):
    """End-to-end: ingest file, search, verify memories."""
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="all-minilm")
    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    path = tmp_path / "sample-integration.txt"
    with open(os.path.join(FIXTURES, "sample.txt"), encoding="utf-8") as fixture:
        path.write_text(fixture.read() + f"\nunique integration path: {tmp_path}\n", encoding="utf-8")

    with (
        patch.object(provider, "embed", side_effect=_mock_embed),
        patch.object(llm, "extract_memories", side_effect=_mock_extract),
    ):
        result = ingest_file(conn=db_conn, file_path=str(path), provider=provider, llm=llm)

    assert result["total_frames"] >= 1
    assert result["skipped"] is False

    # Verify frames exist
    frames = list_frames(db_conn, doc_id=result["doc_id"])
    assert len(frames) == result["total_frames"]

    # Verify memory cards were created
    cards = list_memory_cards(db_conn)
    assert len(cards) >= 1
    oracle_cards = [c for c in cards if c["entity"] == "Oracle"]
    assert len(oracle_cards) >= 1

    # Verify text search works
    text_results = search_text(db_conn, "Oracle", top_k=5)
    assert len(text_results) >= 1

    # Verify vector search works
    with patch.object(provider, "embed", side_effect=_mock_embed):
        vec_results = search_vector(db_conn, "database", provider=provider, top_k=5)
    assert len(vec_results) >= 1

    # Verify hybrid search works
    with patch.object(provider, "embed", side_effect=_mock_embed):
        hybrid_results = search_hybrid(db_conn, "Oracle Database", provider=provider, top_k=5)
    assert len(hybrid_results) >= 1
