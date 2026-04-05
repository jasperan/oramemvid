import pytest
from oramemvid.frames import create_frame, get_frame, list_frames, delete_frame
from oramemvid.embeddings import OllamaEmbedding
from unittest.mock import patch


@pytest.fixture
def mock_ollama_provider():
    return OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )


def _mock_embed(*args, **kwargs):
    return [0.1] * 384


def test_create_and_get_frame(db_conn, mock_ollama_provider):
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        frame_id = create_frame(
            conn=db_conn,
            uri="test://doc1/chunk0",
            content="The quick brown fox jumps over the lazy dog.",
            provider=mock_ollama_provider,
            title="Test Frame",
        )
    assert frame_id is not None
    assert isinstance(frame_id, int)

    frame = get_frame(db_conn, frame_id)
    assert frame["uri"] == "test://doc1/chunk0"
    assert frame["title"] == "Test Frame"
    assert "quick brown fox" in frame["content"]
    assert frame["content_hash"] is not None


def test_create_duplicate_frame_returns_existing(db_conn, mock_ollama_provider):
    content = "Duplicate content test string."
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        id1 = create_frame(
            conn=db_conn, uri="test://dup1", content=content, provider=mock_ollama_provider,
        )
        id2 = create_frame(
            conn=db_conn, uri="test://dup2", content=content, provider=mock_ollama_provider,
        )
    assert id1 == id2


def test_list_frames(db_conn, mock_ollama_provider):
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        create_frame(db_conn, uri="test://list1", content="List test content one unique xyzabc.", provider=mock_ollama_provider)
        create_frame(db_conn, uri="test://list2", content="List test content two unique xyzdef.", provider=mock_ollama_provider)
    frames = list_frames(db_conn, limit=100)
    assert len(frames) >= 2


def test_delete_frame(db_conn, mock_ollama_provider):
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        frame_id = create_frame(
            db_conn, uri="test://delete-me", content="Content to be deleted unique string zzzqqq.", provider=mock_ollama_provider,
        )
    deleted = delete_frame(db_conn, frame_id)
    assert deleted is True
    assert get_frame(db_conn, frame_id) is None


def test_get_nonexistent_frame(db_conn):
    assert get_frame(db_conn, 999999) is None
