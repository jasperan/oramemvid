import pytest
from unittest.mock import patch, MagicMock
from oramemvid.embeddings import (
    EmbeddingProvider,
    OracleONNXEmbedding,
    OllamaEmbedding,
    get_embedding_provider,
)
from oramemvid.config import Settings


def test_oracle_onnx_sql_fragment():
    provider = OracleONNXEmbedding(model_name="ALL_MINILM_L6_V2")
    fragment = provider.sql_fragment()
    assert "VECTOR_EMBEDDING" in fragment
    assert "ALL_MINILM_L6_V2" in fragment


def test_oracle_onnx_is_in_database():
    provider = OracleONNXEmbedding(model_name="ALL_MINILM_L6_V2")
    assert provider.is_in_database is True


def test_ollama_is_not_in_database():
    settings = Settings(oracle_password="test")
    provider = OllamaEmbedding(
        ollama_url=settings.ollama_url,
        model=settings.ollama_embed_model,
    )
    assert provider.is_in_database is False


@patch("httpx.post")
def test_ollama_embed(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [[0.1] * 384]}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    result = provider.embed("hello world")
    assert len(result) == 384
    assert result[0] == pytest.approx(0.1)


@patch("httpx.post")
def test_ollama_embed_batch(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [[0.1] * 384, [0.2] * 384]}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    results = provider.embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 384


def test_get_provider_oracle():
    settings = Settings(oracle_password="test", embedding_provider="oracle_onnx")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, OracleONNXEmbedding)


def test_get_provider_ollama():
    settings = Settings(oracle_password="test", embedding_provider="ollama")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, OllamaEmbedding)
