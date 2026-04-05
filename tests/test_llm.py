import json
import pytest
from unittest.mock import patch, MagicMock
from oramemvid.llm import LLMProvider, OllamaLLM, get_llm_provider
from oramemvid.config import Settings

MOCK_EXTRACTION_RESPONSE = json.dumps([
    {"entity": "Python", "slot": "created_by", "value": "Guido van Rossum", "kind": "Fact", "confidence": 0.95},
    {"entity": "Python", "slot": "first_release", "value": "1991", "kind": "Event", "confidence": 0.90},
])


@patch("httpx.post")
def test_ollama_complete(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Hello world"}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp
    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    result = llm.complete("Say hello")
    assert result == "Hello world"


@patch("httpx.post")
def test_ollama_extract_memories(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": MOCK_EXTRACTION_RESPONSE}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp
    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    cards = llm.extract_memories("Python was created by Guido van Rossum in 1991.")
    assert len(cards) == 2
    assert cards[0]["entity"] == "Python"
    assert cards[0]["slot"] == "created_by"
    assert cards[0]["kind"] == "Fact"
    assert cards[1]["kind"] == "Event"


@patch("httpx.post")
def test_ollama_extract_memories_bad_json_returns_empty(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "not valid json ["}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp
    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    cards = llm.extract_memories("some content")
    assert cards == []


def test_get_llm_provider():
    settings = Settings(oracle_password="test")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OllamaLLM)
