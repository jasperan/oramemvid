import pytest

from oramemvid.mcp_server import _parse_tags, create_server


@pytest.fixture
def server():
    return create_server()


@pytest.mark.asyncio
async def test_server_registers_expected_tools(server):
    tools = {
        t.name: t.description for t in await server.list_tools()
    }
    for expected in (
        "ingest_text", "search", "remember", "recall_entity",
        "list_memory_cards", "get_memory_card", "list_frames",
        "get_frame", "delete_expired_cards", "health",
    ):
        assert expected in tools, f"missing tool {expected}"
    assert "what does the system know about X" in tools["recall_entity"]


@pytest.mark.asyncio
async def test_search_tool_requires_valid_mode(server):
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="Invalid search mode"):
        await server.call_tool("search", {"query": "x", "mode": "bogus"})


def test_parse_tags_accepts_list_and_comma_strings():
    assert _parse_tags(["topic=oracle", "lang=python"]) == {
        "topic": "oracle", "lang": "python",
    }
    assert _parse_tags(["topic=oracle,lang=python"]) == {
        "topic": "oracle", "lang": "python",
    }
    assert _parse_tags(None) is None
    assert _parse_tags([]) is None


def test_parse_tags_rejects_malformed():
    with pytest.raises(ValueError, match="key=value"):
        _parse_tags(["naked"])


# --- Live-Oracle end-to-end MCP tool tests ---


@pytest.mark.asyncio
async def test_ingest_search_remember_recall_roundtrip(db_conn, server):
    """Full agent flow over the MCP surface: ingest -> search -> remember ->
    recall_entity -> delete_expired_cards."""
    from unittest.mock import patch

    from oramemvid.embeddings import OllamaEmbedding
    from oramemvid.mcp_server import _run

    # Route MCP DB calls through the fixture connection so tests share the
    # same transaction semantics as the rest of the suite.
    original_run = _run
    import oramemvid.mcp_server as mcp_module

    def fake_run(callback):
        return callback(db_conn)

    mcp_module._run = fake_run
    mcp_module._provider = None

    # Use a deterministic embedding so search returns consistent results.
    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    mcp_module._provider = provider
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        try:
            result = await server.call_tool("ingest_text", {
                "text": "MCP lets agents talk to memory backends natively mcpunique1.",
                "uri": "test://mcp/unique-1",
            })
            assert not result.is_error, result.content
            import json

            payload = json.loads(result.content[0].text)
            assert payload["total_frames"] >= 1

            search_result = await server.call_tool("search", {
                "query": "MCP memory", "mode": "text", "top_k": 5,
            })
            assert not search_result.is_error
            frames = json.loads(search_result.content[0].text)
            assert len(frames) >= 1

            remember_result = await server.call_tool("remember", {
                "entity": "MCP", "slot": "protocol", "value": "Model Context Protocol",
                "kind": "Fact", "confidence": 0.99,
            })
            assert not remember_result.is_error
            json.loads(remember_result.content[0].text)["card_id"]

            recall_result = await server.call_tool("recall_entity", {
                "entity": "MCP",
            })
            assert not recall_result.is_error
            profile = json.loads(recall_result.content[0].text)
            assert profile["entity"] == "MCP"
            assert profile["total_cards"] >= 1
            assert profile["slots"][0]["slot"] == "protocol"

            cleanup_result = await server.call_tool("delete_expired_cards", {})
            assert not cleanup_result.is_error
        finally:
            mcp_module._run = original_run
            mcp_module._provider = None


@pytest.mark.asyncio
async def test_health_tool_reports_connected(db_conn, server):
    import oramemvid.mcp_server as mcp_module

    original_run = mcp_module._run
    mcp_module._run = lambda callback: callback(db_conn)
    try:
        result = await server.call_tool("health", {})
        assert not result.is_error
        import json

        payload = json.loads(result.content[0].text)
        assert payload["status"] == "ok"
        assert payload["database"] == "connected"
        assert "capabilities" in payload
    finally:
        mcp_module._run = original_run
