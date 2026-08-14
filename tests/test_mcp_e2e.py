"""End-to-end MCP test: spawn the real oramemvid MCP server as a subprocess
over stdio and drive it through the Model Context Protocol client, exactly
as an agent (Claude, Codex, Cursor) would.

Requires a live Oracle database (see conftest.py); skipped otherwise.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.oracle

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_ENTRY = REPO_ROOT / "src" / "oramemvid" / "mcp_server.py"


def test_mcp_stdio_subprocess_roundtrip(db_pool, settings):
    """Full agent flow over stdio: initialize, list tools, ingest, search,
    remember, recall, health."""
    from mcp import ClientSession, StdioServerParameters, stdio_client

    env = dict(os.environ)
    # Point the spawned server at the same database the fixture uses.
    env.update({
        "ORAMEMVID_ORACLE_DSN": settings.oracle_dsn,
        "ORAMEMVID_ORACLE_USER": settings.oracle_user,
        "ORAMEMVID_ORACLE_PASSWORD": settings.oracle_password,
        "ORAMEMVID_EMBEDDING_PROVIDER": "sentence_transformers",
    })

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_ENTRY)],
        env=env,
    )

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert {
                "ingest_text", "search", "remember", "recall_entity",
                "list_memory_cards", "get_memory_card", "list_frames",
                "get_frame", "delete_expired_cards", "health",
            } <= tool_names

            # Ingest a unique text chunk through the real MCP protocol.
            ingest_res = await session.call_tool("ingest_text", {
                "text": "oramemvid speaks MCP to agents over stdio e2eunique9.",
                "uri": "test://mcp-e2e/unique-9",
            })
            payload = json.loads(ingest_res.content[0].text)
            assert payload["total_frames"] >= 1

            # Search back through the protocol.
            search_res = await session.call_tool("search", {
                "query": "MCP", "mode": "text", "top_k": 5,
            })
            frames = json.loads(search_res.content[0].text)
            assert len(frames) >= 1

            # Store a card and recall the entity profile.
            await session.call_tool("remember", {
                "entity": "MCPE2E", "slot": "transport", "value": "stdio",
                "kind": "Fact", "confidence": 0.95,
            })
            recall_res = await session.call_tool("recall_entity", {
                "entity": "MCPE2E",
            })
            profile = json.loads(recall_res.content[0].text)
            assert profile["entity"] == "MCPE2E"
            assert profile["total_cards"] >= 1

            health_res = await session.call_tool("health", {})
            health = json.loads(health_res.content[0].text)
            assert health["status"] == "ok"

    # Run the MCP protocol inside the pytest-asyncio loop.
    asyncio.run(run())
