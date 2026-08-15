"""MCP server exposing oramemvid as a native agent memory backend.

Any MCP-capable agent (Claude, Codex, Cursor, ...) can ingest documents,
search frames with hybrid retrieval, and store/recall structured memory
cards through the Model Context Protocol -- without leaving the REST API.

Run with::

    uv run oramemvid-mcp

or directly::

    uv run python -m oramemvid.mcp_server

The server speaks MCP over stdio. Tools accept the same parameters as the
REST API, so a single database-backed memory layer serves both humans and
agents.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import oracledb
from mcp.server.mcpserver import MCPServer

from oramemvid.config import Settings, get_settings
from oramemvid.db import get_capabilities, get_pool, init_schema
from oramemvid.embeddings import EmbeddingProvider, get_embedding_provider
from oramemvid.entity_profile import get_entity_profile
from oramemvid.frames import get_frame, list_frames
from oramemvid.ingest import ingest_text
from oramemvid.llm import get_llm_provider
from oramemvid.memory_cards import (
    create_memory_card,
    delete_expired_cards,
    get_memory_card,
    list_memory_cards,
)
from oramemvid.search import (
    parse_tag_filters,
    search_hybrid,
    search_text,
    search_vector,
)

_settings: Settings | None = None
_provider: EmbeddingProvider | None = None


def _settings_obj() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _provider_obj() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = get_embedding_provider(_settings_obj())
    return _provider


def _run(callback):
    """Run a database-backed operation on a pooled connection."""
    pool = get_pool(_settings_obj())
    with pool.acquire() as conn:
        return callback(conn)


def create_server(name: str = "oramemvid") -> MCPServer:
    """Build the oramemvid MCP server with all memory tools registered."""
    settings = _settings_obj()

    @asynccontextmanager
    async def lifespan(_server):
        # Bootstrap the schema (tables, indexes, ONNX model) on startup,
        # mirroring the REST API lifespan behavior. Also eagerly load the
        # embedding provider here (single-threaded, deterministic) instead of
        # on the first tool call, which can happen inside a worker thread.
        pool = get_pool(settings)
        with pool.acquire() as conn:
            init_schema(conn, settings)
        _provider_obj()
        yield

    server = MCPServer(
        name=name,
        version="0.1.0",
        description=(
            "AI memory layer for agents backed by Oracle Database: ingest "
            "text, hybrid-search frames, and store/recall structured memory "
            "cards per entity."
        ),
        lifespan=lifespan,
    )

    @server.tool(
        name="ingest_text",
        description=(
            "Ingest a text snippet as one or more frame chunks (with SHA-256 "
            "deduplication) and optionally extract structured memory cards "
            "with the configured LLM. Returns frame ids and counts."
        ),
    )
    def ingest_text_tool(
        text: str,
        uri: str,
        title: str | None = None,
        extract_memories: bool = False,
    ) -> dict:
        llm = get_llm_provider(settings) if extract_memories else None
        return _run(lambda conn: ingest_text(
            conn=conn, text=text, uri=uri, provider=_provider_obj(), llm=llm,
            title=title, chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ))

    @server.tool(
        name="search",
        description=(
            "Search ingested frames. mode is 'hybrid' (text + vector with "
            "reciprocal rank fusion), 'text', or 'vector'. Supports time "
            "ranges (ISO dates), tag key=value filters, and memory card "
            "entity/kind filters. Returns ranked frames with scores."
        ),
    )
    def search_tool(
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        time_from: str | None = None,
        time_to: str | None = None,
        tags: list[str] | None = None,
        entity: str | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        filter_args = {
            "time_from": time_from, "time_to": time_to,
            "tags": parse_tag_filters(tags), "entity": entity, "kind": kind,
        }
        if mode == "text":
            return _run(lambda conn: search_text(
                conn, query, top_k=top_k, **filter_args))
        if mode == "vector":
            return _run(lambda conn: search_vector(
                conn, query, _provider_obj(), top_k=top_k, **filter_args))
        if mode == "hybrid":
            return _run(lambda conn: search_hybrid(
                conn, query, _provider_obj(), top_k=top_k, **filter_args))
        raise ValueError(f"Invalid search mode: {mode}")

    @server.tool(
        name="remember",
        description=(
            "Store a structured memory card (entity, slot, value) with an "
            "optional kind, confidence, source frame id, and expiry. Kinds: "
            "Fact, Preference, Event, Profile, Relationship, Goal. Returns "
            "the new card id."
        ),
    )
    def remember_tool(
        entity: str,
        slot: str,
        value: str,
        kind: str = "Fact",
        confidence: float = 1.0,
        source_frame_id: int | None = None,
        expires_at: str | None = None,
    ) -> dict:
        expires = None
        if expires_at is not None:
            parsed = datetime.fromisoformat(expires_at)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(UTC).replace(tzinfo=None)
            expires = parsed
        card_id = _run(lambda conn: create_memory_card(
            conn=conn, entity=entity, slot=slot, value=value, kind=kind,
            source_frame_id=source_frame_id, confidence=confidence,
            expires_at=expires,
        ))
        return {"card_id": card_id}

    @server.tool(
        name="recall_entity",
        description=(
            "Return a consolidated memory profile for an entity: all "
            "structured cards grouped by slot and ranked by confidence, "
            "with contradictions flagged and backing source frames listed. "
            "This is the 'what does the system know about X?' recall."
        ),
    )
    def recall_entity_tool(entity: str, include_expired: bool = False) -> dict:
        return _run(lambda conn: get_entity_profile(
            conn, entity, include_expired=include_expired))

    @server.tool(
        name="list_memory_cards",
        description=(
            "List structured memory cards with optional entity/kind filters. "
            "Expired cards are excluded by default."
        ),
    )
    def list_memory_cards_tool(
        entity: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[dict]:
        return _run(lambda conn: list_memory_cards(
            conn, entity=entity, kind=kind, limit=limit, offset=offset,
            include_expired=include_expired,
        ))

    @server.tool(
        name="get_memory_card",
        description="Fetch a single memory card by id.",
    )
    def get_memory_card_tool(card_id: int) -> dict | None:
        return _run(lambda conn: get_memory_card(conn, card_id))

    @server.tool(
        name="list_frames",
        description="List ingested frames (content chunks).",
    )
    def list_frames_tool(limit: int = 20, offset: int = 0) -> list[dict]:
        return _run(lambda conn: list_frames(conn, limit=limit, offset=offset))

    @server.tool(
        name="get_frame",
        description="Fetch a single frame (content chunk) by id.",
    )
    def get_frame_tool(frame_id: int) -> dict | None:
        return _run(lambda conn: get_frame(conn, frame_id))

    @server.tool(
        name="delete_expired_cards",
        description=(
            "Delete memory cards whose expires_at has passed. Returns the "
            "number of cards removed (temporal-memory hygiene)."
        ),
    )
    def delete_expired_cards_tool() -> dict:
        return {"deleted": _run(lambda conn: delete_expired_cards(conn))}

    @server.tool(
        name="health",
        description=(
            "Report database connectivity and Oracle capabilities "
            "(Oracle Text, vector index, degraded reasons)."
        ),
    )
    def health_tool() -> dict:
        try:
            return _run(lambda conn: {
                "status": "ok",
                "database": "connected",
                "capabilities": get_capabilities().as_dict(),
            })
        except (oracledb.Error, ValueError) as exc:  # pragma: no cover - defensive
            return {"status": "degraded", "database": str(exc)}

    return server


def main() -> None:
    """stdio entrypoint used by the ``oramemvid-mcp`` console script."""
    server = create_server()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
