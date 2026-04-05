"""FastAPI REST API wiring all oramemvid modules together."""

import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel

from oramemvid.config import get_settings
from oramemvid.db import get_pool, close_pool, init_schema
from oramemvid.embeddings import get_embedding_provider
from oramemvid.frames import create_frame, get_frame, list_frames, delete_frame
from oramemvid.ingest import ingest_file, ingest_text
from oramemvid.llm import get_llm_provider
from oramemvid.memory_cards import (
    create_memory_card,
    get_memory_card,
    list_memory_cards,
    delete_memory_card,
)
from oramemvid.search import search_text, search_vector, search_hybrid

settings = get_settings()
embedding_provider = get_embedding_provider(settings)
llm_provider = get_llm_provider(settings)


def _get_conn():
    return get_pool().acquire()


def _release_conn(conn):
    get_pool().release(conn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = get_pool(settings)
    with pool.acquire() as conn:
        init_schema(conn)
    yield
    close_pool()


app = FastAPI(title="oramemvid", version="0.1.0", lifespan=lifespan)


# --- Request models ---


class IngestTextRequest(BaseModel):
    text: str
    uri: str
    title: str | None = None
    extract_memories: bool = False


# --- Ingest routes ---


@app.post("/ingest/text")
def route_ingest_text(req: IngestTextRequest):
    conn = _get_conn()
    try:
        llm = llm_provider if req.extract_memories else None
        result = ingest_text(
            conn=conn,
            text=req.text,
            uri=req.uri,
            provider=embedding_provider,
            llm=llm,
            title=req.title,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return result
    finally:
        _release_conn(conn)


@app.post("/ingest/file")
def route_ingest_file(
    file: UploadFile,
    extract_memories: bool = Query(False),
):
    conn = _get_conn()
    try:
        suffix = ""
        if file.filename:
            parts = file.filename.rsplit(".", 1)
            if len(parts) == 2:
                suffix = f".{parts[1]}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        llm = llm_provider if extract_memories else None
        result = ingest_file(
            conn=conn,
            file_path=tmp_path,
            provider=embedding_provider,
            llm=llm,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return result
    finally:
        _release_conn(conn)


# --- Frame routes ---


@app.get("/frames")
def route_list_frames(
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    doc_id: int | None = Query(None),
):
    conn = _get_conn()
    try:
        return list_frames(conn, limit=limit, offset=offset, doc_id=doc_id)
    finally:
        _release_conn(conn)


@app.get("/frames/{frame_id}")
def route_get_frame(frame_id: int):
    conn = _get_conn()
    try:
        frame = get_frame(conn, frame_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        return frame
    finally:
        _release_conn(conn)


@app.delete("/frames/{frame_id}")
def route_delete_frame(frame_id: int):
    conn = _get_conn()
    try:
        deleted = delete_frame(conn, frame_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Frame not found")
        return {"deleted": True}
    finally:
        _release_conn(conn)


# --- Search routes ---


@app.get("/search")
def route_search(
    query: str = Query(...),
    mode: str = Query("hybrid", pattern="^(text|vector|hybrid)$"),
    top_k: int = Query(10, ge=1, le=100),
    time_from: str | None = Query(None),
    time_to: str | None = Query(None),
):
    conn = _get_conn()
    try:
        if mode == "text":
            return search_text(conn, query, top_k=top_k, time_from=time_from, time_to=time_to)
        elif mode == "vector":
            return search_vector(conn, query, embedding_provider, top_k=top_k, time_from=time_from, time_to=time_to)
        else:
            return search_hybrid(conn, query, embedding_provider, top_k=top_k, time_from=time_from, time_to=time_to)
    finally:
        _release_conn(conn)


@app.get("/search/text")
def route_search_text(
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    time_from: str | None = Query(None),
    time_to: str | None = Query(None),
):
    conn = _get_conn()
    try:
        return search_text(conn, query, top_k=top_k, time_from=time_from, time_to=time_to)
    finally:
        _release_conn(conn)


@app.get("/search/vector")
def route_search_vector(
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    time_from: str | None = Query(None),
    time_to: str | None = Query(None),
):
    conn = _get_conn()
    try:
        return search_vector(conn, query, embedding_provider, top_k=top_k, time_from=time_from, time_to=time_to)
    finally:
        _release_conn(conn)


# --- Memory card routes ---


@app.get("/memory")
def route_list_memory_cards(
    entity: str | None = Query(None),
    kind: str | None = Query(None),
    source_frame_id: int | None = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    conn = _get_conn()
    try:
        return list_memory_cards(
            conn, entity=entity, kind=kind,
            source_frame_id=source_frame_id, limit=limit, offset=offset,
        )
    finally:
        _release_conn(conn)


@app.get("/memory/{card_id}")
def route_get_memory_card(card_id: int):
    conn = _get_conn()
    try:
        card = get_memory_card(conn, card_id)
        if card is None:
            raise HTTPException(status_code=404, detail="Memory card not found")
        return card
    finally:
        _release_conn(conn)


@app.post("/memory/extract/{frame_id}")
def route_extract_memories(frame_id: int):
    conn = _get_conn()
    try:
        frame = get_frame(conn, frame_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        cards = llm_provider.extract_memories(frame["content"])
        card_ids = []
        for card in cards:
            card_id = create_memory_card(
                conn=conn,
                entity=card.get("entity", "unknown"),
                slot=card.get("slot", "unknown"),
                value=card.get("value", ""),
                kind=card.get("kind", "Fact"),
                source_frame_id=frame_id,
                confidence=card.get("confidence", 1.0),
            )
            card_ids.append(card_id)
        return {"frame_id": frame_id, "cards_created": len(card_ids), "card_ids": card_ids}
    finally:
        _release_conn(conn)


@app.delete("/memory/{card_id}")
def route_delete_memory_card(card_id: int):
    conn = _get_conn()
    try:
        deleted = delete_memory_card(conn, card_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory card not found")
        return {"deleted": True}
    finally:
        _release_conn(conn)


# --- Document routes ---


@app.get("/documents")
def route_list_documents(
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT doc_id, filename, doc_type, file_hash, total_frames, ingested_at
            FROM documents
            ORDER BY doc_id
            OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
        """, {"off": offset, "lim": limit})
        return [
            {
                "doc_id": r[0], "filename": r[1], "doc_type": r[2],
                "file_hash": r[3], "total_frames": r[4],
                "ingested_at": r[5].isoformat() if r[5] else None,
            }
            for r in cursor.fetchall()
        ]
    finally:
        _release_conn(conn)


# --- Stats ---


@app.get("/stats")
def route_stats():
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM frames")
        frame_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM memory_cards")
        card_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM documents")
        document_count = cursor.fetchone()[0]
        return {
            "frame_count": frame_count,
            "card_count": card_count,
            "document_count": document_count,
        }
    finally:
        _release_conn(conn)


# --- Health ---


@app.get("/health")
def route_health():
    try:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM dual")
            cursor.fetchone()
            return {"status": "ok", "database": "connected"}
        finally:
            _release_conn(conn)
    except Exception as exc:
        return {"status": "degraded", "database": str(exc)}
