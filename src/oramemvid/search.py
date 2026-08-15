import array
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import oracledb

from oramemvid.embeddings import EmbeddingProvider
from oramemvid.memory_cards import VALID_KINDS

_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_TAG_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ORACLE_TEXT_TOKEN_LIMIT = 32
_ORACLE_TEXT_OPERATORS = {"AND", "OR", "NOT", "ACCUM", "NEAR", "WITHIN"}
TEXT_SEARCH_MODES = {"auto", "oracle_text", "degraded_lob"}


class SearchCapabilityError(RuntimeError):
    """Raised when a requested non-degraded search capability is unavailable."""


def _read_content(val):
    return val.read() if hasattr(val, "read") else val


def _sanitize_text_query(query: str) -> str:
    tokens = [
        token for token in _TEXT_TOKEN_RE.findall(query)
        if token.upper() not in _ORACLE_TEXT_OPERATORS
    ]
    return " AND ".join(f"{{{token}}}" for token in tokens[:_ORACLE_TEXT_TOKEN_LIMIT])


def parse_tag_filters(tags: list[str] | None) -> dict[str, str] | None:
    """Parse ``key=value`` tag filters (comma-separated, possibly across
    multiple list items) into a dict. Returns None for empty input.

    This is the canonical tag parser for the search layer: the REST API
    translates its ValueError into HTTP 400, while the MCP server lets it
    surface as a tool error.
    """
    if not tags:
        return None
    parsed: dict[str, str] = {}
    for raw in tags:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError("Tag filters must use key=value syntax")
            key, value = item.split("=", 1)
            key, value = key.strip(), value.strip()
            if not key or not value:
                raise ValueError("Tag filters must include both key and value")
            parsed[key] = value
    return parsed or None


def _parse_time_bound(value: str, *, is_end: bool) -> tuple[str, datetime]:
    raw = value.strip()
    try:
        parsed_date = date.fromisoformat(raw)
        if is_end:
            return "<", datetime.combine(parsed_date + timedelta(days=1), time.min)
        return ">=", datetime.combine(parsed_date, time.min)
    except ValueError:
        pass

    try:
        parsed_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "time filters must be ISO dates or datetimes, such as "
            "2026-05-03 or 2026-05-03T12:30:00Z"
        ) from exc

    if parsed_dt.tzinfo is not None:
        parsed_dt = parsed_dt.astimezone(timezone.utc).replace(tzinfo=None)
    return ("<=" if is_end else ">="), parsed_dt


def _build_filters(
    *,
    time_from: str | None = None,
    time_to: str | None = None,
    tags: dict[str, Any] | None = None,
    entity: str | None = None,
    kind: str | None = None,
    frame_alias: str = "f",
) -> tuple[str, dict]:
    conditions = []
    params: dict[str, Any] = {}

    if time_from:
        op, value = _parse_time_bound(time_from, is_end=False)
        conditions.append(f"{frame_alias}.created_at {op} :time_from")
        params["time_from"] = value
    if time_to:
        op, value = _parse_time_bound(time_to, is_end=True)
        conditions.append(f"{frame_alias}.created_at {op} :time_to")
        params["time_to"] = value

    for index, (key, value) in enumerate((tags or {}).items()):
        if not _TAG_KEY_RE.fullmatch(key):
            raise ValueError(f"Invalid tag key: {key!r}")
        param_name = f"tag_{index}"
        conditions.append(
            f"JSON_VALUE({frame_alias}.tags, '$.\"{key}\"') = :{param_name}"
        )
        params[param_name] = str(value)

    card_conditions = []
    if entity:
        card_conditions.append("mc.entity = :entity")
        params["entity"] = entity
    if kind:
        if kind not in VALID_KINDS:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
        card_conditions.append("mc.kind = :kind")
        params["kind"] = kind
    if card_conditions:
        conditions.append(
            "EXISTS ("
            "SELECT 1 FROM memory_cards mc "
            f"WHERE mc.source_frame_id = {frame_alias}.frame_id "
            f"AND {' AND '.join(card_conditions)}"
            ")"
        )

    if not conditions:
        return "", params
    return " AND " + " AND ".join(conditions), params


def search_text(
    conn: oracledb.Connection, query: str, top_k: int = 10,
    time_from: str | None = None, time_to: str | None = None,
    tags: dict[str, Any] | None = None,
    entity: str | None = None,
    kind: str | None = None,
    mode: str = "auto",
) -> list[dict]:
    if mode not in TEXT_SEARCH_MODES:
        raise ValueError(f"Invalid text search mode: {mode}")
    cursor = conn.cursor()
    safe_query = _sanitize_text_query(query)
    if mode == "degraded_lob":
        safe_query = ""
    filter_sql, filter_params = _build_filters(
        time_from=time_from, time_to=time_to, tags=tags,
        entity=entity, kind=kind,
    )
    params: dict = {"query": safe_query, "top_k": top_k, **filter_params}

    # Try Oracle Text CONTAINS first unless the caller explicitly selected
    # the degraded CLOB-safe fallback.
    if safe_query and mode != "degraded_lob":
        try:
            cursor.execute(f"""
                SELECT f.frame_id, f.uri, f.title, f.content, f.content_hash, f.created_at,
                       SCORE(1) AS relevance
                FROM frames f
                WHERE CONTAINS(f.content, :query, 1) > 0 {filter_sql}
                ORDER BY relevance DESC
                FETCH FIRST :top_k ROWS ONLY
            """, params)
        except oracledb.DatabaseError as exc:
            if mode == "oracle_text":
                raise SearchCapabilityError(
                    "Oracle Text search is required but CONTAINS failed."
                ) from exc
            safe_query = ""
        else:
            return [
                {
                    "frame_id": r[0], "uri": r[1], "title": r[2],
                    "content": _read_content(r[3]), "content_hash": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                    "score": float(r[6]),
                }
                for r in cursor.fetchall()
            ]

    if mode == "oracle_text" and not safe_query:
        return []

    if not safe_query:
        # Fallback to DBMS_LOB.INSTR for CLOB-safe text search
        if not query.strip():
            return []
        params["like_query"] = query
        cursor.execute(f"""
            SELECT f.frame_id, f.uri, f.title, f.content, f.content_hash, f.created_at,
                   1 AS relevance
            FROM frames f
            WHERE DBMS_LOB.INSTR(f.content, :like_query) > 0 {filter_sql}
            ORDER BY f.created_at DESC
            FETCH FIRST :top_k ROWS ONLY
        """, params)

    return [
        {
            "frame_id": r[0], "uri": r[1], "title": r[2],
            "content": _read_content(r[3]), "content_hash": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
            "score": float(r[6]),
        }
        for r in cursor.fetchall()
    ]


def search_vector(
    conn: oracledb.Connection, query: str, provider: EmbeddingProvider,
    top_k: int = 10, time_from: str | None = None, time_to: str | None = None,
    tags: dict[str, Any] | None = None,
    entity: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    cursor = conn.cursor()
    filter_sql, filter_params = _build_filters(
        time_from=time_from, time_to=time_to, tags=tags,
        entity=entity, kind=kind,
    )
    params: dict = {"top_k": top_k, **filter_params}

    if provider.is_in_database:
        params["query"] = query
        sql = f"""
            SELECT f.frame_id, f.uri, f.title, f.content, f.content_hash, f.created_at,
                   VECTOR_DISTANCE(f.embedding, {provider.sql_fragment(':query')}, COSINE) AS distance
            FROM frames f WHERE f.embedding IS NOT NULL {filter_sql}
            ORDER BY distance ASC
            FETCH FIRST :top_k ROWS ONLY
        """
    else:
        query_vec = provider.embed(query)
        params["query_vec"] = array.array('f', query_vec)
        sql = f"""
            SELECT f.frame_id, f.uri, f.title, f.content, f.content_hash, f.created_at,
                   VECTOR_DISTANCE(f.embedding, :query_vec, COSINE) AS distance
            FROM frames f WHERE f.embedding IS NOT NULL {filter_sql}
            ORDER BY distance ASC
            FETCH FIRST :top_k ROWS ONLY
        """

    cursor.execute(sql, params)
    return [
        {
            "frame_id": r[0], "uri": r[1], "title": r[2],
            "content": _read_content(r[3]), "content_hash": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
            "score": 1.0 - float(r[6]) if r[6] is not None else 0.0,
        }
        for r in cursor.fetchall()
    ]


def search_hybrid(
    conn: oracledb.Connection, query: str, provider: EmbeddingProvider,
    top_k: int = 10, rrf_k: int = 60,
    time_from: str | None = None, time_to: str | None = None,
    tags: dict[str, Any] | None = None,
    entity: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    text_results = search_text(
        conn, query, top_k=top_k * 2, time_from=time_from, time_to=time_to,
        tags=tags, entity=entity, kind=kind,
    )
    vector_results = search_vector(
        conn, query, provider, top_k=top_k * 2, time_from=time_from, time_to=time_to,
        tags=tags, entity=entity, kind=kind,
    )

    scores: dict[int, float] = {}
    frame_data: dict[int, dict] = {}
    for rank, result in enumerate(text_results):
        fid = result["frame_id"]
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (rrf_k + rank + 1)
        frame_data[fid] = result
    for rank, result in enumerate(vector_results):
        fid = result["frame_id"]
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (rrf_k + rank + 1)
        frame_data[fid] = result

    sorted_ids = sorted(scores, key=lambda fid: scores[fid], reverse=True)[:top_k]
    return [{**frame_data[fid], "score": scores[fid]} for fid in sorted_ids]
