import array

import oracledb

from oramemvid.embeddings import EmbeddingProvider


def _read_content(val):
    return val.read() if hasattr(val, "read") else val


def search_text(
    conn: oracledb.Connection, query: str, top_k: int = 10,
    time_from: str | None = None, time_to: str | None = None,
) -> list[dict]:
    cursor = conn.cursor()
    params: dict = {"query": query, "top_k": top_k}
    time_conditions = ""
    if time_from:
        time_conditions += " AND created_at >= TO_TIMESTAMP(:time_from, 'YYYY-MM-DD')"
        params["time_from"] = time_from
    if time_to:
        time_conditions += " AND created_at <= TO_TIMESTAMP(:time_to, 'YYYY-MM-DD')"
        params["time_to"] = time_to

    # Try Oracle Text CONTAINS first
    try:
        cursor.execute(f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   SCORE(1) AS relevance
            FROM frames
            WHERE CONTAINS(content, :query, 1) > 0 {time_conditions}
            ORDER BY relevance DESC
            FETCH FIRST :top_k ROWS ONLY
        """, params)
    except oracledb.DatabaseError:
        # Fallback to DBMS_LOB.INSTR for CLOB-safe text search
        params["like_query"] = query
        cursor.execute(f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   1 AS relevance
            FROM frames
            WHERE DBMS_LOB.INSTR(content, :like_query) > 0 {time_conditions}
            ORDER BY created_at DESC
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
) -> list[dict]:
    cursor = conn.cursor()
    time_conditions = ""
    params: dict = {"top_k": top_k}
    if time_from:
        time_conditions += " AND created_at >= TO_TIMESTAMP(:time_from, 'YYYY-MM-DD')"
        params["time_from"] = time_from
    if time_to:
        time_conditions += " AND created_at <= TO_TIMESTAMP(:time_to, 'YYYY-MM-DD')"
        params["time_to"] = time_to

    if provider.is_in_database:
        params["query"] = query
        sql = f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   VECTOR_DISTANCE(embedding, {provider.sql_fragment(':query')}, COSINE) AS distance
            FROM frames WHERE embedding IS NOT NULL {time_conditions}
            ORDER BY distance ASC
            FETCH FIRST :top_k ROWS ONLY
        """
    else:
        query_vec = provider.embed(query)
        params["query_vec"] = array.array('f', query_vec)
        sql = f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   VECTOR_DISTANCE(embedding, :query_vec, COSINE) AS distance
            FROM frames WHERE embedding IS NOT NULL {time_conditions}
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
) -> list[dict]:
    text_results = search_text(conn, query, top_k=top_k * 2, time_from=time_from, time_to=time_to)
    vector_results = search_vector(conn, query, provider, top_k=top_k * 2, time_from=time_from, time_to=time_to)

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
