import array
import hashlib
import json

import oracledb

from oramemvid.embeddings import EmbeddingProvider


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_frame(
    conn: oracledb.Connection,
    uri: str,
    content: str,
    provider: EmbeddingProvider,
    title: str | None = None,
    doc_id: int | None = None,
    tags: dict | None = None,
) -> int:
    content_hash = _hash_content(content)
    cursor = conn.cursor()

    # Check for duplicate
    cursor.execute(
        "SELECT frame_id FROM frames WHERE content_hash = :hash",
        {"hash": content_hash},
    )
    existing = cursor.fetchone()
    if existing:
        return existing[0]

    tags_json = json.dumps(tags) if tags else None

    if provider.is_in_database:
        sql = f"""
            INSERT INTO frames (uri, title, content, content_hash, doc_id, tags, embedding)
            VALUES (:uri, :title, :content, :hash, :doc_id, :tags,
                    {provider.sql_fragment(':content')})
            RETURNING frame_id INTO :frame_id
        """
        frame_id_var = cursor.var(oracledb.NUMBER)
        cursor.execute(sql, {
            "uri": uri, "title": title, "content": content,
            "hash": content_hash, "doc_id": doc_id, "tags": tags_json,
            "frame_id": frame_id_var,
        })
    else:
        embedding_list = provider.embed(content)
        # Oracle VECTOR columns need array.array('f', ...) for proper binding
        embedding = array.array("f", embedding_list)
        sql = """
            INSERT INTO frames (uri, title, content, content_hash, doc_id, tags, embedding)
            VALUES (:uri, :title, :content, :hash, :doc_id, :tags, :embedding)
            RETURNING frame_id INTO :frame_id
        """
        frame_id_var = cursor.var(oracledb.NUMBER)
        cursor.execute(sql, {
            "uri": uri, "title": title, "content": content,
            "hash": content_hash, "doc_id": doc_id, "tags": tags_json,
            "embedding": embedding, "frame_id": frame_id_var,
        })

    conn.commit()
    return int(frame_id_var.getvalue()[0])


def get_frame(conn: oracledb.Connection, frame_id: int) -> dict | None:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT frame_id, uri, title, content, content_hash,
               encoding, doc_id, tags, created_at
        FROM frames WHERE frame_id = :id
        """,
        {"id": frame_id},
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "frame_id": row[0],
        "uri": row[1],
        "title": row[2],
        "content": row[3].read() if hasattr(row[3], "read") else row[3],
        "content_hash": row[4],
        "encoding": row[5],
        "doc_id": row[6],
        "tags": json.loads(row[7]) if row[7] else None,
        "created_at": row[8].isoformat() if row[8] else None,
    }


def list_frames(
    conn: oracledb.Connection,
    limit: int = 20,
    offset: int = 0,
    doc_id: int | None = None,
) -> list[dict]:
    cursor = conn.cursor()
    if doc_id is not None:
        cursor.execute(
            """
            SELECT frame_id, uri, title, content_hash, encoding, doc_id, created_at
            FROM frames WHERE doc_id = :doc_id
            ORDER BY frame_id
            OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
            """,
            {"doc_id": doc_id, "off": offset, "lim": limit},
        )
    else:
        cursor.execute(
            """
            SELECT frame_id, uri, title, content_hash, encoding, doc_id, created_at
            FROM frames
            ORDER BY frame_id
            OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
            """,
            {"off": offset, "lim": limit},
        )
    return [
        {
            "frame_id": r[0], "uri": r[1], "title": r[2],
            "content_hash": r[3], "encoding": r[4], "doc_id": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in cursor.fetchall()
    ]


def delete_frame(conn: oracledb.Connection, frame_id: int) -> bool:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_cards WHERE source_frame_id = :id", {"id": frame_id})
    cursor.execute("DELETE FROM frames WHERE frame_id = :id", {"id": frame_id})
    deleted = cursor.rowcount > 0
    conn.commit()
    return deleted
