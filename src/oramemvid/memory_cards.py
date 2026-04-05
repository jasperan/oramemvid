import oracledb

VALID_KINDS = {"Fact", "Preference", "Event", "Profile", "Relationship", "Goal"}


def create_memory_card(
    conn: oracledb.Connection,
    entity: str, slot: str, value: str, kind: str,
    source_frame_id: int | None = None,
    confidence: float = 1.0, expires_at=None,
) -> int:
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of: {VALID_KINDS}")
    cursor = conn.cursor()
    card_id_var = cursor.var(oracledb.NUMBER)
    cursor.execute("""
        INSERT INTO memory_cards
            (entity, slot, card_value, kind, source_frame_id, confidence, expires_at)
        VALUES (:entity, :slot, :value, :kind, :frame_id, :confidence, :expires_at)
        RETURNING card_id INTO :card_id
    """, {
        "entity": entity, "slot": slot, "value": value, "kind": kind,
        "frame_id": source_frame_id, "confidence": confidence,
        "expires_at": expires_at, "card_id": card_id_var,
    })
    conn.commit()
    return int(card_id_var.getvalue()[0])


def get_memory_card(conn: oracledb.Connection, card_id: int) -> dict | None:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT card_id, entity, slot, card_value, kind,
               source_frame_id, confidence, created_at, expires_at
        FROM memory_cards WHERE card_id = :id
    """, {"id": card_id})
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "card_id": row[0], "entity": row[1], "slot": row[2],
        "value": row[3].read() if hasattr(row[3], "read") else row[3],
        "kind": row[4], "source_frame_id": row[5],
        "confidence": float(row[6]) if row[6] is not None else None,
        "created_at": row[7].isoformat() if row[7] else None,
        "expires_at": row[8].isoformat() if row[8] else None,
    }


def list_memory_cards(
    conn: oracledb.Connection, entity: str | None = None,
    kind: str | None = None, source_frame_id: int | None = None,
    limit: int = 50, offset: int = 0,
) -> list[dict]:
    conditions = []
    params: dict = {"off": offset, "lim": limit}
    if entity is not None:
        conditions.append("entity = :entity")
        params["entity"] = entity
    if kind is not None:
        conditions.append("kind = :kind")
        params["kind"] = kind
    if source_frame_id is not None:
        conditions.append("source_frame_id = :frame_id")
        params["frame_id"] = source_frame_id
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT card_id, entity, slot, card_value, kind,
               source_frame_id, confidence, created_at
        FROM memory_cards {where}
        ORDER BY card_id
        OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
    """, params)
    return [
        {
            "card_id": r[0], "entity": r[1], "slot": r[2],
            "value": r[3].read() if hasattr(r[3], "read") else r[3],
            "kind": r[4], "source_frame_id": r[5],
            "confidence": float(r[6]) if r[6] is not None else None,
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in cursor.fetchall()
    ]


def delete_memory_card(conn: oracledb.Connection, card_id: int) -> bool:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_cards WHERE card_id = :id", {"id": card_id})
    deleted = cursor.rowcount > 0
    conn.commit()
    return deleted
