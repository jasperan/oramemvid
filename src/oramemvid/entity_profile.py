"""Entity memory profiles: consolidated, confidence-ranked knowledge about
an entity, built from the structured memory cards and their source frames.

A profile answers the agent-facing question "what does the system know
about X?" by grouping cards by slot, ranking values by confidence, and
flagging contradictions (the same slot holding conflicting values).
"""

from __future__ import annotations

import oracledb

from oramemvid.frames import get_frame
from oramemvid.memory_cards import count_memory_cards, list_memory_cards

CONTRADICTION_CONFIDENCE_THRESHOLD = 0.5
PROFILE_MAX_SOURCES = 10


def _normalize_value(value: str) -> str:
    return " ".join(value.strip().lower().split())


def get_entity_profile(
    conn: oracledb.Connection,
    entity: str,
    *,
    include_expired: bool = False,
    max_sources: int = PROFILE_MAX_SOURCES,
) -> dict:
    """Build a consolidated memory profile for a single entity.

    Returns a dict with:

    - ``entity``: the entity name
    - ``total_cards`` / ``expired_cards``: card counts (expired excluded from
      the profile by default)
    - ``slots``: list of ``{slot, values, contradiction}`` where ``values``
      are ranked by confidence (descending) and ``contradiction`` is True
      when the slot holds two or more confident, distinct values
    - ``sources``: up to ``max_sources`` source frames backing the cards
    """
    cards = list_memory_cards(
        conn, entity=entity, limit=10000, include_expired=include_expired,
    )

    total = count_memory_cards(conn, entity=entity, include_expired=True)
    valid = count_memory_cards(conn, entity=entity, include_expired=False)
    expired_count = max(0, total - valid)

    slots: dict[str, dict] = {}
    source_ids: set[int] = set()
    for card in cards:
        slot = card["slot"]
        entry = slots.setdefault(
            slot,
            {"slot": slot, "values": [], "contradiction": False},
        )
        entry["values"].append({
            "card_id": card["card_id"],
            "value": card["value"],
            "kind": card["kind"],
            "confidence": card["confidence"],
            "created_at": card["created_at"],
            "expires_at": card["expires_at"],
            "source_frame_id": card["source_frame_id"],
        })
        if card["source_frame_id"] is not None:
            source_ids.add(card["source_frame_id"])

    slot_list = []
    for slot in slots.values():
        slot["values"].sort(
            key=lambda v: (v["confidence"] if v["confidence"] is not None else -1.0),
            reverse=True,
        )
        slot["contradiction"] = _detect_contradiction(slot["values"])
        slot_list.append(slot)
    slot_list.sort(key=lambda s: s["slot"].lower())

    sources = []
    for frame_id in sorted(source_ids)[:max_sources]:
        frame = get_frame(conn, frame_id)
        if frame is not None:
            sources.append({
                "frame_id": frame["frame_id"],
                "uri": frame["uri"],
                "title": frame["title"],
            })

    return {
        "entity": entity,
        "total_cards": len(cards),
        "expired_cards": expired_count,
        "slots": slot_list,
        "sources": sources,
    }


def _detect_contradiction(values: list[dict]) -> bool:
    """Return True when a slot holds two confident, distinct values.

    A value is "confident" when its confidence is at least
    CONTRADICTION_CONFIDENCE_THRESHOLD. Distinctness is decided on
    normalized (case/whitespace folded) text, so "Oracle" and "oracle"
    are treated as the same value while "Oracle" and "MySQL" conflict.
    """
    distinct: set[str] = set()
    for value in values:
        confidence = value["confidence"] if value["confidence"] is not None else 0.0
        if confidence >= CONTRADICTION_CONFIDENCE_THRESHOLD:
            distinct.add(_normalize_value(value["value"]))
    return len(distinct) >= 2
