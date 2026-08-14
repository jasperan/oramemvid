from datetime import UTC

import pytest

from oramemvid.entity_profile import (
    CONTRADICTION_CONFIDENCE_THRESHOLD,
    _detect_contradiction,
    _normalize_value,
    get_entity_profile,
)
from oramemvid.memory_cards import create_memory_card

# --- Pure unit tests (no Oracle) ---


def test_normalize_value_folds_case_and_whitespace():
    assert _normalize_value("  Oracle  Database ") == "oracle database"


def test_contradiction_requires_two_confident_distinct_values():
    assert _detect_contradiction([
        {"value": "Oracle", "confidence": 0.9},
        {"value": "oracle", "confidence": 0.8},  # same normalized value
    ]) is False


def test_contradiction_detected_on_distinct_confident_values():
    assert _detect_contradiction([
        {"value": "Oracle", "confidence": 0.9},
        {"value": "MySQL", "confidence": 0.8},
    ]) is True


def test_contradiction_ignores_low_confidence_values():
    assert _detect_contradiction([
        {"value": "Oracle", "confidence": 0.9},
        {"value": "MySQL", "confidence": 0.1},  # below threshold
    ]) is False


def test_contradiction_threshold_constant_is_reasonable():
    assert 0.0 < CONTRADICTION_CONFIDENCE_THRESHOLD <= 1.0


# --- Live-Oracle integration tests ---


@pytest.fixture
def seeded_profile(db_conn):
    """Entity with two slots: one consistent, one contradictory."""
    created = []
    created.append(create_memory_card(
        db_conn, entity="RecallEntity", slot="created_by",
        value="Ada Lovelace", kind="Fact", confidence=0.95,
    ))
    created.append(create_memory_card(
        db_conn, entity="RecallEntity", slot="created_by",
        value="Ada Lovelace", kind="Fact", confidence=0.7,
    ))
    created.append(create_memory_card(
        db_conn, entity="RecallEntity", slot="hometown",
        value="London", kind="Fact", confidence=0.9,
    ))
    created.append(create_memory_card(
        db_conn, entity="RecallEntity", slot="hometown",
        value="Venice", kind="Fact", confidence=0.8,
    ))
    yield created
    for card_id in created:
        from oramemvid.memory_cards import delete_memory_card

        delete_memory_card(db_conn, card_id)


def test_entity_profile_groups_and_ranks_slots(db_conn, seeded_profile):
    profile = get_entity_profile(db_conn, "RecallEntity")

    assert profile["entity"] == "RecallEntity"
    assert profile["total_cards"] == 4
    slots = {s["slot"]: s for s in profile["slots"]}

    # Values within a slot are ranked by confidence descending.
    created_by = slots["created_by"]["values"]
    assert created_by[0]["confidence"] == pytest.approx(0.95)
    assert created_by[1]["confidence"] == pytest.approx(0.7)


def test_entity_profile_flags_contradictions(db_conn, seeded_profile):
    profile = get_entity_profile(db_conn, "RecallEntity")
    slots = {s["slot"]: s for s in profile["slots"]}

    # created_by has one distinct value -> no contradiction.
    assert slots["created_by"]["contradiction"] is False
    # hometown has two distinct confident values -> contradiction.
    assert slots["hometown"]["contradiction"] is True


def test_entity_profile_expired_cards_excluded_by_default(db_conn):
    from datetime import datetime, timedelta

    past = datetime.now(UTC) - timedelta(days=1)
    future = datetime.now(UTC) + timedelta(days=1)
    expired_id = create_memory_card(
        db_conn, entity="ExpiryEntity", slot="status", value="stale",
        kind="Fact", expires_at=past,
    )
    active_id = create_memory_card(
        db_conn, entity="ExpiryEntity", slot="status", value="fresh",
        kind="Fact", expires_at=future,
    )
    try:
        profile = get_entity_profile(db_conn, "ExpiryEntity")
        assert profile["total_cards"] == 1
        assert profile["expired_cards"] == 1
        assert profile["slots"][0]["values"][0]["value"] == "fresh"

        profile_all = get_entity_profile(
            db_conn, "ExpiryEntity", include_expired=True,
        )
        assert profile_all["total_cards"] == 2
        assert profile_all["expired_cards"] == 1
    finally:
        from oramemvid.memory_cards import delete_memory_card

        delete_memory_card(db_conn, expired_id)
        delete_memory_card(db_conn, active_id)


def test_entity_profile_sources_listed(db_conn):
    from unittest.mock import patch

    from oramemvid.embeddings import OllamaEmbedding
    from oramemvid.frames import create_frame

    provider = OllamaEmbedding(ollama_url="http://localhost:11434", model="nomic-embed-text")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        frame_id = create_frame(
            db_conn, uri="test://profile-source", content="Ada Lovelace wrote notes on engines.",
            provider=provider,
        )
    card_id = create_memory_card(
        db_conn, entity="SourceEntity", slot="occupation", value="Mathematician",
        kind="Fact", source_frame_id=frame_id,
    )
    try:
        profile = get_entity_profile(db_conn, "SourceEntity")
        assert any(s["frame_id"] == frame_id for s in profile["sources"])
        assert profile["slots"][0]["values"][0]["source_frame_id"] == frame_id
    finally:
        from oramemvid.frames import delete_frame
        from oramemvid.memory_cards import delete_memory_card

        delete_memory_card(db_conn, card_id)
        delete_frame(db_conn, frame_id)
