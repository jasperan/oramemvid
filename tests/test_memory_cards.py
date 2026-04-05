import pytest
from oramemvid.memory_cards import (
    create_memory_card, get_memory_card, list_memory_cards, delete_memory_card,
)


def test_create_and_get_card(db_conn):
    card_id = create_memory_card(
        conn=db_conn, entity="Python", slot="created_by",
        value="Guido van Rossum", kind="Fact", confidence=0.95,
    )
    assert card_id is not None
    card = get_memory_card(db_conn, card_id)
    assert card["entity"] == "Python"
    assert card["slot"] == "created_by"
    assert card["value"] == "Guido van Rossum"
    assert card["kind"] == "Fact"
    assert card["confidence"] == pytest.approx(0.95)


def test_create_card_with_source_frame(db_conn):
    card_id = create_memory_card(
        conn=db_conn, entity="Oracle", slot="type",
        value="Database", kind="Fact",
    )
    card = get_memory_card(db_conn, card_id)
    assert card["source_frame_id"] is None


def test_list_cards_by_entity(db_conn):
    create_memory_card(db_conn, "Alice", "role", "Engineer", "Profile")
    create_memory_card(db_conn, "Alice", "team", "Platform", "Profile")
    create_memory_card(db_conn, "Bob", "role", "Manager", "Profile")
    alice_cards = list_memory_cards(db_conn, entity="Alice")
    assert len(alice_cards) >= 2
    assert all(c["entity"] == "Alice" for c in alice_cards)


def test_list_cards_by_kind(db_conn):
    create_memory_card(db_conn, "TestEntity", "pref", "dark mode", "Preference")
    cards = list_memory_cards(db_conn, kind="Preference")
    assert len(cards) >= 1
    assert all(c["kind"] == "Preference" for c in cards)


def test_delete_card(db_conn):
    card_id = create_memory_card(db_conn, "DeleteMe", "slot", "val", "Fact")
    assert delete_memory_card(db_conn, card_id) is True
    assert get_memory_card(db_conn, card_id) is None


def test_invalid_kind_raises(db_conn):
    with pytest.raises(Exception):
        create_memory_card(db_conn, "E", "S", "V", "InvalidKind")
