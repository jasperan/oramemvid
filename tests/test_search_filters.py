from datetime import datetime

import pytest

from oramemvid.search import (
    _build_filters,
    _parse_time_bound,
    _sanitize_text_query,
)


def test_sanitize_text_query_strips_oracle_text_operators():
    query = 'Oracle (Database) AND "vector-search"'

    sanitized = _sanitize_text_query(query)

    assert sanitized == "{Oracle} AND {Database} AND {vector} AND {search}"
    assert "(" not in sanitized
    assert '"' not in sanitized


def test_time_to_date_uses_exclusive_next_day_bound():
    op, value = _parse_time_bound("2026-05-03", is_end=True)

    assert op == "<"
    assert value == datetime(2026, 5, 4)


def test_time_datetime_z_is_normalized_to_naive_utc():
    op, value = _parse_time_bound("2026-05-03T01:02:03Z", is_end=False)

    assert op == ">="
    assert value == datetime(2026, 5, 3, 1, 2, 3)


def test_build_filters_includes_tags_and_memory_card_filters():
    sql, params = _build_filters(
        time_from="2026-05-01",
        time_to="2026-05-03",
        tags={"topic": "oracle"},
        entity="Oracle",
        kind="Fact",
    )

    assert "JSON_VALUE(f.tags" in sql
    assert "EXISTS (SELECT 1 FROM memory_cards mc" in sql
    assert "mc.entity = :entity" in sql
    assert "mc.kind = :kind" in sql
    assert params["tag_0"] == "oracle"
    assert params["entity"] == "Oracle"
    assert params["kind"] == "Fact"


def test_build_filters_rejects_invalid_tag_key():
    with pytest.raises(ValueError, match="Invalid tag key"):
        _build_filters(tags={"bad.key": "value"})


def test_build_filters_rejects_invalid_memory_kind():
    with pytest.raises(ValueError, match="Invalid kind"):
        _build_filters(kind="Unknown")
