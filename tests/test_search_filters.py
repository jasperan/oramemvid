from datetime import datetime

import oracledb
import pytest

from oramemvid.search import (
    SearchCapabilityError,
    _build_filters,
    _parse_time_bound,
    _sanitize_text_query,
    search_text,
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


class _TextSearchCursor:
    def __init__(self, *, fail_contains: bool = False):
        self.fail_contains = fail_contains
        self.calls: list[str] = []
        self.rows = [
            (1, "mem://one", "One", "Oracle text", "hash", datetime(2026, 5, 1), 1)
        ]

    def execute(self, sql, params=None):
        self.calls.append(sql)
        if self.fail_contains and "CONTAINS" in sql:
            raise oracledb.DatabaseError("Oracle Text unavailable")

    def fetchall(self):
        return self.rows


class _TextSearchConn:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj


def test_search_text_auto_uses_named_degraded_lob_fallback():
    cursor = _TextSearchCursor(fail_contains=True)

    results = search_text(_TextSearchConn(cursor), "Oracle", mode="auto")

    assert results[0]["frame_id"] == 1
    assert any("CONTAINS" in sql for sql in cursor.calls)
    assert any("DBMS_LOB.INSTR" in sql for sql in cursor.calls)


def test_search_text_strict_mode_raises_when_oracle_text_fails():
    cursor = _TextSearchCursor(fail_contains=True)

    with pytest.raises(SearchCapabilityError):
        search_text(_TextSearchConn(cursor), "Oracle", mode="oracle_text")


def test_search_text_degraded_mode_skips_oracle_text():
    cursor = _TextSearchCursor(fail_contains=True)

    search_text(_TextSearchConn(cursor), "Oracle", mode="degraded_lob")

    assert not any("CONTAINS" in sql for sql in cursor.calls)
    assert any("DBMS_LOB.INSTR" in sql for sql in cursor.calls)
