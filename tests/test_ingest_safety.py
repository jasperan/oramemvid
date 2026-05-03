from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oramemvid.ingest import chunk_text, ingest_file, ingest_text


class _FakeVar:
    def getvalue(self):
        return [1]


class _FakeCursor:
    def __init__(self):
        self._fetchone = None

    def execute(self, sql, params=None):
        if "SELECT doc_id" in sql or "SELECT frame_id" in sql:
            self._fetchone = None

    def fetchone(self):
        return self._fetchone

    def var(self, _type):
        return _FakeVar()


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.commit = MagicMock()
        self.rollback = MagicMock()

    def cursor(self):
        return self.cursor_obj


class _FailingEmbeddingProvider:
    is_in_database = False

    def embed(self, _text):
        raise RuntimeError("embedding failed")


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_text_rejects_invalid_chunk_windows(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_ingest_file_rolls_back_when_frame_insert_fails(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("Oracle vector search content", encoding="utf-8")
    conn = _FakeConnection()

    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest_file(conn, str(path), provider=_FailingEmbeddingProvider())

    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_ingest_text_empty_input_does_not_commit():
    conn = _FakeConnection()

    result = ingest_text(conn, "", "test://empty", provider=_FailingEmbeddingProvider())

    assert result == {"uri": "test://empty", "total_frames": 0, "frame_ids": []}
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()
