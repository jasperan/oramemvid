from unittest.mock import MagicMock

import oracledb
import pytest

from oramemvid.config import Settings
from oramemvid.db import OnnxModelLoadError, _ensure_onnx_model


class _FakeBlob:
    def write(self, data):
        self.data = data


class _FakeVar:
    def getvalue(self):
        return [1]


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params or {}))
        if "DBMS_VECTOR.LOAD_ONNX_MODEL" in sql and "model_data" in (params or {}):
            raise oracledb.DatabaseError("blob loading failed")

    def fetchone(self):
        return [0]

    def var(self, _type):
        return _FakeVar()


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.commit = MagicMock()

    def cursor(self):
        return self.cursor_obj

    def createlob(self, _type):
        return _FakeBlob()


def test_onnx_directory_fallback_requires_admin_credentials(monkeypatch):
    monkeypatch.setattr("oramemvid.db._download_onnx_model", lambda _url: b"raw")
    monkeypatch.setattr("oramemvid.db._fix_onnx_for_oracle", lambda data: data)

    settings = Settings(oracle_password="test")

    with pytest.raises(OnnxModelLoadError, match="ORAMEMVID_ORACLE_ADMIN_USER"):
        _ensure_onnx_model(_FakeConnection(), "ALL_MINILM_L6_V2", settings)


def test_onnx_directory_fallback_uses_configured_admin_credentials(monkeypatch):
    monkeypatch.setattr("oramemvid.db._download_onnx_model", lambda _url: b"raw")
    monkeypatch.setattr("oramemvid.db._fix_onnx_for_oracle", lambda data: data)

    captured = {}
    admin_conn = MagicMock()
    admin_conn.cursor.return_value = MagicMock()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return admin_conn

    monkeypatch.setattr("oramemvid.db.oracledb.connect", fake_connect)

    settings = Settings(
        oracle_password="test",
        oracle_admin_user="system",
        oracle_admin_password="secret",
    )

    _ensure_onnx_model(_FakeConnection(), "ALL_MINILM_L6_V2", settings)

    assert captured["user"] == "system"
    assert captured["password"] == "secret"
