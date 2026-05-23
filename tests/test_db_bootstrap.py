from types import SimpleNamespace
from unittest.mock import MagicMock

import oracledb
import pytest

from oramemvid.config import Settings
from oramemvid.db import (
    DatabaseCapabilities,
    OnnxModelLoadError,
    _detect_capabilities,
    _enforce_required_capabilities,
    _ensure_onnx_model,
    _onnx2oracle_spec,
)


class _FakeVar:
    def getvalue(self):
        return [1]


class _FakeCursor:
    def __init__(self, fail_load=False):
        self.calls = []
        self.fail_load = fail_load

    def execute(self, sql, params=None):
        self.calls.append((sql, params or {}))
        if self.fail_load and "DBMS_VECTOR.LOAD_ONNX_MODEL" in sql:
            raise oracledb.DatabaseError("model loading failed")

    def fetchone(self):
        return [0]

    def var(self, _type):
        return _FakeVar()


class _FakeConnection:
    def __init__(self, fail_load=False):
        self.cursor_obj = _FakeCursor(fail_load=fail_load)
        self.commit = MagicMock()

    def cursor(self):
        return self.cursor_obj


def test_onnx_bootstrap_loads_onnx2oracle_model(monkeypatch):
    monkeypatch.setattr("oramemvid.db._build_oracle_onnx_model", lambda _name: b"augmented")
    monkeypatch.setattr("oramemvid.db._onnx_metadata_json", lambda: '{"function":"embedding"}')

    settings = Settings(oracle_password="test")
    conn = _FakeConnection()

    _ensure_onnx_model(conn, "ALL_MINILM_L6_V2", settings)

    load_calls = [
        (sql, params)
        for sql, params in conn.cursor_obj.calls
        if "DBMS_VECTOR.LOAD_ONNX_MODEL" in sql
    ]
    assert len(load_calls) == 1
    assert load_calls[0][1]["model_name"] == "ALL_MINILM_L6_V2"
    assert load_calls[0][1]["model_data"] == b"augmented"
    assert load_calls[0][1]["metadata"] == '{"function":"embedding"}'
    conn.commit.assert_called_once()


def test_onnx_bootstrap_rejects_presets_with_wrong_dimensions():
    presets = {
        "bad": SimpleNamespace(oracle_name="NOMIC_EMBED_TEXT_V1", dims=768),
    }

    with pytest.raises(OnnxModelLoadError, match="schema requires 384"):
        _onnx2oracle_spec("NOMIC_EMBED_TEXT_V1", presets=presets)


def test_onnx_bootstrap_reports_missing_onnx2oracle(monkeypatch):
    def missing_dependency(_name):
        raise ImportError("no onnx2oracle")

    monkeypatch.setattr("oramemvid.db._build_oracle_onnx_model", missing_dependency)

    settings = Settings(oracle_password="test")

    with pytest.raises(OnnxModelLoadError, match="onnx2oracle optional dependency"):
        _ensure_onnx_model(_FakeConnection(), "ALL_MINILM_L6_V2", settings)


def test_onnx_bootstrap_reports_oracle_load_errors(monkeypatch):
    monkeypatch.setattr("oramemvid.db._build_oracle_onnx_model", lambda _name: b"augmented")
    monkeypatch.setattr("oramemvid.db._onnx_metadata_json", lambda: '{"function":"embedding"}')

    settings = Settings(oracle_password="test")

    with pytest.raises(OnnxModelLoadError, match="onnx2oracle-built ONNX model"):
        _ensure_onnx_model(_FakeConnection(fail_load=True), "ALL_MINILM_L6_V2", settings)


class _CapabilityCursor:
    def __init__(self, indexes: set[str]):
        self.indexes = indexes
        self._count = 0

    def execute(self, _sql, params=None):
        self._count = int((params or {}).get("name", "") in self.indexes)

    def fetchone(self):
        return [self._count]


def test_detect_capabilities_reports_missing_indexes():
    capabilities = _detect_capabilities(_CapabilityCursor(set()))

    assert not capabilities.oracle_text
    assert not capabilities.vector_index
    assert "oracle_text" in capabilities.degraded_reasons
    assert "vector_index" in capabilities.degraded_reasons


def test_enforce_required_capabilities_fails_explicitly():
    capabilities = DatabaseCapabilities(
        oracle_text=False,
        vector_index=True,
        degraded_reasons={"oracle_text": "missing text index"},
    )
    settings = Settings(oracle_password="test", require_oracle_text=True)

    with pytest.raises(RuntimeError, match="missing text index"):
        _enforce_required_capabilities(capabilities, settings)
