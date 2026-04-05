from oramemvid.config import Settings


def test_default_settings():
    s = Settings(oracle_password="test")
    assert s.oracle_dsn == "localhost:1523/FREEPDB1"
    assert s.oracle_user == "oramemvid"
    assert s.ollama_url == "http://localhost:11434"
    assert s.ollama_model == "qwen3.5:9b"
    assert s.embedding_provider in ("oracle_onnx", "ollama")
    assert s.onnx_model_name == "all_minilm_l6_v2"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("ORAMEMVID_ORACLE_DSN", "remotehost:1521/ORCLPDB1")
    monkeypatch.setenv("ORAMEMVID_ORACLE_PASSWORD", "secret")
    s = Settings()
    assert s.oracle_dsn == "remotehost:1521/ORCLPDB1"
    assert s.oracle_password == "secret"


def test_embedding_provider_validation():
    s = Settings(oracle_password="test", embedding_provider="ollama")
    assert s.embedding_provider == "ollama"
