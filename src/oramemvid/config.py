from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    oracle_dsn: str = "localhost:1523/FREEPDB1"
    oracle_user: str = "oramemvid"
    oracle_password: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    embedding_provider: str = "oracle_onnx"
    onnx_model_name: str = "all_minilm_l6_v2"
    ollama_embed_model: str = "nomic-embed-text"
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = {"env_prefix": "ORAMEMVID_", "env_file": ".env"}


def get_settings() -> Settings:
    return Settings()
