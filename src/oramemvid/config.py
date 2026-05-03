from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    oracle_dsn: str = "localhost:1523/FREEPDB1"
    oracle_user: str = "oramemvid"
    oracle_password: str = ""
    oracle_admin_user: str | None = None
    oracle_admin_password: str | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    embedding_provider: Literal["oracle_onnx", "ollama", "sentence_transformers"] = "oracle_onnx"
    onnx_model_name: str = "all_minilm_l6_v2"
    onnx_directory_name: str = "onnx_model_dir"
    ollama_embed_model: str = "nomic-embed-text"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_upload_bytes: int = 50 * 1024 * 1024
    allowed_upload_extensions: str = ".txt,.pdf,.docx,.xlsx,.pptx"

    model_config = {"env_prefix": "ORAMEMVID_", "env_file": ".env"}

    @field_validator("chunk_size", "max_upload_bytes")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def _validate_chunk_window(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def upload_extension_set(self) -> set[str]:
        return {
            ext.strip().lower()
            for ext in self.allowed_upload_extensions.split(",")
            if ext.strip()
        }


def get_settings() -> Settings:
    return Settings()
