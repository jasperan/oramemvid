from abc import ABC, abstractmethod

import httpx

from oramemvid.config import Settings


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def is_in_database(self) -> bool:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class OracleONNXEmbedding(EmbeddingProvider):
    """In-database embeddings via VECTOR_EMBEDDING(). No data leaves Oracle.

    This provider does NOT compute embeddings in Python. Instead, it provides
    SQL fragments for use in INSERT/SELECT statements.
    """

    def __init__(self, model_name: str = "ALL_MINILM_L6_V2"):
        self.model_name = model_name

    @property
    def is_in_database(self) -> bool:
        return True

    def sql_fragment(self, bind_var: str = ":content") -> str:
        return f"VECTOR_EMBEDDING({self.model_name} USING {bind_var} AS data)"

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "OracleONNXEmbedding computes embeddings in SQL. "
            "Use sql_fragment() in your INSERT/SELECT statement."
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "OracleONNXEmbedding computes embeddings in SQL. "
            "Use sql_fragment() in your INSERT/SELECT statement."
        )


class OllamaEmbedding(EmbeddingProvider):
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    @property
    def is_in_database(self) -> bool:
        return False

    def embed(self, text: str) -> list[float]:
        resp = httpx.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "oracle_onnx":
        return OracleONNXEmbedding(model_name=settings.onnx_model_name.upper())
    elif settings.embedding_provider == "ollama":
        return OllamaEmbedding(
            ollama_url=settings.ollama_url,
            model=settings.ollama_embed_model,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
