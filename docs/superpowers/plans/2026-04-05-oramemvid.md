# oramemvid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python AI memory layer backed by Oracle Database, reimplementing memvid's core concepts (frames, memory cards, search) with Oracle Text, Oracle AI Vector Search, and in-database ONNX embeddings.

**Architecture:** Flat Python package with pluggable embedding and LLM providers. Oracle DB handles storage, full-text search (Oracle Text CONTAINS), vector similarity (VECTOR_DISTANCE with HNSW index), and embedding generation (VECTOR_EMBEDDING with loaded ONNX model). FastAPI serves the REST API. Document ingestion pipeline extracts text, chunks it into frames, and optionally runs LLM-based memory extraction.

**Tech Stack:** Python 3.12, oracledb, FastAPI, uvicorn, pymupdf, python-docx, openpyxl, python-pptx, httpx, pydantic-settings, pytest

**Spec:** `docs/superpowers/specs/2026-04-05-oramemvid-design.md`

---

## File Map

| File | Responsibility | Created in |
|------|---------------|------------|
| `pyproject.toml` | Package metadata, dependencies | Task 1 |
| `docker-compose.yml` | Oracle 26ai Free container | Task 1 |
| `.gitignore` | Ignore patterns | Task 1 |
| `.env.example` | Environment template | Task 1 |
| `CLAUDE.md` | Project-specific Claude instructions | Task 1 |
| `src/oramemvid/__init__.py` | Package init, version | Task 1 |
| `src/oramemvid/config.py` | Settings from env vars | Task 2 |
| `tests/conftest.py` | Shared fixtures (DB connection, cleanup) | Task 3 |
| `src/oramemvid/db.py` | Connection pool, schema init, ONNX loading | Task 3 |
| `src/oramemvid/embeddings.py` | Pluggable embedding providers | Task 4 |
| `src/oramemvid/frames.py` | Frame CRUD + queries | Task 5 |
| `src/oramemvid/memory_cards.py` | MemoryCard CRUD + queries | Task 6 |
| `src/oramemvid/llm.py` | Pluggable LLM interface | Task 7 |
| `src/oramemvid/ingest.py` | Document extraction + chunking pipeline | Task 8 |
| `src/oramemvid/search.py` | Unified text/vector/hybrid search | Task 9 |
| `src/oramemvid/api.py` | FastAPI routes | Task 10 |
| `tests/test_config.py` | Config tests | Task 2 |
| `tests/test_db.py` | DB schema tests | Task 3 |
| `tests/test_embeddings.py` | Embedding provider tests | Task 4 |
| `tests/test_frames.py` | Frame CRUD tests | Task 5 |
| `tests/test_memory_cards.py` | MemoryCard CRUD tests | Task 6 |
| `tests/test_llm.py` | LLM provider tests | Task 7 |
| `tests/test_ingest.py` | Ingestion pipeline tests | Task 8 |
| `tests/test_search.py` | Search tests | Task 9 |
| `tests/test_api.py` | API endpoint tests | Task 10 |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `CLAUDE.md`
- Create: `src/oramemvid/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/ubuntu/git/personal/oramemvid
git init
```

- [ ] **Step 2: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "oramemvid"
version = "0.1.0"
description = "AI memory layer backed by Oracle Database, inspired by memvid"
requires-python = ">=3.12"
dependencies = [
    "oracledb>=2.0",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "pymupdf>=1.24",
    "python-docx>=1.1",
    "openpyxl>=3.1",
    "python-pptx>=0.6",
    "httpx>=0.27",
    "pydantic-settings>=2.0",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  oracle-db:
    image: container-registry.oracle.com/database/free:latest
    ports:
      - "1523:1521"
    environment:
      ORACLE_PWD: oramemvid_dev
    volumes:
      - oracle_data:/opt/oracle/oradata
    healthcheck:
      test: ["CMD", "sqlplus", "-L", "sys/oramemvid_dev@//localhost:1521/FREEPDB1 as sysdba", "@/dev/null"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 120s

volumes:
  oracle_data:
```

- [ ] **Step 4: Create .gitignore**

Create `.gitignore`:

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.env
*.onnx
.agents/
.claude/
.crush/
.openhands/
.serena/
dogfood-output/
docs/plans/
task_plan.md
findings.md
progress.md
.playwright-mcp/
content/
.omc/
```

- [ ] **Step 5: Create .env.example**

Create `.env.example`:

```
ORAMEMVID_ORACLE_DSN=localhost:1523/FREEPDB1
ORAMEMVID_ORACLE_USER=oramemvid
ORAMEMVID_ORACLE_PASSWORD=oramemvid_dev
ORAMEMVID_OLLAMA_URL=http://localhost:11434
ORAMEMVID_OLLAMA_MODEL=qwen3.5:9b
ORAMEMVID_EMBEDDING_PROVIDER=oracle_onnx
ORAMEMVID_ONNX_MODEL_NAME=all_minilm_l6_v2
```

- [ ] **Step 6: Create src/oramemvid/__init__.py**

Create `src/oramemvid/__init__.py`:

```python
"""oramemvid: AI memory layer backed by Oracle Database."""

__version__ = "0.1.0"
```

- [ ] **Step 7: Create CLAUDE.md**

Create `CLAUDE.md`:

```markdown
# oramemvid

AI memory layer backed by Oracle Database, inspired by memvid.

## Quick Start

```bash
conda activate oramemvid
docker compose up -d
pip install -e ".[dev]"
python -m oramemvid.db  # init schema
uvicorn oramemvid.api:app --reload
```

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

- Python 3.12, conda env: oramemvid
- Oracle 26ai Free on port 1523/FREEPDB1
- FastAPI on port 8000
- Ollama at localhost:11434 (qwen3.5:9b for extraction, nomic-embed-text fallback)
- In-database ONNX embeddings (all-MiniLM-L6-v2)

## Key Modules

- `config.py` — pydantic-settings from ORAMEMVID_* env vars
- `db.py` — Oracle connection pool, schema init, ONNX model loading
- `embeddings.py` — pluggable: OracleONNXEmbedding (default), OllamaEmbedding (fallback)
- `frames.py` — frame CRUD (content chunks with embeddings)
- `memory_cards.py` — structured memory CRUD (entity/slot/value)
- `llm.py` — pluggable LLM, default OllamaLLM (qwen3.5:9b)
- `ingest.py` — document pipeline (PDF, DOCX, XLSX, PPTX, TXT)
- `search.py` — unified text/vector/hybrid search with RRF
- `api.py` — FastAPI REST routes
```

- [ ] **Step 8: Create conda env and install**

```bash
conda create -n oramemvid python=3.12 -y
conda activate oramemvid
pip install -e ".[dev]"
```

- [ ] **Step 9: Commit scaffolding**

```bash
git add pyproject.toml docker-compose.yml .gitignore .env.example CLAUDE.md src/oramemvid/__init__.py
git commit -m "feat: project scaffolding for oramemvid"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/oramemvid/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config defaults**

Create `tests/test_config.py`:

```python
from oramemvid.config import Settings


def test_default_settings():
    s = Settings(oracle_password="test")
    assert s.oracle_dsn == "localhost:1523/FREEPDB1"
    assert s.oracle_user == "oramemvid"
    assert s.ollama_url == "http://localhost:11434"
    assert s.ollama_model == "qwen3.5:9b"
    assert s.embedding_provider == "oracle_onnx"
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.config'`

- [ ] **Step 3: Implement config.py**

Create `src/oramemvid/config.py`:

```python
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

    model_config = {"env_prefix": "ORAMEMVID_"}


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/config.py tests/test_config.py
git commit -m "feat: config module with pydantic-settings"
```

---

## Task 3: Database Module

**Files:**
- Create: `src/oramemvid/db.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing test for schema init**

Create `tests/conftest.py`:

```python
import pytest
import oracledb
from oramemvid.config import Settings


@pytest.fixture(scope="session")
def settings():
    return Settings()


@pytest.fixture(scope="session")
def db_pool(settings):
    pool = oracledb.create_pool(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn,
        min=1,
        max=5,
    )
    yield pool
    pool.close()


@pytest.fixture
def db_conn(db_pool):
    conn = db_pool.acquire()
    yield conn
    conn.rollback()
    db_pool.release(conn)


@pytest.fixture(scope="session", autouse=True)
def init_schema(db_pool):
    from oramemvid.db import init_schema
    with db_pool.acquire() as conn:
        init_schema(conn)
```

Create `tests/test_db.py`:

```python
def test_frames_table_exists(db_conn):
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT table_name FROM user_tables WHERE table_name = 'FRAMES'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "FRAMES"


def test_memory_cards_table_exists(db_conn):
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT table_name FROM user_tables WHERE table_name = 'MEMORY_CARDS'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "MEMORY_CARDS"


def test_documents_table_exists(db_conn):
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT table_name FROM user_tables WHERE table_name = 'DOCUMENTS'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "DOCUMENTS"


def test_frames_columns(db_conn):
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT column_name FROM user_tab_columns "
        "WHERE table_name = 'FRAMES' ORDER BY column_id"
    )
    cols = [r[0] for r in cursor.fetchall()]
    expected = [
        "FRAME_ID", "URI", "TITLE", "CONTENT", "CONTENT_HASH",
        "ENCODING", "DOC_ID", "TAGS", "EMBEDDING", "CREATED_AT",
    ]
    assert cols == expected


def test_schema_version_table(db_conn):
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT version FROM schema_version ORDER BY applied_at DESC FETCH FIRST 1 ROW ONLY"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.db'`

- [ ] **Step 3: Implement db.py**

Create `src/oramemvid/db.py`:

```python
import oracledb
from oramemvid.config import Settings

SCHEMA_VERSION = 1

_pool: oracledb.ConnectionPool | None = None


def get_pool(settings: Settings | None = None) -> oracledb.ConnectionPool:
    global _pool
    if _pool is None:
        if settings is None:
            settings = Settings()
        _pool = oracledb.create_pool(
            user=settings.oracle_user,
            password=settings.oracle_password,
            dsn=settings.oracle_dsn,
            min=2,
            max=10,
        )
    return _pool


def close_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM user_tables WHERE table_name = :name",
        {"name": table_name.upper()},
    )
    return cursor.fetchone()[0] > 0


def init_schema(conn: oracledb.Connection):
    cursor = conn.cursor()

    if not _table_exists(cursor, "SCHEMA_VERSION"):
        cursor.execute("""
            CREATE TABLE schema_version (
                version    NUMBER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT SYSTIMESTAMP
            )
        """)

    cursor.execute(
        "SELECT MAX(version) FROM schema_version"
    )
    row = cursor.fetchone()
    current = row[0] if row[0] is not None else 0

    if current < 1:
        _apply_v1(cursor)
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (1)"
        )

    conn.commit()


def _apply_v1(cursor):
    # documents first (referenced by frames)
    if not _table_exists(cursor, "DOCUMENTS"):
        cursor.execute("""
            CREATE TABLE documents (
                doc_id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                filename      VARCHAR2(1000) NOT NULL,
                doc_type      VARCHAR2(20) NOT NULL,
                file_hash     VARCHAR2(64) NOT NULL,
                total_frames  NUMBER DEFAULT 0,
                ingested_at   TIMESTAMP DEFAULT SYSTIMESTAMP,
                CONSTRAINT documents_hash_uk UNIQUE (file_hash)
            )
        """)

    if not _table_exists(cursor, "FRAMES"):
        cursor.execute("""
            CREATE TABLE frames (
                frame_id      NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                uri           VARCHAR2(1000) NOT NULL,
                title         VARCHAR2(500),
                content       CLOB NOT NULL,
                content_hash  VARCHAR2(64) NOT NULL,
                encoding      VARCHAR2(20) DEFAULT 'raw',
                doc_id        NUMBER REFERENCES documents(doc_id),
                tags          JSON,
                embedding     VECTOR(384, FLOAT32),
                created_at    TIMESTAMP DEFAULT SYSTIMESTAMP,
                CONSTRAINT frames_hash_uk UNIQUE (content_hash)
            )
        """)

        # Oracle Text index for full-text search
        cursor.execute("""
            CREATE INDEX frames_content_idx ON frames(content)
                INDEXTYPE IS CTXSYS.CONTEXT
                PARAMETERS ('SYNC (ON COMMIT)')
        """)

        # Vector index for similarity search
        cursor.execute("""
            CREATE VECTOR INDEX frames_vec_idx ON frames(embedding)
                ORGANIZATION INMEMORY NEIGHBOR GRAPH
        """)

    if not _table_exists(cursor, "MEMORY_CARDS"):
        cursor.execute("""
            CREATE TABLE memory_cards (
                card_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                entity          VARCHAR2(500) NOT NULL,
                slot            VARCHAR2(500) NOT NULL,
                card_value      CLOB NOT NULL,
                kind            VARCHAR2(50) NOT NULL,
                source_frame_id NUMBER REFERENCES frames(frame_id),
                confidence      NUMBER(3,2) DEFAULT 1.0,
                created_at      TIMESTAMP DEFAULT SYSTIMESTAMP,
                expires_at      TIMESTAMP,
                CONSTRAINT memory_cards_kind_ck CHECK (
                    kind IN ('Fact','Preference','Event','Profile','Relationship','Goal')
                )
            )
        """)


def load_onnx_model(conn: oracledb.Connection, model_name: str, model_path: str):
    """Load an ONNX embedding model into Oracle DB."""
    cursor = conn.cursor()
    # Check if model already loaded
    cursor.execute(
        "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :name",
        {"name": model_name.upper()},
    )
    if cursor.fetchone()[0] > 0:
        return  # already loaded

    cursor.execute("""
        BEGIN
            DBMS_VECTOR.LOAD_ONNX_MODEL(
                :model_name,
                :model_path,
                JSON('{"function":"embedding","input":{"input":["DATA"]},"output":{"embedding":["DATA"]}}')
            );
        END;
    """, {"model_name": model_name, "model_path": model_path})
    conn.commit()


if __name__ == "__main__":
    settings = Settings()
    pool = get_pool(settings)
    with pool.acquire() as conn:
        init_schema(conn)
        print("Schema initialized successfully.")
    close_pool()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: 5 passed

Note: If Oracle Text index creation fails (CTXSYS not available in Free edition), wrap it in a try/except and skip. The test for `frames_columns` validates the table structure independently of indexes. Adjust `_apply_v1` if needed:

```python
try:
    cursor.execute("""
        CREATE INDEX frames_content_idx ON frames(content)
            INDEXTYPE IS CTXSYS.CONTEXT
            PARAMETERS ('SYNC (ON COMMIT)')
    """)
except oracledb.DatabaseError:
    pass  # Oracle Text not available, text search will use LIKE fallback
```

Similarly for the vector index, wrap in try/except in case the Oracle Free version doesn't support `ORGANIZATION INMEMORY NEIGHBOR GRAPH`:

```python
try:
    cursor.execute("""
        CREATE VECTOR INDEX frames_vec_idx ON frames(embedding)
            ORGANIZATION INMEMORY NEIGHBOR GRAPH
    """)
except oracledb.DatabaseError:
    pass  # vector index not supported, vector search will do exact scan
```

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/db.py tests/conftest.py tests/test_db.py
git commit -m "feat: database module with schema init and connection pooling"
```

---

## Task 4: Embeddings Module

**Files:**
- Create: `src/oramemvid/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embeddings.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from oramemvid.embeddings import (
    EmbeddingProvider,
    OracleONNXEmbedding,
    OllamaEmbedding,
    get_embedding_provider,
)
from oramemvid.config import Settings


def test_oracle_onnx_sql_fragment():
    provider = OracleONNXEmbedding(model_name="ALL_MINILM_L6_V2")
    fragment = provider.sql_fragment()
    assert "VECTOR_EMBEDDING" in fragment
    assert "ALL_MINILM_L6_V2" in fragment


def test_oracle_onnx_is_in_database():
    provider = OracleONNXEmbedding(model_name="ALL_MINILM_L6_V2")
    assert provider.is_in_database is True


def test_ollama_is_not_in_database():
    settings = Settings(oracle_password="test")
    provider = OllamaEmbedding(
        ollama_url=settings.ollama_url,
        model=settings.ollama_embed_model,
    )
    assert provider.is_in_database is False


@patch("httpx.post")
def test_ollama_embed(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [[0.1] * 384]}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    result = provider.embed("hello world")
    assert len(result) == 384
    assert result[0] == pytest.approx(0.1)


@patch("httpx.post")
def test_ollama_embed_batch(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [[0.1] * 384, [0.2] * 384]}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    results = provider.embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 384


def test_get_provider_oracle():
    settings = Settings(oracle_password="test", embedding_provider="oracle_onnx")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, OracleONNXEmbedding)


def test_get_provider_ollama():
    settings = Settings(oracle_password="test", embedding_provider="ollama")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, OllamaEmbedding)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_embeddings.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.embeddings'`

- [ ] **Step 3: Implement embeddings.py**

Create `src/oramemvid/embeddings.py`:

```python
from abc import ABC, abstractmethod

import httpx

from oramemvid.config import Settings


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def is_in_database(self) -> bool:
        """True if embeddings are computed inside Oracle SQL."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text. Only used by external providers."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Only used by external providers."""
        ...


class OracleONNXEmbedding(EmbeddingProvider):
    """In-database embeddings via VECTOR_EMBEDDING(). No data leaves Oracle.

    This provider does NOT compute embeddings in Python. Instead, it provides
    SQL fragments for use in INSERT/SELECT statements. The embed() and
    embed_batch() methods raise NotImplementedError because the embedding
    happens inside Oracle SQL.
    """

    def __init__(self, model_name: str = "ALL_MINILM_L6_V2"):
        self.model_name = model_name

    @property
    def is_in_database(self) -> bool:
        return True

    def sql_fragment(self, bind_var: str = ":content") -> str:
        """SQL expression for in-database embedding generation."""
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
    """Fallback: calls Ollama /api/embed endpoint."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embeddings.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/embeddings.py tests/test_embeddings.py
git commit -m "feat: pluggable embedding providers (Oracle ONNX + Ollama)"
```

---

## Task 5: Frames Module

**Files:**
- Create: `src/oramemvid/frames.py`
- Create: `tests/test_frames.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_frames.py`:

```python
import hashlib
import pytest
from oramemvid.frames import create_frame, get_frame, list_frames, delete_frame
from oramemvid.embeddings import OllamaEmbedding
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_ollama_provider():
    """Use Ollama provider with mocked HTTP for tests."""
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    return provider


def _mock_embed(*args, **kwargs):
    return [0.1] * 384


def _mock_embed_batch(texts):
    return [[0.1] * 384 for _ in texts]


def test_create_and_get_frame(db_conn, mock_ollama_provider):
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        frame_id = create_frame(
            conn=db_conn,
            uri="test://doc1/chunk0",
            content="The quick brown fox jumps over the lazy dog.",
            provider=mock_ollama_provider,
            title="Test Frame",
        )

    assert frame_id is not None
    assert isinstance(frame_id, int)

    frame = get_frame(db_conn, frame_id)
    assert frame["uri"] == "test://doc1/chunk0"
    assert frame["title"] == "Test Frame"
    assert "quick brown fox" in frame["content"]
    assert frame["content_hash"] is not None


def test_create_duplicate_frame_returns_existing(db_conn, mock_ollama_provider):
    content = "Duplicate content test string."
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        id1 = create_frame(
            conn=db_conn,
            uri="test://dup1",
            content=content,
            provider=mock_ollama_provider,
        )
        id2 = create_frame(
            conn=db_conn,
            uri="test://dup2",
            content=content,
            provider=mock_ollama_provider,
        )
    assert id1 == id2


def test_list_frames(db_conn, mock_ollama_provider):
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        create_frame(
            conn=db_conn,
            uri="test://list1",
            content="List test content one unique.",
            provider=mock_ollama_provider,
        )
        create_frame(
            conn=db_conn,
            uri="test://list2",
            content="List test content two unique.",
            provider=mock_ollama_provider,
        )
    frames = list_frames(db_conn, limit=100)
    assert len(frames) >= 2


def test_delete_frame(db_conn, mock_ollama_provider):
    with patch.object(mock_ollama_provider, "embed", side_effect=_mock_embed):
        frame_id = create_frame(
            conn=db_conn,
            uri="test://delete-me",
            content="Content to be deleted unique string.",
            provider=mock_ollama_provider,
        )
    deleted = delete_frame(db_conn, frame_id)
    assert deleted is True
    assert get_frame(db_conn, frame_id) is None


def test_get_nonexistent_frame(db_conn):
    assert get_frame(db_conn, 999999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_frames.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.frames'`

- [ ] **Step 3: Implement frames.py**

Create `src/oramemvid/frames.py`:

```python
import hashlib
import json

import oracledb

from oramemvid.embeddings import EmbeddingProvider


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_frame(
    conn: oracledb.Connection,
    uri: str,
    content: str,
    provider: EmbeddingProvider,
    title: str | None = None,
    doc_id: int | None = None,
    tags: dict | None = None,
) -> int:
    """Insert a frame. Returns existing frame_id if content already exists."""
    content_hash = _hash_content(content)
    cursor = conn.cursor()

    # Check for duplicate
    cursor.execute(
        "SELECT frame_id FROM frames WHERE content_hash = :hash",
        {"hash": content_hash},
    )
    existing = cursor.fetchone()
    if existing:
        return existing[0]

    tags_json = json.dumps(tags) if tags else None

    if provider.is_in_database:
        # In-database embedding via VECTOR_EMBEDDING()
        sql = f"""
            INSERT INTO frames (uri, title, content, content_hash, doc_id, tags, embedding)
            VALUES (:uri, :title, :content, :hash, :doc_id, :tags,
                    {provider.sql_fragment(':content')})
            RETURNING frame_id INTO :frame_id
        """
        frame_id_var = cursor.var(oracledb.NUMBER)
        cursor.execute(sql, {
            "uri": uri,
            "title": title,
            "content": content,
            "hash": content_hash,
            "doc_id": doc_id,
            "tags": tags_json,
            "frame_id": frame_id_var,
        })
    else:
        # External embedding provider
        embedding = provider.embed(content)
        sql = """
            INSERT INTO frames (uri, title, content, content_hash, doc_id, tags, embedding)
            VALUES (:uri, :title, :content, :hash, :doc_id, :tags, :embedding)
            RETURNING frame_id INTO :frame_id
        """
        frame_id_var = cursor.var(oracledb.NUMBER)
        cursor.execute(sql, {
            "uri": uri,
            "title": title,
            "content": content,
            "hash": content_hash,
            "doc_id": doc_id,
            "tags": tags_json,
            "embedding": embedding,
            "frame_id": frame_id_var,
        })

    conn.commit()
    return int(frame_id_var.getvalue()[0])


def get_frame(conn: oracledb.Connection, frame_id: int) -> dict | None:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT frame_id, uri, title, content, content_hash,
               encoding, doc_id, tags, created_at
        FROM frames WHERE frame_id = :id
        """,
        {"id": frame_id},
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "frame_id": row[0],
        "uri": row[1],
        "title": row[2],
        "content": row[3].read() if hasattr(row[3], "read") else row[3],
        "content_hash": row[4],
        "encoding": row[5],
        "doc_id": row[6],
        "tags": json.loads(row[7]) if row[7] else None,
        "created_at": row[8].isoformat() if row[8] else None,
    }


def list_frames(
    conn: oracledb.Connection,
    limit: int = 20,
    offset: int = 0,
    doc_id: int | None = None,
) -> list[dict]:
    cursor = conn.cursor()
    if doc_id is not None:
        cursor.execute(
            """
            SELECT frame_id, uri, title, content_hash, encoding, doc_id, created_at
            FROM frames WHERE doc_id = :doc_id
            ORDER BY frame_id
            OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
            """,
            {"doc_id": doc_id, "off": offset, "lim": limit},
        )
    else:
        cursor.execute(
            """
            SELECT frame_id, uri, title, content_hash, encoding, doc_id, created_at
            FROM frames
            ORDER BY frame_id
            OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
            """,
            {"off": offset, "lim": limit},
        )
    rows = cursor.fetchall()
    return [
        {
            "frame_id": r[0],
            "uri": r[1],
            "title": r[2],
            "content_hash": r[3],
            "encoding": r[4],
            "doc_id": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]


def delete_frame(conn: oracledb.Connection, frame_id: int) -> bool:
    cursor = conn.cursor()
    # Delete associated memory cards first
    cursor.execute(
        "DELETE FROM memory_cards WHERE source_frame_id = :id",
        {"id": frame_id},
    )
    cursor.execute(
        "DELETE FROM frames WHERE frame_id = :id",
        {"id": frame_id},
    )
    conn.commit()
    return cursor.rowcount > 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_frames.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/frames.py tests/test_frames.py
git commit -m "feat: frames module with CRUD and deduplication"
```

---

## Task 6: Memory Cards Module

**Files:**
- Create: `src/oramemvid/memory_cards.py`
- Create: `tests/test_memory_cards.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_memory_cards.py`:

```python
import pytest
from oramemvid.memory_cards import (
    create_memory_card,
    get_memory_card,
    list_memory_cards,
    delete_memory_card,
)


def test_create_and_get_card(db_conn):
    card_id = create_memory_card(
        conn=db_conn,
        entity="Python",
        slot="created_by",
        value="Guido van Rossum",
        kind="Fact",
        confidence=0.95,
    )
    assert card_id is not None

    card = get_memory_card(db_conn, card_id)
    assert card["entity"] == "Python"
    assert card["slot"] == "created_by"
    assert card["value"] == "Guido van Rossum"
    assert card["kind"] == "Fact"
    assert card["confidence"] == pytest.approx(0.95)


def test_create_card_with_source_frame(db_conn):
    """Card can reference a source frame (but we don't create one here, use None)."""
    card_id = create_memory_card(
        conn=db_conn,
        entity="Oracle",
        slot="type",
        value="Database",
        kind="Fact",
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
    card_id = create_memory_card(
        db_conn, "DeleteMe", "slot", "val", "Fact"
    )
    assert delete_memory_card(db_conn, card_id) is True
    assert get_memory_card(db_conn, card_id) is None


def test_invalid_kind_raises(db_conn):
    with pytest.raises(Exception):
        create_memory_card(db_conn, "E", "S", "V", "InvalidKind")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_cards.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.memory_cards'`

- [ ] **Step 3: Implement memory_cards.py**

Create `src/oramemvid/memory_cards.py`:

```python
import oracledb


VALID_KINDS = {"Fact", "Preference", "Event", "Profile", "Relationship", "Goal"}


def create_memory_card(
    conn: oracledb.Connection,
    entity: str,
    slot: str,
    value: str,
    kind: str,
    source_frame_id: int | None = None,
    confidence: float = 1.0,
    expires_at=None,
) -> int:
    cursor = conn.cursor()
    card_id_var = cursor.var(oracledb.NUMBER)
    cursor.execute(
        """
        INSERT INTO memory_cards
            (entity, slot, card_value, kind, source_frame_id, confidence, expires_at)
        VALUES
            (:entity, :slot, :value, :kind, :frame_id, :confidence, :expires_at)
        RETURNING card_id INTO :card_id
        """,
        {
            "entity": entity,
            "slot": slot,
            "value": value,
            "kind": kind,
            "frame_id": source_frame_id,
            "confidence": confidence,
            "expires_at": expires_at,
            "card_id": card_id_var,
        },
    )
    conn.commit()
    return int(card_id_var.getvalue()[0])


def get_memory_card(conn: oracledb.Connection, card_id: int) -> dict | None:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT card_id, entity, slot, card_value, kind,
               source_frame_id, confidence, created_at, expires_at
        FROM memory_cards WHERE card_id = :id
        """,
        {"id": card_id},
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "card_id": row[0],
        "entity": row[1],
        "slot": row[2],
        "value": row[3].read() if hasattr(row[3], "read") else row[3],
        "kind": row[4],
        "source_frame_id": row[5],
        "confidence": float(row[6]) if row[6] is not None else None,
        "created_at": row[7].isoformat() if row[7] else None,
        "expires_at": row[8].isoformat() if row[8] else None,
    }


def list_memory_cards(
    conn: oracledb.Connection,
    entity: str | None = None,
    kind: str | None = None,
    source_frame_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    conditions = []
    params: dict = {"off": offset, "lim": limit}

    if entity is not None:
        conditions.append("entity = :entity")
        params["entity"] = entity
    if kind is not None:
        conditions.append("kind = :kind")
        params["kind"] = kind
    if source_frame_id is not None:
        conditions.append("source_frame_id = :frame_id")
        params["frame_id"] = source_frame_id

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT card_id, entity, slot, card_value, kind,
               source_frame_id, confidence, created_at
        FROM memory_cards {where}
        ORDER BY card_id
        OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
        """,
        params,
    )
    return [
        {
            "card_id": r[0],
            "entity": r[1],
            "slot": r[2],
            "value": r[3].read() if hasattr(r[3], "read") else r[3],
            "kind": r[4],
            "source_frame_id": r[5],
            "confidence": float(r[6]) if r[6] is not None else None,
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in cursor.fetchall()
    ]


def delete_memory_card(conn: oracledb.Connection, card_id: int) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM memory_cards WHERE card_id = :id",
        {"id": card_id},
    )
    conn.commit()
    return cursor.rowcount > 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_cards.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/memory_cards.py tests/test_memory_cards.py
git commit -m "feat: memory cards module with CRUD and filtering"
```

---

## Task 7: LLM Module

**Files:**
- Create: `src/oramemvid/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm.py`:

```python
import json
import pytest
from unittest.mock import patch, MagicMock
from oramemvid.llm import LLMProvider, OllamaLLM, get_llm_provider
from oramemvid.config import Settings


MOCK_EXTRACTION_RESPONSE = json.dumps([
    {
        "entity": "Python",
        "slot": "created_by",
        "value": "Guido van Rossum",
        "kind": "Fact",
        "confidence": 0.95,
    },
    {
        "entity": "Python",
        "slot": "first_release",
        "value": "1991",
        "kind": "Event",
        "confidence": 0.90,
    },
])


@patch("httpx.post")
def test_ollama_complete(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Hello world"}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    result = llm.complete("Say hello")
    assert result == "Hello world"


@patch("httpx.post")
def test_ollama_extract_memories(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": MOCK_EXTRACTION_RESPONSE}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    cards = llm.extract_memories("Python was created by Guido van Rossum in 1991.")
    assert len(cards) == 2
    assert cards[0]["entity"] == "Python"
    assert cards[0]["slot"] == "created_by"
    assert cards[0]["kind"] == "Fact"
    assert cards[1]["kind"] == "Event"


@patch("httpx.post")
def test_ollama_extract_memories_bad_json_returns_empty(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "not valid json ["}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    llm = OllamaLLM(ollama_url="http://localhost:11434", model="qwen3.5:9b")
    cards = llm.extract_memories("some content")
    assert cards == []


def test_get_llm_provider():
    settings = Settings(oracle_password="test")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OllamaLLM)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.llm'`

- [ ] **Step 3: Implement llm.py**

Create `src/oramemvid/llm.py`:

```python
import json
from abc import ABC, abstractmethod

import httpx

from oramemvid.config import Settings

EXTRACTION_PROMPT = """Extract structured memory cards from the following text.
Return a JSON array of objects with these fields:
- entity: the subject (person, thing, concept)
- slot: the relationship or attribute
- value: the specific information
- kind: one of Fact, Preference, Event, Profile, Relationship, Goal
- confidence: 0.0 to 1.0

Return ONLY the JSON array, no other text. If no memories can be extracted, return [].

Text:
{content}"""


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...

    @abstractmethod
    def extract_memories(self, content: str) -> list[dict]:
        ...


class OllamaLLM(LLMProvider):
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def extract_memories(self, content: str) -> list[dict]:
        prompt = EXTRACTION_PROMPT.format(content=content)
        raw = self.complete(prompt)

        # Try to parse JSON from the response
        try:
            # Handle case where LLM wraps JSON in markdown code blocks
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            cards = json.loads(cleaned)
            if not isinstance(cards, list):
                return []
            return cards
        except (json.JSONDecodeError, IndexError):
            return []


def get_llm_provider(settings: Settings) -> LLMProvider:
    return OllamaLLM(
        ollama_url=settings.ollama_url,
        model=settings.ollama_model,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/llm.py tests/test_llm.py
git commit -m "feat: pluggable LLM module with Ollama provider"
```

---

## Task 8: Ingest Module

**Files:**
- Create: `src/oramemvid/ingest.py`
- Create: `tests/test_ingest.py`
- Create: `tests/fixtures/sample.txt`

- [ ] **Step 1: Create test fixture**

Create `tests/fixtures/sample.txt`:

```
Oracle Database is a multi-model database management system produced by Oracle Corporation.
It supports SQL, JSON, XML, spatial data, and graph data, among others.
Oracle AI Vector Search enables similarity search using vector embeddings stored directly in the database.
This eliminates the need for a separate vector database.
The VECTOR data type and VECTOR_DISTANCE function provide native vector operations.
DBMS_VECTOR package allows loading ONNX models for in-database embedding generation.
Oracle Text provides full-text search with BM25 ranking via the CONTAINS operator.
Combined, these features make Oracle Database a powerful foundation for AI applications.
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_ingest.py`:

```python
import os
import pytest
from unittest.mock import patch, MagicMock
from oramemvid.ingest import (
    extract_text,
    chunk_text,
    ingest_file,
    ingest_text,
)
from oramemvid.embeddings import OllamaEmbedding


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_chunk_text_basic():
    text = "word " * 1000  # 1000 words
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    # Each chunk should be roughly chunk_size words or fewer
    for chunk in chunks:
        words = chunk.split()
        assert len(words) <= 110  # allow slight overflow


def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_chunk_text_overlap():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    # Second chunk should start with words that overlapped from first
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    overlap = set(first_words[-10:]) & set(second_words[:10])
    assert len(overlap) > 0


def test_extract_text_txt():
    path = os.path.join(FIXTURES, "sample.txt")
    text = extract_text(path)
    assert "Oracle Database" in text
    assert "VECTOR_DISTANCE" in text


def test_ingest_text_creates_frames(db_conn):
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        result = ingest_text(
            conn=db_conn,
            text="Test content for ingest. " * 50,
            uri="test://ingest-text",
            provider=provider,
        )
    assert result["total_frames"] >= 1
    assert result["uri"] == "test://ingest-text"


def test_ingest_file_txt(db_conn):
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    path = os.path.join(FIXTURES, "sample.txt")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        result = ingest_file(
            conn=db_conn,
            file_path=path,
            provider=provider,
        )
    assert result["doc_id"] is not None
    assert result["total_frames"] >= 1
    assert result["filename"] == "sample.txt"


def test_ingest_duplicate_file_skips(db_conn):
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    path = os.path.join(FIXTURES, "sample.txt")
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        r1 = ingest_file(conn=db_conn, file_path=path, provider=provider)
        r2 = ingest_file(conn=db_conn, file_path=path, provider=provider)
    assert r2["skipped"] is True
    assert r2["doc_id"] == r1["doc_id"]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.ingest'`

- [ ] **Step 4: Implement ingest.py**

Create `src/oramemvid/ingest.py`:

```python
import hashlib
import os

import oracledb

from oramemvid.embeddings import EmbeddingProvider
from oramemvid.frames import create_frame
from oramemvid.llm import LLMProvider
from oramemvid.memory_cards import create_memory_card


def extract_text(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(file_path, read_only=True)
        texts = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    texts.append(" | ".join(cells))
        wb.close()
        return "\n".join(texts)

    elif ext == ".pptx":
        from pptx import Presentation
        prs = Presentation(file_path)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    texts.append(shape.text_frame.text)
        return "\n".join(texts)

    else:
        # Fallback: try reading as text
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def _hash_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(
    conn: oracledb.Connection,
    file_path: str,
    provider: EmbeddingProvider,
    llm: LLMProvider | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Ingest a document file into frames. Returns ingestion result."""
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    file_hash = _hash_file(file_path)

    cursor = conn.cursor()

    # Check for duplicate document
    cursor.execute(
        "SELECT doc_id, total_frames FROM documents WHERE file_hash = :hash",
        {"hash": file_hash},
    )
    existing = cursor.fetchone()
    if existing:
        return {
            "doc_id": existing[0],
            "filename": filename,
            "total_frames": existing[1],
            "skipped": True,
        }

    # Register document
    doc_id_var = cursor.var(oracledb.NUMBER)
    cursor.execute(
        """
        INSERT INTO documents (filename, doc_type, file_hash)
        VALUES (:filename, :doc_type, :hash)
        RETURNING doc_id INTO :doc_id
        """,
        {
            "filename": filename,
            "doc_type": ext or "txt",
            "hash": file_hash,
            "doc_id": doc_id_var,
        },
    )
    conn.commit()
    doc_id = int(doc_id_var.getvalue()[0])

    # Extract and chunk
    text = extract_text(file_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Create frames
    frame_ids = []
    for i, chunk in enumerate(chunks):
        uri = f"file://{filename}/chunk/{i}"
        frame_id = create_frame(
            conn=conn,
            uri=uri,
            content=chunk,
            provider=provider,
            title=f"{filename} (chunk {i})",
            doc_id=doc_id,
        )
        frame_ids.append(frame_id)

        # Optional LLM extraction
        if llm is not None:
            cards = llm.extract_memories(chunk)
            for card in cards:
                create_memory_card(
                    conn=conn,
                    entity=card.get("entity", "unknown"),
                    slot=card.get("slot", "unknown"),
                    value=card.get("value", ""),
                    kind=card.get("kind", "Fact"),
                    source_frame_id=frame_id,
                    confidence=card.get("confidence", 1.0),
                )

    # Update frame count
    cursor.execute(
        "UPDATE documents SET total_frames = :count WHERE doc_id = :id",
        {"count": len(frame_ids), "id": doc_id},
    )
    conn.commit()

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "skipped": False,
    }


def ingest_text(
    conn: oracledb.Connection,
    text: str,
    uri: str,
    provider: EmbeddingProvider,
    llm: LLMProvider | None = None,
    title: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Ingest raw text directly into frames."""
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    frame_ids = []

    for i, chunk in enumerate(chunks):
        chunk_uri = f"{uri}/chunk/{i}" if len(chunks) > 1 else uri
        frame_id = create_frame(
            conn=conn,
            uri=chunk_uri,
            content=chunk,
            provider=provider,
            title=title or f"text (chunk {i})",
        )
        frame_ids.append(frame_id)

        if llm is not None:
            cards = llm.extract_memories(chunk)
            for card in cards:
                create_memory_card(
                    conn=conn,
                    entity=card.get("entity", "unknown"),
                    slot=card.get("slot", "unknown"),
                    value=card.get("value", ""),
                    kind=card.get("kind", "Fact"),
                    source_frame_id=frame_id,
                    confidence=card.get("confidence", 1.0),
                )

    return {
        "uri": uri,
        "total_frames": len(frame_ids),
        "frame_ids": frame_ids,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/oramemvid/ingest.py tests/test_ingest.py tests/fixtures/sample.txt
git commit -m "feat: document ingestion pipeline with text extraction and chunking"
```

---

## Task 9: Search Module

**Files:**
- Create: `src/oramemvid/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_search.py`:

```python
import pytest
from unittest.mock import patch
from oramemvid.search import search_text, search_vector, search_hybrid
from oramemvid.frames import create_frame
from oramemvid.embeddings import OllamaEmbedding


@pytest.fixture
def seeded_frames(db_conn):
    """Insert some frames for search testing."""
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    contents = [
        "Oracle Database supports vector embeddings natively.",
        "Python is a popular programming language for data science.",
        "FastAPI is a modern web framework for building APIs with Python.",
    ]
    ids = []
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        for i, c in enumerate(contents):
            fid = create_frame(
                conn=db_conn,
                uri=f"test://search/{i}",
                content=c,
                provider=provider,
            )
            ids.append(fid)
    return ids


def test_search_text(db_conn, seeded_frames):
    results = search_text(db_conn, "Oracle Database", top_k=5)
    assert len(results) >= 1
    assert any("Oracle" in r["content"] for r in results)


def test_search_text_no_results(db_conn, seeded_frames):
    results = search_text(db_conn, "xyznonexistent999", top_k=5)
    assert len(results) == 0


def test_search_vector(db_conn, seeded_frames):
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        results = search_vector(
            db_conn, "database embeddings", provider=provider, top_k=5
        )
    assert len(results) >= 1


def test_search_hybrid(db_conn, seeded_frames):
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    with patch.object(provider, "embed", return_value=[0.1] * 384):
        results = search_hybrid(
            db_conn, "Python web framework", provider=provider, top_k=5
        )
    assert len(results) >= 1


def test_search_text_with_time_filter(db_conn, seeded_frames):
    results = search_text(
        db_conn,
        "Python",
        top_k=5,
        time_from="2020-01-01",
        time_to="2099-12-31",
    )
    assert len(results) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.search'`

- [ ] **Step 3: Implement search.py**

Create `src/oramemvid/search.py`:

```python
import oracledb

from oramemvid.embeddings import EmbeddingProvider


def _read_content(val):
    """Read CLOB or return string as-is."""
    return val.read() if hasattr(val, "read") else val


def search_text(
    conn: oracledb.Connection,
    query: str,
    top_k: int = 10,
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[dict]:
    """Full-text search using Oracle Text CONTAINS or LIKE fallback."""
    cursor = conn.cursor()
    params: dict = {"query": query, "top_k": top_k}
    time_conditions = ""

    if time_from:
        time_conditions += " AND created_at >= TO_TIMESTAMP(:time_from, 'YYYY-MM-DD')"
        params["time_from"] = time_from
    if time_to:
        time_conditions += " AND created_at <= TO_TIMESTAMP(:time_to, 'YYYY-MM-DD')"
        params["time_to"] = time_to

    # Try Oracle Text CONTAINS first
    try:
        cursor.execute(
            f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   SCORE(1) AS relevance
            FROM frames
            WHERE CONTAINS(content, :query, 1) > 0 {time_conditions}
            ORDER BY relevance DESC
            FETCH FIRST :top_k ROWS ONLY
            """,
            params,
        )
    except oracledb.DatabaseError:
        # Fallback to LIKE if Oracle Text index not available
        params["like_query"] = f"%{query}%"
        cursor.execute(
            f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   1 AS relevance
            FROM frames
            WHERE LOWER(content) LIKE LOWER(:like_query) {time_conditions}
            ORDER BY created_at DESC
            FETCH FIRST :top_k ROWS ONLY
            """,
            params,
        )

    return [
        {
            "frame_id": r[0],
            "uri": r[1],
            "title": r[2],
            "content": _read_content(r[3]),
            "content_hash": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
            "score": float(r[6]),
        }
        for r in cursor.fetchall()
    ]


def search_vector(
    conn: oracledb.Connection,
    query: str,
    provider: EmbeddingProvider,
    top_k: int = 10,
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[dict]:
    """Vector similarity search using VECTOR_DISTANCE."""
    cursor = conn.cursor()
    time_conditions = ""
    params: dict = {"top_k": top_k}

    if time_from:
        time_conditions += " AND created_at >= TO_TIMESTAMP(:time_from, 'YYYY-MM-DD')"
        params["time_from"] = time_from
    if time_to:
        time_conditions += " AND created_at <= TO_TIMESTAMP(:time_to, 'YYYY-MM-DD')"
        params["time_to"] = time_to

    if provider.is_in_database:
        # In-database: embed the query inside SQL
        params["query"] = query
        sql = f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   VECTOR_DISTANCE(embedding,
                       {provider.sql_fragment(':query')},
                       COSINE) AS distance
            FROM frames
            WHERE embedding IS NOT NULL {time_conditions}
            ORDER BY distance ASC
            FETCH FIRST :top_k ROWS ONLY
        """
    else:
        # External: compute query embedding in Python
        query_vec = provider.embed(query)
        params["query_vec"] = query_vec
        sql = f"""
            SELECT frame_id, uri, title, content, content_hash, created_at,
                   VECTOR_DISTANCE(embedding, :query_vec, COSINE) AS distance
            FROM frames
            WHERE embedding IS NOT NULL {time_conditions}
            ORDER BY distance ASC
            FETCH FIRST :top_k ROWS ONLY
        """

    cursor.execute(sql, params)
    return [
        {
            "frame_id": r[0],
            "uri": r[1],
            "title": r[2],
            "content": _read_content(r[3]),
            "content_hash": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
            "score": 1.0 - float(r[6]) if r[6] is not None else 0.0,
        }
        for r in cursor.fetchall()
    ]


def search_hybrid(
    conn: oracledb.Connection,
    query: str,
    provider: EmbeddingProvider,
    top_k: int = 10,
    rrf_k: int = 60,
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[dict]:
    """Hybrid search: reciprocal rank fusion of text + vector results."""
    text_results = search_text(
        conn, query, top_k=top_k * 2,
        time_from=time_from, time_to=time_to,
    )
    vector_results = search_vector(
        conn, query, provider, top_k=top_k * 2,
        time_from=time_from, time_to=time_to,
    )

    # RRF scoring
    scores: dict[int, float] = {}
    frame_data: dict[int, dict] = {}

    for rank, result in enumerate(text_results):
        fid = result["frame_id"]
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (rrf_k + rank + 1)
        frame_data[fid] = result

    for rank, result in enumerate(vector_results):
        fid = result["frame_id"]
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (rrf_k + rank + 1)
        frame_data[fid] = result

    # Sort by RRF score, take top_k
    sorted_ids = sorted(scores, key=lambda fid: scores[fid], reverse=True)[:top_k]

    return [
        {**frame_data[fid], "score": scores[fid]}
        for fid in sorted_ids
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_search.py -v
```

Expected: 5 passed

Note: If Oracle Text CONTAINS fails (no CTXSYS index), the LIKE fallback kicks in. Tests should still pass since the fallback handles the query. If vector search fails because embeddings are all identical mock values, adjust test expectations: `search_vector` should return results (all at the same distance), but `search_hybrid` may merge them meaningfully.

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/search.py tests/test_search.py
git commit -m "feat: unified search with text, vector, and hybrid RRF modes"
```

---

## Task 10: FastAPI Module

**Files:**
- Create: `src/oramemvid/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_api.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from oramemvid.api import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


@pytest.mark.anyio
async def test_ingest_text(client):
    resp = await client.post(
        "/ingest/text",
        json={
            "text": "Oracle Database is great for AI workloads.",
            "uri": "test://api/text1",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_frames"] >= 1


@pytest.mark.anyio
async def test_list_frames(client):
    resp = await client.get("/frames")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.anyio
async def test_search(client):
    # Ingest something first
    await client.post(
        "/ingest/text",
        json={
            "text": "FastAPI makes building REST APIs simple and fast.",
            "uri": "test://api/search1",
        },
    )
    resp = await client.get("/search", params={"query": "FastAPI", "mode": "text"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.anyio
async def test_list_memory_cards(client):
    resp = await client.get("/memory")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_list_documents(client):
    resp = await client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_stats(client):
    resp = await client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "frame_count" in data
    assert "card_count" in data
    assert "document_count" in data
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'oramemvid.api'`

- [ ] **Step 3: Implement api.py**

Create `src/oramemvid/api.py`:

```python
import os
import tempfile
from contextlib import asynccontextmanager

import oracledb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from oramemvid.config import Settings, get_settings
from oramemvid.db import get_pool, close_pool, init_schema
from oramemvid.embeddings import get_embedding_provider
from oramemvid.frames import create_frame, get_frame, list_frames, delete_frame
from oramemvid.ingest import ingest_file, ingest_text as _ingest_text
from oramemvid.llm import get_llm_provider
from oramemvid.memory_cards import (
    get_memory_card,
    list_memory_cards,
    delete_memory_card,
)
from oramemvid.search import search_text, search_vector, search_hybrid


settings = get_settings()
embedding_provider = get_embedding_provider(settings)
llm_provider = get_llm_provider(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = get_pool(settings)
    with pool.acquire() as conn:
        init_schema(conn)
    yield
    close_pool()


app = FastAPI(title="oramemvid", version="0.1.0", lifespan=lifespan)


def _get_conn() -> oracledb.Connection:
    return get_pool().acquire()


def _release_conn(conn: oracledb.Connection):
    get_pool().release(conn)


# --- Ingest ---


class IngestTextRequest(BaseModel):
    text: str
    uri: str
    title: str | None = None
    extract_memories: bool = False


@app.post("/ingest/text")
def api_ingest_text(req: IngestTextRequest):
    conn = _get_conn()
    try:
        llm = llm_provider if req.extract_memories else None
        result = _ingest_text(
            conn=conn,
            text=req.text,
            uri=req.uri,
            provider=embedding_provider,
            llm=llm,
            title=req.title,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return result
    finally:
        _release_conn(conn)


@app.post("/ingest/file")
def api_ingest_file(
    file: UploadFile = File(...),
    extract_memories: bool = False,
):
    conn = _get_conn()
    try:
        # Save upload to temp file
        suffix = os.path.splitext(file.filename or "upload.txt")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = file.file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            llm = llm_provider if extract_memories else None
            result = ingest_file(
                conn=conn,
                file_path=tmp_path,
                provider=embedding_provider,
                llm=llm,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            # Use original filename
            result["filename"] = file.filename
            return result
        finally:
            os.unlink(tmp_path)
    finally:
        _release_conn(conn)


# --- Frames ---


@app.get("/frames")
def api_list_frames(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    doc_id: int | None = None,
):
    conn = _get_conn()
    try:
        return list_frames(conn, limit=limit, offset=offset, doc_id=doc_id)
    finally:
        _release_conn(conn)


@app.get("/frames/{frame_id}")
def api_get_frame(frame_id: int):
    conn = _get_conn()
    try:
        frame = get_frame(conn, frame_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        return frame
    finally:
        _release_conn(conn)


@app.delete("/frames/{frame_id}")
def api_delete_frame(frame_id: int):
    conn = _get_conn()
    try:
        if not delete_frame(conn, frame_id):
            raise HTTPException(status_code=404, detail="Frame not found")
        return {"deleted": True}
    finally:
        _release_conn(conn)


# --- Search ---


@app.get("/search")
def api_search(
    query: str = Query(...),
    mode: str = Query("hybrid", pattern="^(text|vector|hybrid)$"),
    top_k: int = Query(10, ge=1, le=100),
    time_from: str | None = None,
    time_to: str | None = None,
):
    conn = _get_conn()
    try:
        if mode == "text":
            return search_text(conn, query, top_k, time_from, time_to)
        elif mode == "vector":
            return search_vector(
                conn, query, embedding_provider, top_k, time_from, time_to
            )
        else:
            return search_hybrid(
                conn, query, embedding_provider, top_k,
                time_from=time_from, time_to=time_to,
            )
    finally:
        _release_conn(conn)


@app.get("/search/text")
def api_search_text(
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    time_from: str | None = None,
    time_to: str | None = None,
):
    conn = _get_conn()
    try:
        return search_text(conn, query, top_k, time_from, time_to)
    finally:
        _release_conn(conn)


@app.get("/search/vector")
def api_search_vector(
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    time_from: str | None = None,
    time_to: str | None = None,
):
    conn = _get_conn()
    try:
        return search_vector(
            conn, query, embedding_provider, top_k, time_from, time_to
        )
    finally:
        _release_conn(conn)


# --- Memory Cards ---


@app.get("/memory")
def api_list_memory_cards(
    entity: str | None = None,
    kind: str | None = None,
    source_frame_id: int | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    conn = _get_conn()
    try:
        return list_memory_cards(
            conn, entity=entity, kind=kind,
            source_frame_id=source_frame_id,
            limit=limit, offset=offset,
        )
    finally:
        _release_conn(conn)


@app.get("/memory/{card_id}")
def api_get_memory_card(card_id: int):
    conn = _get_conn()
    try:
        card = get_memory_card(conn, card_id)
        if card is None:
            raise HTTPException(status_code=404, detail="Memory card not found")
        return card
    finally:
        _release_conn(conn)


@app.post("/memory/extract/{frame_id}")
def api_extract_memories(frame_id: int):
    conn = _get_conn()
    try:
        frame = get_frame(conn, frame_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        cards = llm_provider.extract_memories(frame["content"])
        from oramemvid.memory_cards import create_memory_card
        card_ids = []
        for card in cards:
            cid = create_memory_card(
                conn=conn,
                entity=card.get("entity", "unknown"),
                slot=card.get("slot", "unknown"),
                value=card.get("value", ""),
                kind=card.get("kind", "Fact"),
                source_frame_id=frame_id,
                confidence=card.get("confidence", 1.0),
            )
            card_ids.append(cid)
        return {"frame_id": frame_id, "cards_created": len(card_ids), "card_ids": card_ids}
    finally:
        _release_conn(conn)


@app.delete("/memory/{card_id}")
def api_delete_memory_card(card_id: int):
    conn = _get_conn()
    try:
        if not delete_memory_card(conn, card_id):
            raise HTTPException(status_code=404, detail="Memory card not found")
        return {"deleted": True}
    finally:
        _release_conn(conn)


# --- Documents ---


@app.get("/documents")
def api_list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT doc_id, filename, doc_type, file_hash, total_frames, ingested_at
            FROM documents
            ORDER BY doc_id DESC
            OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
            """,
            {"off": offset, "lim": limit},
        )
        return [
            {
                "doc_id": r[0],
                "filename": r[1],
                "doc_type": r[2],
                "file_hash": r[3],
                "total_frames": r[4],
                "ingested_at": r[5].isoformat() if r[5] else None,
            }
            for r in cursor.fetchall()
        ]
    finally:
        _release_conn(conn)


# --- Stats & Health ---


@app.get("/stats")
def api_stats():
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM frames")
        frame_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM memory_cards")
        card_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        return {
            "frame_count": frame_count,
            "card_count": card_count,
            "document_count": doc_count,
        }
    finally:
        _release_conn(conn)


@app.get("/health")
def api_health():
    try:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM dual")
            return {"status": "ok", "database": "connected"}
        finally:
            _release_conn(conn)
    except Exception as e:
        return {"status": "error", "database": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 7 passed

Note: API tests require the Oracle DB to be running and schema initialized. The `lifespan` handler in FastAPI calls `init_schema` on startup. Tests use `httpx.AsyncClient` with `ASGITransport` for in-process testing.

- [ ] **Step 5: Commit**

```bash
git add src/oramemvid/api.py tests/test_api.py
git commit -m "feat: FastAPI REST API with all endpoints"
```

---

## Task 11: Integration Test and Final Verification

**Files:**
- Modify: `tests/conftest.py` (add cleanup fixture)
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
import os
import pytest
from unittest.mock import patch
from oramemvid.config import Settings
from oramemvid.db import get_pool, init_schema
from oramemvid.embeddings import OllamaEmbedding
from oramemvid.llm import OllamaLLM
from oramemvid.ingest import ingest_file
from oramemvid.search import search_text, search_vector, search_hybrid
from oramemvid.frames import list_frames
from oramemvid.memory_cards import list_memory_cards


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _mock_embed(text):
    """Deterministic mock embedding based on text hash."""
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    # Create a pseudo-random but deterministic vector
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 384 * 2, 2)][:384] + [0.0] * max(0, 384 - 128)


def _mock_extract(content):
    """Mock memory extraction."""
    return [
        {
            "entity": "Oracle",
            "slot": "feature",
            "value": "vector search",
            "kind": "Fact",
            "confidence": 0.9,
        }
    ]


def test_full_pipeline(db_conn):
    """End-to-end: ingest file, search, verify memories."""
    provider = OllamaEmbedding(
        ollama_url="http://localhost:11434",
        model="nomic-embed-text",
    )
    llm = OllamaLLM(
        ollama_url="http://localhost:11434",
        model="qwen3.5:9b",
    )

    path = os.path.join(FIXTURES, "sample.txt")

    with (
        patch.object(provider, "embed", side_effect=_mock_embed),
        patch.object(llm, "extract_memories", side_effect=_mock_extract),
    ):
        result = ingest_file(
            conn=db_conn,
            file_path=path,
            provider=provider,
            llm=llm,
        )

    assert result["total_frames"] >= 1
    assert result["skipped"] is False

    # Verify frames exist
    frames = list_frames(db_conn, doc_id=result["doc_id"])
    assert len(frames) == result["total_frames"]

    # Verify memory cards were created
    cards = list_memory_cards(db_conn)
    assert len(cards) >= 1
    oracle_cards = [c for c in cards if c["entity"] == "Oracle"]
    assert len(oracle_cards) >= 1

    # Verify text search works
    text_results = search_text(db_conn, "Oracle", top_k=5)
    assert len(text_results) >= 1

    # Verify vector search works
    with patch.object(provider, "embed", side_effect=_mock_embed):
        vec_results = search_vector(db_conn, "database", provider=provider, top_k=5)
    assert len(vec_results) >= 1

    # Verify hybrid search works
    with patch.object(provider, "embed", side_effect=_mock_embed):
        hybrid_results = search_hybrid(
            db_conn, "Oracle Database", provider=provider, top_k=5
        )
    assert len(hybrid_results) >= 1
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v
```

Expected: 1 passed

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass (approximately 30+ tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration test for full ingest-search pipeline"
```

- [ ] **Step 5: Create README.md**

Create `README.md`:

```markdown
# oramemvid

AI memory layer for agents, backed by Oracle Database. Inspired by [memvid](https://github.com/memvid/memvid).

Replaces memvid's custom `.mv2` binary format with Oracle AI Vector Search, Oracle Text, and in-database ONNX embeddings. Your data never needs to leave the database.

## Features

- **Frame storage**: Append-only content chunks with SHA-256 deduplication
- **Memory cards**: Structured entity/slot/value knowledge extracted by LLM
- **Hybrid search**: Oracle Text (BM25) + Vector similarity (HNSW), fused with reciprocal rank fusion
- **In-database embeddings**: ONNX model runs inside Oracle via `VECTOR_EMBEDDING()`
- **Document ingestion**: PDF, DOCX, XLSX, PPTX, TXT
- **REST API**: FastAPI with full CRUD and search endpoints

## Quick Start

```bash
# Start Oracle 26ai Free
docker compose up -d

# Create conda env
conda create -n oramemvid python=3.12 -y
conda activate oramemvid

# Install
pip install -e ".[dev]"

# Copy and edit .env
cp .env.example .env

# Initialize schema
python -m oramemvid.db

# Run API
uvicorn oramemvid.api:app --reload --port 8000
```

## API Examples

```bash
# Ingest text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Oracle supports vector embeddings natively.", "uri": "test://example"}'

# Upload a PDF
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@document.pdf" \
  -F "extract_memories=true"

# Hybrid search
curl "http://localhost:8000/search?query=vector+embeddings&mode=hybrid&top_k=5"

# List memory cards
curl "http://localhost:8000/memory?entity=Oracle"

# Health check
curl http://localhost:8000/health
```

## Architecture

| Component | Oracle Feature |
|-----------|---------------|
| Frame storage | `CLOB` + `VECTOR(384)` columns |
| Text search | Oracle Text `CONTAINS` with BM25 |
| Vector search | `VECTOR_DISTANCE` with HNSW index |
| Embeddings | In-database `VECTOR_EMBEDDING()` via ONNX |
| Deduplication | SHA-256 content hashing with unique constraints |
| Transactions | Oracle ACID (replaces memvid's custom WAL) |

## Configuration

All settings via environment variables with `ORAMEMVID_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORAMEMVID_ORACLE_DSN` | `localhost:1523/FREEPDB1` | Oracle connection string |
| `ORAMEMVID_ORACLE_USER` | `oramemvid` | Database user |
| `ORAMEMVID_ORACLE_PASSWORD` | (required) | Database password |
| `ORAMEMVID_EMBEDDING_PROVIDER` | `oracle_onnx` | `oracle_onnx` or `ollama` |
| `ORAMEMVID_OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `ORAMEMVID_OLLAMA_MODEL` | `qwen3.5:9b` | LLM for memory extraction |

## Testing

```bash
pytest tests/ -v
```

Requires a running Oracle 26ai Free instance.
```

- [ ] **Step 6: Final commit and push**

```bash
git add README.md
git commit -m "docs: README with quick start and API examples"
git push -u origin main
```
