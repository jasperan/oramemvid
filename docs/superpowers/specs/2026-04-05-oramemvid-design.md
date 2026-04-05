# oramemvid Design Spec

**Date**: 2026-04-05
**Status**: Approved
**Based on**: [memvid](https://github.com/memvid/memvid) (Rust single-file AI memory layer)

## Overview

Python reimplementation of memvid's core AI memory concepts, backed by Oracle Database 26ai Free instead of the custom `.mv2` binary format. Replaces Tantivy with Oracle Text, HNSW file-based index with Oracle AI Vector Search, and the custom WAL with Oracle's built-in transaction management.

## Scope

**In scope (Core + Documents)**:
- Frame storage and retrieval (append-only content units)
- MemoryCard extraction and storage (structured entity/slot/value triples)
- Document ingestion pipeline (PDF, DOCX, XLSX, PPTX, TXT)
- Text search via Oracle Text (BM25/CONTAINS)
- Vector similarity search via Oracle AI Vector Search (HNSW)
- Hybrid search with reciprocal rank fusion
- In-database ONNX embeddings (all-MiniLM-L6-v2), Ollama fallback
- Pluggable LLM for memory extraction (default: Qwen3.5:9b via Ollama)
- FastAPI REST API

**Out of scope (future)**:
- Encryption (.mv2e equivalent)
- Time-travel / replay / branching
- Audio transcription (Whisper)
- Vision embeddings (CLIP)

## Database Schema

Oracle 26ai Free via Docker on port 1523/FREEPDB1.

### frames

Core content storage. Each frame is an immutable chunk of ingested content.

```sql
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
);

CREATE INDEX frames_content_idx ON frames(content)
    INDEXTYPE IS CTXSYS.CONTEXT
    PARAMETERS ('SYNC (ON COMMIT)');

CREATE VECTOR INDEX frames_vec_idx ON frames(embedding)
    ORGANIZATION INMEMORY NEIGHBOR GRAPH;
```

### memory_cards

Structured knowledge extracted from frames by the LLM.

```sql
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
);
```

### documents

Tracks ingested source files to prevent duplicates and provide provenance.

```sql
CREATE TABLE documents (
    doc_id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename      VARCHAR2(1000) NOT NULL,
    doc_type      VARCHAR2(20) NOT NULL,
    file_hash     VARCHAR2(64) NOT NULL,
    total_frames  NUMBER DEFAULT 0,
    ingested_at   TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT documents_hash_uk UNIQUE (file_hash)
);
```

### ONNX Model Loading

```sql
-- Load embedding model into Oracle DB
BEGIN
    DBMS_VECTOR.LOAD_ONNX_MODEL(
        model_name => 'ALL_MINILM_L6_V2',
        model_path => 'all-MiniLM-L6-v2.onnx',
        metadata   => JSON('{"function":"embedding","input":{"input":["DATA"]},"output":{"embedding":["DATA"]}}')
    );
END;
```

Embeddings generated in-database during INSERT:
```sql
INSERT INTO frames (uri, title, content, content_hash, embedding)
VALUES (:uri, :title, :content, :hash,
        VECTOR_EMBEDDING(ALL_MINILM_L6_V2 USING :content AS data));
```

## Module Architecture

```
oramemvid/
├── src/oramemvid/
│   ├── __init__.py
│   ├── db.py              # Connection pool, schema init, ONNX model loading
│   ├── frames.py          # Frame CRUD + query helpers
│   ├── memory_cards.py    # MemoryCard CRUD + query helpers
│   ├── embeddings.py      # Pluggable embedding interface
│   ├── llm.py             # Pluggable LLM interface
│   ├── ingest.py          # Document pipeline (extract, chunk, store)
│   ├── search.py          # Unified search (text, vector, hybrid)
│   ├── api.py             # FastAPI routes
│   └── config.py          # Settings from env vars
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml     # Oracle 26ai Free container
├── tests/
│   ├── test_db.py
│   ├── test_frames.py
│   ├── test_memory_cards.py
│   ├── test_ingest.py
│   ├── test_search.py
│   └── test_api.py
├── CLAUDE.md
└── README.md
```

### db.py

- `oracledb` with connection pooling (thin client by default)
- `init_schema()`: creates tables, indexes, loads ONNX model if not present
- `get_connection()` context manager for transactional operations
- Schema version tracking via a simple `schema_version` table

### embeddings.py

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class OracleONNXEmbedding(EmbeddingProvider):
    """In-database embeddings via VECTOR_EMBEDDING(). No data leaves Oracle."""

class OllamaEmbedding(EmbeddingProvider):
    """Fallback: calls Ollama /api/embed endpoint."""
```

Two embedding paths in `frames.py`:
- **In-database (default)**: `frames.py` uses `VECTOR_EMBEDDING()` directly in the INSERT SQL. The `OracleONNXEmbedding` class is not called per-row; instead it provides the SQL fragment and validates model availability. No Python-side vectors.
- **External (Ollama fallback)**: `frames.py` calls `OllamaEmbedding.embed()` to get a Python `list[float]`, then passes the vector explicitly to INSERT as a bind variable.

### llm.py

```python
class LLMProvider(ABC):
    @abstractmethod
    def extract_memories(self, content: str) -> list[MemoryCard]: ...
    @abstractmethod
    def complete(self, prompt: str) -> str: ...

class OllamaLLM(LLMProvider):
    """Default: qwen3.5:9b at localhost:11434.
    Uses structured output prompt for entity/slot/value extraction."""
```

Memory extraction prompt asks the LLM to return JSON array of:
```json
[{"entity": "...", "slot": "...", "value": "...", "kind": "Fact|Preference|Event|Profile|Relationship|Goal", "confidence": 0.95}]
```

### ingest.py

Pipeline stages:
1. **Detect type** from extension/magic bytes
2. **Extract text**: pymupdf (PDF), python-docx (DOCX), openpyxl (XLSX), python-pptx (PPTX), or plain read (TXT)
3. **Hash file** (SHA-256) and check `documents` table for duplicates
4. **Chunk text** into frame-sized pieces (~512 tokens, 50-token overlap)
5. **Insert frames** with in-database embedding generation
6. **Extract memories** (optional): run LLM on each frame, insert memory_cards
7. **Update document** record with total_frames count

### search.py

```python
def search(query: str, mode: str = "hybrid", top_k: int = 10, filters: dict = None) -> list[SearchResult]:
    """
    mode:
      "text"   — Oracle Text CONTAINS query, BM25 scoring
      "vector" — VECTOR_DISTANCE with query embedding
      "hybrid" — both, fused with reciprocal rank fusion (RRF)
    filters:
      time_from, time_to — temporal range on created_at
      tags — JSON path filter on tags column
      kind — filter memory_cards by kind
      entity — filter memory_cards by entity
    """
```

Hybrid search: runs text and vector queries in parallel (two SQL statements), merges results using RRF: `score = sum(1 / (k + rank))` where k=60.

### config.py

```python
class Settings(BaseSettings):
    oracle_dsn: str = "localhost:1523/FREEPDB1"
    oracle_user: str = "oramemvid"
    oracle_password: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    embedding_provider: str = "oracle_onnx"  # or "ollama"
    onnx_model_name: str = "all_minilm_l6_v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    class Config:
        env_prefix = "ORAMEMVID_"
```

## API Routes

FastAPI on port 8000, uvicorn runner.

```
POST   /ingest/file              # Upload document (multipart form)
POST   /ingest/text              # Ingest raw text (JSON body)

GET    /frames                   # List frames (paginated, filterable)
GET    /frames/{frame_id}        # Get single frame with content
DELETE /frames/{frame_id}        # Remove a frame

GET    /search                   # Unified search (query, mode, top_k, filters)
GET    /search/text              # Oracle Text only
GET    /search/vector            # Vector similarity only

GET    /memory                   # List memory cards (entity, kind filters)
GET    /memory/{card_id}         # Get single memory card
POST   /memory/extract/{frame_id}  # Trigger LLM extraction on existing frame
DELETE /memory/{card_id}         # Remove a memory card

GET    /documents                # List ingested documents
GET    /stats                    # Frame count, card count, index health
GET    /health                   # DB connectivity check
```

## Dependencies

```
oracledb>=2.0
fastapi>=0.110
uvicorn>=0.27
pymupdf>=1.24
python-docx>=1.1
openpyxl>=3.1
python-pptx>=0.6
httpx>=0.27          # Ollama API calls
pydantic-settings>=2.0
python-multipart>=0.0.9  # File uploads
```

## Infrastructure

- **Oracle 26ai Free**: Docker container, port 1523, PDB: FREEPDB1
- **Ollama**: localhost:11434, models: qwen3.5:9b (extraction), nomic-embed-text (fallback embeddings)
- **Python**: 3.12, conda env named `oramemvid`
- **ONNX model**: all-MiniLM-L6-v2 downloaded and loaded into Oracle via DBMS_VECTOR

## Testing Strategy

- pytest with fixtures for Oracle DB connection
- Test schema creation/teardown
- Test frame CRUD and deduplication
- Test document ingestion pipeline (small sample files)
- Test search modes (text, vector, hybrid)
- Test memory extraction with mocked LLM responses
- Test API endpoints via httpx AsyncClient
