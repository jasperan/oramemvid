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
# Start Oracle 26ai Free (if using Docker)
docker compose up -d

# Create conda env
conda create -n oramemvid python=3.12 -y
conda activate oramemvid

# Install. The oracle-onnx extra supports the default embedding provider.
pip install -e ".[dev,oracle-onnx]"

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
| `ORAMEMVID_ORACLE_ADMIN_USER` | unset | Optional admin user for directory-based ONNX loading |
| `ORAMEMVID_ORACLE_ADMIN_PASSWORD` | unset | Optional admin password for directory-based ONNX loading |
| `ORAMEMVID_EMBEDDING_PROVIDER` | `oracle_onnx` | `oracle_onnx`, `ollama`, or `sentence_transformers` |
| `ORAMEMVID_ONNX_MODEL_NAME` | `all_minilm_l6_v2` | Oracle mining model name |
| `ORAMEMVID_OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `ORAMEMVID_OLLAMA_MODEL` | `qwen3.5:9b` | LLM for memory extraction |
| `ORAMEMVID_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for Ollama fallback |
| `ORAMEMVID_CHUNK_SIZE` | `512` | Words per frame chunk |
| `ORAMEMVID_CHUNK_OVERLAP` | `50` | Overlap words; must be smaller than chunk size |
| `ORAMEMVID_MAX_UPLOAD_BYTES` | `52428800` | Maximum `/ingest/file` upload size |
| `ORAMEMVID_ALLOWED_UPLOAD_EXTENSIONS` | `.txt,.pdf,.docx,.xlsx,.pptx` | Upload extension allowlist |

## Testing

```bash
pytest tests/ -v
```

Pure unit tests run without Oracle. Tests that use `db_conn`, `db_pool`, or
the `oracle` marker require a running Oracle 26ai Free instance; if Oracle is
unavailable, those tests are skipped with the connection error.

## Bootstrap Notes

`oracle_onnx` is the default embedding provider. If the model is not already
loaded in Oracle, schema initialization downloads and patches the ONNX model,
then loads it with `DBMS_VECTOR.LOAD_ONNX_MODEL`. If BLOB loading is not
supported by the target Oracle version, directory-based loading is attempted
only when `ORAMEMVID_ORACLE_ADMIN_USER` and
`ORAMEMVID_ORACLE_ADMIN_PASSWORD` are explicitly set. Otherwise startup fails
with instructions instead of silently pretending to fall back to another
provider.

To avoid ONNX setup entirely during local development, set:

```bash
export ORAMEMVID_EMBEDDING_PROVIDER=ollama
```
