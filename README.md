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
| `ORAMEMVID_OLLAMA_EMBED_MODEL` | `all-minilm` | Embedding model for Ollama fallback |

## Testing

```bash
pytest tests/ -v
```

Requires a running Oracle 26ai Free instance.
