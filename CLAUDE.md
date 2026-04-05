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
