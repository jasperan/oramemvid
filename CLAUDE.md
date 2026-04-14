# oramemvid

AI memory layer backed by Oracle Database, inspired by memvid.

## Quick Start

```bash
conda activate oramemvid
docker compose up -d          # Oracle 26ai Free; first boot takes ~2 min
pip install -e ".[dev]"
python -m oramemvid.db        # init schema + auto-load ONNX model
uvicorn oramemvid.api:app --reload
```

Copy `.env.example` to `.env` before running. All settings are prefixed `ORAMEMVID_`.

## Testing

Tests require a live Oracle DB (no mocks). Schema is auto-inited by `conftest.py`.

```bash
pytest tests/ -v
```

Run a single module: `pytest tests/test_search.py -v`

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `ORAMEMVID_ORACLE_DSN` | `localhost:1523/FREEPDB1` | |
| `ORAMEMVID_ORACLE_USER` | `oramemvid` | |
| `ORAMEMVID_ORACLE_PASSWORD` | *(required)* | `oramemvid_dev` in docker-compose |
| `ORAMEMVID_OLLAMA_URL` | `http://localhost:11434` | |
| `ORAMEMVID_OLLAMA_MODEL` | `qwen3.5:9b` | LLM for extraction |
| `ORAMEMVID_EMBEDDING_PROVIDER` | `oracle_onnx` | `oracle_onnx` / `ollama` / `sentence_transformers` |
| `ORAMEMVID_ONNX_MODEL_NAME` | `all_minilm_l6_v2` | Oracle mining model name |
| `ORAMEMVID_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Used when provider=ollama |
| `ORAMEMVID_CHUNK_SIZE` | `512` | Tokens per frame |
| `ORAMEMVID_CHUNK_OVERLAP` | `50` | |

## Tech Stack

- Python 3.12, conda env: `oramemvid`
- Oracle 26ai Free on port 1523/FREEPDB1 (container port 1521)
- FastAPI on port 8000
- Ollama at `localhost:11434` (qwen3.5:9b for extraction, nomic-embed-text fallback)
- In-database ONNX embeddings: all-MiniLM-L6-v2 loaded via `DBMS_VECTOR.LOAD_ONNX_MODEL`
- Embedding vector dimension: 384 (FLOAT32)

## Key Modules

- `config.py` — pydantic-settings from `ORAMEMVID_*` env vars
- `db.py` — connection pool, schema init, ONNX model auto-load
- `embeddings.py` — pluggable: `OracleONNXEmbedding` (default), `OllamaEmbedding`, `SentenceTransformerEmbedding`
- `frames.py` — frame CRUD (content chunks with embeddings)
- `memory_cards.py` — structured memory CRUD (entity/slot/value)
- `llm.py` — pluggable LLM, default `OllamaLLM` (qwen3.5:9b)
- `ingest.py` — document pipeline (PDF via pymupdf, DOCX, XLSX, PPTX, TXT)
- `search.py` — unified text/vector/hybrid search with RRF fusion
- `api.py` — FastAPI REST routes

## Schema

Tables: `documents`, `frames`, `memory_cards`, `schema_version`. All created in an ASSM tablespace (auto-detected; VECTOR and JSON types are incompatible with the SYSTEM tablespace which uses manual segment space management).

`frames.embedding` is `VECTOR(384, FLOAT32)`. An Oracle Text (`CTXSYS.CONTEXT`) index and a vector index are created if available; both degrade gracefully if not present.

`memory_cards.card_value` stores the value (named to avoid the SQL reserved word `VALUE`). `kind` is constrained to: `Fact`, `Preference`, `Event`, `Profile`, `Relationship`, `Goal`.

Schema migrations are tracked in `schema_version`. Current version: 1.

## Gotchas

**ONNX model loading** happens automatically during `init_schema`. It downloads all-MiniLM-L6-v2 from HuggingFace (~90 MB), patches the ONNX graph to fix dynamic axes Oracle doesn't support, then loads it via `DBMS_VECTOR.LOAD_ONNX_MODEL`. This requires the `onnx` package (`pip install onnx`) which is not in `pyproject.toml` — add it if you need oracle_onnx embeddings.

**ONNX directory fallback**: if BLOB loading fails, the code falls back to a directory-based load using a hardcoded `system` password `Welcome12345*`. This only works against the docker-compose Oracle image. Override in `db.py` if your Oracle setup differs.

**ASSM tablespace**: `db.py` auto-detects an ASSM tablespace. If your Oracle user's default tablespace isn't ASSM, it picks the first available one. The hardcoded fallback name `PYTHIA_DATA` is vestigial and may not exist — the detection logic will handle it.

**First `pytest` run** initializes the schema and may download the ONNX model. Can take several minutes. Subsequent runs are fast.

**`asyncio_mode = "auto"`** is set in `pyproject.toml` — all async tests work without `@pytest.mark.asyncio`.
