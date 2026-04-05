import os
import tempfile

import httpx
import oracledb
from oramemvid.config import Settings

# Hugging Face ONNX model URL for all-MiniLM-L6-v2
_ONNX_MODEL_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
    "/resolve/main/onnx/model.onnx"
)

SCHEMA_VERSION = 1

_pool: oracledb.ConnectionPool | None = None

# Tablespace with ASSM required for VECTOR and JSON types.
# SYSTEM tablespace uses manual segment space management, so we
# target an ASSM-enabled tablespace instead.
_TABLESPACE = "PYTHIA_DATA"


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


def _detect_tablespace(cursor) -> str:
    """Pick an ASSM tablespace. Prefer the user's default if it's ASSM,
    otherwise fall back to the first available ASSM tablespace."""
    cursor.execute(
        "SELECT default_tablespace FROM user_users"
    )
    default_ts = cursor.fetchone()[0]
    cursor.execute(
        "SELECT segment_space_management FROM user_tablespaces "
        "WHERE tablespace_name = :ts",
        {"ts": default_ts},
    )
    row = cursor.fetchone()
    if row and row[0] == "AUTO":
        return default_ts

    # Default isn't ASSM; find one that is (exclude SYSAUX/UNDO/TEMP)
    cursor.execute(
        "SELECT tablespace_name FROM user_tablespaces "
        "WHERE segment_space_management = 'AUTO' "
        "AND tablespace_name NOT IN ('SYSAUX','UNDOTBS1','TEMP') "
        "FETCH FIRST 1 ROW ONLY"
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return default_ts  # last resort


def init_schema(conn: oracledb.Connection, settings: Settings | None = None):
    global _TABLESPACE
    cursor = conn.cursor()
    _TABLESPACE = _detect_tablespace(cursor)

    if not _table_exists(cursor, "SCHEMA_VERSION"):
        cursor.execute(f"""
            CREATE TABLE schema_version (
                version    NUMBER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT SYSTIMESTAMP
            ) TABLESPACE {_TABLESPACE}
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

    # Auto-load ONNX embedding model if configured for in-database embeddings
    if settings is None:
        settings = Settings()
    if settings.embedding_provider == "oracle_onnx":
        try:
            _ensure_onnx_model(conn, settings.onnx_model_name.upper())
        except Exception as e:
            print(f"Warning: ONNX auto-load failed: {e}")
            print("Falling back to Ollama embeddings. Set ORAMEMVID_EMBEDDING_PROVIDER=ollama or =sentence_transformers")


def _apply_v1(cursor):
    ts = _TABLESPACE

    # documents first (referenced by frames)
    if not _table_exists(cursor, "DOCUMENTS"):
        cursor.execute(f"""
            CREATE TABLE documents (
                doc_id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                filename      VARCHAR2(1000) NOT NULL,
                doc_type      VARCHAR2(20) NOT NULL,
                file_hash     VARCHAR2(64) NOT NULL,
                total_frames  NUMBER DEFAULT 0,
                ingested_at   TIMESTAMP DEFAULT SYSTIMESTAMP,
                CONSTRAINT documents_hash_uk UNIQUE (file_hash)
            ) TABLESPACE {ts}
        """)

    if not _table_exists(cursor, "FRAMES"):
        cursor.execute(f"""
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
            ) TABLESPACE {ts}
        """)

        # Oracle Text index for full-text search
        try:
            cursor.execute("""
                CREATE INDEX frames_content_idx ON frames(content)
                    INDEXTYPE IS CTXSYS.CONTEXT
                    PARAMETERS ('SYNC (ON COMMIT)')
            """)
        except oracledb.DatabaseError:
            pass  # Oracle Text not available, text search will use LIKE fallback

        # Vector index for similarity search
        try:
            cursor.execute("""
                CREATE VECTOR INDEX frames_vec_idx ON frames(embedding)
                    ORGANIZATION INMEMORY NEIGHBOR GRAPH
            """)
        except oracledb.DatabaseError:
            pass  # vector index not supported, vector search will do exact scan

    if not _table_exists(cursor, "MEMORY_CARDS"):
        cursor.execute(f"""
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
            ) TABLESPACE {ts}
        """)


def _onnx_model_exists(cursor, model_name: str) -> bool:
    """Check if an ONNX model is already loaded in Oracle."""
    cursor.execute(
        "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :name",
        {"name": model_name.upper()},
    )
    return cursor.fetchone()[0] > 0


def _download_onnx_model(url: str) -> bytes:
    """Download ONNX model file from URL, return raw bytes."""
    print(f"Downloading ONNX model from {url}...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
        resp.raise_for_status()
        chunks = []
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
    data = b"".join(chunks)
    print(f"Downloaded {len(data) / 1024 / 1024:.1f} MB")
    return data


def _fix_onnx_for_oracle(model_bytes: bytes) -> bytes:
    """Prepare ONNX model for Oracle DBMS_VECTOR.LOAD_ONNX_MODEL.

    Standard HuggingFace transformer ONNX exports output per-token
    hidden states [batch, seq_len, hidden_dim]. Oracle needs a model
    that outputs a single embedding vector [hidden_dim].

    This function:
    1. Fixes batch_size to 1 (Oracle allows at most 1 dynamic dim)
    2. Adds a ReduceMean pooling node over the sequence dimension
    3. Replaces the output with the pooled embedding [1, hidden_dim]
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    model = onnx.load_from_string(model_bytes)

    # Fix batch_size to 1 on all inputs (keep sequence_length dynamic)
    for tensor in model.graph.input:
        shape = tensor.type.tensor_type.shape
        if shape is None:
            continue
        for dim in shape.dim:
            if dim.dim_param == "batch_size":
                dim.ClearField("dim_param")
                dim.dim_value = 1

    # Find the original output (last_hidden_state: [batch, seq_len, 384])
    orig_output = model.graph.output[0]
    orig_output_name = orig_output.name

    # Get hidden_dim from the output shape (last dimension)
    hidden_dim = orig_output.type.tensor_type.shape.dim[-1].dim_value

    # Add ReduceMean node: mean over axis=1 (sequence dimension)
    # This pools [1, seq_len, 384] -> [1, 384]
    reduce_mean_node = helper.make_node(
        "ReduceMean",
        inputs=[orig_output_name],
        outputs=["embedding"],
        axes=[1],
        keepdims=0,
    )
    model.graph.node.append(reduce_mean_node)

    # Replace the output with the pooled embedding
    del model.graph.output[:]
    new_output = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [1, hidden_dim]
    )
    model.graph.output.append(new_output)

    return model.SerializeToString()


def _ensure_onnx_model(conn: oracledb.Connection, model_name: str):
    """Check if ONNX model is loaded; if not, download, fix, and load it."""
    cursor = conn.cursor()
    if _onnx_model_exists(cursor, model_name):
        return

    print(f"ONNX model '{model_name}' not found in Oracle. Auto-loading...")

    model_bytes = _download_onnx_model(_ONNX_MODEL_URL)

    # Fix dynamic axes for Oracle compatibility
    print("Fixing ONNX model dimensions for Oracle compatibility...")
    model_bytes = _fix_onnx_for_oracle(model_bytes)

    metadata = '{"function":"embedding","input":{"input":["DATA"]},"output":{"embedding":["DATA"]}}'

    # Try BLOB-based loading (Oracle 23ai+)
    try:
        model_blob = conn.createlob(oracledb.DB_TYPE_BLOB)
        model_blob.write(model_bytes)

        cursor.execute("""
            BEGIN
                DBMS_VECTOR.LOAD_ONNX_MODEL(
                    :model_name,
                    :model_data,
                    JSON(:metadata)
                );
            END;
        """, {
            "model_name": model_name,
            "model_data": model_blob,
            "metadata": metadata,
        })
        conn.commit()
        print(f"ONNX model '{model_name}' loaded into Oracle successfully.")
        return

    except oracledb.DatabaseError as e:
        print(f"BLOB loading failed: {e}")

    # Fallback: directory-based loading via SYSTEM user
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name

    try:
        tmp_dir = os.path.dirname(tmp_path)
        tmp_file = os.path.basename(tmp_path)
        settings = Settings()

        sys_conn = oracledb.connect(
            user="system", password="Welcome12345*", dsn=settings.oracle_dsn,
        )
        sys_cursor = sys_conn.cursor()
        sys_cursor.execute(
            f"CREATE OR REPLACE DIRECTORY onnx_model_dir AS '{tmp_dir}'"
        )
        sys_cursor.execute(
            f"GRANT READ ON DIRECTORY onnx_model_dir TO {settings.oracle_user}"
        )
        sys_conn.commit()
        sys_conn.close()

        cursor.execute("""
            BEGIN
                DBMS_VECTOR.LOAD_ONNX_MODEL(
                    :model_name,
                    'onnx_model_dir/' || :filename,
                    JSON(:metadata)
                );
            END;
        """, {
            "model_name": model_name,
            "filename": tmp_file,
            "metadata": metadata,
        })
        conn.commit()
        print(f"ONNX model '{model_name}' loaded via directory successfully.")

    except oracledb.DatabaseError as e2:
        print(
            f"Warning: Could not auto-load ONNX model: {e2}\n"
            f"Falling back to Ollama embeddings."
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    settings = Settings()
    pool = get_pool(settings)
    with pool.acquire() as conn:
        init_schema(conn, settings)
        print("Schema initialized successfully.")
    close_pool()
