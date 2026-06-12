import logging
import re
from dataclasses import dataclass, field

import oracledb
from oramemvid.config import Settings

SCHEMA_VERSION = 1
EMBEDDING_DIMS = 384

_pool: oracledb.ConnectionPool | None = None
_DB_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9_$#]*$")
_capabilities: "DatabaseCapabilities | None" = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseCapabilities:
    oracle_text: bool = False
    vector_index: bool = False
    degraded_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)

    def as_dict(self) -> dict:
        return {
            "oracle_text": self.oracle_text,
            "vector_index": self.vector_index,
            "degraded": self.degraded,
            "degraded_reasons": dict(self.degraded_reasons),
        }


class OnnxModelLoadError(RuntimeError):
    """Raised when oracle_onnx is configured but the model cannot be loaded."""


def _validate_db_identifier(value: str, label: str) -> str:
    value = value.upper()
    if not _DB_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid Oracle {label}: {value!r}")
    return value


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


def get_capabilities() -> DatabaseCapabilities:
    return _capabilities or DatabaseCapabilities(
        degraded_reasons={"schema": "Schema has not been initialized in this process."}
    )


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM user_tables WHERE table_name = :name",
        {"name": table_name.upper()},
    )
    return cursor.fetchone()[0] > 0


def _index_exists(cursor, index_name: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM user_indexes WHERE index_name = :name",
        {"name": index_name.upper()},
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
    global _capabilities
    cursor = conn.cursor()
    tablespace = _detect_tablespace(cursor)

    if not _table_exists(cursor, "SCHEMA_VERSION"):
        cursor.execute(f"""
            CREATE TABLE schema_version (
                version    NUMBER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT SYSTIMESTAMP
            ) TABLESPACE {tablespace}
        """)

    cursor.execute(
        "SELECT MAX(version) FROM schema_version"
    )
    row = cursor.fetchone()
    current = row[0] if row[0] is not None else 0

    if current < 1:
        _apply_v1(cursor, tablespace)
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (1)"
        )

    capability_reasons = _ensure_optional_indexes(cursor)
    _capabilities = _detect_capabilities(cursor, capability_reasons)

    conn.commit()

    # Auto-load ONNX embedding model if configured for in-database embeddings.
    # If this fails, startup fails explicitly. The embedding provider cannot be
    # switched from here because it is selected before schema initialization.
    if settings is None:
        settings = Settings()
    if settings.embedding_provider == "oracle_onnx":
        _ensure_onnx_model(
            conn,
            _validate_db_identifier(settings.onnx_model_name, "model name"),
            settings,
        )

    _enforce_required_capabilities(_capabilities, settings)


def _apply_v1(cursor, ts: str):
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
                embedding     VECTOR({EMBEDDING_DIMS}, FLOAT32),
                created_at    TIMESTAMP DEFAULT SYSTIMESTAMP,
                CONSTRAINT frames_hash_uk UNIQUE (content_hash)
            ) TABLESPACE {ts}
        """)

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


def _ensure_optional_indexes(cursor) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if _table_exists(cursor, "FRAMES") and not _index_exists(cursor, "FRAMES_CONTENT_IDX"):
        try:
            cursor.execute("""
                CREATE INDEX frames_content_idx ON frames(content)
                    INDEXTYPE IS CTXSYS.CONTEXT
                    PARAMETERS ('SYNC (ON COMMIT)')
            """)
        except oracledb.DatabaseError as exc:
            reasons["oracle_text"] = (
                f"Oracle Text index FRAMES_CONTENT_IDX unavailable: {exc}"
            )

    if _table_exists(cursor, "FRAMES") and not _index_exists(cursor, "FRAMES_VEC_IDX"):
        try:
            cursor.execute("""
                CREATE VECTOR INDEX frames_vec_idx ON frames(embedding)
                    ORGANIZATION INMEMORY NEIGHBOR GRAPH
            """)
        except oracledb.DatabaseError as exc:
            reasons["vector_index"] = (
                f"Vector index FRAMES_VEC_IDX unavailable: {exc}"
            )
    return reasons


def _detect_capabilities(cursor, reasons: dict[str, str] | None = None) -> DatabaseCapabilities:
    reasons = dict(reasons or {})
    oracle_text = _index_exists(cursor, "FRAMES_CONTENT_IDX")
    vector_index = _index_exists(cursor, "FRAMES_VEC_IDX")
    if not oracle_text:
        reasons.setdefault(
            "oracle_text",
            "Oracle Text index FRAMES_CONTENT_IDX is not present; "
            "text search will use degraded DBMS_LOB.INSTR mode.",
        )
    if not vector_index:
        reasons.setdefault(
            "vector_index",
            "Vector index FRAMES_VEC_IDX is not present; vector search will use exact scan.",
        )
    return DatabaseCapabilities(
        oracle_text=oracle_text,
        vector_index=vector_index,
        degraded_reasons=reasons,
    )


def _enforce_required_capabilities(
    capabilities: DatabaseCapabilities,
    settings: Settings,
) -> None:
    missing = []
    if settings.require_oracle_text and not capabilities.oracle_text:
        missing.append(capabilities.degraded_reasons.get("oracle_text", "Oracle Text unavailable"))
    if settings.require_vector_index and not capabilities.vector_index:
        missing.append(capabilities.degraded_reasons.get("vector_index", "Vector index unavailable"))
    if missing:
        raise RuntimeError("Required Oracle capabilities are unavailable: " + "; ".join(missing))


def _onnx_model_exists(cursor, model_name: str) -> bool:
    """Check if an ONNX model is already loaded in Oracle."""
    cursor.execute(
        "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :name",
        {"name": model_name.upper()},
    )
    return cursor.fetchone()[0] > 0


def _onnx2oracle_spec(model_name: str, presets=None):
    if presets is None:
        from onnx2oracle.presets import PRESETS

        presets = PRESETS

    for spec in presets.values():
        if spec.oracle_name == model_name:
            if spec.dims != EMBEDDING_DIMS:
                raise OnnxModelLoadError(
                    f"Unsupported ONNX model {model_name!r}: onnx2oracle preset "
                    f"outputs {spec.dims} dimensions, but the schema requires "
                    f"{EMBEDDING_DIMS}."
                )
            return spec
    available = ", ".join(
        sorted(spec.oracle_name for spec in presets.values() if spec.dims == EMBEDDING_DIMS)
    )
    raise OnnxModelLoadError(
        f"Unsupported ONNX model {model_name!r}. "
        f"Compatible onnx2oracle presets for ORAMEMVID_ONNX_MODEL_NAME: {available}."
    )


def _build_oracle_onnx_model(model_name: str) -> bytes:
    from onnx2oracle.pipeline import build_augmented

    return build_augmented(_onnx2oracle_spec(model_name))


def _onnx_metadata_json() -> str:
    from onnx2oracle.loader import build_metadata_json

    return build_metadata_json()


def _ensure_onnx_model(
    conn: oracledb.Connection,
    model_name: str,
    settings: Settings | None = None,
):
    """Check if ONNX model is loaded; if not, download, fix, and load it."""
    if settings is None:
        settings = Settings()
    cursor = conn.cursor()
    if _onnx_model_exists(cursor, model_name):
        return

    logger.info("ONNX model '%s' not found in Oracle. Auto-loading with onnx2oracle...", model_name)

    try:
        model_bytes = _build_oracle_onnx_model(model_name)
        metadata = _onnx_metadata_json()
    except ImportError as exc:
        raise OnnxModelLoadError(
            "oracle_onnx embeddings require the onnx2oracle optional dependency. "
            "Install with `pip install -e '.[oracle-onnx]'`, or set "
            "ORAMEMVID_EMBEDDING_PROVIDER=ollama."
        ) from exc
    except OnnxModelLoadError:
        raise
    except Exception as exc:
        raise OnnxModelLoadError(
            "Could not build an Oracle-compatible ONNX model with onnx2oracle. "
            "Pre-load the model manually or set ORAMEMVID_EMBEDDING_PROVIDER=ollama."
        ) from exc

    try:
        cursor.execute("""
            BEGIN
                DBMS_VECTOR.LOAD_ONNX_MODEL(
                    model_name => :model_name,
                    model_data => :model_data,
                    metadata   => JSON(:metadata)
                );
            END;
        """, {
            "model_name": model_name,
            "model_data": model_bytes,
            "metadata": metadata,
        })
        conn.commit()
        logger.info("ONNX model '%s' loaded into Oracle successfully.", model_name)

    except oracledb.DatabaseError as e:
        raise OnnxModelLoadError(
            "Could not load the onnx2oracle-built ONNX model with DBMS_VECTOR. "
            "Pre-load the model manually or set ORAMEMVID_EMBEDDING_PROVIDER=ollama."
        ) from e


if __name__ == "__main__":
    settings = Settings()
    pool = get_pool(settings)
    with pool.acquire() as conn:
        init_schema(conn, settings)
        print("Schema initialized successfully.")
    close_pool()
