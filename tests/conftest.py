import pytest
import oracledb
from oramemvid.config import Settings

_schema_initialized = False


@pytest.fixture(scope="session")
def settings():
    return Settings()


@pytest.fixture(scope="session")
def db_pool(settings):
    try:
        pool = oracledb.create_pool(
            user=settings.oracle_user,
            password=settings.oracle_password,
            dsn=settings.oracle_dsn,
            min=1,
            max=5,
        )
    except oracledb.Error as exc:
        pytest.skip(f"Oracle DB unavailable: {exc}")
    yield pool
    pool.close()


@pytest.fixture
def db_conn(db_pool):
    conn = db_pool.acquire()
    yield conn
    conn.rollback()
    db_pool.release(conn)


@pytest.fixture(autouse=True)
def init_schema_for_oracle_tests(request):
    global _schema_initialized
    needs_oracle = (
        "db_pool" in request.fixturenames
        or "db_conn" in request.fixturenames
        or request.node.get_closest_marker("oracle") is not None
    )
    if not needs_oracle or _schema_initialized:
        return

    from oramemvid.db import OnnxModelLoadError, init_schema

    db_pool = request.getfixturevalue("db_pool")
    settings = request.getfixturevalue("settings")
    try:
        with db_pool.acquire() as conn:
            init_schema(conn, settings)
    except OnnxModelLoadError as exc:
        pytest.skip(f"Oracle ONNX bootstrap unavailable: {exc}")
    _schema_initialized = True
