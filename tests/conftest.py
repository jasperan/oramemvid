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
def init_schema(db_pool, settings):
    from oramemvid.db import init_schema
    with db_pool.acquire() as conn:
        init_schema(conn, settings)
