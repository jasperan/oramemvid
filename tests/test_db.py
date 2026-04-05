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
