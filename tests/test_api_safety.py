import os
import zipfile
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from oramemvid import api
from oramemvid.config import Settings


def test_upload_suffix_rejects_unsupported_file_type():
    with pytest.raises(HTTPException) as exc_info:
        api._validate_upload_suffix("payload.exe")

    assert exc_info.value.status_code == 415


def test_file_upload_route_cleans_temp_file_after_ingest(monkeypatch):
    conn = object()
    captured = {}

    monkeypatch.setattr(api, "_get_conn", lambda: conn)
    monkeypatch.setattr(api, "_release_conn", lambda _conn: None)

    def fake_ingest_file(**kwargs):
        captured["path"] = kwargs["file_path"]
        assert kwargs["conn"] is conn
        assert kwargs["filename"] == "sample.txt"
        assert os.path.exists(captured["path"])
        return {"ok": True}

    monkeypatch.setattr(api, "ingest_file", fake_ingest_file)

    upload = UploadFile(file=BytesIO(b"hello"), filename="sample.txt")

    assert api.route_ingest_file(upload) == {"ok": True}
    assert not os.path.exists(captured["path"])


def test_file_upload_limit_rejects_before_db_connection(monkeypatch):
    def fail_get_conn():
        raise AssertionError("DB connection should not be acquired")

    monkeypatch.setattr(api, "_get_conn", fail_get_conn)
    monkeypatch.setattr(
        api,
        "settings",
        Settings(oracle_password="test", max_upload_bytes=3),
    )

    upload = UploadFile(file=BytesIO(b"hello"), filename="sample.txt")

    with pytest.raises(HTTPException) as exc_info:
        api.route_ingest_file(upload)

    assert exc_info.value.status_code == 413


def test_file_upload_rejects_spoofed_pdf_before_db_connection(monkeypatch):
    def fail_get_conn():
        raise AssertionError("DB connection should not be acquired")

    monkeypatch.setattr(api, "_get_conn", fail_get_conn)

    upload = UploadFile(file=BytesIO(b"not a pdf"), filename="report.pdf")

    with pytest.raises(HTTPException) as exc_info:
        api.route_ingest_file(upload)

    assert exc_info.value.status_code == 415
    assert "not a PDF" in exc_info.value.detail


def test_file_upload_rejects_office_archive_with_too_many_files_before_db_connection(monkeypatch):
    def fail_get_conn():
        raise AssertionError("DB connection should not be acquired")

    monkeypatch.setattr(api, "_get_conn", fail_get_conn)

    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("word/document.xml", "<document />")
        for index in range(api._MAX_OFFICE_ARCHIVE_MEMBERS):
            archive.writestr(f"word/extra-{index}.xml", "")
    payload.seek(0)

    upload = UploadFile(file=payload, filename="report.docx")

    with pytest.raises(HTTPException) as exc_info:
        api.route_ingest_file(upload)

    assert exc_info.value.status_code == 413
    assert "too many files" in exc_info.value.detail


def test_file_upload_rejects_office_archive_unsafe_path_before_db_connection(monkeypatch):
    def fail_get_conn():
        raise AssertionError("DB connection should not be acquired")

    monkeypatch.setattr(api, "_get_conn", fail_get_conn)

    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("word/document.xml", "<document />")
        archive.writestr("../evil.xml", "")
    payload.seek(0)

    upload = UploadFile(file=payload, filename="report.docx")

    with pytest.raises(HTTPException) as exc_info:
        api.route_ingest_file(upload)

    assert exc_info.value.status_code == 415
    assert "unsafe path" in exc_info.value.detail


def test_search_tag_parse_error_rejects_before_db_connection(monkeypatch):
    def fail_get_conn():
        raise AssertionError("DB connection should not be acquired")

    monkeypatch.setattr(api, "_get_conn", fail_get_conn)

    with pytest.raises(HTTPException) as exc_info:
        api.route_search(query="Oracle", tags=["missing_equals"])

    assert exc_info.value.status_code == 400
