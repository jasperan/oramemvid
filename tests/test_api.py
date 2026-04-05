import pytest
from httpx import AsyncClient, ASGITransport
from oramemvid.api import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_ingest_text(client):
    resp = await client.post("/ingest/text", json={
        "text": "Oracle Database is great for AI workloads unique api test 1.",
        "uri": "test://api/text-unique-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_frames"] >= 1


@pytest.mark.asyncio
async def test_list_frames(client):
    resp = await client.get("/frames")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_search(client):
    await client.post("/ingest/text", json={
        "text": "FastAPI makes building REST APIs simple and fast unique api test 2.",
        "uri": "test://api/search-unique-1",
    })
    resp = await client.get("/search", params={"query": "FastAPI", "mode": "text"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_list_memory_cards(client):
    resp = await client.get("/memory")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_list_documents(client):
    resp = await client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_stats(client):
    resp = await client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "frame_count" in data
    assert "card_count" in data
    assert "document_count" in data
