from datetime import UTC

import pytest
from httpx import ASGITransport, AsyncClient

from oramemvid.api import app
from oramemvid.memory_cards import create_memory_card, delete_memory_card

pytestmark = pytest.mark.oracle


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_memory_profile_endpoint(client, db_conn):
    card_id = create_memory_card(
        db_conn, entity="ApiProfileEntity", slot="language",
        value="Python", kind="Fact", confidence=0.9,
    )
    try:
        resp = await client.get("/memory/profile", params={"entity": "ApiProfileEntity"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "ApiProfileEntity"
        assert data["total_cards"] == 1
        assert data["slots"][0]["slot"] == "language"
        assert data["slots"][0]["values"][0]["value"] == "Python"
    finally:
        delete_memory_card(db_conn, card_id)


@pytest.mark.asyncio
async def test_memory_profile_missing_entity_is_400(client):
    resp = await client.get("/memory/profile")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_delete_expired_cards_endpoint(client, db_conn):
    from datetime import datetime, timedelta

    past = datetime.now(UTC) - timedelta(days=1)
    expired_id = create_memory_card(
        db_conn, entity="ExpiryApiEntity", slot="state", value="old",
        kind="Fact", expires_at=past,
    )
    try:
        resp = await client.delete("/memory/expired")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] >= 1
    finally:
        delete_memory_card(db_conn, expired_id)
