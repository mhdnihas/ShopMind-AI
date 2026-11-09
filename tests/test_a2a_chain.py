"""R2 + R3 — orchestrator client -> sub-agent -> store, across HTTP boundaries.

Proves:
* the orchestrator discovers the sub-agent from its card (no import of its code);
* the caller's token arrives at the store byte-for-byte unchanged;
* merchant A can never read or change merchant B's orders through the chain.
"""

import httpx
import pytest
import respx

from orchestrator.a2a_client import A2AClient, A2AError
from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
STORE = "http://store.test"
ACME = "tok_acme_alice"
GLOBEX = "tok_globex_bob"


@pytest.fixture
def agent_transport():
    """Sub-agent wired to a fresh in-process store."""
    store = httpx.ASGITransport(app=create_store())
    app = create_agent(store_url=STORE, public_url=AGENT, store_transport=store)
    return httpx.ASGITransport(app=app)


@pytest.fixture
async def client(agent_transport):
    return await A2AClient.discover(AGENT, transport=agent_transport)


async def test_discovery_reads_endpoint_and_skills_from_card(client):
    assert client.card.name == "orders-agent"
    assert client.card.url == f"{AGENT}/"
    assert client.card.skill("preview_action") is not None
    assert client.card.skill("refund_order").mutating


async def test_discovery_failure_is_an_a2a_error():
    with respx.mock:
        respx.get(f"{AGENT}/.well-known/agent-card.json").respond(404)
        with pytest.raises(A2AError, match="discovery failed"):
            await A2AClient.discover(AGENT)


async def test_call_returns_result(client):
    result = await client.call("get_order", {"order_id": 1042}, ACME)
    assert result["ok"] is True
    assert result["data"]["customer_name"] == "Maria Lopez"


async def test_unknown_skill_is_refused_locally(client):
    with pytest.raises(A2AError) as e:
        await client.call("drop_tables", {}, ACME)
    assert e.value.code == -32601


async def test_invalid_params_surface_as_a2a_error(client):
    with pytest.raises(A2AError) as e:
        await client.call("get_order", {"order_id": "abc"}, ACME)
    assert e.value.code == -32602


async def test_response_with_the_wrong_id_is_rejected(client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "other", "result": {"ok": True}})

    other = A2AClient(client.card, transport=httpx.MockTransport(handler))
    with pytest.raises(A2AError, match="Malformed"):
        await other.call("get_order", {"order_id": 1}, ACME)


# --- R3: credential forwarding -------------------------------------------------


async def test_token_reaches_store_unchanged_through_both_hops():
    """Real sub-agent in-process; the store is mocked by respx so we can inspect
    the exact Authorization header the sub-agent sends."""
    token = "tok_acme_alice"
    agent = create_agent(store_url=STORE, public_url=AGENT)  # real HTTP to the store
    transport = httpx.ASGITransport(app=agent)  # not intercepted by respx

    with respx.mock(base_url=STORE) as store:
        store.get("/orders").respond(200, json=[])
        store.get("/orders/lookup").respond(200, json={"status": False, "message": "x"})
        client = await A2AClient.discover(AGENT, transport=transport)
        await client.call("list_orders", {}, token)
        await client.call("get_order", {"order_id": 1}, token)

        assert len(store.calls) == 2
        for call in store.calls:
            assert call.request.headers["Authorization"] == f"Bearer {token}"


# --- R3: tenant isolation through the whole chain ----------------------------------


async def test_merchant_a_cannot_see_merchant_b(client):
    acme_orders = (await client.call("list_orders", {}, ACME))["data"]
    globex_orders = (await client.call("list_orders", {}, GLOBEX))["data"]
    assert {o["merchant"] for o in acme_orders} == {"m_acme"}
    assert {o["merchant"] for o in globex_orders} == {"m_globex"}

    # Asking for B's order by id, or B's customer by name, gets nothing.
    assert (await client.call("get_order", {"order_id": 2001}, ACME))["ok"] is False
    by_name = await client.call("list_orders", {"customer_name": "Garcia"}, ACME)
    assert by_name["data"] == []


async def test_merchant_a_cannot_change_merchant_b(client):
    for skill in ("preview_action", "refund_order", "cancel_order"):
        params = {"order_id": 2001}
        if skill == "preview_action":
            params["action"] = "refund"
        result = await client.call(skill, params, ACME)
        assert result["ok"] is False
        assert result["error"] == "Order 2001 was not found in your store."

    still = await client.call("get_order", {"order_id": 2001}, GLOBEX)
    assert still["data"]["status"] == "PAID"


async def test_orphan_token_gets_nothing(client):
    result = await client.call("list_orders", {}, "tok_orphan_carol")
    assert result["ok"] is False
    assert "not linked to a merchant" in result["error"]
