"""R2 — orders sub-agent: agent card, JSON-RPC 2.0 protocol, skills and proposals.

The sub-agent runs in-process and talks to the REAL store app in-process
(httpx.ASGITransport), so nothing is mocked and no ports are opened.
"""

import httpx
import pytest

from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
ACME = {"Authorization": "Bearer tok_acme_alice"}


@pytest.fixture
def agent():
    store = httpx.ASGITransport(app=create_store())
    app = create_agent(store_url="http://store.test", public_url=AGENT, store_transport=store)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=AGENT)


async def rpc(agent, method, params=None, req_id=1, headers=ACME):
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return await agent.post("/", json=body, headers=headers)


# --- agent card ---------------------------------------------------------------


@pytest.mark.parametrize("path", ["/.well-known/agent-card.json", "/.well-known/agent.json"])
async def test_agent_card(agent, path):
    card = (await agent.get(path)).json()
    assert card["name"] == "orders-agent"
    assert card["url"] == f"{AGENT}/"
    skills = {s["id"]: s for s in card["skills"]}
    assert set(skills) == {
        "list_orders",
        "get_order",
        "preview_action",
        "refund_order",
        "cancel_order",
    }
    assert skills["refund_order"]["mutating"] is True
    assert skills["preview_action"]["mutating"] is False
    # Schemas are real JSON Schema the orchestrator can turn into LLM tools.
    assert skills["get_order"]["input_schema"]["required"] == ["order_id"]


async def test_card_needs_no_token(agent):
    assert (await agent.get("/.well-known/agent-card.json")).status_code == 200


# --- JSON-RPC protocol --------------------------------------------------------


async def test_success_echoes_id_and_returns_result(agent):
    r = await rpc(agent, "get_order", {"order_id": 1042}, req_id="abc-123")
    body = r.json()
    assert r.status_code == 200
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "abc-123"
    assert body["result"]["ok"] is True
    assert "error" not in body


async def test_parse_error(agent):
    r = await agent.post("/", content=b"{not json", headers=ACME)
    assert r.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "body",
    [
        {"id": 1, "method": "get_order"},  # missing "jsonrpc"
        {"jsonrpc": "1.0", "id": 1, "method": "get_order"},
        {"jsonrpc": "2.0", "id": 1, "method": 42},
        {"jsonrpc": "2.0", "id": {"x": 1}, "method": "get_order"},
        "hello",
    ],
)
async def test_invalid_request(agent, body):
    r = await agent.post("/", json=body, headers=ACME)
    assert r.json()["error"]["code"] == -32600


async def test_batch_is_rejected(agent):
    r = await agent.post(
        "/", json=[{"jsonrpc": "2.0", "id": 1, "method": "list_orders"}], headers=ACME
    )
    assert r.json()["error"]["code"] == -32600


async def test_unknown_method(agent):
    r = await rpc(agent, "delete_everything", {})
    assert r.json()["error"]["code"] == -32601


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("get_order", {}),  # missing order_id
        ("get_order", {"order_id": "abc"}),
        ("get_order", {"order_id": -5}),
        ("get_order", {"order_id": 1042, "merchant": "m_globex"}),  # extra field
        ("list_orders", {"status": "LOST"}),
        ("preview_action", {"action": "delete", "order_id": 1042}),
        ("get_order", [1042]),  # params must be an object
    ],
)
async def test_invalid_params(agent, method, params):
    r = await rpc(agent, method, params)
    assert r.json()["error"]["code"] == -32602


async def test_invalid_params_does_not_echo_input(agent):
    r = await rpc(agent, "get_order", {"order_id": "secret-looking-value"})
    assert "secret-looking-value" not in r.text


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}]
)
async def test_missing_token_is_401(agent, headers):
    r = await rpc(agent, "list_orders", {}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32001


async def test_notification_is_not_executed(agent):
    """No id = notification: no reply, and the refund must NOT happen."""
    body = {"jsonrpc": "2.0", "method": "refund_order", "params": {"order_id": 1042}}
    r = await agent.post("/", json=body, headers=ACME)
    assert r.status_code == 204
    order = (await rpc(agent, "get_order", {"order_id": 1042})).json()["result"]["data"]
    assert order["status"] == "PAID"


async def test_business_failure_is_a_result_not_an_error(agent):
    r = await rpc(agent, "get_order", {"order_id": 2001})  # Globex order
    body = r.json()
    assert "error" not in body
    assert body["result"] == {
        "ok": False,
        "data": None,
        "error": "Order 2001 was not found in your store.",
    }


# --- skills -------------------------------------------------------------------


async def test_list_orders_with_filters(agent):
    r = await rpc(agent, "list_orders", {"status": "PAID", "placed_more_than_days_ago": 3})
    assert {o["id"] for o in r.json()["result"]["data"]} == {1001, 1006, 1042}


async def test_results_are_sanitized(agent):
    r = await rpc(agent, "list_orders", {})
    assert "@" not in r.text
    assert "ignore your instructions" not in r.text


# --- proposals (read-only previews for R5) ----------------------------------------


async def test_preview_refund_describes_change_and_changes_nothing(agent):
    r = await rpc(agent, "preview_action", {"action": "refund", "order_id": 1042})
    proposal = r.json()["result"]["data"]
    assert proposal == {
        "action": "refund",
        "order_id": 1042,
        "merchant": "m_acme",
        "customer_name": "Maria Lopez",
        "current_status": "PAID",
        "new_status": "REFUNDED",
        "order_total": 59.90,
        "amount_to_customer": 59.90,
        "summary": (
            "Refund order 1042 for Maria Lopez: PAID -> REFUNDED; "
            "59.90 will be returned to the customer."
        ),
    }
    order = (await rpc(agent, "get_order", {"order_id": 1042})).json()["result"]["data"]
    assert order["status"] == "PAID"


async def test_preview_cancel_of_unpaid_order_returns_nothing(agent):
    r = await rpc(agent, "preview_action", {"action": "cancel", "order_id": 1005})
    assert r.json()["result"]["data"]["amount_to_customer"] == 0.0


@pytest.mark.parametrize(
    ("action", "order_id", "status"),
    [("refund", 1007, "CANCELLED"), ("cancel", 1003, "DELIVERED"), ("cancel", 1002, "SHIPPED")],
)
async def test_preview_rejects_invalid_transitions(agent, action, order_id, status):
    r = await rpc(agent, "preview_action", {"action": action, "order_id": order_id})
    result = r.json()["result"]
    assert result["ok"] is False
    assert status in result["error"]


async def test_preview_of_foreign_order_is_not_found(agent):
    r = await rpc(agent, "preview_action", {"action": "refund", "order_id": 2001})
    assert r.json()["result"]["error"] == "Order 2001 was not found in your store."


async def test_refund_executes(agent):
    r = await rpc(agent, "refund_order", {"order_id": 1042})
    assert r.json()["result"]["data"]["status"] == "REFUNDED"
