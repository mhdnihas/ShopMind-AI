"""R2 — orders sub-agent on the A2A protocol: agent card, SendMessage, skills, proposals.

The sub-agent runs in-process and talks to the REAL store app in-process
(httpx.ASGITransport), so nothing is mocked and no ports are opened. The requests here
are written by hand on purpose, so they pin down the actual A2A wire format.
"""

import httpx
import pytest

from common.a2a_contract import SKILL_EXTENSION_URI, restore_integers
from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
A2A_VERSION = {"A2A-Version": "1.0"}
ACME = {"Authorization": "Bearer tok_acme_alice", **A2A_VERSION}
CARD_PATH = "/.well-known/agent-card.json"


@pytest.fixture
def agent():
    store = httpx.ASGITransport(app=create_store())
    app = create_agent(store_url="http://store.test", public_url=AGENT, store_transport=store)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=AGENT)


def envelope(parts, req_id=1):
    body = {
        "jsonrpc": "2.0",
        "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": parts}},
    }
    if req_id is not None:
        body["id"] = req_id
    return body


async def send(agent, skill, params=None, req_id=1, headers=ACME):
    call = {"skill": skill}
    if params is not None:
        call["params"] = params
    return await agent.post("/", json=envelope([{"data": call}], req_id), headers=headers)


def result_of(response):
    """The ToolResult inside an A2A agent message."""
    message = response.json()["result"]["message"]
    assert message["role"] == "ROLE_AGENT"
    return restore_integers(message["parts"][0]["data"])


# --- agent card -------------------------------------------------------------------


async def test_agent_card_follows_a2a(agent):
    card = (await agent.get(CARD_PATH)).json()
    assert card["name"] == "orders-agent"
    assert card["supportedInterfaces"] == [
        {"url": f"{AGENT}/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    assert card["defaultInputModes"] == ["application/json"]
    assert card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"] == "bearer"
    assert {s["id"] for s in card["skills"]} == {
        "list_orders",
        "get_order",
        "preview_action",
        "refund_order",
        "cancel_order",
    }


async def test_card_extension_publishes_schemas_and_the_mutating_flag(agent):
    card = (await agent.get(CARD_PATH)).json()
    extension = next(
        e for e in card["capabilities"]["extensions"] if e["uri"] == SKILL_EXTENSION_URI
    )
    skills = restore_integers(extension["params"])["skills"]
    assert skills["refund_order"]["mutating"] is True
    assert skills["preview_action"]["mutating"] is False
    assert skills["get_order"]["input_schema"]["required"] == ["order_id"]


async def test_card_needs_no_token(agent):
    assert (await agent.get(CARD_PATH)).status_code == 200


# --- the protocol -----------------------------------------------------------------


async def test_success_echoes_id_and_returns_an_agent_message(agent):
    response = await send(agent, "get_order", {"order_id": 1042}, req_id="abc-123")
    body = response.json()
    assert response.status_code == 200
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "abc-123"
    assert "error" not in body
    assert result_of(response)["ok"] is True


async def test_parse_error(agent):
    response = await agent.post("/", content=b"{not json", headers=ACME)
    assert response.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "body",
    [
        {"id": 1, "method": "SendMessage"},  # missing "jsonrpc"
        {"jsonrpc": "1.0", "id": 1, "method": "SendMessage"},
        {"jsonrpc": "2.0", "id": 1, "method": 42},
        [{"jsonrpc": "2.0", "id": 1, "method": "SendMessage"}],  # batch
    ],
)
async def test_invalid_request(agent, body):
    response = await agent.post("/", json=body, headers=ACME)
    assert response.json()["error"]["code"] == -32600


async def test_old_style_skill_method_is_not_part_of_the_protocol(agent):
    body = {"jsonrpc": "2.0", "id": 1, "method": "get_order", "params": {"order_id": 1042}}
    response = await agent.post("/", json=body, headers=ACME)
    assert response.json()["error"]["code"] == -32601


async def test_request_without_the_protocol_version_is_refused(agent):
    headers = {"Authorization": "Bearer tok_acme_alice"}
    response = await send(agent, "get_order", {"order_id": 1042}, headers=headers)
    assert response.json()["error"]["code"] == -32009


@pytest.mark.parametrize(
    "parts",
    [
        [{"text": "show me order 1042"}],
        [{"data": ["not", "an", "object"]}],
        [{"data": {"params": {"order_id": 1042}}}],  # no skill named
        [
            {"data": {"skill": "get_order", "params": {"order_id": 1}}},
            {"data": {"skill": "get_order", "params": {"order_id": 2}}},
        ],
    ],
)
async def test_message_must_carry_exactly_one_skill_call(agent, parts):
    response = await agent.post("/", json=envelope(parts), headers=ACME)
    assert response.json()["error"]["code"] == -32602


async def test_unknown_skill_is_invalid_params(agent):
    response = await send(agent, "delete_everything", {})
    assert response.json()["error"]["code"] == -32602


@pytest.mark.parametrize(
    ("skill", "params"),
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
async def test_invalid_params(agent, skill, params):
    response = await send(agent, skill, params)
    assert response.json()["error"]["code"] == -32602


async def test_invalid_params_does_not_echo_input(agent):
    response = await send(agent, "get_order", {"order_id": "secret-looking-value"})
    assert "secret-looking-value" not in response.text


@pytest.mark.parametrize(
    "headers", [A2A_VERSION, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}]
)
async def test_missing_token_is_401(agent, headers):
    response = await send(agent, "list_orders", {}, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == -32001


async def test_notification_is_not_executed(agent):
    """No id = notification: no reply, and the refund must NOT happen."""
    response = await send(agent, "refund_order", {"order_id": 1042}, req_id=None)
    assert response.status_code == 204
    order = result_of(await send(agent, "get_order", {"order_id": 1042}))["data"]
    assert order["status"] == "PAID"


async def test_business_failure_is_a_result_not_an_error(agent):
    response = await send(agent, "get_order", {"order_id": 2001})  # Globex order
    assert "error" not in response.json()
    assert result_of(response) == {
        "ok": False,
        "data": None,
        "error": "Order 2001 was not found in your store.",
    }


# --- skills -----------------------------------------------------------------------


async def test_numbers_survive_the_protocol(agent):
    order = result_of(await send(agent, "get_order", {"order_id": 1042}))["data"]
    assert order["id"] == 1042
    assert isinstance(order["id"], int)
    assert order["total"] == 59.90


async def test_list_orders_with_filters(agent):
    params = {"status": "PAID", "placed_more_than_days_ago": 3}
    result = result_of(await send(agent, "list_orders", params))
    assert {o["id"] for o in result["data"]} == {1001, 1006, 1042}


async def test_results_are_sanitized(agent):
    response = await send(agent, "list_orders", {})
    assert "@" not in response.text
    assert "ignore your instructions" not in response.text


# --- proposals (read-only previews for R5) ----------------------------------------


async def test_preview_refund_describes_change_and_changes_nothing(agent):
    response = await send(agent, "preview_action", {"action": "refund", "order_id": 1042})
    assert result_of(response)["data"] == {
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
    order = result_of(await send(agent, "get_order", {"order_id": 1042}))["data"]
    assert order["status"] == "PAID"


async def test_preview_cancel_of_unpaid_order_returns_nothing(agent):
    response = await send(agent, "preview_action", {"action": "cancel", "order_id": 1005})
    assert result_of(response)["data"]["amount_to_customer"] == 0.0


@pytest.mark.parametrize(
    ("action", "order_id", "status"),
    [("refund", 1007, "CANCELLED"), ("cancel", 1003, "DELIVERED"), ("cancel", 1002, "SHIPPED")],
)
async def test_preview_rejects_invalid_transitions(agent, action, order_id, status):
    response = await send(agent, "preview_action", {"action": action, "order_id": order_id})
    result = result_of(response)
    assert result["ok"] is False
    assert status in result["error"]


async def test_preview_of_foreign_order_is_not_found(agent):
    response = await send(agent, "preview_action", {"action": "refund", "order_id": 2001})
    assert result_of(response)["error"] == "Order 2001 was not found in your store."


async def test_refund_executes(agent):
    response = await send(agent, "refund_order", {"order_id": 1042})
    assert result_of(response)["data"]["status"] == "REFUNDED"
