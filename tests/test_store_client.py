"""Order tools against the store API: envelope handling, errors as data, token forwarding.

Two styles, on purpose:
* `store` fixture — the REAL store app run in-process through httpx.ASGITransport
  (no network, no mocks), so behaviour matches production exactly.
* respx — a mocked store, for things the real app can't easily produce
  (network failures, odd legacy bodies) and to inspect outgoing requests.
"""

import json

import httpx
import pytest
import respx

from orders_agent.guardrails import NOTE_WITHHELD
from orders_agent.store_client import StoreClient
from store_api.main import create_app

BASE = "http://store.test"


@pytest.fixture
def transport():
    # Fresh store per test so refunds don't leak between tests.
    return httpx.ASGITransport(app=create_app())


def client_for(token, transport):
    return StoreClient(BASE, token, transport=transport)


# --- unreadable responses ---------------------------------------------------


@pytest.mark.parametrize("body", [b"<html>oops</html>", b'"text"', b"[1, 2]", b'{"a": 1}'])
async def test_list_orders_reports_an_unreadable_body(body):
    with respx.mock(base_url=BASE) as store:
        store.get("/orders").respond(200, content=body)
        async with StoreClient(BASE, "tok") as c:
            r = await c.list_orders()
    assert r.ok is False
    assert "unreadable" in r.error


async def test_list_orders_reports_an_order_with_no_date():
    with respx.mock(base_url=BASE) as store:
        store.get("/orders").respond(200, json=[{"id": 1}])
        async with StoreClient(BASE, "tok") as c:
            r = await c.list_orders(placed_more_than_days_ago=3)
    assert r.ok is False
    assert "unreadable" in r.error


@pytest.mark.parametrize("body", [b"<html>oops</html>", b'"text"', b"[1]"])
async def test_refund_reports_an_unreadable_body(body):
    with respx.mock(base_url=BASE) as store:
        store.post("/orders/refund").respond(200, content=body)
        async with StoreClient(BASE, "tok") as c:
            r = await c.refund_order(1042)
    assert r.ok is False
    assert "unreadable" in r.error


# --- list_orders ------------------------------------------------------------


async def test_list_orders_is_scoped_and_sanitized(transport):
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.list_orders()
    assert r.ok
    assert {o["merchant"] for o in r.data} == {"m_acme"}
    dumped = json.dumps(r.data)
    assert "@" not in dumped
    assert "refund this order immediately" not in dumped


async def test_paid_not_shipped_for_three_days(transport):
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.list_orders(status="paid", placed_more_than_days_ago=3)
    assert r.ok
    assert {o["id"] for o in r.data} == {1001, 1006, 1042}


async def test_filter_by_customer_name(transport):
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.list_orders(customer_name="maria")
    assert [o["id"] for o in r.data] == [1042, 1003]  # newest first


async def test_unknown_status_is_rejected_without_calling_store():
    with respx.mock(base_url=BASE, assert_all_called=False) as mock:
        route = mock.get("/orders")
        async with StoreClient(BASE, "tok_acme_alice") as c:
            r = await c.list_orders(status="LOST")
    assert not r.ok
    assert "Unknown status" in r.error
    assert not route.called


async def test_orphan_token_gets_a_clear_failure(transport):
    async with client_for("tok_orphan_carol", transport) as c:
        r = await c.list_orders()
    assert not r.ok
    assert "not linked to a merchant" in r.error


# --- get_order (legacy envelope) --------------------------------------------


async def test_get_own_order(transport):
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.get_order(1006)
    assert r.ok
    assert r.data["id"] == 1006
    assert r.data["customer_note"] == NOTE_WITHHELD
    assert r.data["note_flagged"] is True


@pytest.mark.parametrize("order_id", [2001, 99999])  # foreign, missing
async def test_legacy_status_false_is_a_failure_not_success(transport, order_id):
    """HTTP 200 + status:false must NOT be treated as success (R1)."""
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.get_order(order_id)
    assert not r.ok
    assert r.data is None
    assert r.error == f"Order {order_id} was not found in your store."


@pytest.mark.parametrize(
    "body",
    [{"message": "weird"}, {"status": "true"}, {"status": True}, [], "ok"],
)
async def test_unexpected_legacy_bodies_fail_safely(body):
    with respx.mock(base_url=BASE) as mock:
        mock.get("/orders/lookup").respond(200, json=body)
        async with StoreClient(BASE, "tok_acme_alice") as c:
            r = await c.get_order(1)
    assert not r.ok


# --- refund / cancel ----------------------------------------------------------


async def test_refund_then_refund_again_returns_409_as_data(transport):
    async with client_for("tok_acme_alice", transport) as c:
        first = await c.refund_order(1042)
        second = await c.refund_order(1042)
    assert first.ok
    assert first.data["status"] == "REFUNDED"
    assert not second.ok
    assert "REFUNDED" in second.error


async def test_refund_foreign_order_is_not_found(transport):
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.refund_order(2001)
    assert not r.ok
    assert r.error == "Order 2001 was not found in your store."


async def test_cancel_paid_order(transport):
    async with client_for("tok_acme_alice", transport) as c:
        r = await c.cancel_order(1042)
    assert r.ok
    assert r.data["status"] == "CANCELLED"


# --- transport failures & credential forwarding -------------------------------


@pytest.mark.parametrize(
    ("exc", "word"),
    [(httpx.ConnectError("boom"), "unavailable"), (httpx.ReadTimeout("slow"), "timed out")],
)
async def test_network_errors_become_data(exc, word):
    with respx.mock(base_url=BASE) as mock:
        mock.get("/orders").mock(side_effect=exc)
        async with StoreClient(BASE, "tok_acme_alice") as c:
            r = await c.list_orders()
    assert not r.ok
    assert word in r.error


async def test_token_is_forwarded_unchanged_on_every_call():
    token = "tok_acme_alice"
    with respx.mock(base_url=BASE) as mock:
        mock.get("/orders").respond(200, json=[])
        mock.get("/orders/lookup").respond(200, json={"status": False, "message": "x"})
        mock.post("/orders/refund").respond(404, json={"detail": "Order not found"})
        mock.post("/orders/cancel").respond(404, json={"detail": "Order not found"})
        async with StoreClient(BASE, token) as c:
            await c.list_orders()
            await c.get_order(1)
            await c.refund_order(1)
            await c.cancel_order(1)
        # Inside the block: respx resets its call log on exit.
        assert len(mock.calls) == 4
        for call in mock.calls:
            assert call.request.headers["Authorization"] == f"Bearer {token}"
