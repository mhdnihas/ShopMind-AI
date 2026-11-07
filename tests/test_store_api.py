"""R0 — mock store API: auth, tenant scoping, legacy envelope, state transitions."""

import pytest
from fastapi.testclient import TestClient

from store_api.data import INJECTION_NOTES, MERCHANTS, seed_orders
from store_api.main import create_app

ACME = {"Authorization": "Bearer tok_acme_alice"}
GLOBEX = {"Authorization": "Bearer tok_globex_bob"}
ORPHAN = {"Authorization": "Bearer tok_orphan_carol"}

ACME_ORDER = 1042  # PAID, belongs to m_acme
GLOBEX_ORDER = 2001  # PAID, belongs to m_globex


@pytest.fixture
def client():
    # Fresh app + store per test, so refunds in one test can't affect another.
    return TestClient(create_app())


# --- seed data meets the R0 minimums ----------------------------------------

def test_seed_meets_spec():
    orders = seed_orders()
    assert len(orders) >= 12
    assert len(MERCHANTS) >= 2
    assert {o.merchant for o in orders.values()} == set(MERCHANTS)
    injected = [o for o in orders.values() if o.customer_note in INJECTION_NOTES.values()]
    assert len(injected) >= 2


# --- authentication -----------------------------------------------------------

@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic x"}])
def test_bad_or_missing_token_is_401(client, headers):
    assert client.get("/orders", headers=headers).status_code == 401
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_auth_me(client):
    assert client.get("/auth/me", headers=ACME).json() == {
        "user": "alice@acme-outdoor", "merchant": "m_acme"}
    # Orphan token is valid but has no merchant: /auth/me reports merchant=null.
    assert client.get("/auth/me", headers=ORPHAN).json()["merchant"] is None


def test_orphan_token_fails_closed(client):
    assert client.get("/orders", headers=ORPHAN).status_code == 403
    assert client.post("/orders/refund", json={"id": ACME_ORDER}, headers=ORPHAN).status_code == 403
    assert client.post("/orders/cancel", json={"id": ACME_ORDER}, headers=ORPHAN).status_code == 403
    r = client.get("/orders/lookup", params={"id": ACME_ORDER}, headers=ORPHAN)
    assert r.status_code == 200 and r.json()["status"] is False


# --- tenant scoping -----------------------------------------------------------

def test_list_is_scoped_to_merchant(client):
    acme = client.get("/orders", headers=ACME).json()
    globex = client.get("/orders", headers=GLOBEX).json()
    assert acme and all(o["merchant"] == "m_acme" for o in acme)
    assert globex and all(o["merchant"] == "m_globex" for o in globex)
    assert not {o["id"] for o in acme} & {o["id"] for o in globex}


def test_list_status_filter(client):
    paid = client.get("/orders", params={"status": "PAID"}, headers=ACME).json()
    assert {o["id"] for o in paid} == {1001, 1004, 1006, 1042}
    assert client.get("/orders", params={"status": "BOGUS"}, headers=ACME).status_code == 422


def test_list_newest_first(client):
    dates = [o["placed_at"] for o in client.get("/orders", headers=ACME).json()]
    assert dates == sorted(dates, reverse=True)


# --- legacy lookup envelope ---------------------------------------------------

def test_lookup_own_order(client):
    r = client.get("/orders/lookup", params={"id": ACME_ORDER}, headers=ACME)
    assert r.status_code == 200
    assert r.json()["status"] is True
    assert r.json()["order"]["customer_name"] == "Maria Lopez"


@pytest.mark.parametrize("order_id", [GLOBEX_ORDER, 99999, "abc"])
def test_lookup_failure_is_http_200_with_status_false(client, order_id):
    r = client.get("/orders/lookup", params={"id": order_id}, headers=ACME)
    assert r.status_code == 200
    assert r.json() == {"status": False, "message": "Order not found"}


def test_foreign_and_missing_orders_look_identical(client):
    """No oracle: a foreign id must be indistinguishable from a non-existent one."""
    foreign = client.get("/orders/lookup", params={"id": GLOBEX_ORDER}, headers=ACME)
    missing = client.get("/orders/lookup", params={"id": 99999}, headers=ACME)
    assert foreign.json() == missing.json()

    foreign = client.post("/orders/refund", json={"id": GLOBEX_ORDER}, headers=ACME)
    missing = client.post("/orders/refund", json={"id": 99999}, headers=ACME)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


# --- refund / cancel ----------------------------------------------------------

def test_refund_paid_order(client):
    r = client.post("/orders/refund", json={"id": ACME_ORDER}, headers=ACME)
    assert r.status_code == 200 and r.json()["status"] == "REFUNDED"
    # Second refund is an invalid transition.
    r = client.post("/orders/refund", json={"id": ACME_ORDER}, headers=ACME)
    assert r.status_code == 409


def test_cancel_rules(client):
    assert client.post("/orders/cancel", json={"id": 1005}, headers=ACME).json()["status"] == "CANCELLED"
    # DELIVERED cannot be cancelled; already-CANCELLED cannot be refunded.
    assert client.post("/orders/cancel", json={"id": 1003}, headers=ACME).status_code == 409
    assert client.post("/orders/refund", json={"id": 1007}, headers=ACME).status_code == 409


# --- money tracking (paid_amount / refunded_amount) ---------------------------

def _order(client, order_id, headers=ACME):
    return client.get("/orders/lookup", params={"id": order_id}, headers=headers).json()["order"]


def test_seed_money_fields(client):
    assert _order(client, 1042)["paid_amount"] == 59.90  # PAID: charged
    assert _order(client, 1005)["paid_amount"] == 0.0  # PLACED: not charged yet
    assert _order(client, 1007)["paid_amount"] == 0.0  # cancelled before payment
    refunded = _order(client, 2005, GLOBEX)  # REFUNDED: charged and returned
    assert refunded["paid_amount"] == refunded["refunded_amount"] == 33.00


def test_refund_returns_paid_amount(client):
    r = client.post("/orders/refund", json={"id": 1042}, headers=ACME).json()
    assert r["status"] == "REFUNDED"
    assert r["refunded_amount"] == r["paid_amount"] == 59.90


def test_cancel_paid_order_also_refunds(client):
    """Paid then cancelled must never leave the customer out of pocket."""
    r = client.post("/orders/cancel", json={"id": 1042}, headers=ACME).json()
    assert r["status"] == "CANCELLED"
    assert r["refunded_amount"] == r["paid_amount"] == 59.90
    # Money already returned, so a later refund is correctly refused.
    assert client.post("/orders/refund", json={"id": 1042}, headers=ACME).status_code == 409


def test_cancel_unpaid_order_refunds_nothing(client):
    r = client.post("/orders/cancel", json={"id": 1005}, headers=ACME).json()
    assert r["status"] == "CANCELLED"
    assert r["paid_amount"] == r["refunded_amount"] == 0.0


def test_cross_merchant_refund_does_not_change_order(client):
    assert client.post("/orders/refund", json={"id": ACME_ORDER}, headers=GLOBEX).status_code == 404
    assert client.post("/orders/cancel", json={"id": ACME_ORDER}, headers=GLOBEX).status_code == 404
    r = client.get("/orders/lookup", params={"id": ACME_ORDER}, headers=ACME)
    assert r.json()["order"]["status"] == "PAID"


def test_action_requires_integer_id(client):
    assert client.post("/orders/refund", json={}, headers=ACME).status_code == 422
    assert client.post("/orders/refund", json={"id": "x"}, headers=ACME).status_code == 422
