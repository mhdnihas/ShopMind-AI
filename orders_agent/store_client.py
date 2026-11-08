"""Order tools: an async client for the store API that returns results as data.

Design rules:
* The caller's bearer token is forwarded unchanged on every request (R3). It is
  passed in per request and never stored beyond the client's lifetime, never
  shared between users, never swapped for a different credential.
* Nothing raises into the agent. Every outcome — success, legacy
  `200 + {"status": false}`, 4xx, timeout — becomes a `ToolResult`, so the model
  can tell the user what went wrong instead of the graph crashing (R1).
* Every order is passed through `sanitize_order` before it is returned (R4).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel

from orders_agent.guardrails import sanitize_order

VALID_STATUSES = {"PLACED", "PAID", "SHIPPED", "DELIVERED", "CANCELLED", "REFUNDED"}


class ToolResult(BaseModel):
    """Uniform tool output. `ok=False` is a normal result the model must explain."""

    ok: bool
    data: Any = None
    error: str | None = None

    @classmethod
    def success(cls, data: Any) -> ToolResult:
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, error=error)


# Messages are written for the end user (via the model) and never reveal whether
# an order exists in another store.
_HTTP_ERRORS = {
    401: "Your session is not valid. Please sign in again.",
    403: "Your account is not linked to a merchant store, so no order data is available.",
    404: "Order {order_id} was not found in your store.",
}


class StoreClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # `transport` lets tests run the real store app in-process (no network).
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> StoreClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    # --- low-level -------------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response | ToolResult:
        try:
            return await self._http.request(method, path, **kwargs)
        except httpx.TimeoutException:
            return ToolResult.failure("The store system timed out. Please try again shortly.")
        except httpx.HTTPError:
            return ToolResult.failure("The store system is unavailable. Please try again shortly.")

    @staticmethod
    def _http_failure(resp: httpx.Response, order_id: int | None = None) -> ToolResult:
        if resp.status_code in _HTTP_ERRORS:
            return ToolResult.failure(_HTTP_ERRORS[resp.status_code].format(order_id=order_id))
        if resp.status_code == 409:
            # The store's 409 message only describes the caller's own order.
            detail = _detail(resp) or "The order is not in a state that allows this action."
            return ToolResult.failure(detail)
        if resp.status_code == 422:
            return ToolResult.failure("The request was invalid (check the order id or status).")
        return ToolResult.failure(f"The store system returned an error (HTTP {resp.status_code}).")

    # --- tools -----------------------------------------------------------------

    async def list_orders(
        self,
        status: str | None = None,
        customer_name: str | None = None,
        placed_more_than_days_ago: int | None = None,
    ) -> ToolResult:
        """List the caller's orders, optionally filtered.

        The date and name filters run here, not in the model: language models are
        unreliable at date arithmetic, so "not shipped in 3 days" is computed in code.
        """
        if status is not None:
            status = status.upper()
            if status not in VALID_STATUSES:
                valid = ", ".join(sorted(VALID_STATUSES))
                return ToolResult.failure(f"Unknown status '{status}'. Use one of: {valid}.")

        resp = await self._request("GET", "/orders", params={"status": status} if status else None)
        if isinstance(resp, ToolResult):
            return resp
        if resp.status_code != 200:
            return self._http_failure(resp)

        orders = resp.json()
        if customer_name:
            needle = customer_name.casefold()
            orders = [o for o in orders if needle in o.get("customer_name", "").casefold()]
        if placed_more_than_days_ago is not None:
            cutoff = datetime.now(UTC) - timedelta(days=placed_more_than_days_ago)
            orders = [o for o in orders if datetime.fromisoformat(o["placed_at"]) < cutoff]

        return ToolResult.success([sanitize_order(o) for o in orders])

    async def get_order(self, order_id: int) -> ToolResult:
        """Look up one order via the legacy endpoint.

        The legacy endpoint answers HTTP 200 even on failure, so success is decided
        by the body's `status` field, never by the HTTP status alone.
        """
        resp = await self._request("GET", "/orders/lookup", params={"id": order_id})
        if isinstance(resp, ToolResult):
            return resp
        if resp.status_code != 200:
            return self._http_failure(resp, order_id)

        try:
            body = resp.json()
        except ValueError:
            return ToolResult.failure("The store system returned an unreadable response.")

        if not isinstance(body, dict) or body.get("status") is not True:
            # Covers {"status": false, ...}, a missing status key and any other shape.
            message = body.get("message") if isinstance(body, dict) else None
            if message == "Order not found":
                return ToolResult.failure(f"Order {order_id} was not found in your store.")
            return ToolResult.failure(message or "The order could not be retrieved.")

        order = body.get("order")
        if not isinstance(order, dict):
            return ToolResult.failure("The store system returned an incomplete order.")
        return ToolResult.success(sanitize_order(order))

    async def refund_order(self, order_id: int) -> ToolResult:
        """Execute a refund. Only call after a human approved the proposal (R5)."""
        return await self._action("/orders/refund", order_id)

    async def cancel_order(self, order_id: int) -> ToolResult:
        """Execute a cancel. Only call after a human approved the proposal (R5)."""
        return await self._action("/orders/cancel", order_id)

    async def _action(self, path: str, order_id: int) -> ToolResult:
        resp = await self._request("POST", path, json={"id": order_id})
        if isinstance(resp, ToolResult):
            return resp
        if resp.status_code != 200:
            return self._http_failure(resp, order_id)
        return ToolResult.success(sanitize_order(resp.json()))


def _detail(resp: httpx.Response) -> str | None:
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail if isinstance(detail, str) else None
