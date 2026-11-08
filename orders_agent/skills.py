"""The orders sub-agent's skills: what it can do, with typed parameters.

Each skill has a Pydantic parameter model. The same model is used twice:
* to validate incoming JSON-RPC params (bad input -> JSON-RPC "invalid params"), and
* to publish a JSON Schema in the agent card, so the orchestrator can build its
  LLM tools from the card instead of importing this code (R2).

`preview_action` is read-only: it builds the proposal a human approves (R5).
`refund_order` / `cancel_order` change state and are marked `mutating` in the card;
the orchestrator only calls them after an explicit approval.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orders_agent.store_client import StoreClient, ToolResult

StatusName = Literal["PLACED", "PAID", "SHIPPED", "DELIVERED", "CANCELLED", "REFUNDED"]
ActionName = Literal["refund", "cancel"]


class _Params(BaseModel):
    # Unknown fields are rejected, so a typo or an injected extra argument fails loudly.
    model_config = ConfigDict(extra="forbid")


class ListOrdersParams(_Params):
    status: StatusName | None = Field(None, description="Only orders in this status.")
    customer_name: str | None = Field(
        None, max_length=100, description="Case-insensitive part of the customer's name."
    )
    placed_more_than_days_ago: int | None = Field(
        None, ge=0, le=3650, description="Only orders placed more than this many days ago."
    )


class OrderIdParams(_Params):
    order_id: int = Field(..., gt=0, description="Numeric order id, e.g. 1042.")


class PreviewActionParams(_Params):
    action: ActionName = Field(..., description="'refund' or 'cancel'.")
    order_id: int = Field(..., gt=0, description="Numeric order id, e.g. 1042.")


# --- Proposal rules -----------------------------------------------------------
# These mirror the store's transition rules so the proposal can say up front
# whether an action is possible. The store's 409 remains the final authority.

_ALLOWED_FROM = {
    "refund": {"PAID", "SHIPPED", "DELIVERED"},
    "cancel": {"PLACED", "PAID"},
}
_NEW_STATUS = {"refund": "REFUNDED", "cancel": "CANCELLED"}
_PAST_TENSE = {"refund": "refunded", "cancel": "cancelled"}


def build_proposal(action: ActionName, order: dict[str, Any]) -> ToolResult:
    """Describe exactly what an action would change, without changing anything."""
    status = order["status"]
    if status not in _ALLOWED_FROM[action]:
        allowed = ", ".join(sorted(_ALLOWED_FROM[action]))
        return ToolResult.failure(
            f"Order {order['id']} is {status}, so it cannot be {_PAST_TENSE[action]} "
            f"(only orders that are {allowed} can)."
        )

    # Both actions return whatever the customer has paid and not yet got back.
    amount = round(order.get("paid_amount", 0.0) - order.get("refunded_amount", 0.0), 2)
    new_status = _NEW_STATUS[action]
    summary = (
        f"{action.capitalize()} order {order['id']} for {order['customer_name']}: "
        f"{status} -> {new_status}; {amount:.2f} will be returned to the customer."
    )
    return ToolResult.success(
        {
            "action": action,
            "order_id": order["id"],
            "merchant": order["merchant"],
            "customer_name": order["customer_name"],
            "current_status": status,
            "new_status": new_status,
            "order_total": order["total"],
            "amount_to_customer": amount,
            "summary": summary,
        }
    )


# --- Skill handlers -------------------------------------------------------------


async def _list_orders(client: StoreClient, p: ListOrdersParams) -> ToolResult:
    return await client.list_orders(p.status, p.customer_name, p.placed_more_than_days_ago)


async def _get_order(client: StoreClient, p: OrderIdParams) -> ToolResult:
    return await client.get_order(p.order_id)


async def _preview_action(client: StoreClient, p: PreviewActionParams) -> ToolResult:
    found = await client.get_order(p.order_id)
    if not found.ok:
        return found
    return build_proposal(p.action, found.data)


async def _refund_order(client: StoreClient, p: OrderIdParams) -> ToolResult:
    return await client.refund_order(p.order_id)


async def _cancel_order(client: StoreClient, p: OrderIdParams) -> ToolResult:
    return await client.cancel_order(p.order_id)


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    params: type[_Params]
    handler: Callable[[StoreClient, Any], Awaitable[ToolResult]]
    mutating: bool = False

    def card_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.params.model_json_schema(),
            "mutating": self.mutating,
        }


SKILLS: dict[str, Skill] = {
    s.id: s
    for s in [
        Skill(
            id="list_orders",
            name="List orders",
            description=(
                "List the merchant's orders, newest first. Optional filters: status, part of "
                "the customer's name, and minimum age in days (e.g. PAID orders placed more "
                "than 3 days ago = paid but not shipped for 3 days)."
            ),
            params=ListOrdersParams,
            handler=_list_orders,
        ),
        Skill(
            id="get_order",
            name="Get order",
            description="Get one order by its numeric id.",
            params=OrderIdParams,
            handler=_get_order,
        ),
        Skill(
            id="preview_action",
            name="Preview refund or cancel",
            description=(
                "Read-only. Check whether an order can be refunded or cancelled and describe "
                "exactly what would change and how much the customer gets back. Changes nothing."
            ),
            params=PreviewActionParams,
            handler=_preview_action,
        ),
        Skill(
            id="refund_order",
            name="Refund order",
            description="Refund an order. Requires prior human approval of a preview.",
            params=OrderIdParams,
            handler=_refund_order,
            mutating=True,
        ),
        Skill(
            id="cancel_order",
            name="Cancel order",
            description=(
                "Cancel an order; a paid order is refunded as part of the cancel. "
                "Requires prior human approval of a preview."
            ),
            params=OrderIdParams,
            handler=_cancel_order,
            mutating=True,
        ),
    ]
}
