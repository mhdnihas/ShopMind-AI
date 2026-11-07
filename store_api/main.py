"""R0 — Mock ShopHub store API (FastAPI, in-memory).

Every order endpoint is scoped to the merchant that owns the caller's bearer token.
`/orders/lookup` deliberately imitates a legacy API: it ALWAYS returns HTTP 200 and
signals failure in the body with {"status": false, "message": ...}.
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from store_api.data import (
    TOKENS,
    InvalidTransition,
    Order,
    OrderNotFound,
    OrderStatus,
    Principal,
    Store,
)

NOT_FOUND = "Order not found"


class OrderAction(BaseModel):
    id: int


def create_app(store: Store | None = None) -> FastAPI:
    """App factory: each call gets its own Store, so tests never share state."""
    app = FastAPI(title="ShopHub mock store API", version="1.0")
    app.state.store = store or Store()

    bearer = HTTPBearer(auto_error=False)

    def get_principal(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> Principal:
        principal = TOKENS.get(creds.credentials) if creds else None
        if principal is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return principal

    def get_merchant(principal: Annotated[Principal, Depends(get_principal)]) -> str:
        # Fail closed: a valid user without a store gets nothing (R4c).
        if principal.merchant is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Token is not linked to a merchant")
        return principal.merchant

    def get_store() -> Store:
        return app.state.store

    StoreDep = Annotated[Store, Depends(get_store)]
    MerchantDep = Annotated[str, Depends(get_merchant)]

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/auth/me")
    def auth_me(principal: Annotated[Principal, Depends(get_principal)]) -> dict:
        return {"user": principal.user, "merchant": principal.merchant}

    @app.get("/orders")
    def list_orders(
        store: StoreDep, merchant: MerchantDep, status: OrderStatus | None = None
    ) -> list[Order]:
        return store.list(merchant, status)

    @app.get("/orders/lookup")
    def lookup_order(
        store: StoreDep,
        principal: Annotated[Principal, Depends(get_principal)],
        id: Annotated[str, Query()],
    ) -> dict:
        # Legacy behaviour: business failures are HTTP 200 + status:false.
        # Only authentication (401) is a real HTTP error, because it is handled
        # by the platform layer before this legacy handler runs.
        if principal.merchant is None:
            return {"status": False, "message": "Not authorised for any store"}
        try:
            order = store.get(principal.merchant, int(id))
        except (ValueError, OrderNotFound):
            # Same message for "does not exist" and "belongs to another merchant",
            # so the endpoint can't be used to discover other stores' order ids.
            return {"status": False, "message": NOT_FOUND}
        return {"status": True, "order": order.model_dump(mode="json")}

    def _act(action, merchant: str, order_id: int) -> Order:
        try:
            return action(merchant, order_id)
        except OrderNotFound:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND) from None
        except InvalidTransition as e:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from None

    @app.post("/orders/refund")
    def refund_order(body: OrderAction, store: StoreDep, merchant: MerchantDep) -> Order:
        return _act(store.refund, merchant, body.id)

    @app.post("/orders/cancel")
    def cancel_order(body: OrderAction, store: StoreDep, merchant: MerchantDep) -> Order:
        return _act(store.cancel, merchant, body.id)

    return app


app = create_app()
