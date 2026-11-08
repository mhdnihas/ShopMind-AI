"""R2 — Orders sub-agent served over JSON-RPC 2.0, discoverable through an agent card.

    GET  /.well-known/agent-card.json   -> agent card (name, skills, endpoint)
    POST /                              -> JSON-RPC 2.0; method = skill id

Two kinds of failure, kept apart on purpose:
* Protocol failures (bad JSON, unknown method, invalid params, no token) are
  JSON-RPC `error` objects — the caller sent something wrong.
* Business failures (order not found, 409, store down) are a normal `result`
  with ok=false — the call worked and the answer is "no". The orchestrator's
  model reads these and explains them to the user.

The caller's bearer token is read from the Authorization header and handed,
unchanged, to the StoreClient for this one request (R3). The sub-agent never
stores it and never uses a credential of its own.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from common.settings import get_settings
from orders_agent.skills import SKILLS
from orders_agent.store_client import StoreClient

log = logging.getLogger("orders_agent")

AGENT_NAME = "orders-agent"
AGENT_VERSION = "1.0.0"
CARD_PATHS = ("/.well-known/agent-card.json", "/.well-known/agent.json")

# JSON-RPC 2.0 error codes (spec) plus one server-defined code for auth.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED = -32001


def build_card(public_url: str) -> dict[str, Any]:
    return {
        "name": AGENT_NAME,
        "description": (
            "Answers questions about a merchant's orders and performs refunds/cancels. "
            "All data is scoped to the merchant of the caller's bearer token."
        ),
        "version": AGENT_VERSION,
        "url": public_url.rstrip("/") + "/",
        "protocol": "jsonrpc-2.0",
        "authentication": {
            "schemes": ["bearer"],
            "description": "Send the end user's own token; it is forwarded unchanged.",
        },
        "skills": [skill.card_entry() for skill in SKILLS.values()],
    }


def _error(req_id: Any, code: int, message: str, data: Any = None, http: int = 200) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": error}, status_code=http)


def _bearer_token(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


def create_app(
    *,
    store_url: str | None = None,
    public_url: str | None = None,
    store_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """App factory. `store_transport` lets tests run the real store app in-process."""
    settings = get_settings()
    store_url = store_url or settings.store_api_url
    card = build_card(public_url or settings.orders_agent_url)

    app = FastAPI(title="ShopHub orders sub-agent", version=AGENT_VERSION)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"ok": True}

    async def agent_card() -> dict:
        return card

    for path in CARD_PATHS:
        app.add_api_route(path, agent_card, methods=["GET"])

    @app.post("/")
    async def rpc(request: Request) -> Response:
        try:
            payload = await request.json()
        except ValueError:
            return _error(None, PARSE_ERROR, "Parse error")

        if isinstance(payload, list):
            return _error(None, INVALID_REQUEST, "Batch requests are not supported")
        if (
            not isinstance(payload, dict)
            or payload.get("jsonrpc") != "2.0"
            or not isinstance(payload.get("method"), str)
        ):
            return _error(None, INVALID_REQUEST, "Invalid Request")

        if "id" not in payload:
            # A notification expects no reply. We do not run skills nobody is
            # waiting on (that would include unobserved refunds), so it is ignored.
            return Response(status_code=204)
        req_id = payload["id"]
        if req_id is not None and not isinstance(req_id, str | int):
            return _error(None, INVALID_REQUEST, "Invalid Request: id must be a string or number")

        token = _bearer_token(request)
        if token is None:
            return _error(req_id, UNAUTHORIZED, "Missing bearer token", http=401)

        skill = SKILLS.get(payload["method"])
        if skill is None:
            return _error(req_id, METHOD_NOT_FOUND, f"Method not found: {payload['method']}")

        raw_params = payload.get("params", {})
        if not isinstance(raw_params, dict):
            return _error(req_id, INVALID_PARAMS, "params must be an object")
        try:
            params = skill.params.model_validate(raw_params)
        except ValidationError as e:
            # Field locations and messages only: never echo the input back.
            details = [
                {"field": ".".join(map(str, err["loc"])), "message": err["msg"]}
                for err in e.errors()
            ]
            return _error(req_id, INVALID_PARAMS, "Invalid params", details)

        try:
            async with StoreClient(
                store_url,
                token,
                timeout=settings.http_timeout_seconds,
                transport=store_transport,
            ) as client:
                result = await skill.handler(client, params)
        except Exception:
            # The client already turns expected failures into results; this is a bug.
            log.exception("skill %s failed", skill.id)
            return _error(req_id, INTERNAL_ERROR, "Internal error")

        log.info("skill=%s ok=%s", skill.id, result.ok)  # never log the token or data
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result.model_dump()})

    return app


app = create_app()
