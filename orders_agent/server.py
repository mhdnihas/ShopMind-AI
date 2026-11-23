"""R2 — The orders sub-agent, served over Google's A2A protocol (a2a-sdk).

    GET  /.well-known/agent-card.json   -> A2A agent card
    POST /                              -> A2A JSON-RPC binding (method: SendMessage)

A skill call is an A2A message with one data part: {"skill": ..., "params": ...}.
The reply is an A2A agent message whose data part is a ToolResult.

Two kinds of failure, kept apart on purpose:
* Protocol failures (bad message, unknown skill, invalid params, no token) are A2A
  errors: the caller sent something wrong.
* Business failures (order not found, 409, store down) are a normal reply with
  ok=false: the call worked and the answer is "no". The orchestrator's model reads
  these and explains them to the user.

The caller's bearer token is read from the Authorization header and handed,
unchanged, to the StoreClient for this one request (R3). The sub-agent never
stores it and never uses a credential of its own.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.errors import InternalError, InvalidParamsError, UnsupportedOperationError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from pydantic import ValidationError

from common.a2a_contract import SKILL_EXTENSION_URI
from common.settings import get_settings
from orders_agent.skills import SKILLS, Skill
from orders_agent.store_client import StoreClient

log = logging.getLogger("orders_agent")

AGENT_NAME = "orders-agent"
AGENT_VERSION = "2.0.0"
A2A_PROTOCOL_VERSION = "1.0"
JSON_MODE = "application/json"
SECURITY_SCHEME = "bearer"
UNAUTHORIZED = -32001  # server-defined JSON-RPC code for a missing token


def build_card(public_url: str) -> AgentCard:
    entries = [skill.card_entry() for skill in SKILLS.values()]
    schemas = {
        entry["id"]: {"input_schema": entry["input_schema"], "mutating": entry["mutating"]}
        for entry in entries
    }
    return AgentCard(
        name=AGENT_NAME,
        description=(
            "Answers questions about a merchant's orders and performs refunds/cancels. "
            "All data is scoped to the merchant of the caller's bearer token."
        ),
        version=AGENT_VERSION,
        supported_interfaces=[
            AgentInterface(
                url=public_url.rstrip("/") + "/",
                protocol_binding="JSONRPC",
                protocol_version=A2A_PROTOCOL_VERSION,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=False,
            extensions=[
                AgentExtension(
                    uri=SKILL_EXTENSION_URI,
                    description="Argument schema of each skill, and whether it changes data.",
                    params=ParseDict({"skills": schemas}, Struct()),
                )
            ],
        ),
        security_schemes={
            SECURITY_SCHEME: SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    description="Send the end user's own token; it is forwarded unchanged.",
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={SECURITY_SCHEME: StringList()})],
        default_input_modes=[JSON_MODE],
        default_output_modes=[JSON_MODE],
        skills=[
            AgentSkill(
                id=entry["id"],
                name=entry["name"],
                description=entry["description"],
                tags=["orders", "mutating" if entry["mutating"] else "read-only"],
            )
            for entry in entries
        ],
    )


def _bearer_token(header: str | None) -> str | None:
    scheme, _, token = (header or "").partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


def _is_notification(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return isinstance(payload, dict) and "id" not in payload


def _read_call(context: RequestContext) -> tuple[Skill, Any]:
    """Turn an incoming A2A message into a skill and validated parameters."""
    parts = get_data_parts(context.message.parts) if context.message else []
    call = parts[0] if len(parts) == 1 else None
    if not isinstance(call, dict) or not isinstance(call.get("skill"), str):
        raise InvalidParamsError(message='Expected one data part: {"skill": ..., "params": ...}')

    skill = SKILLS.get(call["skill"])
    if skill is None:
        raise InvalidParamsError(message=f"Unknown skill: {call['skill']}")
    params = call.get("params", {})
    if not isinstance(params, dict):
        raise InvalidParamsError(message="params must be an object")
    try:
        return skill, skill.params.model_validate(params)
    except ValidationError as e:
        # Field locations and messages only: never echo the input back.
        details = [
            {"field": ".".join(map(str, err["loc"])), "message": err["msg"]} for err in e.errors()
        ]
        raise InvalidParamsError(message="Invalid params", data={"errors": details}) from None


class OrdersExecutor(AgentExecutor):
    """Runs one skill per message. Deterministic: this agent has no model of its own."""

    def __init__(
        self, store_url: str, timeout: float, transport: httpx.AsyncBaseTransport | None
    ) -> None:
        self._store_url = store_url
        self._timeout = timeout
        self._transport = transport

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        skill, params = _read_call(context)
        headers = context.call_context.state.get("headers", {})
        token = _bearer_token(headers.get("authorization"))
        try:
            async with StoreClient(
                self._store_url, token, timeout=self._timeout, transport=self._transport
            ) as client:
                result = await skill.handler(client, params)
        except Exception:
            # The client already turns expected failures into results; this is a bug.
            log.exception("skill %s failed", skill.id)
            raise InternalError(message="Internal error") from None
        log.info("skill=%s ok=%s", skill.id, result.ok)  # never log the token or data
        await event_queue.enqueue_event(new_data_message(result.model_dump()))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Skills finish immediately and cannot be cancelled")


def create_app(
    *,
    store_url: str | None = None,
    public_url: str | None = None,
    store_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """App factory. `store_transport` lets tests run the real store app in-process."""
    settings = get_settings()
    card = build_card(public_url or settings.orders_agent_url)
    handler = DefaultRequestHandler(
        agent_executor=OrdersExecutor(
            store_url or settings.store_api_url, settings.http_timeout_seconds, store_transport
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = FastAPI(title="ShopHub orders sub-agent", version=AGENT_VERSION)

    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next: Any) -> Response:
        if request.method == "POST" and request.url.path == "/":
            if _bearer_token(request.headers.get("authorization")) is None:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": UNAUTHORIZED, "message": "Missing bearer token"},
                    },
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # A JSON-RPC notification expects no reply, but the SDK would still run it.
            # Skills nobody is waiting on must not run: that would include refunds.
            if _is_notification(await request.body()):
                return Response(status_code=204)
        return await call_next(request)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"ok": True}

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
    )
    return app


app = create_app()
