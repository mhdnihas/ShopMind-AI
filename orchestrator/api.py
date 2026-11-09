"""R6, R3, R4c — The chat API: the trust boundary of the system.

    POST /chat                       -> send a message, reply streams over SSE
    POST /chat/{thread_id}/approve   -> approve the waiting proposal, streams the result
    POST /chat/{thread_id}/reject    -> reject it, streams the result

Stream events: token, tool_started, tool_finished, proposal, done, error.
A turn ends with `proposal` (a decision is needed), `done` or `error`.

Every request is checked against the store's /auth/me first. A valid token with no
merchant is refused before the model runs (R4c). The merchant also scopes the thread:
the real key is "merchant:thread_id", so one merchant can never read or resume
another's conversation, even with the same thread_id.

The caller's token is only ever passed on, unchanged, through the run context (R3).
"""

# No `from __future__ import annotations` here: FastAPI must read the real types of
# the routes, and CallerDep is defined inside create_app.
import asyncio
import json
import logging
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any
from weakref import WeakValueDictionary

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from common.settings import get_settings
from orchestrator.graph import APPROVE, REJECT, build_agents
from orchestrator.tools import RunContext

log = logging.getLogger("orchestrator")

STATIC_DIR = pathlib.Path(__file__).parent / "static"
# The page shows untrusted text, so even if some got rendered as HTML no script could run.
CSP = "default-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]

UNAVAILABLE = "No reply came from this model. Switch to another model or try again shortly."
RATE_LIMITED = "This model has reached its usage limit. Switch to another model and send it again."
TOO_MANY_STEPS = "That request took too many steps, so it was stopped."
BUSY = "This conversation is already being processed."
OUT_OF_DATE = "This conversation changed. Please try again."


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    model: str | None = None  # one of GET /models; None means the default


@dataclass(frozen=True)
class Caller:
    token: str = field(repr=False)
    merchant: str


def _sse(event: str, **data: Any) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data)}


def _model_label(name: str) -> str:
    return name.replace("-", " ").title().replace("Flash Lite", "Flash-Lite")


def _thread_config(caller: Caller, thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{caller.merchant}:{thread_id}"}}


def create_app(
    *,
    graph: CompiledStateGraph | None = None,
    graphs: dict[str, CompiledStateGraph] | None = None,
    store_url: str | None = None,
    store_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """App factory. Tests inject the graph and run the real store app in-process."""
    settings = get_settings()
    store_url = store_url or settings.store_api_url
    # Entries disappear once no request is using them, so this cannot grow forever.
    locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.graph is None:
            app.state.graphs = await build_agents(settings=settings)
            app.state.graph = next(iter(app.state.graphs.values()))
        yield

    app = FastAPI(title="ShopHub chat API", lifespan=lifespan)
    app.state.graph = graph  # the default model's graph
    app.state.graphs = graphs or {}  # every selectable model, default first
    bearer = HTTPBearer(auto_error=False)

    async def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> Caller:
        if credentials is None:
            raise HTTPException(401, "Missing bearer token")
        token = credentials.credentials
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout_seconds, transport=store_transport
            ) as http:
                resp = await http.get(
                    f"{store_url}/auth/me", headers={"Authorization": f"Bearer {token}"}
                )
            if resp.status_code == 401:
                raise HTTPException(401, "Invalid token")
            resp.raise_for_status()
            merchant = resp.json().get("merchant")
        except (httpx.HTTPError, ValueError, AttributeError) as e:
            # If identity cannot be checked, the request does not go ahead.
            raise HTTPException(503, "Could not verify your account right now") from e
        if not merchant:
            raise HTTPException(403, "Your account is not linked to a merchant store")
        return Caller(token=token, merchant=merchant)

    CallerDep = Annotated[Caller, Depends(authenticate)]

    async def pending_proposals(config: dict[str, Any]) -> list[dict[str, Any]]:
        state = await app.state.graph.aget_state(config)
        return [item.value for task in state.tasks for item in task.interrupts]

    async def run_turn(
        agent: CompiledStateGraph, graph_input: Any, config: dict[str, Any], caller: Caller
    ) -> AsyncIterator[dict[str, str]]:
        streamed = False
        async for event in agent.astream_events(
            graph_input, config, context=RunContext(auth_token=caller.token), version="v2"
        ):
            kind = event["event"]
            if kind == "on_tool_start":
                yield _sse("tool_started", tool=event["name"])
            elif kind == "on_tool_end":
                yield _sse("tool_finished", tool=event["name"])
            elif kind == "on_chat_model_stream" and (text := event["data"]["chunk"].text):
                streamed = True
                yield _sse("token", text=text)

        proposals = await pending_proposals(config)
        if proposals:
            for proposal in proposals:
                yield _sse("proposal", **proposal)
            return
        if not streamed:
            # A reply that did not stream: the outcome of a decision, or a non-streaming model.
            state = await agent.aget_state(config)
            yield _sse("token", text=state.values["messages"][-1].text)
        yield _sse("done")

    async def events(
        agent: CompiledStateGraph,
        graph_input: Any,
        config: dict[str, Any],
        caller: Caller,
        *,
        expect_pending: bool,
    ) -> AsyncIterator[dict[str, str]]:
        key = config["configurable"]["thread_id"]
        lock = locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            # For example a double click on approve: only the first request runs.
            yield _sse("error", message=BUSY)
            return
        async with lock:
            try:
                # Checked again now that this request holds the conversation.
                if bool(await pending_proposals(config)) != expect_pending:
                    yield _sse("error", message=OUT_OF_DATE)
                    return
                async for item in run_turn(agent, graph_input, config, caller):
                    yield item
            except GraphRecursionError:
                yield _sse("error", message=TOO_MANY_STEPS)
            except Exception as e:  # noqa: BLE001
                # Deliberately broad: whatever failed, the stream must end with an event.
                # Only the type is logged: provider messages can hold request details.
                log.error("turn failed: %s", type(e).__name__)
                limited = "RateLimit" in type(e).__name__
                yield _sse("error", message=RATE_LIMITED if limited else UNAVAILABLE)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/models")
    async def models() -> dict:
        return {"models": [{"id": n, "label": _model_label(n)} for n in app.state.graphs]}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def web_chat() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Content-Security-Policy": CSP})

    @app.post("/chat")
    async def chat(body: ChatRequest, caller: CallerDep) -> EventSourceResponse:
        # Only models from the server's own list can be used.
        if body.model is not None and body.model not in app.state.graphs:
            raise HTTPException(422, "Unknown model")
        agent = app.state.graphs[body.model] if body.model else app.state.graph
        config = _thread_config(caller, body.thread_id)
        if await pending_proposals(config):
            raise HTTPException(409, "A proposal is waiting. Approve or reject it first.")
        message = {"messages": [HumanMessage(content=body.message)]}
        return EventSourceResponse(events(agent, message, config, caller, expect_pending=False))

    async def decide(thread_id: str, caller: Caller, decision: str) -> EventSourceResponse:
        config = _thread_config(caller, thread_id)
        if not await pending_proposals(config):
            raise HTTPException(409, "Nothing is waiting for a decision in this conversation.")
        return EventSourceResponse(
            events(app.state.graph, Command(resume=decision), config, caller, expect_pending=True)
        )

    @app.post("/chat/{thread_id}/approve")
    async def approve(thread_id: ThreadId, caller: CallerDep) -> EventSourceResponse:
        return await decide(thread_id, caller, APPROVE)

    @app.post("/chat/{thread_id}/reject")
    async def reject(thread_id: ThreadId, caller: CallerDep) -> EventSourceResponse:
        return await decide(thread_id, caller, REJECT)

    return app


app = create_app()
