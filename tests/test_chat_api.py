"""R6, R3, R4c — the chat API over HTTP: SSE events, auth and thread ownership.

The graph, sub-agent and store are real and run in-process; only the model is scripted.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from orchestrator.a2a_client import A2AClient
from orchestrator.api import BUSY, OUT_OF_DATE, TOO_MANY_STEPS, UNAVAILABLE, create_app
from orchestrator.client import parse_events
from orchestrator.fake_llm import ScriptedChatModel
from orchestrator.graph import build_graph
from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
STORE = "http://store.test"
ACME = "tok_acme_alice"
GLOBEX = "tok_globex_bob"
ORPHAN = "tok_orphan_carol"


class NeverCalledModel(ScriptedChatModel):
    def _generate(self, messages, *args, **kwargs):
        raise AssertionError("the model must not run for this request")


class BrokenModel(ScriptedChatModel):
    def _generate(self, messages, *args, **kwargs):
        raise RuntimeError("private-provider-detail")


class LoopingModel(ScriptedChatModel):
    def _generate(self, messages, *args, **kwargs):
        call = {"name": "get_order", "args": {"order_id": 1042}, "id": "c", "type": "tool_call"}
        reply = AIMessage(content="", tool_calls=[call])
        return ChatResult(generations=[ChatGeneration(message=reply)])


class SlowModel(ScriptedChatModel):
    """Stays busy long enough for a second request to arrive."""

    async def _agenerate(self, messages, stop=None, run_manager=None, tools=None, **kwargs):
        await asyncio.sleep(0.05)
        return self._generate(messages, stop, None, tools=tools, **kwargs)


class StreamingModel(ScriptedChatModel):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for word in ("Hello ", "there"):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))


@pytest.fixture
def store():
    return httpx.ASGITransport(app=create_store())


@pytest.fixture
async def a2a(store):
    app = create_agent(store_url=STORE, public_url=AGENT, store_transport=store)
    return await A2AClient.discover(AGENT, transport=httpx.ASGITransport(app=app))


def make_http(model, a2a, store_transport) -> httpx.AsyncClient:
    api = create_app(
        graph=build_graph(model, a2a), store_url=STORE, store_transport=store_transport
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://chat.test")


@pytest.fixture
def http(a2a, store):
    return make_http(ScriptedChatModel(), a2a, store)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


async def chat(http, text, token=ACME, thread="t1"):
    return await http.post(
        "/chat", json={"message": text, "thread_id": thread}, headers=auth(token)
    )


async def decide(http, decision, token=ACME, thread="t1"):
    return await http.post(f"/chat/{thread}/{decision}", headers=auth(token))


def events(response):
    assert response.status_code == 200, response.text
    return list(parse_events(iter(response.text.splitlines())))


def names(response):
    return [name for name, _ in events(response)]


def text_of(response):
    return "".join(data["text"] for name, data in events(response) if name == "token")


async def status_of(a2a, order_id, token=ACME):
    return (await a2a.call("get_order", {"order_id": order_id}, token))["data"]["status"]


# --- streaming events (R6) -----------------------------------------------------------


async def test_turn_streams_tool_events_then_the_answer_then_done(http):
    response = await chat(http, "Show me order 1042")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert names(response) == ["tool_started", "tool_finished", "token", "done"]
    assert events(response)[0][1] == {"tool": "get_order"}
    assert "Maria Lopez" in text_of(response)


async def test_reply_without_tools_has_no_tool_events(http):
    assert names(await chat(http, "Hello there")) == ["token", "done"]


async def test_model_tokens_stream_one_by_one_without_repeating(a2a, store):
    http = make_http(StreamingModel(), a2a, store)
    response = await chat(http, "Hello there")
    assert events(response) == [
        ("token", {"text": "Hello "}),
        ("token", {"text": "there"}),
        ("done", {}),
    ]


async def test_turns_in_one_thread_share_history(http):
    await chat(http, "Hello there")
    assert names(await chat(http, "Show me order 1042"))[-1] == "done"
    state = await http._transport.app.state.graph.aget_state(
        {"configurable": {"thread_id": "m_acme:t1"}}
    )
    assert len(state.values["messages"]) == 6


def test_client_reads_sse_lines():
    lines = iter([": ping", "event: token", 'data: {"text": "hi"}', "", "event: done", "data: {}"])
    assert list(parse_events(lines)) == [("token", {"text": "hi"}), ("done", {})]


# --- authentication and fail-closed (R4c) --------------------------------------------


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 401),
        ({"Authorization": "Basic abc"}, 401),
        ({"Authorization": "Bearer not-a-real-token"}, 401),
        (auth(ORPHAN), 403),
    ],
)
async def test_bad_callers_are_refused_before_the_model_runs(a2a, store, headers, status):
    http = make_http(NeverCalledModel(), a2a, store)
    body = {"message": "Show me order 1042", "thread_id": "t1"}
    assert (await http.post("/chat", json=body, headers=headers)).status_code == status
    assert (await http.post("/chat/t1/approve", headers=headers)).status_code == status
    assert (await http.post("/chat/t1/reject", headers=headers)).status_code == status


async def test_store_outage_fails_closed(a2a):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    http = make_http(NeverCalledModel(), a2a, httpx.MockTransport(handler))
    assert (await chat(http, "Show me order 1042")).status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"message": "", "thread_id": "t1"},
        {"message": "hi"},
        {"message": "hi", "thread_id": "bad id!"},
        {"message": "hi", "thread_id": "t1", "merchant": "m_globex"},
    ],
)
async def test_invalid_bodies_are_rejected(http, body):
    assert (await http.post("/chat", json=body, headers=auth(ACME))).status_code == 422


async def test_invalid_thread_id_in_path_is_rejected(http):
    assert (await http.post("/chat/bad%20id!/approve", headers=auth(ACME))).status_code == 422


async def test_token_never_appears_in_the_stream(http):
    response = await chat(http, "Show me order 1042")
    assert ACME not in response.text


# --- tenant isolation (R3) -------------------------------------------------------------


async def test_merchant_a_cannot_read_merchant_b_over_http(http):
    assert "not found in your store" in text_of(await chat(http, "Show me order 2001"))
    assert "m_globex" in text_of(await chat(http, "Show me order 2001", token=GLOBEX))


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_another_merchant_cannot_touch_a_proposal(http, a2a, decision):
    await chat(http, "Please refund order 1042", token=ACME, thread="shared")

    assert (await decide(http, decision, token=GLOBEX, thread="shared")).status_code == 409
    assert await status_of(a2a, 1042) == "PAID"

    # Acme's proposal is still waiting.
    response = await decide(http, "approve", token=ACME, thread="shared")
    assert "is now REFUNDED" in text_of(response)


async def test_same_thread_id_gives_each_merchant_its_own_conversation(http):
    await chat(http, "Please refund order 1042", token=ACME, thread="shared")
    # Acme's waiting proposal must not block Globex.
    assert names(await chat(http, "Hello there", token=GLOBEX, thread="shared")) == [
        "token",
        "done",
    ]


# --- approval over HTTP (R5) ---------------------------------------------------------


async def test_refund_streams_a_proposal_and_changes_nothing(http, a2a):
    response = await chat(http, "Please refund order 1042")
    assert names(response) == ["tool_started", "tool_finished", "proposal"]
    proposal = events(response)[-1][1]
    assert proposal["order_id"] == 1042
    assert proposal["merchant"] == "m_acme"
    assert proposal["amount_to_customer"] == 59.90
    assert await status_of(a2a, 1042) == "PAID"


async def test_approve_over_http(http, a2a):
    await chat(http, "Please refund order 1042")
    response = await decide(http, "approve")
    assert names(response) == ["token", "done"]
    assert text_of(response) == "Order 1042 is now REFUNDED. 59.90 was returned to the customer."
    assert await status_of(a2a, 1042) == "REFUNDED"


async def test_reject_over_http(http, a2a):
    await chat(http, "Please refund order 1042")
    response = await decide(http, "reject")
    assert text_of(response) == "Okay, I did not refund order 1042. Nothing was changed."
    assert await status_of(a2a, 1042) == "PAID"


async def test_new_message_is_refused_while_a_proposal_waits(http, a2a):
    await chat(http, "Please refund order 1042")
    assert (await chat(http, "Show me order 1001")).status_code == 409
    assert "is now REFUNDED" in text_of(await decide(http, "approve"))


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_decision_with_nothing_waiting_is_refused(http, decision):
    assert (await decide(http, decision)).status_code == 409


async def test_approving_twice_in_a_row_is_refused(http):
    await chat(http, "Please refund order 1042")
    await decide(http, "approve")
    assert (await decide(http, "approve")).status_code == 409


async def test_double_click_on_approve_refunds_once(http, a2a):
    await chat(http, "Please refund order 1042")
    responses = await asyncio.gather(*[decide(http, "approve") for _ in range(5)])

    outcomes = [event for r in responses if r.status_code == 200 for event in events(r)]
    assert sum("is now REFUNDED" in e[1].get("text", "") for e in outcomes) == 1
    assert not any("could not confirm" in e[1].get("text", "") for e in outcomes)
    others = {e[1]["message"] for e in outcomes if e[0] == "error"}
    assert others <= {BUSY, OUT_OF_DATE}
    assert await status_of(a2a, 1042) == "REFUNDED"


async def test_second_request_is_refused_while_a_turn_is_running(a2a, store):
    http = make_http(SlowModel(), a2a, store)
    first, second = await asyncio.gather(chat(http, "Hello there"), chat(http, "Hello there"))

    results = sorted([names(first), names(second)])
    assert results == [["error"], ["token", "done"]]
    busy = first if names(first) == ["error"] else second
    assert events(busy) == [("error", {"message": BUSY})]


async def test_state_is_checked_again_once_the_conversation_is_held(http, a2a, monkeypatch):
    await chat(http, "Please refund order 1042")
    graph = http._transport.app.state.graph
    real_get_state = graph.aget_state
    answers = [SimpleNamespace(tasks=[])]  # the first check sees an out-of-date state

    async def stale_then_real(config, **kwargs):
        return answers.pop() if answers else await real_get_state(config, **kwargs)

    monkeypatch.setattr(graph, "aget_state", stale_then_real)
    response = await chat(http, "Show me order 1001")

    assert events(response) == [("error", {"message": OUT_OF_DATE})]
    assert "is now REFUNDED" in text_of(await decide(http, "approve"))  # proposal survived


# --- failures become events ----------------------------------------------------------


async def test_model_failure_becomes_an_error_event(a2a, store):
    http = make_http(BrokenModel(), a2a, store)
    response = await chat(http, "Show me order 1042")
    assert events(response) == [("error", {"message": UNAVAILABLE})]
    assert "private-provider-detail" not in response.text


async def test_conversation_still_works_after_a_failure(a2a, store):
    broken = BrokenModel()
    graph = build_graph(broken, a2a)
    api = create_app(graph=graph, store_url=STORE, store_transport=store)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://chat.test")
    await chat(http, "Show me order 1042")

    api.state.graph = build_graph(ScriptedChatModel(), a2a, checkpointer=graph.checkpointer)
    assert names(await chat(http, "Show me order 1042"))[-1] == "done"


async def test_endless_loop_becomes_an_error_event(a2a, store):
    http = make_http(LoopingModel(), a2a, store)
    assert events(await chat(http, "Show me order 1042"))[-1] == (
        "error",
        {"message": TOO_MANY_STEPS},
    )


async def test_healthz_needs_no_token(http):
    response = await http.get("/healthz")
    assert response.status_code == 200
    assert json.loads(response.text) == {"ok": True}
