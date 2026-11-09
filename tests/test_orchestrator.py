"""R1, R3, R4 — full turns: graph -> A2A client -> sub-agent -> store.

Everything except the model's decisions is real, and runs in-process.
"""

import asyncio
import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from common.settings import Settings
from orchestrator.a2a_client import A2AClient
from orchestrator.fake_llm import ScriptedChatModel
from orchestrator.graph import SYSTEM_PROMPT, build_agent, build_graph
from orchestrator.tools import RunContext, build_tools
from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
STORE = "http://store.test"
ACME = "tok_acme_alice"
GLOBEX = "tok_globex_bob"


@pytest.fixture
def agent_transport():
    store = httpx.ASGITransport(app=create_store())
    app = create_agent(store_url=STORE, public_url=AGENT, store_transport=store)
    return httpx.ASGITransport(app=app)


@pytest.fixture
async def client(agent_transport):
    return await A2AClient.discover(AGENT, transport=agent_transport)


@pytest.fixture
def graph(client):
    return build_graph(ScriptedChatModel(), client)


async def ask(graph, text, token=ACME, config=None):
    config = config or {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = RunContext(auth_token=token) if token else None
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]}, config, context=context
    )
    return result["messages"]


def tool_call(name, args):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": "c1", "type": "tool_call"}]
    )


class RecordingModel(ScriptedChatModel):
    """Behaves normally and keeps everything it was shown."""

    seen: list

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class RogueModel(ScriptedChatModel):
    """Asks for a refund directly, as a manipulated model might."""

    def _generate(self, messages, *args, **kwargs):
        done = isinstance(messages[-1], ToolMessage)
        reply = AIMessage(content="done") if done else tool_call("refund_order", {"order_id": 1042})
        return ChatResult(generations=[ChatGeneration(message=reply)])


class LoopingModel(ScriptedChatModel):
    """Never stops calling tools."""

    def _generate(self, messages, *args, **kwargs):
        reply = tool_call("get_order", {"order_id": 1042})
        return ChatResult(generations=[ChatGeneration(message=reply)])


# --- tools built from the card --------------------------------------------------


def test_tools_come_from_the_card_and_exclude_mutating_skills(client):
    tools = {tool.name: tool for tool in build_tools(client)}
    assert set(tools) == {"list_orders", "get_order", "preview_action"}
    assert tools["get_order"].description == client.card.skill("get_order").description


def test_model_cannot_see_or_set_a_token(client):
    for tool in build_tools(client):
        assert not any("token" in arg or "auth" in arg for arg in tool.args)


# --- a full turn ----------------------------------------------------------------


async def test_full_turn_calls_the_tool_and_answers_from_it(graph):
    messages = await ask(graph, "Show me order 1042")
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert messages[1].tool_calls[0]["name"] == "get_order"
    assert messages[2].content.startswith("<untrusted_data>")
    assert "Maria Lopez" in messages[-1].content


async def test_list_with_filters_returns_the_right_orders(graph):
    messages = await ask(graph, "Which orders are paid but not shipped for 3 days?")
    answer = messages[-1].content
    assert all(f'"id": {order_id}' in answer for order_id in (1001, 1006, 1042))
    assert '"id": 1002' not in answer


async def test_no_tool_is_called_when_none_is_needed(graph):
    messages = await ask(graph, "Hello there")
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]


async def test_legacy_not_found_is_treated_as_a_failure(graph):
    messages = await ask(graph, "Show me order 9999")
    assert messages[-1].content.startswith("I couldn't complete that")
    assert "not found" in messages[-1].content


async def test_build_agent_discovers_the_sub_agent_from_its_card(agent_transport):
    settings = Settings(_env_file=None, llm_provider="fake", orders_agent_url=AGENT)
    agent = await build_agent(settings=settings, transport=agent_transport)
    messages = await ask(agent, "Show me order 1042")
    assert "Maria Lopez" in messages[-1].content


# --- R3: tenant isolation and token handling ------------------------------------


async def test_merchant_a_cannot_read_merchant_b(graph):
    refused = await ask(graph, "Show me order 2001", token=ACME)
    assert refused[-1].content == (
        "I couldn't complete that: Order 2001 was not found in your store."
    )
    allowed = await ask(graph, "Show me order 2001", token=GLOBEX)
    assert '"merchant": "m_globex"' in allowed[-1].content


async def test_concurrent_users_on_one_graph_stay_separate(graph):
    runs = [ask(graph, "List all orders", token=t) for t in (ACME, GLOBEX) * 5]
    results = await asyncio.gather(*runs)
    for i, messages in enumerate(results):
        mine, other = ("m_acme", "m_globex") if i % 2 == 0 else ("m_globex", "m_acme")
        assert mine in messages[-1].content
        assert other not in messages[-1].content


async def test_token_is_sent_unchanged_on_each_call(client):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        body = httpx.Response(200, content=request.content).json()
        result = {"ok": True, "data": [], "error": None}
        message = {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"data": result}]}
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"message": message}}
        )

    spy = A2AClient(client.card, client.sdk_card, transport=httpx.MockTransport(handler))
    graph = build_graph(ScriptedChatModel(), spy)
    await ask(graph, "List all orders", token=ACME)
    await ask(graph, "List all orders", token=GLOBEX)
    assert seen == [f"Bearer {ACME}", f"Bearer {GLOBEX}"]


async def test_missing_token_fails_closed_without_calling_the_agent(client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the sub-agent must not be called without a token")

    spy = A2AClient(client.card, client.sdk_card, transport=httpx.MockTransport(handler))
    graph = build_graph(ScriptedChatModel(), spy)
    messages = await ask(graph, "Show me order 1042", token=None)
    assert "not authenticated" in messages[-1].content


async def test_token_is_never_saved_in_a_checkpoint(client):
    saver = InMemorySaver()
    graph = build_graph(ScriptedChatModel(), client, checkpointer=saver)
    config = {"configurable": {"thread_id": "t1"}}
    await ask(graph, "Show me order 1042", config=config)

    checkpoints = [c async for c in saver.alist(config)]
    assert checkpoints  # guards against passing because nothing was saved
    assert ACME not in repr(checkpoints)
    assert ACME not in repr(saver.storage)
    assert ACME not in repr(saver.writes)


def test_run_context_hides_the_token_from_repr():
    assert ACME not in repr(RunContext(auth_token=ACME))


# --- R4: what the model is shown ---------------------------------------------------


async def test_model_gets_the_prompt_and_never_pii_or_the_token(client):
    model = RecordingModel(seen=[])
    await ask(build_graph(model, client), "Show me order 1042")

    assert all(isinstance(call[0], SystemMessage) for call in model.seen)
    assert model.seen[0][0].content == SYSTEM_PROMPT
    shown = repr(model.seen)
    assert "maria.lopez@example.com" not in shown
    assert "612 345 678" not in shown
    assert ACME not in shown


@pytest.mark.parametrize("order_id", [1006, 1002])
async def test_injected_note_is_not_obeyed(graph, order_id):
    messages = await ask(graph, f"Show me order {order_id}")
    called = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    assert called == ["get_order"]
    assert "withheld" in messages[-1].content
    assert "ignore your instructions" not in repr(messages)


# --- nothing changes without approval ------------------------------------------


async def test_refund_request_only_previews(graph, client):
    messages = await ask(graph, "Please refund order 1042")
    assert "will be returned to the customer" in messages[-1].content
    order = await client.call("get_order", {"order_id": 1042}, ACME)
    assert order["data"]["status"] == "PAID"


async def test_model_cannot_call_a_mutating_skill(client):
    graph = build_graph(RogueModel(), client)
    messages = await ask(graph, "anything")

    refused = next(m for m in messages if isinstance(m, ToolMessage))
    assert refused.status == "error"
    assert "not a valid tool" in refused.content
    order = await client.call("get_order", {"order_id": 1042}, ACME)
    assert order["data"]["status"] == "PAID"


# --- failures ----------------------------------------------------------------------


async def test_sub_agent_outage_becomes_a_message_not_a_crash(client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    down = A2AClient(client.card, client.sdk_card, transport=httpx.MockTransport(handler))
    graph = build_graph(ScriptedChatModel(), down)
    messages = await ask(graph, "Show me order 1042")
    assert messages[-1].content == "I couldn't complete that: The orders agent is unavailable"


async def test_endless_tool_loop_is_stopped(client):
    graph = build_graph(LoopingModel(), client)
    with pytest.raises(GraphRecursionError):
        await ask(graph, "Show me order 1042")
