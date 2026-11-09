"""R5 — human approval: a refund or cancel needs an explicit decision.

Mandatory R8 test: proposal, approve, reject. The sub-agent and store are real and
run in-process; only the model's decisions are scripted.
"""

import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orchestrator.a2a_client import A2AClient
from orchestrator.fake_llm import ScriptedChatModel
from orchestrator.graph import APPROVE, REJECT, build_graph
from orchestrator.tools import RunContext
from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
STORE = "http://store.test"
ACME = "tok_acme_alice"
GLOBEX = "tok_globex_bob"


@pytest.fixture
async def client():
    store = httpx.ASGITransport(app=create_store())
    app = create_agent(store_url=STORE, public_url=AGENT, store_transport=store)
    return await A2AClient.discover(AGENT, transport=httpx.ASGITransport(app=app))


@pytest.fixture
def saver():
    return InMemorySaver()


@pytest.fixture
def graph(client, saver):
    return build_graph(ScriptedChatModel(), client, checkpointer=saver)


@pytest.fixture
def thread():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


async def ask(graph, thread, text, token=ACME):
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]}, thread, context=RunContext(auth_token=token)
    )


async def decide(graph, thread, decision, token=ACME):
    context = RunContext(auth_token=token) if token else None
    return await graph.ainvoke(Command(resume=decision), thread, context=context)


async def order(client, order_id, token=ACME):
    return (await client.call("get_order", {"order_id": order_id}, token))["data"]


class TwoPreviewsModel(ScriptedChatModel):
    """Asks for two refunds in one reply, as real models sometimes do."""

    def _generate(self, messages, *args, **kwargs):
        calls = [
            {
                "name": "preview_action",
                "args": {"action": "refund", "order_id": order_id},
                "id": f"c{order_id}",
                "type": "tool_call",
            }
            for order_id in (1042, 1001)
        ]
        reply = AIMessage(content="", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=reply)])


# --- the proposal ------------------------------------------------------------------


async def test_refund_request_pauses_with_a_full_proposal(graph, thread, client):
    result = await ask(graph, thread, "Please refund order 1042")

    proposal = result["__interrupt__"][0].value
    assert proposal["action"] == "refund"
    assert proposal["order_id"] == 1042
    assert proposal["merchant"] == "m_acme"
    assert proposal["current_status"] == "PAID"
    assert proposal["new_status"] == "REFUNDED"
    assert proposal["amount_to_customer"] == 59.90

    assert (await graph.aget_state(thread)).next == ("approval",)
    assert (await order(client, 1042))["status"] == "PAID"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("refund order 1007", "cannot be refunded"),  # already cancelled
        ("cancel order 1003", "cannot be cancelled"),  # already delivered
        ("refund order 2001", "not found"),  # another merchant's order
    ],
)
async def test_impossible_action_is_explained_without_a_pause(graph, thread, text, error):
    result = await ask(graph, thread, text)
    assert "__interrupt__" not in result
    assert error in result["messages"][-1].content


# --- approve -----------------------------------------------------------------------


async def test_approve_refunds_the_order(graph, thread, client):
    await ask(graph, thread, "Please refund order 1042")
    result = await decide(graph, thread, APPROVE)

    assert result["messages"][-1].content == (
        "Order 1042 is now REFUNDED. 59.90 was returned to the customer."
    )
    assert (await order(client, 1042))["status"] == "REFUNDED"
    assert (await graph.aget_state(thread)).next == ()


async def test_approve_cancels_an_unpaid_order_and_returns_nothing(graph, thread, client):
    await ask(graph, thread, "cancel order 1005")
    result = await decide(graph, thread, APPROVE)

    assert result["messages"][-1].content == (
        "Order 1005 is now CANCELLED. 0.00 was returned to the customer."
    )
    assert (await order(client, 1005))["status"] == "CANCELLED"


# --- reject and anything that is not an approval ------------------------------------


async def test_reject_changes_nothing(graph, thread, client):
    await ask(graph, thread, "Please refund order 1042")
    result = await decide(graph, thread, REJECT)

    assert result["messages"][-1].content == (
        "Okay, I did not refund order 1042. Nothing was changed."
    )
    assert (await order(client, 1042))["status"] == "PAID"
    assert (await graph.aget_state(thread)).next == ()


@pytest.mark.parametrize("decision", ["yes", "APPROVE", {"decision": "approve"}])
async def test_only_an_exact_approve_counts(graph, thread, client, decision):
    await ask(graph, thread, "Please refund order 1042")
    await decide(graph, thread, decision)
    assert (await order(client, 1042))["status"] == "PAID"


# --- the approver's token ------------------------------------------------------------


async def test_another_merchant_cannot_approve(graph, thread, client):
    await ask(graph, thread, "Please refund order 1042", token=ACME)
    result = await decide(graph, thread, APPROVE, token=GLOBEX)

    assert result["messages"][-1].content.startswith("I could not confirm the refund")
    assert (await order(client, 1042))["status"] == "PAID"


async def test_approve_without_a_token_fails_closed(graph, thread, client):
    await ask(graph, thread, "Please refund order 1042")
    result = await decide(graph, thread, APPROVE, token=None)

    assert "not authenticated" in result["messages"][-1].content
    assert (await order(client, 1042))["status"] == "PAID"


async def test_token_is_not_saved_during_or_after_a_pause(graph, thread, saver):
    await ask(graph, thread, "Please refund order 1042")
    assert ACME not in repr(saver.storage)
    assert ACME not in repr(saver.writes)

    await decide(graph, thread, APPROVE)
    assert saver.storage  # guards against passing because nothing was saved
    assert ACME not in repr(saver.storage)
    assert ACME not in repr(saver.writes)


# --- things that can go wrong ---------------------------------------------------------


async def test_order_changed_while_waiting_is_reported(graph, thread, client):
    await ask(graph, thread, "Please refund order 1042")
    await client.call("refund_order", {"order_id": 1042}, ACME)  # someone else got there first

    result = await decide(graph, thread, APPROVE)
    assert result["messages"][-1].content.startswith("I could not confirm the refund")


async def test_second_approve_does_not_refund_twice(graph, thread, client):
    await ask(graph, thread, "Please refund order 1042")
    await decide(graph, thread, APPROVE)
    before = await order(client, 1042)

    await decide(graph, thread, APPROVE)
    assert await order(client, 1042) == before
    assert before["refunded_amount"] == 59.90


async def test_two_proposals_change_nothing_until_both_are_decided(client, thread):
    graph = build_graph(TwoPreviewsModel(), client)
    first = await ask(graph, thread, "refund both")
    assert first["__interrupt__"][0].value["order_id"] == 1042

    second = await decide(graph, thread, APPROVE)
    assert second["__interrupt__"][0].value["order_id"] == 1001
    assert (await order(client, 1042))["status"] == "PAID"

    final = await decide(graph, thread, REJECT)
    assert (await order(client, 1042))["status"] == "REFUNDED"
    assert (await order(client, 1001))["status"] == "PAID"
    assert "Order 1042 is now REFUNDED" in final["messages"][-1].content
    assert "did not refund order 1001" in final["messages"][-1].content


async def test_a_new_graph_can_resume_a_saved_pause(graph, thread, saver, client):
    await ask(graph, thread, "Please refund order 1042")

    after_restart = build_graph(ScriptedChatModel(), client, checkpointer=saver)
    await decide(after_restart, thread, APPROVE)
    assert (await order(client, 1042))["status"] == "REFUNDED"
